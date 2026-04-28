"""
ESE 6510 Phase 2 — Team deployment policy.

This file mirrors the TA-provided reference (`controller_simple_policy_vineet_reference.py.bak`)
but adapts the Actor architecture to match our Isaac Lab training configuration.

Observation layout (36D, same as the reference):
    [ lin_vel_b               (3)   linear velocity in body frame
      rot_matrix_wb flatten   (9)   drone body-to-world rotation matrix
      current_gate_corners_b  (12)  current gate's 4 corners in body frame
      next_gate_corners_b     (12)  next gate's 4 corners in body frame ]

Actor network (differs from reference):
    Linear(36->128) -> ELU -> Linear(128->128) -> ELU -> Linear(128->4) -> Tanh

This matches `src/isaac_quad_sim2real/tasks/race/config/crazyflie/agents/rsl_rl_ppo_cfg.py`:
    actor_hidden_dims=[128, 128], activation="elu"

Network checkpoint: the best_model.pt from our Isaac Lab training run. The
rsl_rl `ActorCritic.state_dict()` uses the exact same key layout (`actor.0.*`,
`actor.2.*`, `actor.4.*`), so the "actor in k" filter below recovers our actor
weights directly.
"""

import numpy as np
import torch
import torch.nn as nn
from scipy.spatial.transform import Rotation as R


class Actor(nn.Module):
    """MLP actor that mirrors rsl_rl's ActorCritic actor sub-module.

    Given ``actor_hidden_dims=[h1, h2]``, the layer order is:
        Linear(obs->h1) -> act -> Linear(h1->h2) -> act -> Linear(h2->num_actions) -> Tanh

    which is identical to rsl_rl's key ordering (`actor.0`, `actor.2`,
    `actor.4`), so we can load the weights directly from a rsl_rl checkpoint.
    """

    def __init__(self, mlp_input_dim, actor_hidden_dims, num_actions, activation):
        super(Actor, self).__init__()

        actor_layers = []
        actor_layers.append(nn.Linear(mlp_input_dim, actor_hidden_dims[0]))
        actor_layers.append(activation())
        for layer_index in range(len(actor_hidden_dims)):
            if layer_index == len(actor_hidden_dims) - 1:
                actor_layers.append(nn.Linear(actor_hidden_dims[layer_index], num_actions))
                actor_layers.append(nn.Tanh())
            else:
                actor_layers.append(
                    nn.Linear(actor_hidden_dims[layer_index], actor_hidden_dims[layer_index + 1])
                )
                actor_layers.append(activation())
        self.actor = nn.Sequential(*actor_layers)

    def forward(self, x):
        return self.actor(x)


class SimpleRacingPolicy:
    """Deployment wrapper around our Isaac Lab-trained circle/powerloop policy."""

    # ------------------------------------------------------------------
    # Architecture constants — keep in sync with rsl_rl_ppo_cfg.py
    # ------------------------------------------------------------------
    ACTOR_HIDDEN_DIMS = [128, 128]
    ACTOR_ACTIVATION = nn.ELU

    def __init__(self, vehicle, model_path, params, device="cpu", use_cond=False):
        """
        Args:
            vehicle: Quadrotor parameters (unused here, kept for interface
                compatibility with the controller node).
            model_path: Absolute path to our trained rsl_rl checkpoint
                (typically ``best_model.pt`` from an Isaac Lab log dir).
            params: Dict with at least the following keys:
                ``waypoints``       (G, 6) float32  — [x,y,z,roll,pitch,yaw]
                ``waypoints_quat``  (G, 4) float32  — quaternion (w, x, y, z)
                ``gate_side``       float            — gate side length (m)
                ``initial_waypoint`` int             — starting gate index
                ``max_roll_br``, ``max_pitch_br``, ``max_yaw_br`` (rad/s)
                ``pass_gate_thr`` (optional)         — default 0.10 m
            device: Torch device for inference, e.g. ``"cpu"`` or ``"cuda"``.
            use_cond: Present for compatibility with the TA's controller_node
                call signature. We do NOT use conditional policies in our
                training, so any truthy value is rejected at construction.
        """
        if use_cond:
            raise ValueError(
                "controller_simple_policy.py (ours): use_cond=True is not supported — "
                "our rsl_rl training does not produce a conditional actor."
            )

        self.quadrotor = vehicle
        self.device = torch.device(device)
        self.obs_dim = 3 + 9 + 12 + 12  # 36
        self.action_dim = 4

        self.waypoints = params["waypoints"]
        self.waypoints_quat = params["waypoints_quat"]

        self.gate_side = params["gate_side"]
        d = self.gate_side / 2
        self.local_square = np.array(
            [
                [0,  d,  d],
                [0, -d,  d],
                [0, -d, -d],
                [0,  d, -d],
            ],
            dtype=np.float32,
        )

        # ----- Build actor and load weights from rsl_rl checkpoint ---------
        self.model = Actor(
            self.obs_dim,
            self.ACTOR_HIDDEN_DIMS,
            self.action_dim,
            self.ACTOR_ACTIVATION,
        ).to(self.device)

        checkpoint = torch.load(model_path, map_location=self.device)
        # rsl_rl's ActorCritic state dict has keys like:
        #   actor.0.weight, actor.0.bias, actor.2.weight, ..., critic.*, std
        # Filter to actor keys only; strict=True catches shape mismatches.
        actor_state_dict = {
            k: v for k, v in checkpoint["model_state_dict"].items() if "actor" in k
        }
        self.model.load_state_dict(actor_state_dict, strict=True)
        self.model = torch.compile(self.model)
        self.model.eval()

        # Warm-up compile before the first race step.
        with torch.no_grad():
            dummy_obs = torch.zeros(self.obs_dim, dtype=torch.float32, device=self.device)
            _ = self.model(dummy_obs)

        # ----- Race state ---------------------------------------------------
        self.idx_wp = params["initial_waypoint"]

        # Body-rate caps (rad/s). Must match our training config:
        # body_rate_scale_xy = 100 deg/s, body_rate_scale_z = 200 deg/s.
        self.max_roll_br = params["max_roll_br"]
        self.max_pitch_br = params["max_pitch_br"]
        self.max_yaw_br = params["max_yaw_br"]

        # Gate pass threshold in the gate's body-frame x coordinate.
        self.pass_gate_thr = params.get("pass_gate_thr", 0.10)
        self.prev_gate_x = None

    # ----------------------------------------------------------------------
    # Main inference step
    # ----------------------------------------------------------------------
    def update(self, state):
        """Run one policy inference step.

        Args:
            state: Dict filled by ``controller_utils.single_update``, containing
                ``x``   (3,)   world position
                ``v_b`` (3,)   linear velocity in body frame
                ``R``   (3,3)  body-to-world rotation matrix

        Returns:
            control_input: Dict with keys
                ``cmd_thrust`` — normalized thrust in [0, 1]
                ``cmd_w``      — ndarray of body rate targets [roll, pitch, yaw]
            obs: Torch tensor of the 36D observation fed to the policy.
        """
        pos_drone = state["x"]
        lin_vel_drone = state["v_b"]
        rot_drone = state["R"]

        # ----- Gate pass detection & idx_wp advancement --------------------
        curr_idx = self.idx_wp
        next_idx = (self.idx_wp + 1) % self.waypoints.shape[0]

        wp_curr_pos = self.waypoints[curr_idx, :3]
        wp_next_pos = self.waypoints[next_idx, :3]
        quat_curr = self.waypoints_quat[curr_idx, :]
        quat_next = self.waypoints_quat[next_idx, :]
        rot_curr = R.from_quat(quat_curr, scalar_first=True).as_matrix()
        rot_next = R.from_quat(quat_next, scalar_first=True).as_matrix()

        pose_drone_wrt_gate = self._subtract_frame_transforms(wp_curr_pos, rot_curr, pos_drone)
        current_gate_x = pose_drone_wrt_gate[0]
        within_gate_opening = (
            np.abs(pose_drone_wrt_gate[1]) < self.gate_side / 2
            and np.abs(pose_drone_wrt_gate[2]) < self.gate_side / 2
        )
        x_crossed_gate_plane = (
            self.prev_gate_x is not None
            and self.prev_gate_x > self.pass_gate_thr
            and current_gate_x <= self.pass_gate_thr
        )

        if x_crossed_gate_plane and within_gate_opening:
            self.idx_wp = (self.idx_wp + 1) % self.waypoints.shape[0]

            # Keep the post-crossing observation aligned with the new current gate.
            curr_idx = self.idx_wp
            next_idx = (self.idx_wp + 1) % self.waypoints.shape[0]
            wp_curr_pos = self.waypoints[curr_idx, :3]
            wp_next_pos = self.waypoints[next_idx, :3]
            quat_curr = self.waypoints_quat[curr_idx, :]
            quat_next = self.waypoints_quat[next_idx, :]
            rot_curr = R.from_quat(quat_curr, scalar_first=True).as_matrix()
            rot_next = R.from_quat(quat_next, scalar_first=True).as_matrix()
            pose_drone_wrt_gate = self._subtract_frame_transforms(wp_curr_pos, rot_curr, pos_drone)

        # Track gate-frame x for the active gate so same-direction gate pairs
        # still require a real crossing from the valid entry side.
        self.prev_gate_x = pose_drone_wrt_gate[0]

        # ----- Build 36D observation in body frame -------------------------
        verts_curr = self.local_square @ rot_curr.T + wp_curr_pos
        verts_next = self.local_square @ rot_next.T + wp_next_pos

        waypoint_pos_b_curr = self._subtract_frame_transforms(
            pos_drone, rot_drone, verts_curr
        ).reshape(4, 3)
        waypoint_pos_b_next = self._subtract_frame_transforms(
            pos_drone, rot_drone, verts_next
        ).reshape(4, 3)

        obs_parts = [
            torch.from_numpy(lin_vel_drone).float().flatten(),
            torch.from_numpy(rot_drone).float().flatten(),
            torch.from_numpy(waypoint_pos_b_curr).float().flatten(),
            torch.from_numpy(waypoint_pos_b_next).float().flatten(),
        ]
        obs = torch.cat(obs_parts).float().to(self.device)

        # ----- Actor forward pass & action decoding ------------------------
        with torch.no_grad():
            actions = self.model(obs).squeeze(0).cpu().numpy()
        actions = np.clip(actions, -1, 1)

        # Thrust: training maps actions[0] in [-1, 1] to [0, 1] fraction of
        # max thrust via ((a + 1) / 2). See quadcopter_env.py:_pre_physics_step.
        cmd_thrust = 0.5 * (actions[0] + 1.0)

        # Body rates: training uses body_rate_scale_xy/z to map [-1,1] to rad/s.
        roll_br = actions[1] * self.max_roll_br
        pitch_br = actions[2] * self.max_pitch_br
        yaw_br = actions[3] * self.max_yaw_br

        control_input = {
            "cmd_thrust": cmd_thrust,
            "cmd_w": np.array([roll_br, pitch_br, yaw_br]),
        }
        return control_input, obs

    # ----------------------------------------------------------------------
    # Frame helpers (numpy, identical to the TA reference)
    # ----------------------------------------------------------------------
    def _subtract_frame_transforms(self, pos, rot, pos_des):
        """Transform world-frame point(s) into the body frame defined by (pos, rot).

        ``rot`` is the body-to-world rotation. For a single point the result is
        ``rot.T @ (pos_des - pos)``; for a (N, 3) batch of row vectors we use
        the equivalent ``(pos_des - pos) @ rot``.
        """
        if pos_des.ndim == 1:
            return rot.T @ (pos_des - pos)
        elif pos_des.ndim == 2:
            return (pos_des - pos) @ rot
