# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Modular strategy classes for quadcopter environment rewards, observations, and resets."""

from __future__ import annotations

import torch
import numpy as np
from typing import TYPE_CHECKING, Dict, Optional, Tuple

from isaaclab.utils.math import subtract_frame_transforms, quat_from_euler_xyz, euler_xyz_from_quat, wrap_to_pi, matrix_from_quat, quat_apply_inverse

if TYPE_CHECKING:
    from .quadcopter_env import QuadcopterEnv

D2R = np.pi / 180.0
R2D = 180.0 / np.pi


class DefaultQuadcopterStrategy:
    """Default strategy implementation for quadcopter environment."""

    def __init__(self, env: QuadcopterEnv):
        """Initialize the default strategy.

        Args:
            env: The quadcopter environment instance.
        """
        self.env = env
        self.device = env.device
        self.num_envs = env.num_envs
        self.cfg = env.cfg

        # Initialize episode sums for logging if in training mode
        if self.cfg.is_train and hasattr(env, 'rew'):
            keys = [key.split("_reward_scale")[0] for key in env.rew.keys() if key != "death_cost"]
            self._episode_sums = {
                key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
                for key in keys
            }

        self._num_gates = int(self.env._waypoints.shape[0])
        self._gate_pass_counts_total = torch.zeros(self._num_gates, dtype=torch.long, device=self.device)
        self._gate_pass_counts_since_log = torch.zeros(self._num_gates, dtype=torch.long, device=self.device)
        self._episode_gate_pass_counts = torch.zeros(
            self.num_envs, self._num_gates, dtype=torch.float, device=self.device
        )
        self._progress_distance_scale = 3.0

        # Domain randomization of physics parameters for sim-to-real transfer
        # Evaluation alters TWR ±5%, aero drag 0.5-2x, PID ±15%/30%

        # TWR: ±5%
        twr_factor = 1.0 + (torch.rand(self.num_envs, device=self.device) * 0.10 - 0.05)
        self.env._thrust_to_weight[:] = self.env._twr_value * twr_factor

        # Aero drag: 0.5x to 2.0x
        aero_xy_factor = 0.5 + torch.rand(self.num_envs, device=self.device) * 1.5
        aero_z_factor = 0.5 + torch.rand(self.num_envs, device=self.device) * 1.5
        self.env._K_aero[:, :2] = self.env._k_aero_xy_value * aero_xy_factor.unsqueeze(1)
        self.env._K_aero[:, 2] = self.env._k_aero_z_value * aero_z_factor

        # PID roll/pitch P and I gains: ±15%
        rp_factor = 1.0 + (torch.rand(self.num_envs, device=self.device) * 0.30 - 0.15)
        self.env._kp_omega[:, :2] = self.env._kp_omega_rp_value * rp_factor.unsqueeze(1)
        self.env._ki_omega[:, :2] = self.env._ki_omega_rp_value * rp_factor.unsqueeze(1)
        # PID roll/pitch D gain: ±30%
        rp_d_factor = 1.0 + (torch.rand(self.num_envs, device=self.device) * 0.60 - 0.30)
        self.env._kd_omega[:, :2] = self.env._kd_omega_rp_value * rp_d_factor.unsqueeze(1)

        # PID yaw P and I gains: ±15%
        y_factor = 1.0 + (torch.rand(self.num_envs, device=self.device) * 0.30 - 0.15)
        self.env._kp_omega[:, 2] = self.env._kp_omega_y_value * y_factor
        self.env._ki_omega[:, 2] = self.env._ki_omega_y_value * y_factor
        # PID yaw D gain: ±30%
        y_d_factor = 1.0 + (torch.rand(self.num_envs, device=self.device) * 0.60 - 0.30)
        self.env._kd_omega[:, 2] = self.env._kd_omega_y_value * y_d_factor

        # Motor time constant: keep fixed
        self.env._tau_m[:] = self.env._tau_m_value

        # Tracking buffers for reward computation
        self._prev_global_progress = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self._prev_in_entry_half_plane = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._entry_half_plane_rewarded = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._prev_step_actions = torch.zeros(self.num_envs, 4, device=self.device)

        # Reward only tight same-direction gate pairs where the policy must first move back
        # to the correct entry side before a valid pass can happen. In the current powerloop
        # track this isolates gate 3 without hard-coding the gate index.
        gate_dirs = -self.env._normal_vectors
        prev_gate_dirs = torch.roll(gate_dirs, shifts=1, dims=0)
        seg_to_gate = self.env._waypoints[:, :3] - torch.roll(self.env._waypoints[:, :3], shifts=1, dims=0)
        seg_len = torch.linalg.norm(seg_to_gate, dim=1)
        same_direction = torch.sum(gate_dirs * prev_gate_dirs, dim=1) > 0.95
        self._half_plane_shape_gate_mask = same_direction & (seg_len < 2.0)

        # Pre-compute gate corners in world frame (static, computed once)
        gate_rot = matrix_from_quat(self.env._waypoints_quat)       # (G, 3, 3)
        local_c = self.env._local_square[0]                          # (4, 3)
        self._gate_corners_world = torch.einsum(
            'gij,cj->gci', gate_rot, local_c
        ) + self.env._waypoints[:, :3].unsqueeze(1)                  # (G, 4, 3)

    def _compute_global_progress(self) -> torch.Tensor:
        """Completed gates plus smooth progress toward the current target gate."""
        dist_to_gate = torch.linalg.norm(self.env._pose_drone_wrt_gate, dim=1)
        local_progress = 1.0 - torch.tanh(dist_to_gate / self._progress_distance_scale)
        return self.env._n_gates_passed.float() + local_progress

    def get_rewards(self) -> torch.Tensor:
        """Compute rewards with gate-crossing detection and racing-specific shaping."""

        n_gates = self.env._waypoints.shape[0]

        # ---- Gate passage detection via gate-frame x-axis crossing ----
        # Gate frame x-axis = gate normal (opposite to passing direction).
        # Drone approaches with x > 0, passes through when x crosses to <= 0.
        current_x = self.env._pose_drone_wrt_gate[:, 0]
        prev_x = self.env._prev_x_drone_wrt_gate
        x_crossed = (prev_x > 0) & (current_x <= 0)

        # Check drone is within the gate opening (gate_side=1.0 => ±0.5m)
        yz = self.env._pose_drone_wrt_gate[:, 1:3]
        in_bounds = (yz.abs() < 0.5).all(dim=1)

        gate_passed = x_crossed & in_bounds
        self.env._prev_x_drone_wrt_gate[:] = current_x.clone()

        # Advance waypoint for passed envs
        ids = torch.where(gate_passed)[0]
        if len(ids) > 0:
            passed_gate_indices = self.env._idx_wp[ids].clone().long()
            gate_count_delta = torch.bincount(passed_gate_indices, minlength=n_gates)
            self._gate_pass_counts_total += gate_count_delta
            self._gate_pass_counts_since_log += gate_count_delta
            self._episode_gate_pass_counts.index_put_(
                (ids, passed_gate_indices),
                torch.ones(len(ids), device=self.device),
                accumulate=True,
            )

            self.env._n_gates_passed[ids] += 1
            self.env._idx_wp[ids] = (self.env._idx_wp[ids] + 1) % n_gates
            self.env._desired_pos_w[ids, :3] = self.env._waypoints[self.env._idx_wp[ids], :3]
            self.env._prev_x_drone_wrt_gate[ids] = 1.0  # reset for next gate
            # Recompute gate-frame pose for the new gate so rewards & obs are consistent
            self.env._pose_drone_wrt_gate[ids], _ = subtract_frame_transforms(
                self.env._waypoints[self.env._idx_wp[ids], :3],
                self.env._waypoints_quat[self.env._idx_wp[ids], :],
                self.env._robot.data.root_link_pos_w[ids, :3]
            )
            self._prev_in_entry_half_plane[ids] = self.env._pose_drone_wrt_gate[ids, 0] > 0.15
            self._entry_half_plane_rewarded[ids] = self._prev_in_entry_half_plane[ids]

        # ---- Reward component tensors ----

        # 1. Gate pass (sparse)
        gate_pass = gate_passed.float()

        # 2. Global progress: completed gates plus smooth approach to the current gate.
        # Reward the increase in this global progress so hovering near one gate cannot farm reward.
        global_progress = self._compute_global_progress()
        progress = torch.clamp(global_progress - self._prev_global_progress, -1.0, 1.0)
        progress[gate_passed] = 0.0  # avoid double reward with gate_pass on crossing step
        self._prev_global_progress[:] = global_progress

        # 3. Speed bonus
        vel_world = self.env._robot.data.root_link_state_w[:, 7:10]
        speed = torch.linalg.norm(vel_world, dim=1)
        speed_reward = torch.tanh(speed / 3.0)

        # # 4.5 Entry-side shaping: reward returning to the valid entry half-plane for tight
        # # same-direction gate pairs (e.g. gate 2 -> gate 3 in powerloop).
        # current_gate_mask = self._half_plane_shape_gate_mask[self.env._idx_wp]
        # in_entry_half_plane = self.env._pose_drone_wrt_gate[:, 0] > 0.15
        # near_gate = torch.linalg.norm(self.env._pose_drone_wrt_gate[:, :2], dim=1) < 2.0
        # entered_entry_half_plane = (
        #     current_gate_mask
        #     & near_gate
        #     & (~self._entry_half_plane_rewarded)
        #     & (~self._prev_in_entry_half_plane)
        #     & in_entry_half_plane
        # ).float()
        # self._prev_in_entry_half_plane[:] = in_entry_half_plane
        # self._entry_half_plane_rewarded |= entered_entry_half_plane.bool()

        # 5. Crash detection feeds only termination/death_cost, not a dense reward term.
        contact_forces = self.env._contact_sensor.data.net_forces_w
        contact_detected = (torch.norm(contact_forces, dim=-1) > 1e-8).squeeze(1)
        mask = self.env.episode_length_buf > 100
        self.env._crashed = self.env._crashed + (contact_detected & mask).int()

        # 6. Action smoothness: penalize large action changes
        action_diff = self.env._actions - self._prev_step_actions
        action_smooth = torch.sum(action_diff ** 2, dim=1)
        self._prev_step_actions[:] = self.env._actions.clone()

        # # 7. Altitude penalty: penalize flying too low
        # altitude = self.env._robot.data.root_link_pos_w[:, 2]
        # altitude_penalty = torch.clamp(0.3 - altitude, min=0.0)

        # # 8. Lateral penalty: use updated gate-frame position (after gate advance)
        # yz_offset = torch.linalg.norm(self.env._pose_drone_wrt_gate[:, 1:3], dim=1)
        # lateral_penalty = torch.clamp(yz_offset - 0.45, min=0.0)

        # 9. Time penalty: constant per-step cost to encourage speed
        time_penalty = torch.ones(self.num_envs, device=self.device)

        if self.cfg.is_train:
            rew = self.env.rew
            rewards = {
                "gate_pass":     gate_pass * rew['gate_pass_reward_scale'],
                "progress":      progress * rew['progress_reward_scale'],
                "speed":         speed_reward * rew['speed_reward_scale'],
                # "entry_half_plane": entered_entry_half_plane * rew['entry_half_plane_reward_scale'],
                "action_smooth": action_smooth * rew['action_smooth_reward_scale'],
                # "altitude":      altitude_penalty * rew['altitude_reward_scale'],
                # "lateral":       lateral_penalty * rew['lateral_reward_scale'],
                "time":          time_penalty * rew['time_reward_scale'],
            }
            reward = torch.sum(torch.stack(list(rewards.values())), dim=0)
            reward = torch.where(self.env.reset_terminated,
                                 torch.ones_like(reward) * rew['death_cost'], reward)

            # Logging
            for key, value in rewards.items():
                self._episode_sums[key] += value
        else:
            reward = torch.zeros(self.num_envs, device=self.device)

        return reward

    def _gate_corners_to_body(self, gate_indices: torch.Tensor,
                              drone_pos_w: torch.Tensor,
                              drone_quat_w: torch.Tensor) -> torch.Tensor:
        """Transform pre-computed gate world corners to body frame.

        Args:
            gate_indices: (N,) indices into self._gate_corners_world
            drone_pos_w: (N, 3) drone world position
            drone_quat_w: (N, 4) drone quaternion wxyz

        Returns:
            (N, 12) flattened gate corners in body frame
        """
        corners_w = self._gate_corners_world[gate_indices]          # (N, 4, 3)
        corners_rel = corners_w - drone_pos_w.unsqueeze(1)          # (N, 4, 3)
        N = drone_quat_w.shape[0]
        quat_exp = drone_quat_w.unsqueeze(1).expand(-1, 4, -1).reshape(N * 4, 4)
        corners_body = quat_apply_inverse(quat_exp, corners_rel.reshape(N * 4, 3))
        return corners_body.reshape(N, 12)

    def get_observations(self) -> Dict[str, torch.Tensor]:
        """Get 36-dim observations: O_ego (12D) + O_env (24D).

        O_ego (12D):
            lin_vel_b           (3)  - linear velocity in body frame
            rotation_matrix_wb  (9)  - drone rotation matrix (body-to-world), flattened
            # prev_actions      (4)  - previous policy actions (disabled)

        O_env (24D):
            current_gate_corners_body (12) - current gate 4 corners in body frame
            next_gate_corners_body    (12) - next gate 4 corners in body frame
        """

        n_gates = self.env._waypoints.shape[0]
        N = self.num_envs

        # --- Raw state ---
        drone_pos_w = self.env._robot.data.root_link_pos_w          # (N, 3)
        drone_quat_w = self.env._robot.data.root_quat_w             # (N, 4) wxyz
        lin_vel_b = self.env._robot.data.root_com_lin_vel_b         # (N, 3)

        # --- O_ego ---
        # 1. Linear velocity in body frame (3D)
        # 2. Flattened R_wb (9D): body-frame basis expressed in world frame.
        rot_matrix_wb = matrix_from_quat(drone_quat_w).reshape(N, 9)  # (N, 9)
        # # 3. Previous actions (4D) - disabled
        # prev_actions = self.env._actions                           # (N, 4)

        # --- O_env ---
        idx_wp = self.env._idx_wp                                    # (N,)
        next_idx = (idx_wp + 1) % n_gates                            # (N,)

        # 4. Current gate 4 corners in body frame (12D)
        current_gate_corners_body = self._gate_corners_to_body(
            idx_wp, drone_pos_w, drone_quat_w)                       # (N, 12)
        # 5. Next gate 4 corners in body frame (12D)
        next_gate_corners_body = self._gate_corners_to_body(
            next_idx, drone_pos_w, drone_quat_w)                     # (N, 12)

        obs = torch.cat(
            [
                lin_vel_b,                  # 3
                rot_matrix_wb,              # 9  = O_ego 12
                # prev_actions,             # 4  (disabled)
                current_gate_corners_body,  # 12
                next_gate_corners_body,     # 12 = O_env 24  -> total 36
            ],
            dim=-1,
        )
        observations = {"policy": obs}

        # --- Privileged critic observations (optional, training only) ---
        # Appends extra state to the policy obs so the critic sees 36 + G+9 D.
        # Privileged info (G=num_gates, e.g. 7 for powerloop -> 16D):
        #   ang_vel_b              (3)       - angular velocity in body frame
        #   drone_pos_w            (3)       - world position
        #   pose_drone_wrt_gate    (3)       - drone position in current gate frame
        #   gate_index_onehot      (n_gates) - one-hot current target gate index
        if self.cfg.use_privileged_critic:
            ang_vel_b = self.env._robot.data.root_ang_vel_b             # (N, 3)
            pose_gate = self.env._pose_drone_wrt_gate                   # (N, 3)
            gate_onehot = torch.nn.functional.one_hot(
                idx_wp.long(), num_classes=n_gates
            ).float()                                                   # (N, n_gates)

            critic_obs = torch.cat(
                [
                    obs,            # 36       (actor obs as base)
                    ang_vel_b,      # 3
                    drone_pos_w,    # 3
                    pose_gate,      # 3
                    gate_onehot,    # n_gates  -> critic total 36+9+n_gates
                ],
                dim=-1,
            )
            observations["critic"] = critic_obs

        return observations

    def reset_idx(self, env_ids: Optional[torch.Tensor]):
        """Reset specific environments to initial states."""
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self.env._robot._ALL_INDICES

        # Logging for training mode
        if self.cfg.is_train and hasattr(self, '_episode_sums'):
            extras = dict()
            for key in self._episode_sums.keys():
                episodic_sum_avg = torch.mean(self._episode_sums[key][env_ids])
                extras["Episode_Reward/" + key] = episodic_sum_avg / self.env.max_episode_length_s
                self._episode_sums[key][env_ids] = 0.0
            self.env.extras["log"] = dict()
            self.env.extras["log"].update(extras)
            extras = dict()
            episode_gate_pass_totals = self._episode_gate_pass_counts[env_ids].sum(dim=1)
            completed_laps = torch.floor(episode_gate_pass_totals / self._num_gates)
            extras["Episode_Progress/gates_passed"] = torch.mean(episode_gate_pass_totals).item()
            extras["Episode_Progress/laps_completed"] = (
                torch.mean(completed_laps).item()
            )
            extras["Episode_Progress/lap_progress"] = torch.mean(
                episode_gate_pass_totals / self._num_gates
            ).item()
            for gate_idx in range(self._num_gates):
                extras[f"Episode_GatePass/gate_{gate_idx}"] = torch.mean(
                    self._episode_gate_pass_counts[env_ids, gate_idx]
                ).item()
            extras["Episode_Termination/died"] = torch.count_nonzero(self.env.reset_terminated[env_ids]).item()
            extras["Episode_Termination/out_of_bounds"] = torch.count_nonzero(self.env._out_of_bounds[env_ids]).item()
            extras["Episode_Termination/time_out"] = torch.count_nonzero(self.env.reset_time_outs[env_ids]).item()
            self.env.extras["log"].update(extras)
            self._episode_gate_pass_counts[env_ids] = 0.0

        # Call robot reset first
        self.env._robot.reset(env_ids)

        # Initialize model paths if needed
        if not self.env._models_paths_initialized:
            num_models_per_env = self.env._waypoints.size(0)
            model_prim_names_in_env = [f"{self.env.target_models_prim_base_name}_{i}" for i in range(num_models_per_env)]

            self.env._all_target_models_paths = []
            for env_path in self.env.scene.env_prim_paths:
                paths_for_this_env = [f"{env_path}/{name}" for name in model_prim_names_in_env]
                self.env._all_target_models_paths.append(paths_for_this_env)

            self.env._models_paths_initialized = True

        n_reset = len(env_ids)
        if n_reset == self.num_envs and self.num_envs > 1:
            self.env.episode_length_buf = torch.randint_like(self.env.episode_length_buf,
                                                             high=int(self.env.max_episode_length))

        # Reset action buffers
        self.env._actions[env_ids] = 0.0
        self.env._previous_actions[env_ids] = 0.0
        self.env._previous_yaw[env_ids] = 0.0
        self.env._motor_speeds[env_ids] = 0.0
        self.env._previous_omega_meas[env_ids] = 0.0
        self.env._previous_omega_err[env_ids] = 0.0
        self.env._omega_err_integral[env_ids] = 0.0

        # Reset joints state
        joint_pos = self.env._robot.data.default_joint_pos[env_ids]
        joint_vel = self.env._robot.data.default_joint_vel[env_ids]
        self.env._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        default_root_state = self.env._robot.data.default_root_state[env_ids].clone()

        if self.cfg.is_train:
            n_gates = self.env._waypoints.shape[0]
            if getattr(self.env.cfg, "use_curriculum_reset", True):
                iteration = self.env.iteration
                total_iterations = max(1, getattr(self.env, "total_training_iterations", 1))
                progress_ratio = iteration / max(1, total_iterations - 1)

                # Curriculum: gradually introduce harder gates with smoother transitions.
                # Powerloop (gate 2/3) and chicane (gate 5/6) get extra weight.
                if progress_ratio < 0.05:
                    max_gate, p_start = 1, 1.0
                    gate_weights = torch.ones(max_gate, device=self.device)
                elif progress_ratio < 0.12:
                    max_gate, p_start = 2, 0.55
                    gate_weights = torch.tensor([1.0, 2.0], device=self.device)[:max_gate]
                elif progress_ratio < 0.25:
                    max_gate, p_start = min(4, n_gates), 0.45
                    gate_weights = torch.tensor([1.0, 1.5, 4.0, 3.0], device=self.device)[:max_gate]
                elif progress_ratio < 0.50:
                    max_gate, p_start = min(6, n_gates), 0.40
                    gate_weights = torch.tensor([1.0, 1.0, 3.0, 2.0, 1.5, 1.5], device=self.device)[:max_gate]
                else:
                    max_gate, p_start = n_gates, 0.35
                    gate_weights = torch.ones(max_gate, device=self.device)
                    if max_gate > 2:
                        gate_weights[2] += 2.0   # powerloop entry
                    if max_gate > 3:
                        gate_weights[3] += 1.5   # powerloop exit
                    if max_gate > 5:
                        gate_weights[5] += 1.0   # chicane entry
                    if max_gate > 6:
                        gate_weights[6] += 1.5   # chicane, same physical gate as 3

                gate_weights = gate_weights / gate_weights.sum()
                random_gates = torch.multinomial(gate_weights, n_reset, replacement=True).to(self.env._idx_wp.dtype)
                use_start = torch.rand(n_reset, device=self.device) < p_start
                waypoint_indices = torch.where(use_start, torch.zeros_like(random_gates), random_gates)
            else:
                waypoint_indices = torch.zeros(
                    n_reset, device=self.device, dtype=self.env._idx_wp.dtype
                )

            wp_data = self.env._waypoints[waypoint_indices]
            x0_wp = wp_data[:, 0]
            y0_wp = wp_data[:, 1]
            z_wp = wp_data[:, 2]
            theta = wp_data[:, -1]

            # Match training reset x/y sampling to evaluation for better train-test consistency.
            x_local = torch.empty(n_reset, device=self.device).uniform_(-3.0, -0.5)
            y_local = torch.empty(n_reset, device=self.device).uniform_(-1.0, 1.0)
            z_noise = torch.rand(n_reset, device=self.device) * 0.6 - 0.3      # [-0.3, 0.3]

            cos_theta = torch.cos(theta)
            sin_theta = torch.sin(theta)
            x_rot = cos_theta * x_local - sin_theta * y_local
            y_rot = sin_theta * x_local + cos_theta * y_local
            initial_x = x0_wp - x_rot
            initial_y = y0_wp - y_rot
            initial_z = (z_wp + z_noise).clamp(min=0.2)

            initial_yaw = torch.atan2(y0_wp - initial_y, x0_wp - initial_x)
            roll_noise = torch.rand(n_reset, device=self.device) * 0.2 - 0.1
            pitch_noise = torch.rand(n_reset, device=self.device) * 0.2 - 0.1
            yaw_noise = torch.rand(n_reset, device=self.device) * 0.4 - 0.2
            quat = quat_from_euler_xyz(roll_noise, pitch_noise, initial_yaw + yaw_noise)

            default_root_state[:, 0] = initial_x
            default_root_state[:, 1] = initial_y
            default_root_state[:, 2] = initial_z
            default_root_state[:, 3:7] = quat

            # Gate 0 matches evaluation: start from rest. Mid-track starts get realistic forward speed.
            gate_dirs = -self.env._normal_vectors[waypoint_indices]
            speed0 = (1.5 + torch.rand(n_reset, device=self.device) * 2.5) * (waypoint_indices > 0).float()
            default_root_state[:, 7:10] = gate_dirs * speed0.unsqueeze(1)
            default_root_state[:, 10:13] = 0.0
        else:
            waypoint_indices = torch.full(
                (n_reset,), int(self.env._initial_wp), device=self.device, dtype=self.env._idx_wp.dtype
            )

            x_local = torch.empty(n_reset, device=self.device).uniform_(-3.0, -0.5)
            y_local = torch.empty(n_reset, device=self.device).uniform_(-1.0, 1.0)

            wp_data = self.env._waypoints[waypoint_indices]
            x0_wp = wp_data[:, 0]
            y0_wp = wp_data[:, 1]
            theta = wp_data[:, -1]

            cos_theta = torch.cos(theta)
            sin_theta = torch.sin(theta)
            x_rot = cos_theta * x_local - sin_theta * y_local
            y_rot = sin_theta * x_local + cos_theta * y_local
            initial_x = x0_wp - x_rot
            initial_y = y0_wp - y_rot
            initial_z = torch.full((n_reset,), 0.05, device=self.device)

            initial_yaw = torch.atan2(y0_wp - initial_y, x0_wp - initial_x)
            quat = quat_from_euler_xyz(
                torch.zeros(n_reset, device=self.device),
                torch.zeros(n_reset, device=self.device),
                initial_yaw,
            )

            default_root_state[:, 0] = initial_x
            default_root_state[:, 1] = initial_y
            default_root_state[:, 2] = initial_z
            default_root_state[:, 3:7] = quat
            default_root_state[:, 7:13] = 0.0

        # Set waypoint indices and desired positions
        self.env._idx_wp[env_ids] = waypoint_indices

        self.env._desired_pos_w[env_ids, :3] = self.env._waypoints[waypoint_indices, :3].clone()

        self.env._last_distance_to_goal[env_ids] = torch.linalg.norm(
            self.env._desired_pos_w[env_ids, :2] - default_root_state[:, :2], dim=1
        )
        self.env._n_gates_passed[env_ids] = waypoint_indices

        # Write state to simulation
        self.env._robot.write_root_link_pose_to_sim(default_root_state[:, :7], env_ids)
        self.env._robot.write_root_com_velocity_to_sim(default_root_state[:, 7:], env_ids)

        # Reset variables
        self.env._yaw_n_laps[env_ids] = 0

        self.env._pose_drone_wrt_gate[env_ids], _ = subtract_frame_transforms(
            self.env._waypoints[self.env._idx_wp[env_ids], :3],
            self.env._waypoints_quat[self.env._idx_wp[env_ids], :],
            default_root_state[:, :3]
        )

        self.env._prev_x_drone_wrt_gate[env_ids] = 1.0

        self.env._crashed[env_ids] = 0
        self.env._out_of_bounds[env_ids] = False

        # Reset tracking buffers for reward computation
        self._prev_global_progress[env_ids] = self._compute_global_progress()[env_ids]
        self._prev_in_entry_half_plane[env_ids] = self.env._pose_drone_wrt_gate[env_ids, 0] > 0.15
        self._entry_half_plane_rewarded[env_ids] = self._prev_in_entry_half_plane[env_ids]
        self._prev_step_actions[env_ids] = 0.0
