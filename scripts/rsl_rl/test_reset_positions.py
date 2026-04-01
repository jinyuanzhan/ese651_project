# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Visualize curriculum reset positions for gate 2 / gate 3 (powerloop).

This script spawns drones at the computed reset positions for gate 2 and gate 3
curriculum stages, holds them in place for a few seconds at each configuration,
and records a video so you can visually verify the spawn locations are correct.

Usage:
    python scripts/rsl_rl/test_reset_positions.py --video --num_envs 16
    python scripts/rsl_rl/test_reset_positions.py --video --num_envs 16 --test_mode apex
    python scripts/rsl_rl/test_reset_positions.py --video --num_envs 16 --test_mode all
    python scripts/rsl_rl/test_reset_positions.py --video --headless --camera_view side_y
    python scripts/rsl_rl/test_reset_positions.py --video --headless --camera_view isometric
"""

import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, project_root)

import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Test curriculum reset positions for gate 2/3.")
parser.add_argument("--video", action="store_true", default=False, help="Record video.")
parser.add_argument("--video_length", type=int, default=1500, help="Video length in steps.")
parser.add_argument("--num_envs", type=int, default=16, help="Number of environments.")
parser.add_argument(
    "--camera_view",
    type=str,
    default="topdown",
    choices=["topdown", "side_y", "side_x", "isometric"],
    help="Viewer camera preset for the gate 2/3 region.",
)
parser.add_argument(
    "--camera_eye",
    type=float,
    nargs=3,
    default=None,
    metavar=("X", "Y", "Z"),
    help="Optional custom camera eye position. Overrides --camera_view eye.",
)
parser.add_argument(
    "--camera_lookat",
    type=float,
    nargs=3,
    default=None,
    metavar=("X", "Y", "Z"),
    help="Optional custom camera look-at position. Overrides preset look-at.",
)
parser.add_argument(
    "--test_mode",
    type=str,
    default="all",
    choices=["gate2", "gate3", "apex", "all"],
    help="Which reset mode to test: gate2, gate3, apex (powerloop apex), or all.",
)
parser.add_argument("--disable_fabric", action="store_true", default=False)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if args_cli.video:
    args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import torch
import numpy as np

from isaaclab.utils.math import subtract_frame_transforms, quat_from_euler_xyz

# Import extensions to register environment
import src.isaac_quad_sim2real.tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg


def compute_standard_reset_positions(waypoints, normal_vectors, gate_idx, n_envs, device):
    """Compute standard curriculum reset positions for a given gate index.

    Places drones 0.5-3.0m behind the gate in the gate-frame x direction,
    with +-1.0m lateral offset and +-0.3m height noise.
    """
    wp = waypoints[gate_idx]
    x0_wp, y0_wp, z_wp = wp[0], wp[1], wp[2]
    theta = wp[-1]  # yaw angle

    x_local = torch.empty(n_envs, device=device).uniform_(-3.0, -0.5)
    y_local = torch.empty(n_envs, device=device).uniform_(-1.0, 1.0)
    z_noise = torch.rand(n_envs, device=device) * 0.6 - 0.3

    cos_theta = torch.cos(theta)
    sin_theta = torch.sin(theta)
    x_rot = cos_theta * x_local - sin_theta * y_local
    y_rot = sin_theta * x_local + cos_theta * y_local
    initial_x = x0_wp - x_rot
    initial_y = y0_wp - y_rot
    initial_z = (z_wp + z_noise).clamp(min=0.2)

    initial_yaw = torch.atan2(y0_wp - initial_y, x0_wp - initial_x)
    roll_noise = torch.rand(n_envs, device=device) * 0.2 - 0.1
    pitch_noise = torch.rand(n_envs, device=device) * 0.2 - 0.1
    yaw_noise = torch.rand(n_envs, device=device) * 0.4 - 0.2
    quat = quat_from_euler_xyz(roll_noise, pitch_noise, initial_yaw + yaw_noise)

    # Velocity: forward toward gate
    gate_dir = -normal_vectors[gate_idx]
    speed = 1.5 + torch.rand(n_envs, device=device) * 2.5
    vel = gate_dir.unsqueeze(0) * speed.unsqueeze(1)

    return initial_x, initial_y, initial_z, quat, vel


def compute_apex_reset_positions(n_envs, device, apex_pitch_max=np.pi * 2 / 3):
    """Compute powerloop apex reset positions (between gates 2-3).

    Samples along the loop arc: phase 0 = gate 2 exit, phase 1 = full apex.
    X follows the natural trajectory from gate 2 (x=-0.625) to gate 3 (x=+0.625).
    Roll scales with phase to model banking for the 1.25m lateral shift.
    """
    phase = torch.linspace(0, 1, n_envs, device=device)  # spread evenly for visualization

    pitch_noise = torch.empty(n_envs, device=device).uniform_(-0.15, 0.15)
    apex_pitch = (phase * apex_pitch_max + pitch_noise).clamp(0, apex_pitch_max)

    apex_z = 0.8 + (phase * 1.5)
    apex_z += torch.empty(n_envs, device=device).uniform_(-0.15, 0.15)
    apex_z.clamp_(0.5, 2.8)

    # X: follow trajectory from gate 2 to gate 3
    x_center = -0.625 + phase * 1.25
    x_noise = torch.empty(n_envs, device=device).uniform_(-0.3, 0.3)
    apex_x = (x_center + x_noise).clamp(-1.0, 1.0)
    apex_y = 0.05 + torch.empty(n_envs, device=device).uniform_(-0.1, 0.1)

    # Roll: phase-dependent — larger at mid-loop for banking
    roll_max = 0.3 + phase * 0.5  # [0.3, 0.8] rad
    apex_roll = (torch.rand(n_envs, device=device) * 2 - 1) * roll_max
    apex_yaw = torch.empty(n_envs, device=device).uniform_(
        -np.pi / 2 - 0.3, -np.pi / 2 + 0.3
    )
    quat = quat_from_euler_xyz(apex_roll, apex_pitch, apex_yaw)

    speed_noise = torch.empty(n_envs, device=device).uniform_(-0.3, 0.3)
    loop_speed = (2.0 * (1.0 - phase * 0.7) + speed_noise).clamp(0.3, 3.0)
    vel_noise = torch.randn(n_envs, 3, device=device) * 0.2

    # vx: wider range to match larger banking
    vx = torch.empty(n_envs, device=device).uniform_(-0.8, 0.8) + vel_noise[:, 0]
    vy = -loop_speed * torch.cos(apex_pitch) + vel_noise[:, 1]
    vz = loop_speed * torch.sin(apex_pitch) + vel_noise[:, 2]
    vel = torch.stack([vx, vy, vz], dim=1)

    return apex_x, apex_y, apex_z, quat, vel


def place_drones(
    env,
    positions_x,
    positions_y,
    positions_z,
    quats,
    vels,
    env_ids,
    gate_idx,
    *,
    use_actual_prev_x=False,
):
    """Write drone states into the simulation and sync reset-tracking buffers."""
    quad_env = env.unwrapped
    root_state = quad_env._robot.data.default_root_state[env_ids].clone()

    root_state[:, 0] = positions_x
    root_state[:, 1] = positions_y
    root_state[:, 2] = positions_z
    root_state[:, 3:7] = quats
    root_state[:, 7:10] = vels
    root_state[:, 10:13] = 0.0

    quad_env._robot.write_root_link_pose_to_sim(root_state[:, :7], env_ids)
    quad_env._robot.write_root_com_velocity_to_sim(root_state[:, 7:], env_ids)

    # Update waypoint tracking
    quad_env._idx_wp[env_ids] = gate_idx
    quad_env._desired_pos_w[env_ids, :3] = quad_env._waypoints[gate_idx, :3]
    quad_env._n_gates_passed[env_ids] = gate_idx
    quad_env._last_distance_to_goal[env_ids] = torch.linalg.norm(
        quad_env._desired_pos_w[env_ids, :2] - root_state[:, :2], dim=1
    )

    quad_env._actions[env_ids] = 0.0
    quad_env._previous_actions[env_ids] = 0.0
    quad_env._previous_yaw[env_ids] = 0.0
    quad_env._motor_speeds[env_ids] = 0.0
    quad_env._previous_omega_meas[env_ids] = 0.0
    quad_env._previous_omega_err[env_ids] = 0.0
    quad_env._omega_err_integral[env_ids] = 0.0
    quad_env._crashed[env_ids] = 0
    quad_env._out_of_bounds[env_ids] = False

    pose_drone_wrt_gate, _ = subtract_frame_transforms(
        quad_env._waypoints[quad_env._idx_wp[env_ids], :3],
        quad_env._waypoints_quat[quad_env._idx_wp[env_ids], :],
        root_state[:, :3],
    )
    quad_env._pose_drone_wrt_gate[env_ids] = pose_drone_wrt_gate
    if use_actual_prev_x:
        quad_env._prev_x_drone_wrt_gate[env_ids] = pose_drone_wrt_gate[:, 0]
    else:
        quad_env._prev_x_drone_wrt_gate[env_ids] = 1.0

    if hasattr(quad_env, "strategy"):
        strategy = quad_env.strategy
        if hasattr(strategy, "_prev_global_progress"):
            strategy._prev_global_progress[env_ids] = strategy._compute_global_progress()[env_ids]
        if hasattr(strategy, "_prev_in_entry_half_plane"):
            in_entry_half_plane = pose_drone_wrt_gate[:, 0] > 0.15
            strategy._prev_in_entry_half_plane[env_ids] = in_entry_half_plane
            strategy._entry_half_plane_rewarded[env_ids] = in_entry_half_plane
        if hasattr(strategy, "_prev_step_actions"):
            strategy._prev_step_actions[env_ids] = 0.0


def main():
    env_cfg = parse_env_cfg(
        "Isaac-Quadcopter-Race-v0",
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )

    camera_presets = {
        "topdown": {
            "eye": (0.0, 0.0, 8.0),
            "lookat": (0.0, 0.0, 0.8),
        },
        "side_y": {
            "eye": (0.0, 6.0, 1.8),
            "lookat": (0.0, 0.0, 1.2),
        },
        "side_x": {
            "eye": (6.0, 0.0, 1.8),
            "lookat": (0.0, 0.0, 1.2),
        },
        "isometric": {
            "eye": (4.5, 4.5, 3.2),
            "lookat": (0.0, 0.0, 1.2),
        },
    }
    preset_camera = camera_presets[args_cli.camera_view]
    camera_eye = tuple(args_cli.camera_eye) if args_cli.camera_eye is not None else preset_camera["eye"]
    camera_lookat = (
        tuple(args_cli.camera_lookat)
        if args_cli.camera_lookat is not None
        else preset_camera["lookat"]
    )

    # Camera for gate 2/3 area visualization
    env_cfg.viewer.resolution = (1920, 1080)
    env_cfg.viewer.eye = camera_eye
    env_cfg.viewer.lookat = camera_lookat

    env_cfg.is_train = True  # need train mode so rewards dict is available
    env_cfg.use_curriculum_reset = False  # we handle placement manually
    env_cfg.scene.num_envs = args_cli.num_envs

    # Provide minimal rewards dict to avoid ValueError
    env_cfg.rewards = {
        "gate_pass_reward_scale": 25.0,
        "progress_reward_scale": 25.0,
        "speed_reward_scale": 0.0,
        "entry_half_plane_reward_scale": 0.0,
        "crash_reward_scale": -3.0,
        "action_smooth_reward_scale": 0.0,
        "time_reward_scale": -0.01,
        "powerloop_corridor_reward_scale": 0.0,
        "death_cost": -80.0,
    }

    render_mode = "rgb_array" if args_cli.video else None
    env = gym.make("Isaac-Quadcopter-Race-v0", cfg=env_cfg, render_mode=render_mode)

    if args_cli.video:
        video_dir = os.path.join(project_root, "logs", "test_reset_positions")
        os.makedirs(video_dir, exist_ok=True)
        video_kwargs = {
            "video_folder": video_dir,
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print(f"[INFO] Recording video to: {video_dir}")
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    quad_env = env.unwrapped
    device = quad_env.device
    n_envs = quad_env.num_envs
    waypoints = quad_env._waypoints
    normal_vectors = quad_env._normal_vectors

    print("\n" + "=" * 60)
    print("  GATE POSITIONS (powerloop track)")
    print("=" * 60)
    print(f"  Camera view: {args_cli.camera_view}")
    print(f"  Camera eye: {camera_eye}")
    print(f"  Camera lookat: {camera_lookat}")
    for i, wp in enumerate(waypoints):
        print(f"  Gate {i}: x={wp[0]:.3f}, y={wp[1]:.3f}, z={wp[2]:.3f}, yaw={wp[5]:.4f} rad ({wp[5]*180/np.pi:.1f} deg)")
    print(f"  Normal vectors:")
    for i, nv in enumerate(normal_vectors):
        print(f"    Gate {i}: [{nv[0]:.3f}, {nv[1]:.3f}, {nv[2]:.3f}]")
    print("=" * 60 + "\n")

    # Determine test sequence
    test_configs = []
    if args_cli.test_mode in ("gate2", "all"):
        test_configs.append(("gate2_standard", 2))
    if args_cli.test_mode in ("gate3", "all"):
        test_configs.append(("gate3_standard", 3))
    if args_cli.test_mode in ("apex", "all"):
        test_configs.append(("gate3_apex_phase1", "apex_phase1"))
        test_configs.append(("gate3_apex_phase2", "apex_phase2"))

    steps_per_config = args_cli.video_length // max(len(test_configs), 1)
    env_ids = torch.arange(n_envs, device=device)

    # Initial reset
    obs, _ = env.reset()
    step = 0
    config_idx = 0
    current_display_state = None

    print(f"[INFO] Testing {len(test_configs)} configurations, {steps_per_config} steps each")

    while simulation_app.is_running():
        with torch.inference_mode():
            # Determine which config to show
            current_config_idx = min(step // steps_per_config, len(test_configs) - 1)

            # Switch config when crossing boundary
            if current_config_idx != config_idx or step == 0:
                config_idx = current_config_idx
                name, mode = test_configs[config_idx]

                if isinstance(mode, int):
                    # Standard gate reset
                    gate_idx = mode
                    px, py, pz, q, v = compute_standard_reset_positions(
                        waypoints, normal_vectors, gate_idx, n_envs, device
                    )
                    current_display_state = {
                        "positions_x": px,
                        "positions_y": py,
                        "positions_z": pz,
                        "quats": q,
                        "vels": torch.zeros_like(v),
                        "gate_idx": gate_idx,
                        "use_actual_prev_x": False,
                    }
                    place_drones(env, env_ids=env_ids, **current_display_state)
                    print(f"\n[STEP {step}] Placed drones at STANDARD reset for Gate {gate_idx}")
                    print(f"  X range: [{px.min():.2f}, {px.max():.2f}]")
                    print(f"  Y range: [{py.min():.2f}, {py.max():.2f}]")
                    print(f"  Z range: [{pz.min():.2f}, {pz.max():.2f}]")

                elif mode == "apex_phase1":
                    # Apex with conservative pitch (120 deg)
                    px, py, pz, q, v = compute_apex_reset_positions(
                        n_envs, device, apex_pitch_max=np.pi * 2 / 3
                    )
                    current_display_state = {
                        "positions_x": px,
                        "positions_y": py,
                        "positions_z": pz,
                        "quats": q,
                        "vels": torch.zeros_like(v),
                        "gate_idx": 3,
                        "use_actual_prev_x": True,
                    }
                    place_drones(env, env_ids=env_ids, **current_display_state)
                    print(f"\n[STEP {step}] Placed drones at APEX reset (Phase 1: pitch<=120 deg)")
                    print(f"  X range: [{px.min():.2f}, {px.max():.2f}]")
                    print(f"  Y range: [{py.min():.2f}, {py.max():.2f}]")
                    print(f"  Z range: [{pz.min():.2f}, {pz.max():.2f}]")

                elif mode == "apex_phase2":
                    # Apex with full pitch (180 deg)
                    px, py, pz, q, v = compute_apex_reset_positions(
                        n_envs, device, apex_pitch_max=np.pi
                    )
                    current_display_state = {
                        "positions_x": px,
                        "positions_y": py,
                        "positions_z": pz,
                        "quats": q,
                        "vels": torch.zeros_like(v),
                        "gate_idx": 3,
                        "use_actual_prev_x": True,
                    }
                    place_drones(env, env_ids=env_ids, **current_display_state)
                    print(f"\n[STEP {step}] Placed drones at APEX reset (Phase 2: pitch<=180 deg)")
                    print(f"  X range: [{px.min():.2f}, {px.max():.2f}]")
                    print(f"  Y range: [{py.min():.2f}, {py.max():.2f}]")
                    print(f"  Z range: [{pz.min():.2f}, {pz.max():.2f}]")

            # Re-apply the sampled state each step so the video shows the intended
            # reset positions instead of later auto-resets after crash/ground contact.
            if current_display_state is not None:
                place_drones(env, env_ids=env_ids, **current_display_state)

            actions = torch.zeros(n_envs, 4, device=device)
            obs, rewards, dones, truncated, infos = env.step(actions)
            step += 1

            if step >= args_cli.video_length:
                break

    print(f"\n[DONE] Completed {step} steps.")
    if args_cli.video:
        print(f"[INFO] Video saved to: {os.path.join(project_root, 'logs', 'test_reset_positions')}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
