# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, project_root)

local_rsl_path = os.path.abspath("src/third_parties/rsl_rl_local")
if os.path.exists(local_rsl_path):
    sys.path.insert(0, local_rsl_path)
    print(f"[INFO] Using local rsl_rl from: {local_rsl_path}")
else:
    print(f"[WARNING] Local rsl_rl not found at: {local_rsl_path}")

import argparse

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=2500, help="Length of the recorded video (in steps).")
parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--follow_robot", type=int, default=-1, help="Follow robot index.")
parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
parser.add_argument("--no_ang_vel_obs", action="store_true", default=False, help="Disable ang_vel in obs (for old 36D checkpoints).")

# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import torch

from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.dict import print_dict

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlVecEnvWrapper,
    export_policy_as_jit,
    export_policy_as_onnx,
)

# Import extensions to set up environment tasks
import src.isaac_quad_sim2real.tasks   # noqa: F401

def main():
    """Play with RSL-RL agent."""
    # parse configuratio
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    log_dir = os.path.dirname(resume_path)

    if args_cli.follow_robot == -1:
        env_cfg.viewer.resolution = (1920, 1080)
        env_cfg.viewer.eye = (10.7, 0.4, 7.2)
        env_cfg.viewer.lookat = (-2.7, 0.5, -0.3)
    elif args_cli.follow_robot >= 0:
        env_cfg.viewer.eye = (-0.8, 0.8, 0.8)
        env_cfg.viewer.resolution = (1920, 1080)
        env_cfg.viewer.lookat = (0.0, 0.0, 0.0)
        env_cfg.viewer.origin_type = "asset_root"
        env_cfg.viewer.env_index = args_cli.follow_robot
        env_cfg.viewer.asset_name = "robot"

    env_cfg.is_train = False
    env_cfg.max_motor_noise_std = 0.0
    env_cfg.seed = args_cli.seed
    env_cfg.use_ang_vel_obs = not args_cli.no_ang_vel_obs

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # load previously trained model
    ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    ppo_runner.load(resume_path)

    # obtain the trained policy for inference
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

    # export policy to onnx/jit
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    export_policy_as_jit(
        ppo_runner.alg.actor_critic, ppo_runner.obs_normalizer, path=export_model_dir, filename="policy.pt"
    )
    export_policy_as_onnx(
        ppo_runner.alg.actor_critic, normalizer=ppo_runner.obs_normalizer, path=export_model_dir, filename="policy.onnx"
    )

    # -- Lap timing setup --
    quad_env = env.unwrapped
    num_gates = quad_env._waypoints.shape[0]
    step_dt = quad_env.cfg.sim.dt * quad_env.cfg.decimation
    max_laps = quad_env.cfg.max_n_laps

    lap_start_step = None
    last_lap_end_step = None
    lap_times = []
    all_lap_times = []
    prev_gates_passed = 0

    # reset environment
    obs = env.get_observations()
    # Extract tensor from TensorDict for policy
    if hasattr(obs, "get"):  # Check if it's a TensorDict
        obs = obs["policy"]  # Extract the policy observation
    timestep = 0
    # simulate environment
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            actions = policy(obs)
            # env stepping
            obs, rewards, dones, infos = env.step(actions)
            # Extract tensor from TensorDict for policy
            if hasattr(obs, "get"):  # Check if it's a TensorDict
                obs = obs["policy"]  # Extract the policy observation
        timestep += 1

        # -- Lap timing (env 0) --
        gates_passed = quad_env._n_gates_passed[0].item()

        if gates_passed < prev_gates_passed:
            # Auto-reset detected. The final lap may have been completed in the same
            # env.step() that triggered the reset, so _n_gates_passed is already zeroed.
            # Recover the missing lap if prev_gates_passed indicates completion.
            if lap_start_step is not None and len(lap_times) < max_laps:
                expected_laps = (prev_gates_passed - 1) // num_gates
                while len(lap_times) < expected_laps and len(lap_times) < max_laps:
                    lap_time = (timestep - last_lap_end_step) * step_dt
                    total_time = (timestep - lap_start_step) * step_dt
                    lap_times.append(lap_time)
                    last_lap_end_step = timestep
                    print(f"[LAP TIMER] Lap {len(lap_times)}/{max_laps}: {lap_time:.3f}s  (total: {total_time:.3f}s)")
            all_lap_times.extend(lap_times)
            lap_start_step = None
            last_lap_end_step = None
            lap_times = []

        if gates_passed > prev_gates_passed:
            if lap_start_step is None:
                lap_start_step = timestep
                last_lap_end_step = timestep
                print(f"[LAP TIMER] Timing started (passed gate 0, {num_gates} gates/lap)")

            completed_laps = (gates_passed - 1) // num_gates
            while len(lap_times) < completed_laps and len(lap_times) < max_laps:
                lap_time = (timestep - last_lap_end_step) * step_dt
                total_time = (timestep - lap_start_step) * step_dt
                lap_times.append(lap_time)
                last_lap_end_step = timestep
                print(f"[LAP TIMER] Lap {len(lap_times)}/{max_laps}: {lap_time:.3f}s  (total: {total_time:.3f}s)")

        prev_gates_passed = gates_passed

        if args_cli.video:
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break

    # -- Print lap summary --
    all_lap_times.extend(lap_times)
    if len(all_lap_times) > 0:
        total = sum(all_lap_times)
        print(f"\n{'='*50}")
        print(f"  RACE RESULTS  ({num_gates} gates/lap)")
        print(f"{'='*50}")
        for i, lt in enumerate(all_lap_times):
            print(f"  Lap {i+1}: {lt:.3f}s")
        print(f"  {'-'*30}")
        print(f"  Total ({len(all_lap_times)} laps): {total:.3f}s")
        print(f"  Average lap:  {total / len(all_lap_times):.3f}s")
        print(f"{'='*50}\n")
    else:
        print("\n[LAP TIMER] No laps completed.\n")

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
