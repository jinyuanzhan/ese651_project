# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to train RL agent with RSL-RL."""

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

from rsl_rl.utils import wandb_fix
import argparse
import csv
import json
from isaaclab.app import AppLauncher
import cli_args


def parse_bool_flag(value: str) -> bool:
    value = value.lower()
    if value in {"true", "1", "yes", "y"}:
        return True
    if value in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError("Expected true or false.")


# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument(
    "--disable_curriculum_reset",
    action="store_true",
    default=False,
    help="Disable reset curriculum and always start training episodes from gate 0.",
)
parser.add_argument(
    "--reward_set",
    type=str,
    default="baseline",
    choices={"baseline", "trial1", "trial2"},
    help="Reward preset to use for training.",
)
parser.add_argument(
    "--privileged_critic",
    action="store_true",
    default=False,
    help="Enable asymmetric actor-critic with privileged observations for the critic.",
)
# Reward scale overrides (override values from the selected reward preset)
parser.add_argument("--gate_pass_reward", type=float, default=None, help="Override gate_pass_reward_scale.")
parser.add_argument("--progress_reward", type=float, default=None, help="Override progress_reward_scale.")
parser.add_argument("--speed_reward", type=float, default=None, help="Override speed_reward_scale.")
parser.add_argument("--time_reward", type=float, default=None, help="Override time_reward_scale.")
parser.add_argument("--crash_reward", type=float, default=None, help="Override crash_reward_scale.")
parser.add_argument("--death_cost", type=float, default=None, help="Override death_cost.")
parser.add_argument("--entry_half_plane_reward", type=float, default=None, help="Override entry_half_plane_reward_scale.")
# PPO hyperparameter overrides
parser.add_argument("--gamma", type=float, default=None, help="Override discount factor gamma.")
parser.add_argument(
    "--gate_metrics_summary_csv",
    type=str,
    default=None,
    help="Optional CSV file to append aggregated training Episode_GatePass metrics after training.",
)
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import torch
from datetime import datetime

from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_pickle, dump_yaml

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper

# Import extensions to set up environment tasks
import src.isaac_quad_sim2real.tasks   # noqa: F401

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False
#     113 -    #     'gate_pass_reward_scale': 100.0,      # sparse bonus for passing a gate
#     114 -    #     'progress_reward_scale': 10.0,         # dense shaping toward current gate
#     115 -    #     'vel_align_reward_scale': 1.2,        # fly toward gate, not hover
#     116 -    #     'speed_reward_scale': 0.5,            # encourage fast flight
#     117 -    #     'entry_half_plane_reward_scale': 3.0, # small bonus for re-entering the valid approach side
#     118 -    #     'crash_reward_scale': -5.0,           # collision penalty
#     119 -    #     'action_smooth_reward_scale': -0.05,  # prevent oscillation
#     120 -    #     'altitude_reward_scale': -2.0,        # avoid ground
#     121 -    #     'lateral_reward_scale': 0,         # stay near gate center without over-penalizing hairpins
#     122 -    #     'death_cost': -80.0,                  # termination penalty

REWARD_PRESETS = {
    "baseline": {
        "gate_pass_reward_scale": 50.0,     # 过门是核心目标，给强激励
    "progress_reward_scale": 15,         # 朝门飞的 dense shaping
        "speed_reward_scale": 0,           # 奖励高速飞行
        "entry_half_plane_reward_scale": 0,  # 一次性奖励回到 gate3 的有效进入半平面（+Y 侧）
        "crash_reward_scale": 0,
      "action_smooth_reward_scale": 0,
        # "altitude_reward_scale": -2.0,
        # "lateral_reward_scale": 0.0,
        "time_reward_scale": -0.08,          # 每步惩罚，让慢飞代价高
        "death_cost": -80.0,
    },
    # "trial1": {
    #     "gate_pass_reward_scale": 80.0,
    #     "progress_reward_scale": 6.0,
    #     "speed_reward_scale": 0.5,
    #     # "entry_half_plane_reward_scale": 5.0,
    #     # "crash_reward_scale": -2.0,
    #     "action_smooth_reward_scale": -0.03,
    #     # "altitude_reward_scale": -1.5,
    #     # "lateral_reward_scale": 0.0,
    #     "time_reward_scale": -0.01,
    #     "death_cost": -30.0,
    # },
    # "trial2": {
    #     "gate_pass_reward_scale": 80.0,
    #     "progress_reward_scale": 5.0,
    #     "speed_reward_scale": 0.5,
    #     # "entry_half_plane_reward_scale": 1.5,
    #     # "crash_reward_scale": -4.0,
    #     "action_smooth_reward_scale": -0.03,
    #     # "altitude_reward_scale": -1.5,
    #     # "lateral_reward_scale": 0.0,
    #     "time_reward_scale": -0.01,
    #     "death_cost": -60.0,
    # },
}


def _build_training_gate_metric_summary(
    runner: OnPolicyRunner,
    args_cli: argparse.Namespace,
    agent_cfg: RslRlOnPolicyRunnerCfg,
    env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg,
    rewards: dict,
    log_dir: str,
    num_gates: int,
) -> dict:
    episode_summary = runner.get_episode_metric_summary()
    return {
        "timestamp_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "task": args_cli.task,
        "run_name": agent_cfg.run_name,
        "experiment_name": agent_cfg.experiment_name,
        "log_dir": log_dir,
        "num_envs": env_cfg.scene.num_envs,
        "max_iterations": agent_cfg.max_iterations,
        "seed": agent_cfg.seed,
        "reward_set": args_cli.reward_set,
        "curriculum_reset_enabled": env_cfg.use_curriculum_reset,
        "privileged_critic_enabled": env_cfg.use_privileged_critic,
        "num_gates": int(num_gates),
        "rewards": dict(rewards),
        "episode_metrics": episode_summary,
    }


def _write_training_gate_metric_summary_json(summary: dict, log_dir: str) -> str:
    summary_path = os.path.join(log_dir, "training_gate_metrics_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    return summary_path


def _append_training_gate_metric_summary_csv(csv_path: str, summary: dict):
    rewards = summary["rewards"]
    best_metrics = summary["episode_metrics"].get("best", {})
    latest_metrics = summary["episode_metrics"].get("latest", {})
    num_gates = int(summary["num_gates"])

    row = {
        "timestamp_utc": summary["timestamp_utc"],
        "task": summary["task"],
        "run_name": summary["run_name"],
        "experiment_name": summary["experiment_name"],
        "log_dir": summary["log_dir"],
        "num_envs": summary["num_envs"],
        "max_iterations": summary["max_iterations"],
        "seed": summary["seed"],
        "reward_set": summary["reward_set"],
        "curriculum_reset_enabled": summary["curriculum_reset_enabled"],
        "privileged_critic_enabled": summary["privileged_critic_enabled"],
        "gate_pass_reward_scale": rewards.get("gate_pass_reward_scale"),
        "progress_reward_scale": rewards.get("progress_reward_scale"),
        "speed_reward_scale": rewards.get("speed_reward_scale"),
        "time_reward_scale": rewards.get("time_reward_scale"),
        "powerloop_corridor_reward_scale": rewards.get("powerloop_corridor_reward_scale"),
        "crash_reward_scale": rewards.get("crash_reward_scale"),
        "action_smooth_reward_scale": rewards.get("action_smooth_reward_scale"),
        "entry_half_plane_reward_scale": rewards.get("entry_half_plane_reward_scale"),
        "powerloop_altitude_reward_scale": rewards.get("powerloop_altitude_reward_scale"),
        "death_cost": rewards.get("death_cost"),
        "best_iteration": summary["episode_metrics"].get("best_iteration"),
        "best_episode_progress_gates_passed": best_metrics.get("Episode_Progress/gates_passed"),
        "best_episode_progress_laps_completed": best_metrics.get("Episode_Progress/laps_completed"),
        "best_episode_progress_lap_progress": best_metrics.get("Episode_Progress/lap_progress"),
        "latest_iteration": summary["episode_metrics"].get("latest_iteration"),
        "latest_episode_progress_gates_passed": latest_metrics.get("Episode_Progress/gates_passed"),
        "latest_episode_progress_laps_completed": latest_metrics.get("Episode_Progress/laps_completed"),
        "latest_episode_progress_lap_progress": latest_metrics.get("Episode_Progress/lap_progress"),
    }
    for gate_idx in range(num_gates):
        row[f"best_episode_gate_{gate_idx}"] = best_metrics.get(f"Episode_GatePass/gate_{gate_idx}")
        row[f"latest_episode_gate_{gate_idx}"] = latest_metrics.get(f"Episode_GatePass/gate_{gate_idx}")

    os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)
    file_exists = os.path.exists(csv_path) and os.path.getsize(csv_path) > 0
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


#    111 -    # # Reward scales for 8-component reward structure
#     112 -    # rewards = {
#     113 -    #     'gate_pass_reward_scale': 100.0,      # sparse bonus for passing a gate
#     114 -    #     'progress_reward_scale': 10.0,         # dense shaping toward current gate
#     115 -    #     'vel_align_reward_scale': 1.2,        # fly toward gate, not hover
#     116 -    #     'speed_reward_scale': 0.5,            # encourage fast flight
#     117 -    #     'entry_half_plane_reward_scale': 3.0, # small bonus for re-entering the valid approach side
#     118 -    #     'crash_reward_scale': -5.0,           # collision penalty
#     119 -    #     'action_smooth_reward_scale': -0.05,  # prevent oscillation
#     120 -    #     'altitude_reward_scale': -2.0,        # avoid ground
#     121 -    #     'lateral_reward_scale': 0,         # stay near gate center without over-penalizing hairpins
#     122 -    #     'death_cost': -80.0,                  # termination penalty
#     123 -    # }
#     124 -    rewards = {
#     125 -    'gate_pass_reward_scale': 20.0,
#     126 -    'progress_reward_scale': 8.0,
#     127 -    'vel_align_reward_scale': 1.0,
#     128 -    'speed_reward_scale': 0.3,
#     129 -    'entry_half_plane_reward_scale': 5.0,
#     130 -    'crash_reward_scale': -2.0,
#     131 -    'action_smooth_reward_scale': -0.03,
#     132 -    'altitude_reward_scale': -1.5,
#     133 -    'lateral_reward_scale': 0.0,
#     134 -    'time_reward_scale': -0.1,
#     135 -    'death_cost': -25.0,
#     136 -}
@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    """Train with RSL-RL agent."""
    # override configurations with non-hydra CLI arguments
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    agent_cfg.max_iterations = (
        args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg.max_iterations
    )

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    # specify directory for logging runs: {time-stamp}_{run_name}
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)

    rewards = REWARD_PRESETS[args_cli.reward_set].copy()

    # Apply command-line reward overrides
    _reward_overrides = {
        "gate_pass_reward_scale": args_cli.gate_pass_reward,
        "progress_reward_scale": args_cli.progress_reward,
        "speed_reward_scale": args_cli.speed_reward,
        "time_reward_scale": args_cli.time_reward,
        "crash_reward_scale": args_cli.crash_reward,
        "death_cost": args_cli.death_cost,
        "entry_half_plane_reward_scale": args_cli.entry_half_plane_reward,
    }
    for key, val in _reward_overrides.items():
        if val is not None:
            rewards[key] = val
            print(f"[INFO] Override: {key} = {val}")

    print(f"[INFO] Using reward preset: {args_cli.reward_set}")
    print_dict(rewards, nesting=4)

    env_cfg.is_train = True
    env_cfg.rewards = rewards
    env_cfg.use_curriculum_reset = not args_cli.disable_curriculum_reset
    env_cfg.use_privileged_critic = args_cli.privileged_critic
    print(f"[INFO] Reset curriculum enabled: {env_cfg.use_curriculum_reset}")
    print(f"[INFO] Privileged critic enabled: {env_cfg.use_privileged_critic}")

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None, rewards=rewards)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # save resume path before creating a new log_dir
    if agent_cfg.resume:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env)

    # Apply gamma override if specified
    if args_cli.gamma is not None:
        agent_cfg.algorithm.gamma = args_cli.gamma
        print(f"[INFO] Override: gamma = {args_cli.gamma}")

    # create runner from rsl-rl
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    # write git state to logs
    runner.add_git_repo_to_log(__file__)
    # load the checkpoint
    if agent_cfg.resume:
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        # load previously trained model
        runner.load(resume_path)

    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    dump_pickle(os.path.join(log_dir, "params", "env.pkl"), env_cfg)
    dump_pickle(os.path.join(log_dir, "params", "agent.pkl"), agent_cfg)

    # run training
    # runner.learn() 定义在 rsl_rl/runners/on_policy_runner.py 中的 OnPolicyRunner 类
    # 本地路径: src/third_parties/rsl_rl_local/rsl_rl/runners/on_policy_runner.py
    # 该方法执行 PPO 训练循环：采集数据 -> 计算优势 -> 更新策略网络
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)

    num_gates = int(env.unwrapped._waypoints.shape[0]) if hasattr(env.unwrapped, "_waypoints") else 0
    gate_metric_summary = _build_training_gate_metric_summary(
        runner=runner,
        args_cli=args_cli,
        agent_cfg=agent_cfg,
        env_cfg=env_cfg,
        rewards=rewards,
        log_dir=log_dir,
        num_gates=num_gates,
    )
    summary_json_path = _write_training_gate_metric_summary_json(gate_metric_summary, log_dir)
    print(f"[INFO] Wrote training gate-metric summary to: {summary_json_path}")
    best_gates_passed = gate_metric_summary["episode_metrics"]["best"].get("Episode_Progress/gates_passed")
    if best_gates_passed is not None:
        print(f"[INFO] Best training Episode_Progress/gates_passed: {best_gates_passed:.4f}")
    if args_cli.gate_metrics_summary_csv:
        _append_training_gate_metric_summary_csv(args_cli.gate_metrics_summary_csv, gate_metric_summary)
        print(f"[INFO] Appended training gate-metric summary to: {os.path.abspath(args_cli.gate_metrics_summary_csv)}")

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
