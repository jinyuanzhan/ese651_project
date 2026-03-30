#!/bin/bash

set -euo pipefail

cd "$(dirname "$0")/../.."

COMMON_ARGS=(
  --task Isaac-Quadcopter-Race-v0
  --num_envs 8192
  --max_iterations 5000
  --headless
  --disable_curriculum_reset
)

run_trial() {
  local reward_set="$1"
  local run_name="$2"

  echo "============================================================"
  echo "Starting training with reward preset: ${reward_set}"
  echo "Run name: ${run_name}"
  echo "============================================================"

  python scripts/rsl_rl/train_race.py \
    "${COMMON_ARGS[@]}" \
    --reward_set "${reward_set}" \
    --run_name "${run_name}"
}

run_trial "baseline" "reward_baseline"
run_trial "trial1" "reward_trial_1"
run_trial "trial2" "reward_trial_2"
