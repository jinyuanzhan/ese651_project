#!/bin/bash
# Reward tuning grid search for drone racing
#
# 7-axis sweep:
#   Original axes:
#     1. gate_pass:   25, 50, 80
#     2. time_reward: -0.01, -0.1
#     3. powerloop:   0, 0.5, 1.5
#   New axes:
#     4. entry_half_plane: 0, 0.5
#     5. curriculum:  on, off  (off = --disable_curriculum_reset)
#     6. progress:    10, 15, 30
#     7. enforce_powerloop_entry: true, false  (--enforce_valid_powerloop_entry)
#
# Total: 3×2×3 × 2×2×3×2 = 432 experiments

set -euo pipefail
cd "$(dirname "$0")/../.."

# === Original axes ===
gate_pass_values=(25 50 80)
time_reward_values=(-0.1 -0.5)
powerloop_values=(0 0.5 1.5)

# === New axes ===
entry_hp_values=(0 0.5)
curriculum_values=(on off)
progress_values=(10 30)
enforce_entry_values=(true false)

# === Fixed parameters ===
SUMMARY_CSV="logs/rsl_rl/train_reward_trials_gate_metrics.csv"

COMMON_ARGS=(
  --task Isaac-Quadcopter-Race-v0
  --num_envs 4096
  --max_iterations 800
  --headless
  --seed 42
  --gate_metrics_summary_csv "$SUMMARY_CSV"
)

total=$(( ${#gate_pass_values[@]} * ${#time_reward_values[@]} * ${#powerloop_values[@]} \
        * ${#entry_hp_values[@]} * ${#curriculum_values[@]} * ${#progress_values[@]} \
        * ${#enforce_entry_values[@]} ))

echo "=========================================="
echo "  Reward Tuning Grid Search (7-axis)"
echo "  gate_pass:       ${gate_pass_values[*]}"
echo "  time:            ${time_reward_values[*]}"
echo "  powerloop:       ${powerloop_values[*]}"
echo "  entry_hp:        ${entry_hp_values[*]}"
echo "  curriculum:      ${curriculum_values[*]}"
echo "  progress:        ${progress_values[*]}"
echo "  enforce_entry:   ${enforce_entry_values[*]}"
echo "  Total experiments: $total"
echo "  Gate metrics CSV: $SUMMARY_CSV"
echo "=========================================="

run_id=1
for gp in "${gate_pass_values[@]}"; do
  for tr in "${time_reward_values[@]}"; do
    for pl in "${powerloop_values[@]}"; do
      for ehp in "${entry_hp_values[@]}"; do
        for cur in "${curriculum_values[@]}"; do
          for pr in "${progress_values[@]}"; do
            for enf in "${enforce_entry_values[@]}"; do

              CUR_TAG=$(echo "$cur" | tr '[:lower:]' '[:upper:]')
              ENF_TAG=$(echo "$enf" | head -c1 | tr '[:lower:]' '[:upper:]')  # T or F
              RUN_NAME="gp${gp}_t${tr}_pl${pl}_ehp${ehp}_cur${CUR_TAG}_pr${pr}_enf${ENF_TAG}"
              echo ""
              echo ">>> [$run_id/$total] $RUN_NAME"

              RUN_ARGS=("${COMMON_ARGS[@]}")
              RUN_ARGS+=(--run_name "$RUN_NAME")
              RUN_ARGS+=(--gate_pass_reward "$gp")
              RUN_ARGS+=(--time_reward "$tr")
              RUN_ARGS+=(--powerloop_corridor_reward "$pl")
              RUN_ARGS+=(--entry_half_plane_reward "$ehp")
              RUN_ARGS+=(--progress_reward "$pr")
              RUN_ARGS+=(--enforce_valid_powerloop_entry "$enf")

              if [[ "$cur" == "off" ]]; then
                RUN_ARGS+=(--disable_curriculum_reset)
              fi

              python scripts/rsl_rl/train_race.py "${RUN_ARGS[@]}"

              echo "<<< Finished $RUN_NAME"
              run_id=$((run_id + 1))
              sleep 10
            done
          done
        done
      done
    done
  done
done

echo ""
echo "=========================================="
echo "  All $((run_id - 1)) experiments completed!"
echo "  $(date)"
echo "=========================================="
