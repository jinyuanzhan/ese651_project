#!/bin/bash
# Reward tuning v4 — gate_pass x progress x vel_align x curriculum
#
# Sweep: gate_pass = {50, 55, 65}, progress = {9, 15, 20},
#        vel_align = {0, 1.0, 2.0}, curriculum = {ON, OFF}
# Fixed: time = -0.08, speed = 0, entry_half_plane = 0
# Total: 3 x 3 x 3 x 2 = 54 experiments
#
# Results go to a NEW csv: logs/rsl_rl/train_reward_trials_v4.csv

set -uo pipefail
cd "$(dirname "$0")/../.."

# === Paths ===
SUMMARY_CSV="logs/rsl_rl/train_reward_trials_v4.csv"

# === Sweep axes ===
gate_pass_values=(50 65 85 30)
progress_values=(12 15 20 0)
vel_align_values=(0)
curriculum_values=(ON OFF)

# === Sweep: time_reward ===
time_reward_values=(0 -0.08 -0.12)

# === Fixed reward parameters ===
EHP=0
SR=0

COMMON_ARGS=(
  --task Isaac-Quadcopter-Race-v0
  --num_envs 4096
  --max_iterations 1500
  --headless
  --seed 42
  --gate_metrics_summary_csv "$SUMMARY_CSV"
  --entry_half_plane_reward "$EHP"
  --speed_reward "$SR"
)

total=$(( ${#gate_pass_values[@]} * ${#progress_values[@]} * ${#vel_align_values[@]} * ${#curriculum_values[@]} * ${#time_reward_values[@]} ))
skipped=0
failed=0
completed=0

echo "=========================================="
echo "  Reward Tuning v4 — gp x pr x va x curriculum"
echo "  gate_pass:   ${gate_pass_values[*]}"
echo "  progress:    ${progress_values[*]}"
echo "  vel_align:   ${vel_align_values[*]}"
echo "  curriculum:  ${curriculum_values[*]}"
echo "  time_reward: ${time_reward_values[*]}"
echo "  Fixed: speed=$SR  ehp=$EHP"
echo "  Total experiments: $total"
echo "  Gate metrics CSV: $SUMMARY_CSV"
echo "=========================================="

run_id=1
for gp in "${gate_pass_values[@]}"; do
  for pr in "${progress_values[@]}"; do
    for va in "${vel_align_values[@]}"; do
      for cur in "${curriculum_values[@]}"; do
        for tr in "${time_reward_values[@]}"; do

        RUN_NAME="v4_gp${gp}_pr${pr}_va${va}_t${tr}_cur${cur}"

        # --- Skip if already in CSV ---
        if [[ -f "$SUMMARY_CSV" ]] && grep -q ",${RUN_NAME}," "$SUMMARY_CSV"; then
          echo ""
          echo ">>> [$run_id/$total] $RUN_NAME — SKIP (already in CSV)"
          skipped=$((skipped + 1))
          run_id=$((run_id + 1))
          continue
        fi

        echo ""
        echo ">>> [$run_id/$total] $RUN_NAME"

        CUR_FLAG=()
        if [[ "$cur" == "OFF" ]]; then
          CUR_FLAG=(--disable_curriculum_reset)
        fi

        if python scripts/rsl_rl/train_race.py \
             "${COMMON_ARGS[@]}" \
             --run_name "$RUN_NAME" \
             --gate_pass_reward "$gp" \
             --progress_reward "$pr" \
             --vel_align_reward "$va" \
             --time_reward "$tr" \
             "${CUR_FLAG[@]}"; then
          echo "<<< Finished $RUN_NAME"
          completed=$((completed + 1))
        else
          echo "<<< FAILED $RUN_NAME (exit code $?)"
          failed=$((failed + 1))
        fi

        run_id=$((run_id + 1))
        sleep 10
        done
      done
    done
  done
done

echo ""
echo "=========================================="
echo "  v4 done!  $(date)"
echo "  Total:     $total"
echo "  Completed: $completed"
echo "  Skipped:   $skipped"
echo "  Failed:    $failed"
echo "=========================================="
