#!/bin/bash
# Reward tuning v3 — gate_pass + curriculum ablation
#
# Purpose: test gate_pass {50, 70} x curriculum {ON, OFF} with the gate-pass
#          bug fix (prev_x now uses actual value instead of forced 1.0).
#
# Sweep: gate_pass = {50, 70}, curriculum = {ON, OFF}  (4 experiments)
#
# Fixed (baseline from user):
#   progress        = 15
#   speed           = 0
#   entry_half_plane = 0
#   crash           = 0
#   action_smooth   = 0
#   time            = -0.1
#   death_cost      = -80

set -uo pipefail
cd "$(dirname "$0")/../.."

# === Paths ===
SUMMARY_CSV="logs/rsl_rl/train_reward_trials_gate_metrics.csv"

# === Sweep axes ===
gate_pass_values=(50 60)
curriculum_values=(ON OFF)

# === Fixed reward parameters ===
PR=15
TR=-0.08
EHP=0

COMMON_ARGS=(
  --task Isaac-Quadcopter-Race-v0
  --num_envs 4096
  --max_iterations 800
  --headless
  --seed 42
  --gate_metrics_summary_csv "$SUMMARY_CSV"
  --progress_reward "$PR"
  --time_reward "$TR"
  --entry_half_plane_reward "$EHP"
)

total=$(( ${#gate_pass_values[@]} * ${#curriculum_values[@]} ))
skipped=0
failed=0
completed=0

echo "=========================================="
echo "  Reward Tuning v3 — gate_pass x curriculum"
echo "  gate_pass:   ${gate_pass_values[*]}"
echo "  curriculum:  ${curriculum_values[*]}"
echo "  Fixed: pr=$PR  time=$TR  ehp=$EHP"
echo "  Total experiments: $total"
echo "  Gate metrics CSV: $SUMMARY_CSV"
echo "=========================================="

run_id=1
for gp in "${gate_pass_values[@]}"; do
  for cur in "${curriculum_values[@]}"; do

    RUN_NAME="v3_gp${gp}_pr${PR}_t${TR}_cur${cur}"

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

echo ""
echo "=========================================="
echo "  v3 done!  $(date)"
echo "  Total:     $total"
echo "  Completed: $completed"
echo "  Skipped:   $skipped"
echo "  Failed:    $failed"
echo "=========================================="
