#!/bin/bash
# Reward tuning grid search v2 — pruned based on 27-experiment analysis
#
# Pruning rationale (from gp25_t-0.1 experiments):
#   - enforce_entry=True:  consistently -0.86 laps → drop enfT
#   - curriculum=ON:       consistently -0.44 laps → drop curON
#   - entry_half_plane=0.5: unstable, -1 lap in best config → drop ehp0.5
#   - progress=10:         causes training collapse with curOFF+enfF → drop pr10
#
# Remaining sweep axes (3 × 2 × 3 = 18 experiments):
#   gate_pass:  25, 50, 80
#   time:       -0.1, -0.5
#   powerloop:  0, 0.5, 1.5
#
# Fixed (best from analysis):
#   entry_half_plane = 0
#   curriculum       = off
#   progress         = 30
#   enforce_entry    = false
#
# Features:
#   - Skips experiments already present in the CSV (safe re-run / resume)
#   - Continues on single-experiment failure instead of aborting

set -uo pipefail
cd "$(dirname "$0")/../.."

# === Sweep axes ===
gate_pass_values=(100 80 50)
time_reward_values=(-0.3 -0.1)
progress_values=(35 20 5)

# === Fixed parameters (pruned to best) ===
EHP=0
CUR=off
PL=0
ENF=false

# === Paths ===
SUMMARY_CSV="logs/rsl_rl/train_reward_trials_gate_metrics.csv"

COMMON_ARGS=(
  --task Isaac-Quadcopter-Race-v0
  --num_envs 8192
  --max_iterations 800
  --headless
  --seed 42
  --gate_metrics_summary_csv "$SUMMARY_CSV"
  --entry_half_plane_reward "$EHP"
  --powerloop_corridor_reward "$PL"
  --enforce_valid_powerloop_entry "$ENF"
  --disable_curriculum_reset
)

total=$(( ${#gate_pass_values[@]} * ${#time_reward_values[@]} * ${#progress_values[@]} ))
skipped=0
failed=0
completed=0

echo "=========================================="
echo "  Reward Tuning Grid Search v2 (pruned)"
echo "  gate_pass:   ${gate_pass_values[*]}"
echo "  time:        ${time_reward_values[*]}"
echo "  progress:    ${progress_values[*]}"
echo "  Fixed: ehp=$EHP, cur=$CUR, pl=$PL, enf=$ENF"
echo "  Total experiments: $total"
echo "  Gate metrics CSV:  $SUMMARY_CSV"
echo "=========================================="

run_id=1
for gp in "${gate_pass_values[@]}"; do
  for pr in "${progress_values[@]}"; do
    for tr in "${time_reward_values[@]}"; do

      RUN_NAME="gp${gp}_t${tr}_pl${PL}_ehp${EHP}_curOFF_pr${pr}_enfF"

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

      if python scripts/rsl_rl/train_race.py \
           "${COMMON_ARGS[@]}" \
           --run_name "$RUN_NAME" \
           --gate_pass_reward "$gp" \
           --time_reward "$tr" \
           --progress_reward "$pr"; then
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

echo ""
echo "=========================================="
echo "  Grid search v2 done!  $(date)"
echo "  Total:     $total"
echo "  Completed: $completed"
echo "  Skipped:   $skipped"
echo "  Failed:    $failed"
echo "=========================================="
