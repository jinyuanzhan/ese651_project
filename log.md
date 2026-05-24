# Change Log

## 2026-03-30 Reward & Training Config Tuning

1. **Added time penalty** (`train_race.py`, `quadcopter_strategies.py`)
   - `time_reward_scale: -0.1`, per-step constant cost to push faster lap times

2. **Lowered crash threshold** (`quadcopter_env.py:683`)
   - `crashed > 100` → `crashed > 20`, prevent "rubbing gate frame" behavior

3. **Set min action std** (`rsl_rl_ppo_cfg.py:24`)
   - `min_std=0.0` → `min_std=0.01`, prevent premature exploration collapse

4. **Powerloop reward** — kept current `entry_half_plane=5.0` unchanged, pending training results

## 2026-03-30 Curriculum & Reset Tuning

5. **Raised mid-track initial speed** (`quadcopter_strategies.py`)
   - `0.5~1.5 m/s` → `1.5~4.0 m/s`, match actual racing approach speeds

6. **Added gate 6 weight** (`quadcopter_strategies.py`)
   - Late-stage weights now include gate 5 (+1.0) and gate 6 (+1.5) for chicane segment

7. **Lowered late-stage p_start** (`quadcopter_strategies.py`)
   - `0.35` → `0.20`, reduce gate 0 practice in favor of harder segments

8. **Smoother curriculum transitions** (`quadcopter_strategies.py`)
   - Added intermediate stage at 12% progress (gates 0-1), shifted later stages accordingly

9. **Widened position sampling** (`quadcopter_strategies.py`)
   - x: `[-3, -1]` → `[-3.8, -0.8]`, y: `[-0.3, 0.3]` → `[-0.6, 0.6]`, z: `[-0.2, 0.2]` → `[-0.3, 0.3]`

## 2026-03-30 Gate Pass & Reward Updates

10. **Relaxed gate passage x-threshold** (`quadcopter_strategies.py:122-126`)
    - Gate pass detection: `x <= 0` → `x <= 0.1m`, drone counts as passed 0.1m before the YOZ plane
    - Reduces missed detections for fast approaches

11. **Enabled entry_half_plane reward** (`quadcopter_strategies.py:176-189`, `train_race.py:119`)
    - Uncommented computation and reward dict entry, scale set to `1` (configurable)

12. **Enabled OOB XY termination** (`quadcopter_env.py:692-698`)
    - Out-of-bounds XY check now active, margin reduced from `6.0m` → `3.0m`

## 2026-04-08 Phase 2 Week 1 — Circle Track Zero-shot Setup

Goal: build the training infrastructure for Phase 2 Stage 1 (Circle Track).
Deliverables: Circle policy trained in sim, `controller_simple_policy.py`,
`policy.zip`, and a sim-vs-real comparison plot after the Pennovation session.

13. **Added circle track to tracks dict** (`quadcopter_env.py:1167-1176`)
    - New `'circle'` entry with the 4 waypoints given in the Phase 2-3 handout
    - Gates 0/1/3 at `z=0.75`, gate 2 at `z=1.75` — drone must climb 1m between
      gate 1→2 and descend 1m between gate 2→3 (not a planar circle)

14. **Switched default track to circle** (`quadcopter_env.py:873`)
    - `track_name = 'powerloop'` → `track_name = 'circle'`
    - Week 1 training targets the simpler circle track for sim2real calibration

15. **Added circle branch in reset curriculum** (`quadcopter_strategies.py:449-466`)
    - When `track_name == 'circle'`, use a simple uniform per-gate distribution
      (4 stages: 1 → 2 → 3 → 4 reachable gates) with decreasing `p_start`
    - Avoids powerloop-specific weight tweaks (`gate_weights[2] += 2.0`, etc.)
      that are meaningless for the 4-gate circle
    - Existing `_half_plane_shape_gate_mask` (line 99) is already safe on circle:
      seg lengths ≈ 2.12m > 2.0 threshold, adjacent yaw deltas = 90° (cos=0 < 0.95),
      so the mask is all-False and no shape reward is applied

16. **Added `circle_safe` reward preset** (`train_race.py:143-186`)
    - New preset optimised for safe/slow sim2real on circle:
      `gate_pass=30`, `progress=10`, `vel_align=0`, `action_smooth=-0.10`,
      `crash=-5`, `time=-0.02`, `death=-50`
    - Key change vs baseline: heavy `action_smooth` penalty to suppress
      high-frequency body-rate jitter (main sim2real failure mode on Crazyflie)
    - `vel_align` kept at 0 to avoid overlap with `progress` and keep the
      training signal calm for this safe/slow policy
    - Also added `"circle_safe"` to `--reward_set` argparse choices

## 2026-04-08 First Circle Training Run — `circle_safe` 1500 iter

17. **Trained first circle policy** (run `2026-04-08_23-01-21`, env `env_isaaclab`)
    - Command: `python scripts/rsl_rl/train_race.py --task Isaac-Quadcopter-Race-v0 --reward_set circle_safe --max_iterations 1500 --headless`
    - Wall time: ~55 min on 4096 envs, total timesteps: 147M
    - Final `vel_align_reward_scale` tweak (0.8 → 0) was applied before this run
    - **Best iteration: 734** (NOT the final), captured in `best_model.pt`:
      | Metric | Best (734) | Latest (1499) |
      |---|---|---|
      | `gates_passed/episode` | **37.90** | 27.39 |
      | `laps_completed/episode` | **9.06** | 6.58 |
      | `died/episode` | **0.79** | 2.04 |
      | `action_smooth` | -0.20 | -0.13 |
    - Policy is safely completing ~9 laps/30s in sim on the circle track,
      far exceeding Stage 2's 3-lap requirement
    - Gate pass counts are nearly uniform (9.84/9.80/9.14/9.11) — climb/descent
      gate 2 (z=1.75) is not a bottleneck
    - Post-peak drift in iterations 800-1500: policy slightly regresses on
      crash avoidance (died 0.79 → 2.04). Possible cause: action noise std
      collapsed to 0.01 and the policy exploited local investment patterns.
      Mitigation for future runs: raise `min_std` or stop at ~iter 800.
    - **Deploy with `best_model.pt` from this run, not `model_1499_*.pt`**

## 2026-04-09 Deployment Controller (Phase 2 Week 1 continued)

18. **Recorded sim playback video for best_model.pt** (run `2026-04-08_23-01-21`)
    - `python scripts/rsl_rl/play_race.py --task Isaac-Quadcopter-Race-v0 --num_envs 1 --load_run 2026-04-08_23-01-21 --checkpoint best_model.pt --headless --video --video_length 1600 --follow_robot 0`
    - Output: `logs/rsl_rl/quadcopter_direct/2026-04-08_23-01-21/videos/play/rl-video-step-0.mp4` (20MB, 1600 frames = 32s @50Hz)
    - stdout lap-timer output was buffered out by `tee` and not captured in the log file; visual verification only

19. **Cloned ROS2 deployment repo**
    - `git clone https://github.com/Jirl-upenn/ese651_sim2real.git ~/ese651_sim2real`
    - Reviewed: `controller_simple_policy.py` (Vineet reference), `controller_node.py`,
      `controller_utils.py`, `controller_params.py`, `config/controller.yaml`
    - Confirmed mocap→policy pipeline: `single_update(Odometry)` builds
      `mocap_pose = {x, R, v_b, w_b, ...}` from Vicon `/odom`, calls
      `policy.update(mocap_pose)`, publishes `/ctbr_cmd` + `/observations`

20. **Wrote our `controller_simple_policy.py`** (overwrites the TA reference)
    - Path: `~/ese651_sim2real/src/controller/controller/controller_simple_policy.py`
    - Backup of TA reference: `controller_simple_policy_vineet_reference.py.bak`
    - Project-repo copy (version-controlled alongside training code):
      `src/isaac_quad_sim2real/deploy_controller_simple_policy.py`
    - Delta vs TA reference:
      | Change | TA reference | Ours |
      |---|---|---|
      | Actor hidden dims | `[512, 512, 256, 128]` | **`[128, 128]`** (matches `rsl_rl_ppo_cfg.py`) |
      | `use_cond` kwarg | Not accepted (TypeError) | Accepted, raises if `True` (we don't train conditional) |
      | Docstrings | Minimal | Added observation-layout/checkpoint-format notes |
    - Observation layout is unchanged (36D: `lin_vel_b(3) + rot_wb(9) + 2×gate_corners_b(12)`)
      because our Isaac Lab training already matches the TA reference exactly
    - Actor weight loading: rsl_rl's `ActorCritic.state_dict()` uses the same
      `actor.{0,2,4}.{weight,bias}` key layout, so the filter
      `{k: v for k, v in ckpt["model_state_dict"].items() if "actor" in k}`
      with `strict=True` directly recovers our actor weights from `best_model.pt`

## Week 1 TODO (remaining)

- [x] Train circle policy (✅ `best_model.pt` at iter 734, see entry #17)
- [x] Record a sim flight video via `play_race.py --video` (✅ entry #18)
- [x] Clone `Jirl-upenn/ese651_sim2real`, write `controller_simple_policy.py` (✅ entries #19, #20)
- [ ] Fill ROS2 `controller.yaml` with our circle waypoints + initial_waypoint
      + `policy.path_per_drone` pointing at our `best_model.pt`
- [ ] Export `policy.zip` (params/ + best_model.pt) for hand-in
- [ ] Reserve a Pennovation slot on Ed; bring laptop + flash drive
- [ ] After deployment: run `process_bag_with_br_pos_export.py` on the ROS2 bag
  and produce the sim-vs-real comparison plot
