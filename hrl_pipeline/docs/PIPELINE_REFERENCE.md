# PIPELINE_REFERENCE

Authoritative reference for the full social-nav HRL stack: scripted expert, episode-data
lineage, HL training, LL training, and the end-to-end call chains.

**Conventions**

- All paths are repo-relative to `/home/xinyuan/habicrowd/Simulator/habitat-lab` (host) =
  `/habitat-lab` (container). Every script hard-codes the container path.
- `file.py:NN` anchors were re-read and verified against the working tree on 2026-08-17.
  Where a mined document disagreed with the file, the file wins; the disagreement is listed
  in *Contradictions resolved* at the end.
- Container entry for anything runnable:
  `docker exec -u root wxinyuan bash -c '. activate habitat && cd /habitat-lab && <cmd>'`
- Shorthands used in override strings throughout:
  - `V2 = social_nav/social_nav_hierarchical_overfit_v2.yaml`
  - `HL = habitat_baselines.rl.policy.agent_0.hierarchical_policy.high_level_policy`
  - `SK = habitat_baselines.rl.policy.agent_0.hierarchical_policy.defined_skills.backoff`
  - `R  = habitat.task.measurements.social_nav_reward`
  - `SD = /habitat-lab/hrl_pipeline`

---

# Expert

The "expert" is a two-tier scripted teacher, not a trained model. The HL half is a
monkeypatch harness; the LL half is three observation-driven velocity controllers whose
intelligence lives in two privileged sim-side sensors. Both halves run through the *real*
HRL machinery, so the teacher and the student execute identical plumbing.

## High level

### Strategy

- **Injected, not reimplemented.** `hrl_pipeline/scripted_hl_eval.py:370` replaces
  `SocialNavNeuralHighLevelPolicy.get_next_skill` with `patched`, which first calls the real
  network (`scripted_hl_eval.py:277-286`) and only then overwrites `new_skills[i]` and
  `pad.actions[i,0]` (`scripted_hl_eval.py:347-350`). Rollout storage, value head, log-probs,
  video and eval stats all stay well-formed.
- **`rule_markov` is a pure Markov function** of (7-D obs, prev HL action, three tick
  counters) — no world pose, no door geometry, no monotonic recede counter
  (`scripted_hl_eval.py:104-118`), so the student can reconstruct every teacher input from its
  own observation vector.
- **Trigger tight + immediate, release loose + hysteretic.** Trigger needs
  `d<3.0 ∧ |bearing|<1.75 ∧ |rel_heading|>π/2 ∧ approach>0.15` with `trig_ticks=1`
  (`scripted_hl_eval.py:119-129`); release needs 2 consecutive clear decisions inside a
  *wider* still-conflict region `d<3.5 / approach>0.05` (`scripted_hl_eval.py:137-143`).
- **The release predicate uses only ego-invariant terms** (distance, `human_approach_speed`).
  Bearing was tried and removed: the yield skill spins the robot while backing off, swinging
  the human's bearing out of any release cone and firing release mid-transit
  (`scripted_hl_eval.py:70-74`).
- **`rel_min_dist=1.0` applies only while still retreating** (`st["boff"] < wait_after_ticks`,
  `scripted_hl_eval.py:137-139`). In the wait phase nothing moves, `d` freezes at ~0.99, and
  the gate could never clear (`scripted_hl_eval.py:130-136`).
- **The yield splits by retreat DURATION, not distance.** After `wait_after_ticks=6` counted
  backoff decisions the HL switches to `wait` (`scripted_hl_eval.py:144-147`). A distance split
  (`wait` whenever `d<1.5`) deadlocked: stopping near the human can leave the robot standing
  *in* its way and the firm human freezes against it (`scripted_hl_eval.py:91-96`).
- **The regime switch reads the previous HL action alone** — `yielding = _prev_hl in (0,1)`
  (`scripted_hl_eval.py:118`) — so both backoff and wait enter the release branch, and the whole
  hysteresis is recoverable from the prev-action one-hot the student gets
  (`social_nav_hierarchical_v2.yaml:56`).
- **Per-env state resets on the episode boundary** detected via `masks[i]==0`; the same hook
  latches this episode's door for `rule_yield` by nearest stored `start_position`
  (`scripted_hl_eval.py:314-325`).
- **Every mode logs one jsonl row per HL decision** (feat6 + approach + prev_action +
  yield_branch + num_steps) to `RULE_LOG` (`scripted_hl_eval.py:353-366`). This file is the
  BC/DAPG demo format consumed by `bc_pretrain_hl.py` and by
  `SocialNavNeuralHighLevelPolicy._load_dapg_demos` (`social_nav_neural_policy.py:349-405`).
- **`MODE=learned` is pure pass-through**: the harness reads the network's own choice out of
  `pad.actions` and logs it without overriding (`scripted_hl_eval.py:340-341`) — how learned
  checkpoints are traced with the same tooling.
- **The legacy `rule_yield` teacher (M1) needed privileged world geometry**: it inverts the
  human's polar obs to world XZ and latches a "passed the door" flag by side-sign + 0.3 m
  perpendicular distance (`scripted_hl_eval.py:230-261`). `rule_markov` was built to delete
  exactly that latched state.
- **Nine modes exist**: `always_go`, `rule_backoff`, `rule_wait`, `rule_waitgo`, `rule_yield`,
  `rule_markov`, `smart_wait`, `always_backoff`, `learned` (`scripted_hl_eval.py:3-17`). Six are
  the solvability-certification set; `rule_markov` is production.

### Parameters

| Name | Value | Where |
|---|---|---|
| `RULE_MODE` | `rule_markov` (production); **code default `rule_backoff`** | `hrl_pipeline/scripted_hl_eval.py:33`; set in `_hl_train_stage1.sh:23`, `_recheck_all.sh:22`, `certify_solvability.sh:16` |
| `RULE_DIST` → `DIST_T` | 3.0 (default); 2.0 in `_eval_ll.sh` | `scripted_hl_eval.py:34`; override `hrl_pipeline/_eval_ll.sh:11` |
| `MARKOV_PARAMS.trig_dist` | `= DIST_T` = 3.0 | `scripted_hl_eval.py:65` |
| `.trig_bearing` | 1.75 rad | `scripted_hl_eval.py:66` |
| `.trig_rel_heading` | `π/2` | `scripted_hl_eval.py:67` |
| `.trig_approach` | 0.15 m/s | `scripted_hl_eval.py:68` |
| `.trig_ticks` | 1 (no debounce) | `scripted_hl_eval.py:69` |
| `.rel_dist` | 3.5 m (wider than trigger = the hysteresis band) | `scripted_hl_eval.py:75` |
| `.rel_approach` | 0.05 m/s | `scripted_hl_eval.py:76` |
| `.rel_min_dist` | 1.0 m (retreat phase only) | `scripted_hl_eval.py:82` |
| `.rel_ticks` | 2 | `scripted_hl_eval.py:83` |
| `.wait_after_ticks` | 6 → wait starts on the 8th decision | `scripted_hl_eval.py:97`, used `:138`, `:144-146` |
| legacy `conflict` predicate | `d<DIST_T ∧ |bearing|<1.75 ∧ |rel_heading|>π/2 ∧ human_rel_speed>0.1` | `scripted_hl_eval.py:216-225` |
| `smart_wait` extra gate | `human_rel_speed > 0.15` | `scripted_hl_eval.py:226-229` |
| `rule_yield` recede release | 3 consecutive decisions with `human_dist` increasing | `scripted_hl_eval.py:246-250`, `:255` |
| `rule_yield` door-pass margin | 0.3 m past the door line, robot's side | `scripted_hl_eval.py:244-245` |
| `control_interval` | 30 env steps (~1 s @30 Hz) | `social_nav_hierarchical.yaml:121`; consumed `hierarchical_policy.py:122-123`, `:483-493` |
| `hl_action_names` | `["backoff","wait","go_to_goal"]` → 0/1/2 | `social_nav_hierarchical_v2.yaml:51`; mapping `social_nav_neural_policy.py:84-90` |
| `input_feature_dim` | **7** (legacy default 6) | `social_nav_hierarchical_v2.yaml:55`; default `social_nav_neural_policy.py:98-99` |
| `use_prev_action` | true → GRU input width 10 | `social_nav_hierarchical_v2.yaml:56`; impl `social_nav_neural_policy.py:101-102`, `:630-644` |
| obs `social_nav_policy_state` | `[human_dist, human_bearing, human_rel_heading, human_rel_speed, goal_dist, goal_bearing, human_approach_speed]` | `social_nav_sensors.py:1771` (class), docstring `:1777-1793`, dim `:1831` |
| `include_approach_speed` | true (v2 benchmark); schema default false | `hssd_spot_human_social_nav_twoagent_v2.yaml:53-54`; default `default_structured_configs.py:710` |
| `approach_ema_alpha` | 0.2 | `default_structured_configs.py:711`; applied `social_nav_sensors.py:1810`, `:1925-1930`; reset `:1898` |
| `RULE_EVALS` | code default `"8"`; runs use 1 (cert/demos) or 3 (stability) | `scripted_hl_eval.py:38` → `:435` |
| `RULE_DATASET` | default `overfit_switch.json.gz`; always overridden | `scripted_hl_eval.py:35-37` → `:436` |
| `RULE_CONFIG` | code default `social_nav_hierarchical_overfit.yaml`; **all production runs pass `$V2`** | `scripted_hl_eval.py:40-42` → `:428`; overrides `certify_solvability.sh:13`, `_recheck_all.sh:10`, `_hl_train_stage1.sh:7` |
| `RULE_LOG` | default `$SD/hl_decisions_<MODE>.jsonl`; **append mode** | `scripted_hl_eval.py:45-47`, `:313` |
| `RULE_TRACE` | `""` (off); patches `HierarchicalPolicy.act` for per-step rows | `scripted_hl_eval.py:56`, `:373-421` |
| `EVAL_STATS` | `""` (off) → `habitat_baselines.eval.episode_stats_path` | `scripted_hl_eval.py:43`, `:439-442` |
| `STEP_CAP` | `""` (off); every production run passes 1200 | `scripted_hl_eval.py:443-448` → `habitat.environment.max_episode_steps` |
| `RULE_EXTRA` | `""`; whitespace-split, appended to `sys.argv` | `scripted_hl_eval.py:449-453` |
| `RULE_VIDEO` / `RULE_VIDEO_DIR` | `"0"` / `hrl_pipeline/video_<MODE>` | `scripted_hl_eval.py:39`, `:48` → `:432-433` |
| `DEBUG_PREV` | `"0"`; prints harness `prev_actions` vs teacher `_prev_hl` | `scripted_hl_eval.py:49`, `:331-337` |
| hard-coded harness overrides | `num_environments=1`, `habitat.seed=7`, `eval.should_load_ckpt=False`, `evaluate=True` | `scripted_hl_eval.py:426-438` |
| teacher-run net size (unused for decisions) | hidden 32 / 1 GRU layer (v2 overfit) vs 256 / 2 (base) | `social_nav_hierarchical_overfit_v2.yaml:58-59` vs `social_nav_hierarchical.yaml:129,131` |

### Gotchas (HL expert)

- The in-file `RULE_CONFIG` default resolves to the legacy 6-D benchmark, under which
  `rule_markov` raises `rule_markov needs the 7-dim policy state`
  (`scripted_hl_eval.py:110-114`). Every real run overrides to `$V2`.
- `MARKOV_PARAMS["trig_dist"]` is bound to `DIST_T` at **import** (`scripted_hl_eval.py:65`),
  so `RULE_DIST` silently moves the trigger while `rel_dist=3.5` / `rel_min_dist=1.0` stay
  fixed. `RULE_DIST > 3.5` makes the trigger wider than the release band.
- The backoff phase is **7** HL decisions, not 6: the triggering decision returns 0 with
  `boff` reset to 0 (`:126-129`), and `boff` only increments on subsequent yielding decisions
  (`:146`, checked `:144`).
- `_prev_hl[i] = a` is set for **all** modes including `learned` (`:351`), so the logged
  `prev_action` column is meaningful for learned traces too.
- The `feat` column of the decision jsonl is truncated to 6 dims forever for back-compat; the
  7th dim goes into a separate `approach` key (`:353-366`). `_load_dapg_demos` rebuilds it
  explicitly (`social_nav_neural_policy.py:349-405`).
- `RULE_LOG` is opened in **append** mode (`:313`) — every driver does `rm -f` first
  (e.g. `certify_solvability.sh:15`); forgetting silently concatenates runs.
- `_rule_action` is dead-ended for `rule_markov`: the `human_rel_speed > 0.1` motion gate and
  the `smart_wait` / recede logic do not apply to the production teacher.
- The per-episode door latch assumes sequential episodes (`num_environments=1`,
  `:434`) and picks the nearest stored spawn (`:321-325`).
- Skill INDEX comes from `defined_skills` order (`hierarchical_policy.py:96-100`), HL ACTION
  index from `hl_action_names` (`social_nav_neural_policy.py:84-90`). They coincide only
  because both lists are in the same order; `_skill_name_to_idx.get(name, 0)` falls back to
  skill 0 instead of erroring (`scripted_hl_eval.py:348`).
- `STEP_CAP` deliberately caps via the TASK episode length, not the evaluator's forced reset,
  which would slice long episodes into re-evaluated segments and skip episodes (`:445-447`).

## Low level

### Strategy

- **All three scripted skills are observation-only velocity controllers** writing into the
  located `base_velocity` slot (`social_nav_skills.py:86`, `:304-305`). Per-step pathfinding is
  pushed into task sensors because only they hold a sim handle.
- **`ClearCorridorYieldSkill` drives in REVERSE toward a privileged pocket**: it aims the
  robot's BACK at the waypoint (`back_target = -wp_delta`) and commands negative longitudinal
  velocity scaled by heading alignment (`social_nav_skills.py:281-302`).
- **Reverse-and-drive with a floored angular rate** replaced the old dead in-place spin:
  `ang = _proportional_turn(...)`, floored to `min_turn_rate` when the heading error exceeds
  `turn_thresh`, and `lin = -(v/lin_speed)*max(0,cos(angle))` (`social_nav_skills.py:287-293`).
- **Hard reverse-safety freeze**: `lin=ang=0` whenever `social_nav_policy_state[0] <
  human_stop_dist`, because the reverse drive is blind (`social_nav_skills.py:268-274`). Set to
  1.0 m so it sits above the human's 0.8 m firm block gate
  (`hssd_spot_human_social_nav_twoagent_v2.yaml:61`, `social_nav_hierarchical_v2.yaml:42`).
- **`hold_at_target: true` means the yield skill never self-terminates on arrival**: it parks
  and rotates to face the door using the sensor's `[door_dx, door_dz]` dims, handing control
  back only on the HL's 30-step interval — 1 Hz temporal abstraction instead of a 30 Hz replan
  storm (`social_nav_skills.py:275-280`, `:344-353`).
- **`WaitSkill` is not a no-op**: it holds position and rotates to face the goal using the
  straight-line `goal_world_delta` (`social_nav_skills.py:529-548`), pre-aligning for the resume.
- **`GoToGoalSkill` is deliberately human-blind** — plain navmesh geodesic, no RVO, no
  avoidance — because yielding is the exclusive job of the HL (`social_nav_skills.py:551-561`).
- **The yield pocket is chosen by `BackoffWaypointDeltaSensor._compute_target`**
  (`social_nav_sensors.py:1253`): door-anchored candidates plus a 360° ring around the robot out
  to `1.75×backoff_dist`, filtered to navigable + same-side-of-door + `≥obstacle_clearance` +
  `≥corridor_clearance` from the human's route + `≥human_goal_clearance` + reachable; the valid
  candidate nearest the robot wins.
- **The "human corridor" is the human's ACTUAL navmesh shortest path**
  (`social_nav_sensors.py:1237`, used `:1358`), not the straight `[pos, door, goal]` polyline,
  which under-covered the swept area and let pockets sit on the real route.
- **`retreat_corridor_clearance=0.5` also requires the RETREAT PATH to clear the human's
  route**, relaxing over a 4-pass cascade so a crossing pocket still beats no pocket.
- **When no pocket validates, corridor mode returns an explicit hold-in-place** (target =
  snapped robot position) with `branch=1` exposed as the last observation dim
  (`social_nav_sensors.py:1463`, `:1586-1587`), replacing the old silent reverse-to-spawn.
- **Target churn is suppressed by hysteresis**: the pocket is only re-picked when the human has
  moved `≥recompute_human_motion` AND the current target no longer clears the corridor
  (`social_nav_sensors.py:1496` `_target_still_valid`).
- **`LearnedYieldSkill` is the distilled replacement**: a frozen 22→128→128→2 tanh MLP over
  lidar(16)/human polar(4)/door dir(2), all normalized, emitting `base_vel` directly and never
  self-terminating (`social_nav_skills.py:395-410`, `:430-444`, `:500-511`).
- **DAgger labels are free**: the privileged pocket sensor keeps computing `wp_delta`
  regardless of who drives, so `_teacher_action` re-applies the scripted control law on the
  LEARNER's visited state and writes it to `LL_DAGGER` (`social_nav_skills.py:453-471`, `:474-498`).
- **`FlatNavSkill` subclasses `LearnedYieldSkill`** and appends goal polar for a 24-D flat
  end-to-end baseline (`social_nav_skills.py:689-703`), evaluated by pinning the HL with
  `RULE_MODE=always_backoff`.

### Parameters

| Name | Value | Where |
|---|---|---|
| backoff `hold_at_target` | true | `social_nav_hierarchical_v2.yaml:31`; used `social_nav_skills.py:218`, `:344` |
| backoff `proportional_turn` | true | `social_nav_hierarchical_v2.yaml:32`; `social_nav_skills.py:211`, `:287-299` |
| backoff `drive_while_turning` | true | `social_nav_hierarchical_v2.yaml:35`; `social_nav_skills.py:214`, `:287-293` |
| backoff `min_turn_rate` | 0.35 | `social_nav_hierarchical_v2.yaml:36`; `social_nav_skills.py:215`, `:291-292` |
| backoff `turn_thresh` | 0.3 rad (class default 0.1) | `social_nav_hierarchical_v2.yaml:37`; default `social_nav_skills.py:204` |
| backoff `human_stop_dist` | 1.0 m (class default 0.0 = off) | `social_nav_hierarchical_v2.yaml:42`; `social_nav_skills.py:223`, `:268-274` |
| backoff `lin_speed` / `max_back_speed` | 10.0 / 1.0 (class defaults) | `social_nav_skills.py:202-203` |
| backoff `start_stop_radius` / `start_done_radius` | 0.15 / 0.3 m | `social_nav_skills.py:207`, `:209`; done-radius only if `hold_at_target=false` (`:344-353`) |
| backoff `max_skill_steps` | 200 | `social_nav_hierarchical.yaml:142` |
| `_TURN_GAIN` | 2.0 → `ang = 2·angle/π` clipped to ±1 | `social_nav_skills.py:156`, used `:168` |
| go_to_goal `skill_data` | `proportional_turn/drive_while_turning: true, min_turn_rate 0.35, turn_thresh 0.3` | `social_nav_hierarchical_v2.yaml:45-48`; used `social_nav_skills.py:632-641` |
| go_to_goal `lin_speed` / `max_speed` / stop / done radii | 10.0 / 1.0 / 0.15 / 0.3 | `social_nav_skills.py:576-577`, `:582`, `:584`; done at `:676` |
| `agent_0_base_velocity` | `BaseVelNonCylinderAction`, `ang_speed 4.0`, `navmesh_offset [[0,0],[0.225,0]]`, `rotate_on_collision true` | `hssd_spot_human_social_nav_twoagent_v2.yaml:64-78` |
| action scaling | `clip(lin,-1,1)*longitudinal_lin_speed(10.0)`; `clip(ang,-1,1)*ang_speed(4.0)` | `actions.py:723-724`, `:729`; defaults `default_structured_configs.py:287`, `:290` |
| `backoff_waypoint_delta.yield_mode` | `corridor` (schema default `legacy`) | `hssd_..._v2.yaml:40`; default `default_structured_configs.py:763` |
| `.include_door_dir` | true (default false) → dims 3-4 | `hssd_..._v2.yaml:41`; default `:767`; appended `social_nav_sensors.py:1583-1585` |
| `.backoff_dist` | 2.0 m (ring 0.5×..1.75× = 1.0..3.5 m) | `hssd_..._v2.yaml:43`; default `:759` |
| `.include_branch` | true (default false) → last dim 0=pocket/1=hold | `hssd_..._v2.yaml:46`; default `:773`; appended `social_nav_sensors.py:1586-1587` |
| `.retreat_corridor_clearance` | 0.5 (default 0.0 = off) | `hssd_..._v2.yaml:49`; default `:777` |
| `.corridor_clearance` | 1.1 m (relaxed in the cascade) | `default_structured_configs.py:764` |
| `.human_goal_clearance` | 1.5 m | `default_structured_configs.py:765` |
| `.obstacle_clearance` | 0.3 m | `default_structured_configs.py:760` |
| `.recompute_human_motion` | 0.75 m | `default_structured_configs.py:769` |
| `backoff_waypoint_delta` layout | `[wp_dx, wp_dz, remaining, door_dx, door_dz, branch]` (6-D under v2) | `social_nav_sensors.py:1531-1533`, `:1583-1587`; obs space `:1200-1202` |
| `goal_waypoint_delta` | `[wp_dx, wp_dz, remaining]`; path recomputed every step; agent_0 only | `social_nav_sensors.py:1091` (class), `:1119` (`_compute_target`), helper `:1074-1087` |
| `goal_world_delta` | world (x,z) straight-line vector agent→goal | `social_nav_sensors.py:1021`; used `social_nav_skills.py:529-548` |
| `lidar_scan` | `num_rays 16, max_range 3.0, ray_step 0.1`; ray 0 = forward, CCW | `default_structured_configs.py:717-719`; sensor `social_nav_sensors.py:1715`; enabled `hssd_..._v2.yaml:10`, obs key `:99` |
| `LearnedYieldSkill` `model_path` / `LL_MODEL` | yaml default `$SD/ll_yield_bc.pth`; env `LL_MODEL` takes precedence | `social_nav_skills.py:384-385`; yaml `social_nav_hierarchical_v2_llyield.yaml:20` |
| `LearnedYieldSkill` in_dim / arch | 22 (16+4+2), `22→128→128→2` Tanh; FlatNav 24 | `social_nav_skills.py:396`, `:398-402`; flat `:689-703` |
| DAgger label freeze `human_stop_dist` | 1.0 (from skill_data) — label path only | `social_nav_skills.py:389`, used `:479-480` |
| `LL_LOG` / `LL_DAGGER` | `""` (off); a path enables scripted-demo / DAgger logging | `social_nav_skills.py:229`, `:307-326`; `:394`, `:453-471` |
| simulator rate | `ctrl_freq 120`, `ac_freq_ratio 4` → dt = 1/30 s per env step | `social_nav_hierarchical.yaml:45-48` |
| firm human (the adversary) | `rvo_reciprocity 0.0, rvo_agent_0_radius 0.25, rvo_navmesh_pref_vel true, rvo_resume_speed_floor 0.3`; `firm_mode true, block_clearance 0.8, block_hysteresis_steps 5` | `hssd_..._v2.yaml:30-33`, `:60-62` |

### Gotchas (LL expert)

- With `hold_at_target: true` the yield skill NEVER self-terminates
  (`social_nav_skills.py:344`), so only the 30-step control interval and `max_skill_steps=200`
  end a yield — `start_done_radius` is inert under the production config.
- The `human_stop_dist` freeze reads `social_nav_policy_state[0]`, and `feat_all` is only
  fetched when `human_stop_dist > 0` (`social_nav_skills.py:253-257`); setting it to 0 disables
  the reverse-safety floor entirely.
- `LearnedYieldSkill` does **not** apply the freeze to its own output; `_t_human_stop_dist` is
  used only for DAgger labels (`social_nav_skills.py:479`). Swapping in the learned LL removes
  a safety interlock the scripted LL has.
- `_teacher_action` re-implements the scripted law with HARD-CODED constants (0.15 stop radius,
  1.0 max back speed, 0.3 turn_thresh, 0.35 min_turn_rate, 10.0 lin_speed) that do NOT read
  `skill_data` (`social_nav_skills.py:484-497`). Retuning the yaml silently desynchronizes
  DAgger labels; the comment at `:476-478` says "keep in sync" but nothing enforces it.
- `BackOffSkill` is only an alias for `ClearCorridorYieldSkill` (`social_nav_skills.py:516`);
  the yaml key was deliberately left as `backoff` to avoid a Hydra dict-merge adding a stale
  4th skill (`social_nav_hierarchical_v2.yaml:24-29`).
- The pocket sensor prints `[YieldSensor] branch=...` to stdout on every recompute
  (`social_nav_sensors.py:1443`) — in the run logs, not behind a debug flag, and voluminous.
- `human_rel_speed` (feat[3]) is unsigned and ego-polluted — a parked human reads the robot's
  own speed (`social_nav_sensors.py:1783-1785` docstring). Every rule except `rule_markov` uses
  it; that is why the `>0.1` gate exists and why `rule_markov` switched to feat[6].
- `social_nav_policy_state` supports 2-agent setups only (`social_nav_sensors.py:1795-1797`
  docstring); the pocket sensor likewise hard-codes agent 1 as the human. Both privileged
  sensors return zeros for `agent_id != 0`.
- `social_nav_sensors.py:34` does `from IPython import embed` at import time — a stray debug
  import that makes IPython a hard runtime dependency of the reward module.

---

# Datasets

Five-stage lineage: geometric candidate generation → hand-clicking → scripted-mode
certification → pool/train36 consolidation → teacher-outcome classification and slicing.

### Strategy

- **Geometry is only a PRE-FILTER**; the real SWEET test is a runtime audit (`always_go` fails
  AND the expert succeeds), because offline navmesh checks do not predict whether the Spot body
  physically fits the door (`hrl_pipeline/scene_tools/gen_v3_datasets.py:19-20`).
- **Auto-generation was abandoned for hand-clicking**: the sweet rate was ~4%
  (`hrl_pipeline/doorcand_v3_audit.csv`: 78 rows → unsolv 53 / trivial 18 / pin 4 / SWEET 3).
  The click tool bakes the navmesh outline into the pixels so the pixel↔world map stays strictly
  linear (`hrl_pipeline/scene_tools/make_click_page.py:42`).
- **The clicker enforces a verified geometric RECIPE** (straight corridor, shallow robot_goal
  1–2.5 m past the door, DEEP human_goal 4–6 m past it): a deep robot_goal makes the robot grind
  walls, a shallow human_goal parks the human in the conflict zone so yield never releases
  (`hrl_pipeline/scene_tools/build_manual_episodes.py:96-99`).
- **Hard errors kill an entry, warnings are recorded** — every rejected candidate stays
  auditable (`hrl_pipeline/scene_tools/build_manual_episodes.py:50-54`).
- **Solvability is certified by WHICH scripted mode succeeds**, and the succeeding mode names
  the behavior PPO must discover: `rule_markov`→SWEET, any-other→SOLVABLE, none→UNCERT
  (`hrl_pipeline/certify_merge.py:6-11`).
- **The pool taxonomy separates "expert passes" from "expert fails but something else works"**,
  splitting failures by which certificate is stable: `always_go`→`fail_A_go`,
  wait/waitgo→`fail_B_wait`, yield→`fail_C_dither` (`hrl_pipeline/build_pool_v1.py:4-11`).
- **`trivial` splits by a ≥60-step efficiency gap** so `trivial_eff` becomes explicit
  "student can beat the expert on time" material (`build_pool_v1.py:5-7`).
- **Knife-edge certificates are labelled `challenge`** (eval-only); `dead` episodes are dropped
  from the dataset but still listed in `POOL_v1.csv` with a reason (`build_pool_v1.py:11-13`).
- **The holdout is defined at SCENE level**, not episode level — remaining pool episodes whose
  scenes are entirely unused by train36 (`build_train36.py:10-11`), which is what makes
  `holdout2` a zero-leakage transfer probe.
- **The authoritative split is by TEACHER OUTCOME, not geometric label**: "clean" means success
  AND zero collision on EVERY eval on record, evidence = a 3-eval cert ∪ two historical 1-eval
  passes (`hrl_pipeline/classify_teacher.py:3-17`).
- **Slowness is an attribute, not a defect**: `near_cap` / `inefficient` tag stable-success
  episodes without affecting eligibility (`classify_teacher.py:4-8`).
- **Flaky episodes are excluded from round-1 training** and from mechanism verdicts, but are
  put back for the final full-36 eval (`classify_teacher.py:10-12`, `build_fullset.py:3-4`).
- **A demo-collection failure is treated as NEW evidence**: `verify_demo_jsonl.py` downgrades
  that episode to flaky in `TEACHER_CERT3.csv` so it leaves BC/DAPG and the eval group together.
- **Demo collection uses a two-pass protocol** (run teacher → build a success-only dataset from
  the stats → re-run) because the evaluator visits episodes in its own order and the decision
  jsonl cannot otherwise be mapped back to ids (`_new_teacher_chain.sh:5-7`).
- **Failure upsampling is done by duplicating episodes in the DATASET file**, and the resulting
  rollout mix is MEASURED at train time (`metrics/rollout_fail_frac`) rather than assumed.
- **Analysis always joins stats to `TEACHER_CERT3.csv`** and defines episode-level success as
  ALL evals passing, collided as ANY eval colliding (`by_group.py:9-13`, `:40-42`).

### Parameters

| Name | Value | Where |
|---|---|---|
| `SNAP_MAX` (click→navmesh snap) | 0.3 m (exceeded = hard error) | `hrl_pipeline/scene_tools/build_manual_episodes.py:30`, used `:50-51` |
| `--half` (click-page half-extent) | 6.0 m | `hrl_pipeline/scene_tools/make_click_page.py:97`; mirrored `build_manual_episodes.py:31` |
| robot_goal depth warning | `> 3.0 m` past door (recipe 1–2.5) | `hrl_pipeline/scene_tools/build_manual_episodes.py:96-97` |
| human_goal depth warning | `< 3.0 m` past door (recipe 4–6) | `hrl_pipeline/scene_tools/build_manual_episodes.py:98-99` |
| lateral off-corridor tolerance | 1.2 m | `hrl_pipeline/scene_tools/build_manual_episodes.py:100-102` |
| gen_v3 `START_MIN/MAX` | 2.0 / 2.5 m | `hrl_pipeline/scene_tools/gen_v3_datasets.py:33` |
| gen_v3 `HGOAL_DEPTH_MIN/MAX` | 2.5 / 4.0 m | `hrl_pipeline/scene_tools/gen_v3_datasets.py:40` |
| gen_v3 `NAVMESH_RADIUS` / `CLEAR` | 0.25 / 0.3 (matched to runtime) | `hrl_pipeline/scene_tools/gen_v3_datasets.py:42` |
| gen_v3 `CORRIDOR_CLEAR` / `YIELD_REACH` | 1.1 / 2.8 m | `hrl_pipeline/scene_tools/gen_v3_datasets.py:45-46` |
| certification mode set | `always_go, rule_wait, rule_waitgo, smart_wait, rule_markov, rule_yield` | `certify_solvability.sh:14`; `certify_merge.py:17` |
| `STEP_CAP` (all certification runs) | 1200 | `certify_solvability.sh:16`; `_recheck_all.sh:22` |
| `RULE_EVALS` | 1 = cheap screen; 3 = stability certificate | `certify_solvability.sh:16` (=1); `_recheck_all.sh:28-29` (=3) vs `:30-34` (=1) |
| `habitat.seed` / `num_environments` | 7 / 1 for every certification and readout | `scripted_hl_eval.py:437`, `:434`; `_formal_run.sh:68`, `:66` |
| `RULE_CONFIG` | `$V2` for every certification run | `certify_solvability.sh:13`, `_recheck_all.sh:10` |
| pool stable-pass `ok()` | all evals ≥ 0.5 | `build_pool_v1.py:52-53` |
| `trivial_eff` gap threshold | 60 steps (expert − always_go) | `build_pool_v1.py:5-7`, `:65-67` |
| train36 quotas (docstring) | sweet 18 / trivial_eff 9 / trivial_zero 3 / fail_B 1 / fail_C 5 | `build_train36.py:3` — **disagrees with output** |
| **train36 ACTUAL composition** | 36 eps: sweet 18, trivial_eff 9, trivial_zero 4, fail_C_dither 4, fail_B_wait 1 | verified by counting `hrl_pipeline/TRAIN36.csv:2-37` |
| train36 sources | manual0805 19, manual 7, cleandev17_v2 6, bcsolid6_v2 2, dev20_parked 2 | `hrl_pipeline/TRAIN36.csv:2-37` |
| train36 scenes | 23 distinct over 36 episodes | computed from `TRAIN36.csv:2-37` |
| train36 per-scene cap | nominal 2, **ACTUAL max 3** | `build_train36.py:7-8`; verified from `TRAIN36.csv` |
| holdout candidates | manual0805 idx 26 (`106879080_174887211-e390`), idx 27 (`107734479_176000442-e461`) | `hrl_pipeline/TRAIN36.csv:40-41` |
| stable-success definition | every eval `success > 0.5` AND `did_collide < 0.5` | `classify_teacher.py:3-4` |
| stable-failure definition | every eval `success ≤ 0.5` (collision not considered) | `classify_teacher.py:9` |
| flaky definition | anything mixed / success-with-collision / history contradiction | `classify_teacher.py:10-12` |
| attr `inefficient` | cert3 mean steps ≥ 900, OR pool label `trivial_eff` | `classify_teacher.py:5-6` |
| attr `near_cap` | any eval steps ≥ 1150 (vs the 1200 cap) | `classify_teacher.py:5` |
| **TEACHER_CERT3 ACTUAL counts** | 28 stable-success / 4 stable-failure / 4 flaky | counted from `hrl_pipeline/TEACHER_CERT3.csv:2-37` |
| stable-failure ids | 2, 3, 20, 25 | `TEACHER_CERT3.csv` |
| flaky ids | 0, 1, 4, 29 | `TEACHER_CERT3.csv` (id0 line 2: `2/3`, `1/3` collide) |
| `quick<N>` composition | all stable-failure + 6 sweet + 3 trivial_eff + 2 trivial_zero, hardest-first, fresh scenes first | `build_quickset.py:3-5` |
| `quick<N>` scene assertion | ≥ 8 distinct scenes (num_environments=8) | `build_quickset.py:11-12` |
| upsample factor | ×3 for stable-failure | `build_quickset.py:6`; `build_fullset.py:2` |
| QUICK stem (default vs used) | scripts default `quick17`; **`quick15` is what exists and ran** | `_mech_check.sh:16`, `_dapg_dryrun.sh:14`; `data/social_nav_episode_quick15*.json.gz` |
| `SPLIT_SCENES` | 0 for the mech check, 1 for the formal 3M run | `_mech_check.sh:20` vs `_formal_run.sh:25` |
| `episode_stats_path` | `""` by default; set per run via CLI | default `social_nav_hierarchical_v2.yaml:17`, `habitat-baselines/.../default_structured_configs.py:53`; set `_formal_run.sh:70` |
| eval readout protocol | `evals_per_ep=3`, seed 7, `max_episode_steps=1200`, `video_option=[]`, on the FULL train36_v1 | `_formal_run.sh:63-70`; `_mech_check.sh:48-55` |

### Dataset inventory (verified on disk)

| Dataset | Size | Built by |
|---|---|---|
| `doorcand_v3/v4/v5/v6` | 78 / 114 / 93 / 93 eps | `hrl_pipeline/scene_tools/gen_v3_datasets.py` |
| `manual` / `manual0805` | 17 eps (7 scenes) / 30 eps (22 scenes) | `hrl_pipeline/scene_tools/build_manual_episodes.py` from `clicks.json` / `0805.json` |
| `pool_v1` | 76 kept (sweet 35, trivial_eff 16, trivial_zero 10, fail_C 9, challenge 5, fail_B 1); 19 dead excluded, CSV-listed | `hrl_pipeline/build_pool_v1.py` |
| `train36_v1` | 36 eps / 23 scenes | `hrl_pipeline/build_train36.py` |
| `stableok` | 28 eps (the stable-success set), ascending train36_id | `hrl_pipeline/build_stableset.py` |
| `train32_up3` | 40 entries = 28 ss + 4 sf ×3, 23 scenes, flaky out | `hrl_pipeline/build_fullset.py` |
| `quick15` / `quick15_up3` | 15 eps / 12 scenes; 23 entries | `hrl_pipeline/build_quickset.py` |
| `train36_bc` / `train36_bcwait` | 31 / 30 eps (inline, success-only) | `_hl_train_stage1.sh:10-19` / `_new_teacher_chain.sh:29-45` |
| `es8` / `es14` / `l0set` / `holdout2` | 8 / 14 / 5 / 2 eps | **no build script in repo** — reconstructed from `info["pool"]["pool_v1_id"]` |
| `cd17fam` | 6 eps (cleandev17_v2 indices 0,1,2,3,4,7) | inline at `_recheck_all.sh:12-18` |

### Gotchas (datasets)

- **GOTCHA 1 — `episode_id` is IGNORED at load time.** `RearrangeDatasetV0.from_json`
  overwrites it with the positional index: `rearrangement_episode.episode_id = str(i)` at
  `habitat-lab/habitat/datasets/rearrange/rearrange_dataset.py:83` (identical in `from_binary`
  at `:203`). Identity is POSITION in the `"episodes"` list; the configs use
  `type: RearrangeDataset-v0` (`habitat/config/habitat/dataset/rearrangement/hssd.yaml:7`).
  Every id in `POOL_v1.csv` / `TRAIN36.csv` / `TEACHER_CERT3.csv` is a list index, and
  reordering a dataset file silently re-labels everything.
- **GOTCHA 2 — the eval stats key format.** `habitat_evaluator.py:509` writes
  `f"{scene_ep[0]}|{scene_ep[1]}|{count}"` where `scene_ep[0]` is the FULL absolute
  scene_instance path (`:386`), `scene_ep[1]` is the load-time positional id (`:387`) and
  `count` is a 1-BASED per-(scene,episode) counter (`:389`, stored `:392`). Every consumer does
  `k.split("|")[1]` — except `hrl_pipeline/scene_tools/audit_summary.py:22`, which uses `k.rsplit("|",2)`. The
  file is written **only** if `episode_stats_path` is non-empty
  (`habitat_evaluator.py:500-503`); forgetting the override silently produces no stats.
- `build_train36.py`'s docstring, code and output all disagree. Read `TRAIN36.csv`, not the
  docstring.
- The "max 2 episodes per scene" rule (`build_train36.py:7`) is not what the output has: four
  scenes carry 3 episodes each (`104348361_171513414`, `103997541_171030615`, `102344439`,
  `103997586_171030666`).
- `build_pool_v1.py`'s cleandev17 branch has no final `else`; on different stats an episode
  could be written to neither the dataset nor the CSV. (It did not fire on this data.)
- The `trivial_eff`/`trivial_zero` split uses only the FIRST eval's step count, even when the
  success verdict came from a 3-eval certificate.
- Several pool labels rest on 1-eval evidence only, and the `note` column says so. Only
  `manual`, `manual0805` and the cd17 family carry 3-eval certificates.
- `classify_teacher.py`'s bc-evidence remap (`ok = sorted(ids where wait_t36 succeeded)`, then
  bcwait position `i → ok[i]`) is only correct because `train36_bcwait` was built by exactly
  that construction (`_new_teacher_chain.sh:29-45`). Rebuild it any other way and the `bc`
  column attaches to the wrong episodes with no assertion firing.
- `verify_demo_jsonl.py` REWRITES `TEACHER_CERT3.csv` in place. Re-running
  `classify_teacher.py` afterwards silently erases any demo-pass downgrade. Order matters:
  classify → build_stableset → collect demos → verify → build_quickset/build_fullset.
- `stableok`'s positional order (ascending train36_id) is load-bearing: `verify_demo_jsonl.py`
  maps a demo segment back to a train36 id purely by that ordering.
- ×3 upsampling is DATASET-level and the per-env scene split dilutes it (measured 0.52→0.27,
  `_mech_check.sh:18-19`). `SPLIT_SCENES=0` avoids this but OOMs on the 23-scene set
  (`_formal_run.sh:19-24`), so the formal run's real fail fraction is only knowable from the
  measured `metrics/rollout_fail_frac`.
- `by_group.py` assumes stats from the FULL train36_v1 so positional ids equal train36 ids
  (`by_group.py:5-7`). Handing it `quick15` / `es14` / `holdout2` stats mis-keys every row.
- `cert_*` filenames do not always match the dataset stem: `cert_solid6_*` and `cert_ms4_*`
  come from `_recheck_all.sh:30-34`, not from `certify_solvability.sh`; `cert_cd17_*` exists
  although no `cd17` dataset does (the underlying set is `cleandev17_v2`).
- `make_video_index.py` is NOT read-only: it renames mp4s in place via `os.replace`, matching
  decision-log segments to episodes heuristically by duration (60-step tolerance).
- `dev20_parked` is a diagnostic set, not a conflict set: its `human_start == human_goal` in
  all 20 episodes. Only 8 survive into the pool as `trivial_zero`.
- The container clock is UTC and the host is UTC-4; host `ls` mtimes are 4 h behind the
  timestamps inside run logs and `outputs/<date>/<HH-MM-SS>` names.

---

# HL training

A tiny recurrent categorical policy (10-D input → 1-layer GRU(32) → 3-way `CategoricalNet` +
`CriticHead`, **4356 trainable params**, `hrl_pipeline/_ppo_nr1.log:114`) replaces the
`rule_markov` teacher over three skills, re-deciding every 30 low-level steps. Three stages:
BC → PPO fine-tune → deterministic eval of the extracted HL weights.

## Strategy

- **BC first, PPO second.** PPO from scratch converges to an avoid-timeout local optimum and
  never discovers go→yield→go, so the teacher is cloned into the SAME modules the neural HL
  uses (GRU state encoder + `CategoricalNet` head) and PPO only fine-tunes
  (`hrl_pipeline/bc_pretrain_hl.py:1-6`).
- **The BC warm start loads only `state_encoder` + `policy`**; the critic is deliberately left
  random and PPO must learn it (`social_nav_neural_policy.py:228-251`; verified: the extracted
  `.pth` has exactly those two keys, shapes `rnn.weight_ih_l0 (96,10)`, `rnn.weight_hh_l0
  (96,32)`, `linear.weight (3,32)`).
- **Critic warmup exists because a random critic's normalized advantages destroy the cloned
  actor** within a few dozen updates. During warmup features are detached so the value loss
  reaches only the critic head and the shared GRU stays frozen
  (`social_nav_neural_policy.py:171-172`, `:465`).
- **A critic-specific learning rate (3e-2 vs actor 5e-5) fixes a timescale problem, not a
  capacity problem**: Adam moves each weight ~lr/step, so a |w|≈1 orthogonal-init value head
  needs O(1e5) steps to reach ~±40 returns (`habitat-baselines/.../rl/ppo/ppo.py:130-146`).
- **`critic_output_scale` is the DIAGNOSTIC alternative** — scale the head output so the critic
  only has to rotate, not grow; kept as a probe, default off
  (`social_nav_neural_policy.py:196`).
- **`bc_anchor` (L2-SP) is a weight-space trust region**, motivated by PPO's per-decision credit
  assignment rewarding yield-trimming while the later collision cost lands on other decisions
  (`social_nav_neural_policy.py:182-183`, `:484-497`). Legacy; off in all current runs.
- **DAPG anchors in ACTION space instead of weight space**: a class-weighted CE on teacher
  success-only demos, linearly decayed on a clock that starts when critic warmup ends, entering
  the objective through the aux channel (`social_nav_neural_policy.py:499-527`;
  `hrl_ppo.py:119`).
- **DAPG demo batching is a deterministic round-robin** over demo episodes, so every episode
  recurs with the same cadence and no RNG state is touched
  (`social_nav_neural_policy.py:511-527`).
- **The DAPG gradient probe measures `‖g_dapg‖/‖g_ppo‖` and their cosine**
  (`hrl_ppo.py:129-160`), because a nominally weak coefficient can still be a strong pull once
  batch sizes differ. Measured verdict on arm C: cosine median −0.669 → real suppression → the
  formal round dropped DAPG.
- **`prev_action` is fed as a pre-GRU one-hot** (masked to zero on episode starts) so the
  teacher's hysteresis tick counters are recoverable from the student's own inputs rather than
  hidden teacher state (`social_nav_neural_policy.py:101-102`, `:630-644`).
- **The act-time recurrence manually mirrors `single_forward`** (mask-reset + only replanning
  envs advance their hidden state) so act-time and `evaluate_actions` recurrences match and the
  PPO importance ratio stays ≈1 on the first epoch (`social_nav_neural_policy.py:690-713`).
- **HRLPPO normalizes advantages over VALID slots only** (`hrl_ppo.py:20-49`), because the HRL
  buffer writes one transition per high-level decision and empty slots would otherwise dominate
  mean/std; every loss is reduced over `batch["loss_mask"]` (`hrl_ppo.py:52`, `:62-63`).
- **`SPLIT_SCENES=0` makes every env cycle the FULL episode set**, because per-env scene
  ownership flattens dataset-level failure upsampling back toward uniform
  (`habitat_env_factory.py:49-61`). The formal run nevertheless uses `SPLIT_SCENES=1` because 0
  OOMs on the 23-scene set (~18 GB by u60, two crashes), and the resulting dilution is MEASURED
  (`_formal_run.sh:19-25`).
- **Rollout episode-mix is measured, not assumed**: per-class and per-episode frame counts are
  accumulated and published as `metrics/rollout_fail_frac` + `checkpoint_folder/
  rollout_counts.json` (`ppo_trainer.py:429`, `:670`, `:684`).
- **The three new reward terms are potential-based / terminal-only by construction**:
  efficiency pays only on the success step, the corridor term telescopes, and the release bonus
  is once-per-episode and gated on a kinematic go-counterfactual clearance.
- **Reward thresholds are not tuned free parameters**: `corridor_safe_clear=1.0` and
  `release_mpd_min=0.75` come from the yield-geometry diagnosis, cited at `_formal_run.sh:6-7`.
- **Evaluation is deliberately decoupled from the RL checkpoint**: HL weights are extracted to
  the `pretrained_hl_weights` format and re-loaded with `should_load_ckpt=False`, so eval is
  deterministic-argmax and the (random) critic is irrelevant.
- **Readout points are pre-registered** (1M/1.5M/2M/3M = ckpt.9/14/19/latest,
  `_formal_run.sh:9-12`) so the headline is a protocol point, not a cherry-picked peak.
- **Never evaluate a crashed run**: a missing-weights eval silently falls back to random init,
  so the formal script aborts the eval chain when training exits non-zero or the endpoint
  checkpoint is absent (`_formal_run.sh:52-58`).

## Parameters

### Policy / architecture

| Name | Value | Where |
|---|---|---|
| `high_level_policy.name` | `SocialNavNeuralHighLevelPolicy` | `social_nav_hierarchical.yaml:119` |
| `control_interval` | 30 | `social_nav_hierarchical.yaml:121`; used `hierarchical_policy.py:122-123`, `:483-493` |
| `hl_action_names` | `["backoff","wait","go_to_goal"]` | `social_nav_hierarchical_v2.yaml:51`; default `social_nav_neural_policy.py:84-85` |
| `input_feature_dim` | 7 (default 6) | `social_nav_hierarchical_v2.yaml:55`; `social_nav_neural_policy.py:98-99` |
| `use_prev_action` | true → width 7+3 = 10 | `social_nav_hierarchical_v2.yaml:56`; `social_nav_neural_policy.py:101-102` |
| `hidden_dim` | **32** (base yaml 256) | `social_nav_hierarchical_overfit_v2.yaml:58`; base `social_nav_hierarchical.yaml:129` |
| `num_rnn_layers` | **1** (base yaml 2) | `social_nav_hierarchical_overfit_v2.yaml:59`; base `social_nav_hierarchical.yaml:131` |
| `rnn_type` | GRU | `social_nav_hierarchical.yaml:130` |
| `termination_obs_name` | null → PDDL-success fallback | `social_nav_hierarchical.yaml:133`; used `social_nav_neural_policy.py:560-569` |
| `map_obs_key` / `map_feature_dim` | `""` / 64 → map branch OFF, `_map_dim = 0` | `social_nav_neural_policy.py:121`, `:143`; no script sets it |

### PPO / warm start (all CLI overrides; `HL`/`R` as defined at the top)

| Name | Value | Where (override string / file) |
|---|---|---|
| `pretrained_hl_weights` | `$SD/bc_stable_hl.pth` (train); `$SD/nr1_ck{9,14,19}_hl.pth`, `nr1_final_hl.pth` (eval) | `+$HL.pretrained_hl_weights=...` — `_formal_run.sh:36`, `:65`; `_seed_runs.sh:22`; `_mech_check.sh:29`. Default `""` `social_nav_neural_policy.py:228` |
| `critic_warmup_calls` | 120 (formal/seed/mech); 16 (dry-run) | `+$HL.critic_warmup_calls=120` — `_formal_run.sh:37`, `_seed_runs.sh:23`, `_mech_check.sh:30`; 16 at `_dapg_dryrun.sh:22` |
| `bc_anchor_coef` | 0.0 (OFF everywhere now) | default `social_nav_neural_policy.py:182-183` |
| `critic_output_scale` | 1.0 (probe OFF) | default `social_nav_neural_policy.py:196` |
| `dapg_demo_path` | `$SD/hl_decisions_dapg_success.jsonl` (arm C only) | `+$HL.dapg_demo_path=...` — `_mech_check.sh:70`, `_dapg_dryrun.sh:23` |
| `dapg_coef` | **0.1 actually used** (script default 0.3) | `_mech_check.sh:17` (default) vs `_mech_check_C.log:78`, `:91` (actual) |
| `dapg_decay_calls` | 240 (= 120 updates) | `+$HL.dapg_decay_calls=240` — `_mech_check.sh:71`; default `social_nav_neural_policy.py:209-210` |
| `dapg_batch_eps` | 8 | `+$HL.dapg_batch_eps=8` — `_mech_check.sh:71`; default `:212` |
| `rl.ppo.lr` | 5e-5 (yaml default 2.5e-4) | `_formal_run.sh:42`, `_seed_runs.sh:28`, `_mech_check.sh:35`; yaml `social_nav_hierarchical.yaml:83` |
| `rl.ppo.critic_lr` | 3e-2 (schema default 0.0 = single group) | `_formal_run.sh:38`; impl `ppo.py:53`, `:130-146` |
| `rl.ppo.gamma` | 0.99 | `_formal_run.sh:39`; yaml `social_nav_hierarchical.yaml:88` |
| `rl.ppo.entropy_coef` | 0.0 (yaml 0.004 / base 0.001) | `_formal_run.sh:41`; yaml `social_nav_hierarchical_overfit_v2.yaml:53` |
| `rl.ppo.value_loss_coef` | 0.05 (yaml 0.5) | `_formal_run.sh:41`; yaml `social_nav_hierarchical.yaml:81` |
| `rl.ppo.max_grad_norm` | 2.0 (yaml 10.0 / base 0.5) | `_formal_run.sh:42`; clip at `ppo.py:389-392` |
| `rl.ppo.num_steps` | 1024 → 8192 frames/update at 8 envs | `_formal_run.sh:40`; yaml default 256 `social_nav_hierarchical.yaml:86` |
| `rl.ppo.num_mini_batch` | 1 → exactly 2 `evaluate_actions` calls/update | `_formal_run.sh:40`; yaml default 4 `:80` |
| `rl.ppo.ppo_epoch` | 2 | `social_nav_hierarchical.yaml:79` |
| `rl.ppo.clip_param` | 0.2 (also the clipped value loss) | `social_nav_hierarchical.yaml:78`; `hrl_ppo.py:88-89`, `:101-105` |
| `rl.ppo.use_normalized_advantage` | True | `social_nav_hierarchical_overfit_v2.yaml:51` |
| `rl.ppo.hidden_size` | 32 (must equal `hidden_dim`) | `social_nav_hierarchical_overfit_v2.yaml:50` |
| `rl.ppo.tau` / `use_gae` | 0.95 / true | `social_nav_hierarchical.yaml:87-89` |
| `updater_name` / `distrib_updater_name` | `HRLPPO` / `HRLDDPPO` | `social_nav_hierarchical.yaml:56-57`; classes `hrl_ppo.py` |
| `rollout_storage_name` | `HrlRolloutStorage` (supplies `loss_mask`) | `social_nav_hierarchical.yaml:68` |
| `num_environments` | 8 | `social_nav_hierarchical_overfit_v2.yaml:41` |
| `data_path` | `train32_up3` (formal/seed); `quick15{,_up3}` (mech); `train36_v1`/`holdout2` (eval) | `_formal_run.sh:35`, `:67`; `_mech_check.sh:28`, `:52` |
| `habitat.seed` | default (seed #1), 200, 300 for training; **7 for every eval** | `_seed_runs.sh:17`, `:20`; `_formal_run.sh:68` |
| `total_num_steps` | 3.0e6 (formal); 1.5e6 (seeds/mech); 2.5e5 (dry-run) | `_formal_run.sh:46`; `_seed_runs.sh:32`; `_mech_check.sh:39`; `_dapg_dryrun.sh:33` |
| `num_checkpoints` | 30 (formal) / 15 (seeds/mech) → ckpt every 100k | `_formal_run.sh:46`; `_seed_runs.sh:32` |
| `checkpoint_folder` / `tensorboard_dir` / `video_dir` | `checkpoints_nr1` / `tb_nr1` / `video_nr1` etc. | `_formal_run.sh:47-48`; `_seed_runs.sh:33-34`; `_mech_check.sh:40-41` |
| `SPLIT_SCENES` | 1 (formal/seed) vs 0 (mech/dry-run) | `_formal_run.sh:25`, `_seed_runs.sh:12` vs `_mech_check.sh:20`, `_dapg_dryrun.sh:16`; impl `habitat_env_factory.py:49` |
| `DISABLE_CUDNN` | 1 (formal/seed); unset in mech | `_formal_run.sh:28`, `_seed_runs.sh:13`; impl `ppo_trainer.py:99-101`; confirmed `_ppo_nr1.log:6` |
| `DAPG_PROBE` | 1 (arm C and dry-run only) | `_mech_check.sh:69`, `_dapg_dryrun.sh:19`; read `hrl_ppo.py:133` |

### Reward (`R = habitat.task.measurements.social_nav_reward`)

| Name | Value | Where |
|---|---|---|
| `habitat.task.success_reward` | 50.0 (yaml 30.0 / base 10.0) | `_formal_run.sh:43`; applied OUTSIDE the measure at `habitat/core/environments.py:79-80` |
| `habitat.task.slack_reward` | −0.01 (base −0.1) | `social_nav_hierarchical_overfit_v2.yaml:23`; applied `environments.py:75` |
| `$R.collide_penalty` | 30.0 (yaml 5.0; schema 1.0) | `_formal_run.sh:44`; applied `social_nav_sensors.py:260-261`; schema `default_structured_configs.py:1531` |
| `$R.end_on_collide` | true | `social_nav_hierarchical_overfit_v2.yaml:26`; schema `:1489`; used `social_nav_sensors.py:262`, `:270` |
| `$R.safe_dis_min` | **0.0** → proximity penalty fully disabled | `_formal_run.sh:44`; guard `social_nav_sensors.py:284` can never fire |
| `$R.yield_dis` | 3.0 (schema 1.0) | `social_nav_hierarchical_overfit_v2.yaml:35`; used `social_nav_sensors.py:296` |
| `$R.backoff_reward` | 3.0 (base 2.0) | `social_nav_hierarchical_overfit_v2.yaml:36` |
| `$R.goal_progress_reward` | 2.0 | `social_nav_hierarchical_overfit_v2.yaml:34`; used `social_nav_sensors.py:277-281` |
| `$R.eff_success_reward` | **20.0** (formal+seed); 0.0 = OFF in mech arms | `_formal_run.sh:45`, `_seed_runs.sh:31`; schema `default_structured_configs.py:1495`; impl `social_nav_sensors.py:364-373` |
| `$R.eff_step_cap` | 1200.0 (schema, never overridden) | `default_structured_configs.py:1496`; used `social_nav_sensors.py:78`, `:370` |
| `$R.corridor_potential_coef` | **3.0** (formal+seed); 0.0 in mech | `_formal_run.sh:45`; schema `:1502`; impl `social_nav_sensors.py:313-326` |
| `$R.corridor_safe_clear` | 1.0 (schema; from the yield-geometry diagnosis) | `default_structured_configs.py:1503`; `social_nav_sensors.py:80` |
| `$R.release_bonus` | **3.0** (formal+seed); 0.0 in mech | `_formal_run.sh:45`; schema `:1507`; impl `social_nav_sensors.py:332-356` |
| `$R.release_mpd_min` | 0.75 (schema) | `default_structured_configs.py:1508`; `social_nav_sensors.py:82` |
| `$R.release_hold_steps` | 30 (schema) | `default_structured_configs.py:1509`; `social_nav_sensors.py:83` |
| `$R.release_robot_speed` | 0.008 m/step (schema) | `default_structured_configs.py:1512`; `social_nav_sensors.py:84` |
| `explore_reward` / `near_human_bonus` / `facing_human_reward` | −1.0 (all disabled) | `default_structured_configs.py:1514-1516`, `:1474` |
| `social_nav_to_pos_succ.success_distance` | 0.4 (schema 0.2) | `hssd_spot_human_social_nav_twoagent.yaml:128`; schema `default_structured_configs.py:1122`; read back `social_nav_sensors.py:364-368` |
| `habitat.environment.max_episode_steps` | **1500 training / 1200 eval** | `social_nav_hierarchical_overfit_v2.yaml:17` vs `_formal_run.sh:68` |

### BC stage

| Name | Value | Where |
|---|---|---|
| `BC_JSONL` / `BC_OUT` / `BC_EPOCHS` | defaults `hl_decisions_rule_markov.jsonl` / `bc_markov_hl.pth` / 1000 | `bc_pretrain_hl.py:33-36`, `:102` |
| optimizer / schedule | Adam 1e-3, halved to 2e-4 at epoch 500, 1000 full-batch epochs | `bc_pretrain_hl.py:84`, `:103-106` |
| net | hidden 32, 1-layer GRU, `CategoricalNet(32,3)` | `bc_pretrain_hl.py:32`, `:82-83` |
| loss | class-weighted CE, inverse frequency, `reduction='sum'`, one `opt.step()` per epoch | `bc_pretrain_hl.py:88-91`, `:114-116`, `:121` |
| seed | `torch.manual_seed(0)` | `bc_pretrain_hl.py:81` |
| output format | `{"state_encoder": ..., "policy": ...}` | `bc_pretrain_hl.py:126` |

### Eval

| Name | Value | Where |
|---|---|---|
| `evaluate` / `eval.should_load_ckpt` | True / **False** | `_formal_run.sh:64` |
| `eval.deterministic` | true → `dist.mode()` argmax | `social_nav_hierarchical_v2.yaml:15`; used `social_nav_neural_policy.py:731-732` |
| `eval.evals_per_ep` / `num_environments` | 3 / 1 | `_formal_run.sh:66` |
| `eval.video_option` | `[]` | `_formal_run.sh:69` |
| `eval.episode_stats_path` | `$SD/stats_nr1_<tag>.json` | `_formal_run.sh:70`; consumed `habitat_evaluator.py:500-516` |

## Gotchas (HL training)

- **`_mech_check.sh`'s defaults are not what ran**: `QUICK=quick17` / `DAPG_COEF=0.3`
  (`:16-17`) vs the real invocation `QUICK=quick15` / `DAPG_COEF=0.1`
  (`_mech_check_C.log:78`, full command `:91`).
- **The version of `_formal_run.sh` that ran differs from the file on disk**: the driver
  transcript shows the endpoint guard was `[ ! -f checkpoints_nr1/ckpt.29.pth ]`, which fired
  and aborted with `TRAINING_INCOMPLETE_ABORTING_EVALS`. The script was then edited to test
  `latest.pth` (current `_formal_run.sh:55`) and the evals were re-run under `SKIP_TRAIN=1`.
- `_mech_check.sh` does NOT abort on a failed training run — `train()` only echoes the exit
  code (`:43`), unlike `_formal_run.sh:55-58`. Arm B's retrain exited 1 and was still extracted
  and evaluated from `latest.pth`.
- **The DAPG demo file is byte-identical to the BC training file**
  (`hl_decisions_stable.jsonl` == `hl_decisions_dapg_success.jsonl`, md5
  `3454b5a6145861f3fcf0b5b82500e0b5`, 849 rows). There is no held-out demo signal, which is
  worth stating when interpreting `aux_dapg_acc` (~0.98).
- **`critic_warmup_calls` and `dapg_decay_calls` are counted in `evaluate_actions` CALLS**, not
  updates. With `ppo_epoch=2` and `num_mini_batch=1` there are 2 calls per update, so 120 → 60
  updates and 240 → 120 updates. Changing either silently rescales both schedules.
- The DAPG decay clock starts only AFTER warmup (`t = _eval_calls - _critic_warmup_calls`,
  `social_nav_neural_policy.py:500`), and the term is skipped during warmup because its
  independent forward would bypass the actor freeze (`:499`).
- The DAPG forward path calls `self._state_encoder.rnn` and `self._policy.linear` directly with
  `h0=0` per demo episode (`:520-524`), skipping `rnn_build_seq_info`, the distribution wrapper
  and the map branch — hence the hard error when the map branch is on (`:217-221`).
- **`pretrained_hl_weights` failing to exist is NOT fatal** — it only logs a warning
  (`social_nav_neural_policy.py:252-255`) and the policy stays randomly initialised. Combined
  with `should_load_ckpt=False`, a typo'd eval path yields a silent random-policy evaluation.
- `_extract_hl_map.py` **drops the critic** (verified: `checkpoints_nr1/latest.pth` contains
  `_high_level_policy._critic.fc.*` but the extracted `.pth` has only `state_encoder` +
  `policy`). Fine for argmax eval; such a file cannot resume PPO.
- `safe_dis_min=0.0` completely disables the proximity penalty; the yield/backoff potential
  still fires because it is gated by the separate `yield_dis=3.0`.
- Training and eval use different caps (1500 vs 1200) while `eff_step_cap` is 1200, so during
  TRAINING a success at 1201–1500 steps earns zero efficiency bonus while still paying slack.
- `success_reward` and `slack_reward` are applied in `environments.py:75-80`, OUTSIDE
  `SocialNavReward`. They never appear in `comp_sums` — the breakdown does not sum to the
  environment return.
- `end_on_collide=true` suppresses goal_progress/proximity/backoff (`social_nav_sensors.py:270`)
  and the corridor potential (`:320`) on the colliding step, but NOT the efficiency term
  (`:364`) or the release bonus (`:332`).
- The corridor and release terms call `habitat_sim.ShortestPath` every step whenever either
  coefficient is > 0 (`social_nav_sensors.py:313`), using privileged human goal info —
  acceptable for a training reward, not deployable.
- The efficiency term reads another measure's private config
  (`task.measurements.measures['social_nav_to_pos_success']._config.success_distance`,
  `social_nav_sensors.py:364-368`). The Hydra key is `social_nav_to_pos_succ` while the
  measure's `cls_uuid` is `social_nav_to_pos_success` — easy to mis-grep.
- **Rollout-mix tracking is a no-op unless episodes carry `info.pool.teacher_class`**; it
  self-disables permanently on the first check (`ppo_trainer.py:443`, `:450-451`). Training
  directly on raw `train36_v1` silently produces no `rollout_fail_frac`.
- `SPLIT_SCENES` has **opposite** settings in the two experiment families (0 in mech, fail frac
  0.37; 1 in formal/seed, fail frac 0.157), so the two sets of conclusions rest on different
  effective failure sampling.
- `bc_pretrain_hl.py` hardcodes `HIDDEN=32` (`:32`), `num_layers=1` (`:82`) and
  `ACT2IDX` (`:30`). Any change to `hidden_dim` / `num_rnn_layers` / `hl_action_names` breaks
  the warm start as a `load_state_dict` shape error at `social_nav_neural_policy.py:234`.
- Learner tensorboard tags are agent-prefixed (`learner/agent_0_dapg_grad_ratio`), but the
  dry-run's missing-tag check looks for the unprefixed names (`_dapg_dryrun.sh:50-52`) and
  reports them MISSING even on a healthy run.
- The map branch, `bc_anchor_coef` and `critic_output_scale` are all implemented but inert in
  every current run.
- The config file is named `social_nav_hierarchical_**OVERFIT**_v2.yaml` even though it is the
  production multi-episode recipe; its header still describes the single-episode ep70fix
  experiment and its `data_path` (`:13`) / `total_num_steps` (`:40`) are always overridden.

---

# LL training

The LL layer replaces the scripted `ClearCorridorYieldSkill` with `LearnedYieldSkill`, a
22→128→128→2 tanh MLP seeing only deployment-legal observations. The ladder is
**BC → DAgger → ES**; there is **no RL/PPO anywhere in the LL pipeline**.

## Strategy

- **Distill the privileged planner into observation-only inputs**: the scripted yield reads
  `backoff_waypoint_delta` (a sim-side pathfinder that computes the yield pocket), while the
  student sees only lidar + human polar + door dir, so the planner is baked into a reactive net
  that is deployment-legal (`social_nav_skills.py:363-370`, `hrl_pipeline/bc_ll_yield.py:1-6`).
- **Keep the HL/LL interface byte-identical** so the swap is a single-factor experiment: the
  learned skill occupies the same `backoff` slot, outputs the same 2-D `base_velocity`, and is
  resolved by `cls = eval(skill_config.skill_name)` (`hierarchical_policy.py:96`) — one hydra
  override changes the LL and nothing else.
- **Freeze the LL during HL PPO deliberately**: skills live in a plain dict
  (`hierarchical_policy.py:82`) so their params are never registered submodules, and
  `requires_grad_(False)` (`social_nav_skills.py:410`) makes it explicit and future-proofs
  against a `ModuleDict` refactor. Only `_high_level_policy.get_policy_components()` is
  optimized (`hierarchical_policy.py:289-291`).
- **Never self-terminate the learned skill** (hold semantics): `should_terminate` fires only on
  `max_skill_steps` or the HL's control interval (`social_nav_skills.py:500-511`), preserving
  1 Hz temporal abstraction.
- **Make DAgger labels free**: the privileged sensor keeps computing `wp_delta` regardless of
  who drives, so `LL_DAGGER` logs (student-visited obs → scripted-teacher action) at zero extra
  sim cost (`social_nav_skills.py:390-393`, `:453-471`).
- **Add the `human_stop_dist` freeze branch to the DAgger teacher label**: without it the label
  taught "keep reversing" at close range while the real scripted skill freezes, producing
  back-into-human collisions (`social_nav_skills.py:479-480` mirroring `:268-274`;
  pre-registered as must-fix in `hrl_pipeline/LL_PLAN.md:57-60`).
- **Make demo rows joinable back to episodes**: the `t` field carries
  `HierarchicalPolicy._step_counter` (`hierarchical_policy.py:115`, `:302`, `:362`) plus `env`,
  so the cumsum of per-episode `num_steps` from `EVAL_STATS` partitions the t axis and failed
  segments can be excised (`social_nav_skills.py:319-323`, `verify_ll_demos.py:5-9`).
- **Two-pass clean collection instead of post-hoc cleaning**: certify which episodes the teacher
  passes stably (`build_stableset.py`), then collect only on those. The Aug-15 pass came back
  **28/28 clean** (`stats_ll_demo_pass.json`: n=84, succ 1.000, coll 0.000), so
  `verify_ll_demos.py` dropped 0 of 15490 rows (input and output md5-identical).
- **BC learns from successes only**: rows whose segment was not (success AND zero collision) are
  dropped wholesale rather than reweighted (`verify_ll_demos.py:21-24`, `:37-40`).
- **ES rather than LL PPO for the fine-tune step**: black-box search on the tiny net reuses the
  existing eval harness with zero new sim infrastructure and no continuous-action RL machinery
  (`es_ll_finetune.py:1-8`); LL PPO is gated as an un-opened Phase 5 (`LL_PLAN.md:103-107`).
- **Restrict the ES search space to the final Linear + bias (258 params)**
  (`es_ll_finetune.py:34`, `:106-109`) so a few hundred closed-loop episodes can plausibly move
  the policy without destroying the BC/DAgger feature stack.
- **Score ES candidates on real closed-loop outcomes, not surrogate loss**: fitness =
  `mean_success − 0.5·mean_collide − 0.1·mean(steps/step_cap)` (`es_ll_finetune.py:72`), each
  candidate injected into a genuine subprocess run via `LL_MODEL` (`:55`, `:64-65`).
- **Evaluate one factor at a time against a fixed control**: the LL-only arm is teacher HL
  (`rule_markov`) + student LL vs. the identical run with the scripted LL; the full-learned-stack
  arm adds `pretrained_hl_weights=nr1_ck14_hl.pth` (`LL_PLAN.md:45-48`).
- **Verify artifacts are reproducible from the recorded data**: retraining BC from
  `ll_demos_v2_clean.jsonl` and `ll_train_r1.jsonl` reproduces `ll_yield_bc_v2.pth` and
  `ll_yield_dagger1.pth` bit-for-bit, enabled by `torch.manual_seed(0)` (`bc_ll_yield.py:58`).

## Parameters

### Net / observation

| Name | Value | Where |
|---|---|---|
| architecture | `Linear(in,128)-ReLU-Linear(128,128)-ReLU-Linear(128,2)-Tanh` | `social_nav_skills.py:398-402`; trainer `bc_ll_yield.py:26-30` |
| `in_dim` | 22 (`LearnedYieldSkill`) / 24 (`FlatNavSkill`) — verified on every `.pth` | `social_nav_skills.py:396`; flat `:689-703` |
| input composition | `concat(lidar[16]/3.0, feat[0:4] normalized, door_unit[2])` | `social_nav_skills.py:430-444` |
| normalization | `lidar/=3.0; f0=min(d,6)/6; f1/=π; f2/=π; f3=clip(v,±1.5)/1.5; door/=‖door‖` | `social_nav_skills.py:431-443`; duplicated `bc_ll_yield.py:40-48` |
| `lidar_max_range` (skill) | 3.0 — must match the sensor | `social_nav_skills.py:386`; override `+$SK.skill_data.lidar_max_range=3.0` |
| lidar sensor | `num_rays 16 / max_range 3.0 / ray_step 0.1` | `default_structured_configs.py:717-719`; sensor `social_nav_sensors.py:1715` |
| `model_path` / `LL_MODEL` | yaml default `$SD/ll_yield_bc.pth`; `LL_MODEL` wins | `social_nav_skills.py:384-385`; yaml `social_nav_hierarchical_v2_llyield.yaml:20` |
| weight-file format | `{"model": state_dict (keys `net.0/2/4.*`), "in_dim": int}`; `net.` prefix stripped on load | written `bc_ll_yield.py:77`, `es_ll_finetune.py:113/124/139`; loaded `social_nav_skills.py:395-404` |
| DAgger label freeze | `human_stop_dist` 1.0, from `skill_data` | `social_nav_skills.py:389`, `:479-480` |
| scripted freeze floor | 1.0 m (class default 0.0) | `social_nav_hierarchical_v2.yaml:42`; `social_nav_skills.py:223`, `:268-274` |
| `max_skill_steps` | 200 (explains the ~200-step contiguous-t segments) | `social_nav_hierarchical.yaml:142` |
| `control_interval` | 30 env steps | `social_nav_hierarchical.yaml:121` |
| action scaling | `lin ×10.0`, `ang ×4.0` | `actions.py:723-724`, `:729`; `hssd_..._v2.yaml:67` |

### BC

| Name | Value | Where |
|---|---|---|
| `LL_DATA` / `LL_OUT` | default `ll_demos.jsonl` / `ll_yield_bc.pth` | `bc_ll_yield.py:18-19` |
| optimizer / epochs / batch / loss | Adam 1e-3, 400 epochs, batch 4096, plain unweighted MSE on both dims | `bc_ll_yield.py:60`, `:66-71` |
| seed / split | `manual_seed(0)`; val = first `n//10` of a random **per-row** permutation | `bc_ll_yield.py:58`, `:63-65` |
| BC results | demos-only: 15490 rows, train 0.0018 / val 0.0019. demos+DAgger: 30950 rows, train 0.0025 / val 0.0027 | `_bc_ll_v2_full.log`, `_bc_ll_dagger1_full.log` |

### DAgger

| Name | Value | Where |
|---|---|---|
| `LL_DAGGER` | `""` (off); a path enables logging | `social_nav_skills.py:394`, `:453-471` |
| label function | `_teacher_action` — freeze → arrival-rotate → reverse-and-drive | `social_nav_skills.py:474-498` |
| round-1 data | 15460 rows on `stableok`, student = `ll_yield_bc_v2.pth` | `outputs/2026-08-16/01-26-36/.hydra/overrides.yaml` |
| aggregate | `cat ll_demos_v2_clean.jsonl ll_dagger_r1.jsonl > ll_train_r1.jsonl` → 30950 rows | verified by row counts |

### ES (file defaults vs. what actually ran)

| Name | File default | **Actually used** | Where |
|---|---|---|---|
| `--base` | `ll_yield_bc.pth` | **`ll_yield_dagger1.pth`** | `es_ll_finetune.py:33`; proven by `overrides.yaml` `model_path=` + workdir `center.pth` |
| `--rule-mode` | `rule_yield` | **`learned`** (NR1 HL) | `es_ll_finetune.py:89`; `outputs/2026-08-16/06-00-51/.hydra/overrides.yaml` |
| `--rule-config` | `..._v2_llyield.yaml` | **`$V2`** | `es_ll_finetune.py:90-91`; `.hydra/hydra.yaml config_name` |
| `--episodes-gz` | `bcsolid6_v2` | **`es8` (v2 run) / `es14` (v3 run)** | `outputs/2026-08-16/06-00-51` and `08-32-53` overrides |
| `--pop` / `--gens` | 16 / 15 | 16 / **10** | `es_ll_finetune.py:80-81`; `_es_v3.log` |
| `--workers` | 8 | **6** | `es_ll_finetune.py:83`; inferred from completion waves |
| `--topk` | 4 | 4 (recovered numerically) | `es_ll_finetune.py:86`, `:129-130` |
| `--sigma` / `--sigma-decay` | 0.03 / 0.95 | same | `es_ll_finetune.py:84-85`, `:135` |
| `--evals` / `--step-cap` | 1 / 1200 | same | `es_ll_finetune.py:82`, `:87`; per-candidate overrides |
| `--init` / `--full` | `bc` / off | same (2 tensors, 258 params) | `es_ll_finetune.py:93`, `:92`, `:106` |
| fitness | `succ − 0.5·coll − 0.1·steps/cap`; `−1.0` on subprocess failure | — | `es_ll_finetune.py:69-72`, `:66-67` |
| center update | mean of top-`topk`; `best_vec` tracked separately (elitism only for the saved output) | — | `es_ll_finetune.py:128-134` |
| ES results | v3: gen0 center 0.827 → best 0.930 @gen4. v2: 0.746 → 0.933 @gen3 | — | `_es_v3.log`, `_es_v2.log` |

### Eval / datasets

| Name | Value | Where |
|---|---|---|
| harness fixed settings | `num_environments=1`, `seed=7`, `deterministic=true`, `should_load_ckpt=False` | `scripted_hl_eval.py:426-438`; deterministic `social_nav_hierarchical_v2.yaml:15` |
| `RULE_EVALS` / `STEP_CAP` | 3 / 1200 for the reported evals | `scripted_hl_eval.py:38`, `:443-448` |
| teacher HL for collection | `RULE_MODE=rule_markov`, `RULE_DIST` at default 3.0 | `scripted_hl_eval.py:33-34`; `video_dir=video_rule_markov` in the overrides |
| demo collection dataset | `stableok` (28 eps / 21 scenes) | `outputs/2026-08-16/00-44-56/.hydra/overrides.yaml` |
| L0 probe set | `l0set` (5 eps: 2 sweet, 2 trivial_eff, 1 fail_C) | `LL_PLAN.md:87-90` |
| LL checkpoints | `ll_yield_bc_v2.pth` → `ll_yield_dagger1.pth` → `ll_yield_es_v2/v3.pth`; all `in_dim=22` | verified by hashing all six tensors of each |

### Verified results table

| Arm | Dataset | succ | coll | steps | Stats file |
|---|---|---|---|---|---|
| Teacher HL + **scripted** LL (control) | train36 ×3 | 0.833 | 0.120 | 805.0 | `stats_teacher_cert3.json` |
| Teacher HL + DAgger LL | train36 ×3 | 0.778 | 0.213 | 749.6 | `stats_ll36_teachhl.json` |
| NR1 HL @ck14 + scripted LL (control) | train36 ×3 | 0.806 | 0.148 | 775.5 | `stats_nr1_u183.json` |
| NR1 HL + DAgger LL | train36 ×3 | 0.759 | 0.204 | 759.5 | `stats_ll36_nr1hl.json` |
| NR1 HL + **ES v3** LL | train36 ×3 | 0.833 | 0.120 | 812.1 | `stats_esv3_train36_v1.json` |
| NR1 HL + ES v2 LL | train36 ×3 | 0.769 | 0.176 | 767.9 | `stats_esv2_train36_v1.json` |
| NR1 HL @ck14 + scripted LL | **holdout2** ×3 | 0.833 | 0.167 | 655.5 | `stats_nr1_ck14_holdout.json` |
| NR1 HL + ES v3 LL | **holdout2** ×3 | **0.500** | **0.500** | 647.8 | `stats_esv3_holdout2.json` |

## Gotchas (LL training)

- **There is no RL/PPO in the LL pipeline.** The only PPO in the project trains the HL, during
  which the LL is frozen — confirmed by `Number of params to train: 4356` (the HL alone) in
  every run log.
- `LL_LOG` and `LL_DAGGER` open the file in **APPEND** mode (`social_nav_skills.py:325`,
  `:458`). Without `rm -f` before every collection, rows from different runs interleave and the
  t→segment mapping in `verify_ll_demos.py` silently produces garbage.
- `t` is `HierarchicalPolicy._step_counter` — a RUN-GLOBAL counter never reset on episode
  boundaries — so the cumsum mapping is only valid with `num_environments=1` and sequential
  episode visiting.
- **`es14` (the ES v3 search set) is a SUBSET of `train36_v1`** (train36 ids
  2,13,23,27,21,26,20,5,8,9,10,12,16,30). `stats_esv3_train36_v1.json` (0.833/0.120) is
  therefore NOT held out from the selection. The only clean held-out reading is `holdout2`,
  where ES v3 gives 0.500/0.500 against the control's 0.833/0.167.
- **The ES "best fitness" is a max over 160 stochastic closed-loop evaluations.** Only 4/160
  candidates beat the center on success count; all 115 ep20 failures are collisions; ep26 flips
  between ~900 steps and the 1200 cap in 95/160 candidates
  (`hrl_pipeline/an_es_v3_mine5.py`). The top of the landscape is dominated by evaluation noise.
- **`eval_candidate` copies the parent environment (`es_ll_finetune.py:53`) but never sets
  `RULE_EXTRA` itself.** Launch the ES driver without `RULE_EXTRA` exported and every child
  silently runs the SCRIPTED skill, `LL_MODEL` is ignored, and all candidates score identically.
  Always confirm `Skills: {0: LearnedYieldSkill(` in a candidate log first.
- `eval_candidate` also never sets `RULE_LOG`, so all 161 candidate processes append to the same
  default path. `hrl_pipeline/hl_decisions_learned.jsonl` (28 MB) is interleaved output from 6
  concurrent workers and is not analyzable.
- `_teacher_action` hardcodes the v2 numeric constants and reads ONLY `human_stop_dist` from
  `skill_data`. Retuning the scripted skill silently desynchronizes DAgger labels.
- **The freeze branch is untested by the collected data**: minimum human distance in
  `ll_demos_v2_clean.jsonl` is 1.0153 m and there are 0 exact-zero actions, so neither the
  scripted freeze nor the teacher-label freeze ever fired during collection.
- `bc_ll_yield.py:9` claims the checkpoint carries a `"norm"` key; `:77` saves only
  `{"model","in_dim"}`. Normalization is duplicated by hand in three places and drifts silently.
- The validation split is a random **per-row** 10% over 30 Hz-adjacent frames, so val MSE is
  optimistic by construction. The real acceptance signal is the L0 closed-loop probe.
- `hrl_pipeline/ll_mse_units.py:16` sets `ANG_SPEED = 10.0`, but v2 overrides `ang_speed` to
  4.0 (`hssd_..._v2.yaml:67`). Every "PHYSICAL … rad/s / deg/s" figure in
  `_ll_mse_units.txt` is 2.5× too large; the `lin` figures (×10.0) are correct.
- **The tanh head is badly matched to the label scale**: BC `lin` targets span only
  `[-0.10, 0.00]` (5% of the tanh range) while `ang` saturates at ±0.35 for 25–39% of rows.
  MSE weights both dims equally, so `lin` is effectively unsupervised relative to `ang`.
- **`ll_yield_bc.pth` and `ll_yield_bc_v2.pth` hold BIT-IDENTICAL weights** (verified by hashing
  all six tensors). `ll_yield_bc.pth` was overwritten in place on 2026-08-15, so the July BC
  checkpoint that `ll_yield_es_ms4.pth` was fine-tuned from NO LONGER EXISTS. Any pre-August
  command naming `ll_yield_bc.pth` refers to different weights than the file on disk today —
  and both stale defaults (`social_nav_skills.py:385`,
  `social_nav_hierarchical_v2_llyield.yaml:20`) point at it.
- The July demo set (`ll_demos.jsonl`, 15352 rows) has no `t`/`env`/`branch` fields, so
  `verify_ll_demos.py:28` asserts on it. Declared dead in `LL_PLAN.md:53-56`.
- **`FlatNavSkill` is a failed branch**, not a work in progress: `flat_bc.pth` (in_dim 24)
  scores 0.000 success with every episode hitting the 1200-step cap, and its training data has
  been deleted. It inherits `LearnedYieldSkill.__init__`, so `LL_MODEL` silently overrides its
  `model_path` too.
- **The single-factor LL swap currently LOSES** against the scripted LL under the same teacher
  HL (0.778/0.213 vs 0.833/0.120). "Current best" = `ll_yield_es_v3.pth`, but only on the
  full-learned-stack train36 arm and with the es14-subset caveat; on `holdout2` every
  learned-LL arm sits at 0.500/0.500.
- Line references inside `hrl_pipeline/LL_PLAN.md` are stale relative to the current
  `social_nav_skills.py` (it cites `LearnedYieldSkill` at `:357` — actual `:363`; `LL_LOG` at
  `:226` — actual `:229`; `_teacher_action` at `:460-480` — actual `:474-498`). Its
  `hierarchical_policy.py:96` reference is still correct.
- Two hydra runs both wrote `stats_ll_l0_student.json`
  (`outputs/2026-08-16/01-15-48` and `01-19-51`); the on-disk file is from the later run.

---

# Call chains

All four chains assume the container prefix

```bash
DEX() { docker exec -u root wxinyuan bash -c ". activate habitat && cd /habitat-lab && $1"; }
```

and, inside the container:

```bash
SD=/habitat-lab/hrl_pipeline
V2=social_nav/social_nav_hierarchical_overfit_v2.yaml
HL=habitat_baselines.rl.policy.agent_0.hierarchical_policy.high_level_policy
SK=habitat_baselines.rl.policy.agent_0.hierarchical_policy.defined_skills.backoff
R=habitat.task.measurements.social_nav_reward
```

## (a) Build + certify a dataset

```bash
# 1. Generate narrow-door candidates (host-side render deps need LD_LIBRARY_PATH)
DEX 'LD_LIBRARY_PATH=/opt/conda/envs/habitat/lib python hrl_pipeline/scene_tools/gen_v3_datasets.py \
      data/social_nav_episode_dev_passable.json.gz \
      data/social_nav_episode_doorcand_v4.json.gz'
# -> also writes *_failed.jsonl with a per-rejection reason   (hrl_pipeline/scene_tools/gen_v3_datasets.py:23)

# 2. (optional) screen candidates with an always_go + rule_yield audit pair
DEX 'python hrl_pipeline/scene_tools/audit_summary.py data/social_nav_episode_doorcand_v3.json.gz \
      hrl_pipeline/stats_ag.json hrl_pipeline/stats_ry.json \
      hrl_pipeline/doorcand_v3_audit.csv'                     # hrl_pipeline/scene_tools/audit_summary.py:9

# 3. Render the one-page clicker (navmesh outline baked into the pixels)
DEX 'LD_LIBRARY_PATH=/opt/conda/envs/habitat/lib python hrl_pipeline/scene_tools/make_click_page.py \
      data/social_nav_episode_doorcand_v4.json.gz hrl_pipeline/manual/click30.html --half 6.0'
# 4. Human clicks robot_start/robot_goal/human_start/human_goal per door -> Export -> 0805.json

# 5. Snap + validate + build the episode file (SNAP_MAX 0.3, ep70fix recipe)
DEX 'LD_LIBRARY_PATH=/opt/conda/envs/habitat/lib python hrl_pipeline/scene_tools/build_manual_episodes.py \
      hrl_pipeline/0805.json data/social_nav_episode_manual0805.json.gz \
      --review-dir hrl_pipeline/manual/review'                # build_manual_episodes.py:11-13

# 6. Certify solvability: all six scripted modes, 1 eval each, cap 1200
DEX 'bash hrl_pipeline/certify_solvability.sh manual0805'     # certify_solvability.sh:14-19
#   equivalently, per mode:
#   rm -f $SD/hl_decisions_$mode.jsonl
#   RULE_MODE=$mode RULE_EVALS=1 STEP_CAP=1200 \
#     EVAL_STATS=$SD/cert_manual0805_${mode}.json RULE_CONFIG=$V2 \
#     RULE_DATASET=/habitat-lab/data/social_nav_episode_manual0805.json.gz \
#     python hrl_pipeline/scripted_hl_eval.py

# 7. Merge the six per-mode stats into one certificate CSV (host or container)
python3 hrl_pipeline/certify_merge.py manual0805     # -> hrl_pipeline/cert_manual0805.csv

# 8. Stability sweep (3-eval) on the boundary families
DEX 'bash hrl_pipeline/_recheck_all.sh'                       # _recheck_all.sh:28-34

# 9. Consolidate all batches into the labelled pool, then quota-select train36
DEX 'python hrl_pipeline/build_pool_v1.py'    # -> pool_v1.json.gz + POOL_v1.csv
DEX 'python hrl_pipeline/build_train36.py'    # -> train36_v1.json.gz + TRAIN36.csv (+holdout)

# 10. Teacher certification (3 evals) and the authoritative outcome split
DEX 'RULE_MODE=rule_markov RULE_EVALS=3 STEP_CAP=1200 RULE_CONFIG=$V2 \
      RULE_DATASET=/habitat-lab/data/social_nav_episode_train36_v1.json.gz \
      EVAL_STATS=$SD/stats_teacher_cert3.json \
      python hrl_pipeline/scripted_hl_eval.py'
# NOTE: this exact invocation is RECONSTRUCTED — no script producing
# stats_teacher_cert3.json is checked in (see "Unresolved" below).
DEX 'python hrl_pipeline/classify_teacher.py' # -> hrl_pipeline/TEACHER_CERT3.csv

# 11. Slice the training/eval subsets
DEX 'python hrl_pipeline/build_stableset.py'  # stableok       (28 eps)
DEX 'python hrl_pipeline/build_fullset.py'    # train32_up3    (40 entries)
DEX 'python hrl_pipeline/build_quickset.py'   # quick15 + quick15_up3 + QUICK15.csv
DEX 'python hrl_pipeline/make_subset.py train36_v1 t36_579 5 7 9'   # generic slicer
```

## (b) HL: demos → BC → PPO → extract → eval → by_group

```bash
# 1. Collect teacher decisions on the clean set (APPEND mode -> rm -f first)
DEX 'rm -f $SD/hl_decisions_stable.jsonl && \
     RULE_MODE=rule_markov RULE_EVALS=3 STEP_CAP=1200 RULE_CONFIG=$V2 \
     RULE_DATASET=/habitat-lab/data/social_nav_episode_stableok.json.gz \
     RULE_LOG=$SD/hl_decisions_stable.jsonl \
     EVAL_STATS=$SD/stats_demo_pass.json \
     python hrl_pipeline/scripted_hl_eval.py'
# -> 849 rows / 28 episodes. Row schema: {feat[6], action, env, mask, num_steps,
#    prev_action, yield_branch, approach}            (scripted_hl_eval.py:353-366)

# 2. Verify the demo pass; downgrade any demo-failing episode to flaky
DEX 'python hrl_pipeline/verify_demo_jsonl.py \
      $SD/hl_decisions_stable.jsonl $SD/stats_demo_pass.json \
      $SD/hl_decisions_dapg_success.jsonl'
# On this data nothing was downgraded; output is byte-identical to the input.

# 3. Behaviour-clone into the GRU state encoder + CategoricalNet head
DEX 'LD_LIBRARY_PATH=/opt/conda/envs/habitat/lib \
     BC_JSONL=$SD/hl_decisions_dapg_success.jsonl \
     BC_OUT=$SD/bc_stable_hl.pth \
     python hrl_pipeline/bc_pretrain_hl.py'
# input width auto-follows the log: 6 feat + approach + one-hot3(prev) = 10
# 1000 full-batch epochs, class-weighted CE; acc 0.985 on the 28 clean demos

# 4. PPO fine-tune (the formal NR1 run, verbatim from _formal_run.sh:34-49)
DEX 'export SPLIT_SCENES=1 DISABLE_CUDNN=1 && \
     python -u -m habitat_baselines.run --config-name=$V2 \
       habitat.dataset.data_path=/habitat-lab/data/social_nav_episode_train32_up3.json.gz \
       +$HL.pretrained_hl_weights=$SD/bc_stable_hl.pth \
       +$HL.critic_warmup_calls=120 \
       habitat_baselines.rl.ppo.critic_lr=3e-2 \
       habitat_baselines.rl.ppo.gamma=0.99 \
       habitat_baselines.rl.ppo.num_steps=1024 habitat_baselines.rl.ppo.num_mini_batch=1 \
       habitat_baselines.rl.ppo.entropy_coef=0.0 habitat_baselines.rl.ppo.value_loss_coef=0.05 \
       habitat_baselines.rl.ppo.lr=5e-5 habitat_baselines.rl.ppo.max_grad_norm=2.0 \
       habitat.task.success_reward=50.0 \
       $R.collide_penalty=30.0 $R.safe_dis_min=0.0 \
       $R.eff_success_reward=20.0 $R.corridor_potential_coef=3.0 $R.release_bonus=3.0 \
       habitat_baselines.total_num_steps=3.0e6 habitat_baselines.num_checkpoints=30 \
       habitat_baselines.checkpoint_folder=checkpoints_nr1 \
       habitat_baselines.tensorboard_dir=tb_nr1 habitat_baselines.video_dir=video_nr1 \
       > $SD/_ppo_nr1.log 2>&1'
# Seeds: same + habitat.seed={200,300}, total_num_steps=1.5e6   (_seed_runs.sh:19-35)
# Mech arms A/B/C: same MINUS the three new reward terms, on quick15/quick15_up3,
#   with SPLIT_SCENES=0; arm C adds
#   +$HL.dapg_demo_path=$SD/hl_decisions_dapg_success.jsonl +$HL.dapg_coef=0.1 \
#   +$HL.dapg_decay_calls=240 +$HL.dapg_batch_eps=8   under DAPG_PROBE=1
#                                                       (_mech_check.sh:69-71)

# 5. NEVER evaluate a crashed run
DEX '[ -f checkpoints_nr1/latest.pth ] || echo TRAINING_INCOMPLETE_ABORTING_EVALS'

# 6. Extract HL weights into the pretrained_hl_weights format (critic dropped)
DEX 'for ck in 9 14 19; do python $SD/_extract_hl_map.py \
        checkpoints_nr1/ckpt.$ck.pth $SD/nr1_ck${ck}_hl.pth; done && \
     python $SD/_extract_hl_map.py checkpoints_nr1/latest.pth $SD/nr1_final_hl.pth'

# 7. Deterministic 3-eval on the FULL train36_v1 (+ holdout2 at the endpoint)
DEX 'python -u -m habitat_baselines.run --config-name=$V2 \
       habitat_baselines.evaluate=True habitat_baselines.eval.should_load_ckpt=False \
       +$HL.pretrained_hl_weights=$SD/nr1_ck14_hl.pth \
       habitat_baselines.num_environments=1 habitat_baselines.eval.evals_per_ep=3 \
       habitat.dataset.data_path=/habitat-lab/data/social_nav_episode_train36_v1.json.gz \
       habitat.seed=7 habitat.environment.max_episode_steps=1200 \
       "habitat_baselines.eval.video_option=[]" \
       habitat_baselines.eval.episode_stats_path=$SD/stats_nr1_u183.json \
       > $SD/_eval_nr1_u183.log 2>&1'
# repeat with nr1_ck9/ck19/final -> u122/u244/u366, and final on holdout2 -> u366_holdout

# 8. Group breakdown (first stats file = reference policy for retention/paired steps)
DEX 'python $SD/by_group.py $SD/stats_mechBC.json:BC $SD/stats_mechB.json:B \
       $SD/stats_nr1_u183.json:NR1_1.5M $SD/stats_nr1_u366.json:NR1_3M \
       | tee $SD/_by_group_nr1.txt'
# quick aggregate: python hrl_pipeline/_agg.py LABEL=path.json ...
```

## (c) LL: collect → filter → BC → DAgger → ES → eval

```bash
# 1. COLLECT — teacher HL + SCRIPTED yield LL, with LL_LOG on (rm -f: append mode!)
DEX 'rm -f $SD/ll_demos_v2.jsonl && \
     LL_LOG=$SD/ll_demos_v2.jsonl \
     RULE_MODE=rule_markov RULE_EVALS=3 STEP_CAP=1200 RULE_CONFIG=$V2 \
     RULE_DATASET=/habitat-lab/data/social_nav_episode_stableok.json.gz \
     EVAL_STATS=$SD/stats_ll_demo_pass.json \
     python hrl_pipeline/scripted_hl_eval.py > $SD/_ll_demo_collect.log 2>&1'
# -> 15490 rows; run outcome 84/84 clean (succ 1.000, coll 0.000)

# 2. FILTER — keep only clean-success (episode, eval) segments
DEX 'python hrl_pipeline/verify_ll_demos.py \
      hrl_pipeline/ll_demos_v2.jsonl hrl_pipeline/stats_ll_demo_pass.json \
      hrl_pipeline/ll_demos_v2_clean.jsonl'
# dropped 0/15490 on this data (files md5-identical)

# 3. BC — distill into the 22->128->128->2 tanh MLP
DEX 'LD_LIBRARY_PATH=/opt/conda/envs/habitat/lib \
     LL_DATA=$SD/ll_demos_v2_clean.jsonl LL_OUT=$SD/ll_yield_bc_v2.pth \
     python hrl_pipeline/bc_ll_yield.py'

# 4. (optional) L0 closed-loop probe: scripted vs student, HL held fixed
DEX 'RULE_MODE=rule_markov RULE_EVALS=3 STEP_CAP=1200 RULE_CONFIG=$V2 \
     RULE_DATASET=/habitat-lab/data/social_nav_episode_l0set.json.gz \
     EVAL_STATS=$SD/stats_ll_l0_student.json \
     RULE_EXTRA="$SK.skill_name=LearnedYieldSkill \
+$SK.skill_data.model_path=$SD/ll_yield_bc_v2.pth \
+$SK.skill_data.lidar_max_range=3.0" \
     python hrl_pipeline/scripted_hl_eval.py'
# control arm = the identical command with RULE_EXTRA removed

# 5. DAgger round 1 — the STUDENT drives, the scripted teacher labels
DEX 'rm -f $SD/ll_dagger_r1.jsonl && \
     LL_DAGGER=$SD/ll_dagger_r1.jsonl \
     RULE_MODE=rule_markov RULE_EVALS=3 STEP_CAP=1200 RULE_CONFIG=$V2 \
     RULE_DATASET=/habitat-lab/data/social_nav_episode_stableok.json.gz \
     EVAL_STATS=$SD/stats_ll_dagger_r1.json \
     RULE_EXTRA="$SK.skill_name=LearnedYieldSkill \
+$SK.skill_data.model_path=$SD/ll_yield_bc_v2.pth \
+$SK.skill_data.lidar_max_range=3.0" \
     python hrl_pipeline/scripted_hl_eval.py > $SD/_ll_dagger_r1.log 2>&1'
# -> 15460 rows (lidar/feat/door/act/t/env; no wp/branch — BC does not read those)

# 6. AGGREGATE + RETRAIN on the union
DEX 'cat $SD/ll_demos_v2_clean.jsonl $SD/ll_dagger_r1.jsonl > $SD/ll_train_r1.jsonl && \
     LD_LIBRARY_PATH=/opt/conda/envs/habitat/lib \
     LL_DATA=$SD/ll_train_r1.jsonl LL_OUT=$SD/ll_yield_dagger1.pth \
     python hrl_pipeline/bc_ll_yield.py'
# -> 30950 rows, val_mse 0.0027

# 7. ES fine-tune — RULE_EXTRA MUST BE EXPORTED (the driver only copies os.environ)
DEX 'export RULE_EXTRA="$SK.skill_name=LearnedYieldSkill \
+$SK.skill_data.model_path=$SD/ll_yield_dagger1.pth \
+$SK.skill_data.lidar_max_range=3.0 \
+$HL.pretrained_hl_weights=$SD/nr1_ck14_hl.pth" && \
     python hrl_pipeline/es_ll_finetune.py \
       --base $SD/ll_yield_dagger1.pth \
       --episodes-gz /habitat-lab/data/social_nav_episode_es14.json.gz \
       --rule-mode learned --rule-config $V2 \
       --pop 16 --gens 10 --evals 1 --workers 6 --step-cap 1200 \
       --out $SD/ll_yield_es_v3.pth > $SD/_es_v3.log 2>&1'
# Sanity check BEFORE trusting a generation:
#   grep -m1 "Skills: {0: LearnedYieldSkill(" $SD/es_ll_*/log_center.txt
# v2 run = same with --episodes-gz ...es8.json.gz --out $SD/ll_yield_es_v2.pth

# 8. Final evals — see chain (d)
```

## (d) Single-factor swap-in evals (student LL; full learned stack)

The control arm in both cases is the *identical* command with `RULE_EXTRA` removed (scripted
LL) — that is what makes it single-factor.

```bash
# --- Arm 1: TEACHER HL + student LL (isolates the LL substitution) ---
DEX 'RULE_MODE=rule_markov RULE_EVALS=3 STEP_CAP=1200 RULE_CONFIG=$V2 \
     RULE_DATASET=/habitat-lab/data/social_nav_episode_train36_v1.json.gz \
     EVAL_STATS=$SD/stats_ll36_teachhl.json \
     RULE_EXTRA="$SK.skill_name=LearnedYieldSkill \
+$SK.skill_data.model_path=$SD/ll_yield_dagger1.pth \
+$SK.skill_data.lidar_max_range=3.0" \
     python hrl_pipeline/scripted_hl_eval.py'
# control = stats_teacher_cert3.json (0.833/0.120); this arm = 0.778/0.213

# --- Arm 2: FULL LEARNED STACK (NR1 HL + student LL); RULE_MODE=learned is pass-through ---
DEX 'RULE_MODE=learned RULE_EVALS=3 STEP_CAP=1200 RULE_CONFIG=$V2 \
     RULE_DATASET=/habitat-lab/data/social_nav_episode_train36_v1.json.gz \
     EVAL_STATS=$SD/stats_esv3_train36_v1.json \
     RULE_EXTRA="$SK.skill_name=LearnedYieldSkill \
+$SK.skill_data.model_path=$SD/ll_yield_es_v3.pth \
+$SK.skill_data.lidar_max_range=3.0 \
+$HL.pretrained_hl_weights=$SD/nr1_ck14_hl.pth" \
     python hrl_pipeline/scripted_hl_eval.py'
# control = stats_nr1_u183.json (0.806/0.148); this arm = 0.833/0.120
# ...but es14 ⊂ train36, so repeat on the ZERO-LEAK holdout:

# --- Arm 3: the only clean held-out reading ---
DEX 'RULE_MODE=learned RULE_EVALS=3 STEP_CAP=1200 RULE_CONFIG=$V2 \
     RULE_DATASET=/habitat-lab/data/social_nav_episode_holdout2.json.gz \
     EVAL_STATS=$SD/stats_esv3_holdout2.json \
     RULE_EXTRA="$SK.skill_name=LearnedYieldSkill \
+$SK.skill_data.model_path=$SD/ll_yield_es_v3.pth \
+$SK.skill_data.lidar_max_range=3.0 \
+$HL.pretrained_hl_weights=$SD/nr1_ck14_hl.pth" \
     python hrl_pipeline/scripted_hl_eval.py'
# control = stats_nr1_ck14_holdout.json (0.833/0.167); this arm = 0.500/0.500

# NOTE: do NOT feed holdout2 / es14 / quick15 stats to by_group.py — it assumes the
# FULL train36_v1 so positional ids equal train36 ids (by_group.py:5-7).
```

---

# Cross-check appendix

## Contradictions found and resolved

| # | Claim A | Claim B | Verified |
|---|---|---|---|
| 1 | `input_feature_dim` at `v2.yaml:54`, `use_prev_action` at `:55` | at `:55` / `:56` | **B** — `:55` / `:56` |
| 2 | `hidden_dim`/`num_rnn_layers` at `overfit_v2.yaml:59`/`:60` | at `:56-59` | **Neither** — `:58` / `:59` |
| 3 | `hold_at_target`/`proportional_turn` at `v2.yaml:32`/`:33` | — | **Wrong** — `:31` / `:32` |
| 4 | `eval.deterministic` at `social_nav_hierarchical_v2.yaml:16` | at `:15` | **B** — `:15` (`:16` is `max_episode_steps_override`) |
| 5 | `STEP_CAP` consumed at `scripted_hl_eval.py:441-444` | at `:443-448` | **B** |
| 6 | `scripted_hl_eval.py` invoked by `es_ll_finetune.py:34` | by `:32`/`:64` | **B** (`:34` is `LAST_KEYS`) |
| 7 | `bc_pretrain_hl.py` invoked at `_hl_train_stage1.sh:34` | at `:32-35` | Block `:32-35`, `python` at `:35` |
| 8 | `bc_pretrain_hl.py` in `_new_teacher_chain.sh:54` | at `:52-55` | Block `:52-55`, `python` at `:55` |
| 9 | gen_v3 params at `:34`/`:41`/`:44`/`:47-48` | — | **Off by 1-2** — `:33` / `:40` / `:42` / `:45-46` |
| 10 | `DAPG_COEF` = 0.3 (script default) | = 0.1 (run) | **0.1 ran** (`_mech_check_C.log:78`, `:91`); 0.3 is the stale default at `_mech_check.sh:17` |
| 11 | `QUICK` = `quick17` (script default) | `quick15` | **`quick15`** — no `quick17` file exists; log shows `quick15_up3` |
| 12 | `RULE_CONFIG` default `..._overfit.yaml` | every run passes `..._overfit_v2.yaml` | Both true; default at `scripted_hl_eval.py:40-42` would make `rule_markov` raise |
| 13 | ES ran under a scripted teacher HL (`es_ll_finetune.py:5`, `:89`) | ran with `RULE_MODE=learned` + NR1 HL | **B** — `outputs/2026-08-16/{06-00-51,08-32-53}/.hydra/overrides.yaml` |
| 14 | ES `--base` = `ll_yield_bc.pth` (file default) | = `ll_yield_dagger1.pth` | **B** — proven by the run's `model_path=` override |
| 15 | train36 quotas per docstring (fail_C 5, trivial_zero 3) | per CSV | **CSV** — sweet 18 / trivial_eff 9 / trivial_zero 4 / fail_C 4 / fail_B 1, 23 scenes |
| 16 | train36 "max 2 per scene" | max 3 | **3** — four scenes carry 3 |
| 17 | `bc_ll_yield.py` saves a `"norm"` key (`:9`) | saves `{"model","in_dim"}` (`:77`) | **B** — docstring stale |
| 18 | `ll_mse_units.py` `ANG_SPEED = 10.0` (`:16`) | v2 `ang_speed = 4.0` | **4.0** — every rad/s figure in `_ll_mse_units.txt` is 2.5× too large |
| 19 | holdout candidates at `TRAIN36.csv:39-41` | at `:40-41` | **`:40-41`** are the rows; `:39` is the block header |
| 20 | `LL_PLAN.md` cites `LearnedYieldSkill:357`, `LL_LOG:226`, `_teacher_action:460-480` | — | **Stale** — `:363`, `:229`, `:474-498` |
| 21 | `certify_solvability.sh:19` invokes the harness | env block at `:16-18` | Both — env `:16-18`, `python` `:19` |
| 22 | `_formal_run.sh` on disk == what ran | endpoint guard differed | **Differs** — driver transcript shows `ckpt.29.pth`; disk has `latest.pth` (`:55`); evals re-run under `SKIP_TRAIN=1` |

Two more items were reported by two sections as conflicts but are **not** conflicts: (i) both
`_eval_ll.sh` (July-era, `v2_llyield.yaml`) and the `RULE_EXTRA`-on-`overfit_v2` pattern are
real — the latter supersedes the former; (ii) `ll_yield_bc.pth` and `ll_yield_bc_v2.pth` are
bit-identical *today* because the former was overwritten in place, not because they were ever
the same artifact.

## Unresolved / not determined

- The exact invocation that produced `hrl_pipeline/stats_teacher_cert3.json` — the single most
  load-bearing evidence file in the lineage. No script is checked in; only the run log survives
  (`hrl_pipeline/_teacher_cert3.log`, 2026-08-13 22:31:56). The 108 entries (36×3) are
  consistent with `RULE_MODE=rule_markov RULE_EVALS=3 STEP_CAP=1200` on `train36_v1`, but that
  is inference, not record.
- The exact `BC_JSONL` / `BC_OUT` invocation that produced `hrl_pipeline/bc_stable_hl.pth`.
- The exact invocation of the `stableok` HL demo pass (`hl_decisions_stable.jsonl` +
  `stats_demo_pass.json`, both 2026-08-13 18:41).
- Build commands for `es8`, `es14`, `l0set`, `holdout2`. Contents are recoverable only from
  `info["pool"]["pool_v1_id"]` (or `info["manual"]["click_id"]` for `holdout2`); shape and
  provenance are consistent with `make_subset.py`, but the commands were never recorded. Note
  these sets carry `pool_v1_id` but **not** `train36_id`/`teacher_class`, so the trainer's
  rollout-mix tracking does not work on them.
- The exact command that wrote the `cert_cd17_*` family (no `cd17` dataset exists; the
  underlying set is `cleandev17_v2`).
- ES `--topk=4` and `--workers=6` were recovered numerically (top-4 mean matching the next
  generation's centre; 6/6/4 completion waves), not from a record.
- `dev20_parked`'s build script is not in the repo (`info["v2_transform"]` shows it is
  v2-family, but its robot starts differ from `dev20_v2`'s).
