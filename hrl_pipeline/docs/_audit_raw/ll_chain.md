# LL (yield-skill) pipeline — definitive reference

Verified against source + hydra records + stats JSONs on 2026‑08‑17. Everything below was re‑derived from disk; where a fact could not be recovered it says so.

## 0. Conventions

```bash
# host
DEX() { docker exec -u root wxinyuan bash -c ". activate habitat && cd /habitat-lab && $1"; }
```
```bash
# inside the container
SD=/habitat-lab/hrl_pipeline
V2=social_nav/social_nav_hierarchical_overfit_v2.yaml
SK=habitat_baselines.rl.policy.agent_0.hierarchical_policy.defined_skills.backoff
HL=habitat_baselines.rl.policy.agent_0.hierarchical_policy.high_level_policy
```
**Every script below has stale internal defaults (§5). Pass every path explicitly; never rely on a default.**

Missing prerequisites — the four episode subsets the chain needs were archived by the dataset consolidation and are **not** in `data/` anymore:
```bash
DEX 'cp data/archive_20260817/social_nav_episode_{stableok,l0set,es8,es14,holdout2}.json.gz data/'
# or rebuild (index lists verified by full non-info episode equality against train36_v1):
# make_subset.py train36_v1 stableok 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 21 22 23 24 26 27 28 30 31 32 33 34 35
# make_subset.py train36_v1 l0set   2 20 25 21 26
# make_subset.py train36_v1 es8     2 13 23 27 21 26 20 5
# make_subset.py train36_v1 es14    2 13 23 27 21 26 20 5 8 9 10 12 16 30
```
`stableok` ≡ the 28 `stable-success` episodes of `results/TEACHER_CERT3.csv` in ascending `train36_id` ≡ `TRAIN[0:28]`.

---

## 1. The chain: collect → filter → BC → L0 probe → DAgger → retrain → ES → eval

### Step 1 — LL demo collection (`LL_LOG`)

```bash
DEX 'rm -f $SD/demos/ll_demos_v3.jsonl && \
     LL_LOG=$SD/demos/ll_demos_v3.jsonl \
     RULE_MODE=rule_markov RULE_EVALS=3 STEP_CAP=1200 RULE_CONFIG=$V2 \
     RULE_DATASET=/habitat-lab/data/social_nav_episode_stableok.json.gz \
     EVAL_STATS=$SD/results/stats_ll_demo_pass.json \
     RULE_LOG=$SD/demos/hl_decisions_ll_collect.jsonl \
     python hrl_pipeline/hl/scripted_hl_eval.py > $SD/runs/_ll_demo_collect.log 2>&1'
```
- Driver stack: `rule_markov` teacher HL + **scripted** `ClearCorridorYieldSkill`. `LL_LOG` is read once in `social_nav_skills.py:229` and written per env step at `:307-326`, only while the yield skill is the active skill.
- Row schema: `{lidar[16], feat[4], door[2], wp[3], act[2], branch, t, env}`.
- **`LL_LOG` opens in APPEND mode (`social_nav_skills.py:325`) — the `rm -f` is mandatory.** `t` is `HierarchicalPolicy._step_counter` (injected at `hierarchical_policy.py:362`), a run-global counter, so the mapping in step 2 is valid only at `num_environments=1` (the harness hardcodes it, `scripted_hl_eval.py:434`).
- Historical result (`outputs/2026-08-16/00-44-56`): 15,490 rows, run outcome 84/84 success, 0 human collisions, mean 840.8 steps.
- **Acceptance:** `wc -l` ≥ ~15k; `EVAL_STATS` shows ≥ 27/28 clean; `lin` support non-degenerate. Measured `lin ∈ [-0.10, 0.00]` (never positive — this skill only reverses or holds), `|ang|` mean 0.182, 24.9 % at the ±0.35 floor.

### Step 2 — filter to clean-success segments

```bash
DEX 'python hrl_pipeline/ll/verify_ll_demos.py \
      $SD/demos/ll_demos_v3.jsonl $SD/results/stats_ll_demo_pass.json \
      $SD/demos/ll_demos_v3_clean.jsonl'
```
- Partitions the `t` axis by the cumsum of per-(episode,eval) `num_steps` in stats **insertion order = evaluator visit order** (`verify_ll_demos.py:19-25,34-40`); drops every row from a segment that was not (success ∧ zero collision).
- **Acceptance:** the assert at `:28` must pass (`t` present, `t ≤ cum[-1]`); printed `zero-action share` ≈ 0. Historically it dropped **0 of 15,490** rows (input and output md5-identical).

### Step 3 — BC

```bash
DEX 'LD_LIBRARY_PATH=/opt/conda/envs/habitat/lib \
     LL_DATA=$SD/demos/ll_demos_v3_clean.jsonl LL_OUT=$SD/weights/ll_yield_bc.pth \
     python hrl_pipeline/ll/bc_ll_yield.py > $SD/runs/_bc_ll.log 2>&1'
```
- 22→128→128→2 tanh MLP; Adam lr 1e‑3 constant, 400 epochs, batch 4096, unweighted MSE on both dims, `torch.manual_seed(0)`, CPU, fully deterministic (re-runs reproduce **bit-identically**).
- Saves `{"model": state_dict("net.0/2/4.*"), "in_dim": 22}`. (The docstring `:9` claims a `"norm"` key; `:77` does not write one — normalization is hand-duplicated in three places instead.)
- Historical curve (`runs/_bc_ll_v2_full.txt`): train 0.0368→**0.0018**, val 0.0377→**0.0019** (9 printed points is the entire on-disk resolution).
- **Acceptance is NOT the val MSE.** The split is a random *per-row* 10 % over 30 Hz-adjacent frames, so it measures interpolation inside seen trajectories. Measured open-loop→closed-loop degradation of these exact weights: MSE 0.001886 on held-out rows → **0.020698 (11.0×) on the states its own rollout visits**, `|err ang|` max 1.28 command units. The real gate is step 4.

### Step 4 — L0 closed-loop probe (the real BC acceptance)

```bash
# student arm
DEX 'RULE_MODE=rule_markov RULE_EVALS=3 STEP_CAP=1200 RULE_CONFIG=$V2 \
     RULE_DATASET=/habitat-lab/data/social_nav_episode_l0set.json.gz \
     EVAL_STATS=$SD/results/stats_ll_l0_student.json \
     RULE_EXTRA="$SK.skill_name=LearnedYieldSkill \
+$SK.skill_data.model_path=$SD/weights/ll_yield_bc.pth \
+$SK.skill_data.lidar_max_range=3.0" \
     python hrl_pipeline/hl/scripted_hl_eval.py'
# control arm = the identical command with RULE_EXTRA removed (scripted LL)
```
`l0set` = train36 ids **[2, 20, 25, 21, 26]**. Historical (each 3 evals, `(succ, coll)`):

| LL | ep2 | ep20 | ep25 | ep21 | ep26 |
|---|---|---|---|---|---|
| scripted | 0/3, 3c | 0/3, timeout | 0/3, 3c | **3/3** | 3/3 |
| BC | 0/3, 3c | **3/3** | 0/3, 3c | **1/3, 2c ← regression** | 3/3 |
| DAgger | 0/3, 3c | 0/3, 3c | 0/3, 3c | **3/3 (repaired)** | 3/3 |

**Acceptance:** no episode the scripted LL solves 3/3 may drop below 3/3. Pure BC **fails** this (ep21). That is exactly what DAgger is for.

### Step 5 — DAgger round 1

```bash
DEX 'rm -f $SD/demos/ll_dagger_r1.jsonl && \
     LL_DAGGER=$SD/demos/ll_dagger_r1.jsonl \
     RULE_MODE=rule_markov RULE_EVALS=3 STEP_CAP=1200 RULE_CONFIG=$V2 \
     RULE_DATASET=/habitat-lab/data/social_nav_episode_stableok.json.gz \
     EVAL_STATS=$SD/results/stats_ll_dagger_r1.json \
     RULE_EXTRA="$SK.skill_name=LearnedYieldSkill \
+$SK.skill_data.model_path=$SD/weights/ll_yield_bc.pth \
+$SK.skill_data.lidar_max_range=3.0" \
     python hrl_pipeline/hl/scripted_hl_eval.py > $SD/runs/_ll_dagger_r1.log 2>&1'
```
- The **student drives**; the privileged `backoff_waypoint_delta` sensor keeps computing `wp_delta` regardless of who drives, so `_teacher_action` (`social_nav_skills.py:474-498`) relabels every student-visited state at zero extra sim cost (`:453-471`).
- Rows carry `{lidar, feat, door, act, t, env}` — **no `wp`/`branch`** (the trainer reads neither, but branch-level attribution is impossible for this half).
- Historical: 15,460 rows, 83/84 raw success, 0 human collisions. Note **`verify_ll_demos.py` was never applied to the DAgger half** — the one failed rollout's rows are in the corpus.
- **Acceptance:** rows ≈ demo rows; `|ang|` distribution must have *shifted*, which is what DAgger actually taught. Measured A→B: `|ang|` mean 0.1816→0.2338 (+28.7 %), rows at the 0.35 floor 10.3 %→19.9 %, near-straight `|ang|<0.05` 26.5 %→19.2 %. Freeze rows: **0 in both halves** — the reverse-safety freeze never fired, so that branch is untested by the data. State drift is real: of 15,460 DAgger rows only 2,403 share a `t` with a demo row, and 2,402 of those command a different action (`|Δang|` max 1.405, observed human distance differs by up to 1.296 m).

### Step 6 — aggregate + retrain

```bash
DEX 'cat $SD/demos/ll_demos_v3_clean.jsonl $SD/demos/ll_dagger_r1.jsonl > $SD/demos/ll_train_r1.jsonl && \
     LD_LIBRARY_PATH=/opt/conda/envs/habitat/lib \
     LL_DATA=$SD/demos/ll_train_r1.jsonl LL_OUT=$SD/weights/ll_yield_dagger1.pth \
     python hrl_pipeline/ll/bc_ll_yield.py > $SD/runs/_bc_ll_dagger1.log 2>&1'
```
- 30,950 rows; train 0.0025 / val 0.0027. **Do not compare this val MSE to step 3's** — B is a strictly harder target (angular target variance 0.0707→0.0837; predict-the-mean baseline 0.0354→0.0427).
- **Acceptance:** re-run the L0 probe; the ep21 regression must be gone (it is).
- `demos/ll_train_r1.jsonl` no longer exists on disk; regenerate it with the `cat` above (verified: `ll_demos_v2_clean` ++ `ll_dagger_r1`, elementwise-identical actions).

### Step 7 — ES fine-tune

```bash
DEX 'export RULE_EXTRA="$SK.skill_name=LearnedYieldSkill \
+$SK.skill_data.model_path=$SD/weights/ll_yield_dagger1.pth \
+$SK.skill_data.lidar_max_range=3.0 \
+$HL.pretrained_hl_weights=$SD/weights/nr1_ck14_hl.pth" && \
     python hrl_pipeline/ll/es_ll_finetune.py \
       --base $SD/weights/ll_yield_dagger1.pth \
       --episodes-gz /habitat-lab/data/social_nav_episode_es14.json.gz \
       --rule-mode learned --rule-config $V2 \
       --pop 16 --gens 10 --evals 1 --workers 6 --step-cap 1200 --seed 1 \
       --out $SD/weights/ll_yield_es_v3.pth > $SD/runs/_es_v3.log 2>&1'
```
- **`ll/es_ll_finetune.py:32` `HARNESS` is stale and must be fixed before this runs at all** (§5) — otherwise every `subprocess.run` returns non-zero, `eval_candidate` returns the `-1.0` sentinel for all 160 candidates, and the run "completes" saving the unmodified warm start.
- **`eval_candidate` copies `os.environ` (`:53`) but never sets `RULE_EXTRA` itself.** Forget the `export` and every child silently runs the **scripted** skill and `LL_MODEL` is ignored. Mandatory first check:
  ```bash
  DEX 'grep -m1 "Skills: {0: LearnedYieldSkill(" $SD/es_ll_*/log_center.txt'
  DEX 'grep -m1 "Number of params to train: 4356" $SD/es_ll_*/log_center.txt'
  ```
- Algorithm: (μ=4, λ=16) elitist-**mean** ES on the last Linear only (`net.4.weight` 2×128 + `net.4.bias` = **258 params**), σ 0.03 × 0.95^(g−1), fitness `succ − 0.5·coll − 0.1·steps/1200`, 1 eval per candidate. Round 2 gen‑0 center 0.827101 → record **0.930018 at gen 4** (`g4_c11`), no later candidate beat it.
- Verified provenance: `weights/ll_yield_es_v3.pth` last layer ≡ `_archive/dirs/es_ll_jl5ncpsk/g4_c11.pth`; its `net.0`/`net.2` are **bit-identical** to `ll_yield_dagger1.pth` (last-layer-only confirmed); `‖Δ_last‖ = 0.58276` against `‖W_last‖ = 3.2043` (18.2 % displacement).
- **Known defects you must account for, not just note:** the driver saves `best_vec`, never the final center, which it computes and discards (`:136` vs `:145`) — recover it with `ll/reconstruct_es_center.py <workdir> <base.pth> <out.pth>`. `did_collide` counts robot–human contact only: round 2's winner scored a clean 14/14 / 0 collisions while logging 984 scene collisions on ep10 and 483 on ep9. Once a candidate is perfect the only remaining gradient is the 0.1-weight step term, and that is what picked the shipped artifact (winner 839.79 vs 847.36 / 851.29 / 855.71 mean steps among 4 tied-perfect candidates).

### Step 8 — final single-factor evals

```bash
# LL-only isolation: teacher HL, swap only the LL
DEX 'RULE_MODE=rule_markov RULE_EVALS=3 STEP_CAP=1200 RULE_CONFIG=$V2 \
     RULE_DATASET=/habitat-lab/data/social_nav_episode_train36_v1.json.gz \
     EVAL_STATS=$SD/results/stats_ll36_teachhl.json \
     RULE_EXTRA="$SK.skill_name=LearnedYieldSkill \
+$SK.skill_data.model_path=$SD/weights/ll_yield_es_v3.pth \
+$SK.skill_data.lidar_max_range=3.0" \
     python hrl_pipeline/hl/scripted_hl_eval.py'
# control = results/stats_teacher_cert3.json

# full learned stack (RULE_MODE=learned is pure pass-through logging)
DEX 'RULE_MODE=learned RULE_EVALS=3 STEP_CAP=1200 RULE_CONFIG=$V2 \
     RULE_DATASET=/habitat-lab/data/social_nav_episode_train36_v1.json.gz \
     EVAL_STATS=$SD/results/stats_esv3_train36_v1.json \
     RULE_EXTRA="$SK.skill_name=LearnedYieldSkill \
+$SK.skill_data.model_path=$SD/weights/ll_yield_es_v3.pth \
+$SK.skill_data.lidar_max_range=3.0 \
+$HL.pretrained_hl_weights=$SD/weights/nr1_ck14_hl.pth" \
     python hrl_pipeline/hl/scripted_hl_eval.py'
# repeat on holdout2 -> stats_esv3_holdout2.json  (the ONLY zero-leak reading: es14 ⊂ train36)
DEX 'python hrl_pipeline/diag/eval_summary.py \
      "script=$SD/results/stats_nr1_u183.json,$SD/results/stats_nr1_ck14_holdout.json" \
      "esv3=$SD/results/stats_esv3_train36_v1.json,$SD/results/stats_esv3_holdout2.json"'
```

**Hard methodological rule (measured, §6):** episode visit order is shuffled (`iterator_options.shuffle=True`, `group_by_scene=True`, nothing overrides them) and depends on the episode-file length. All `train36_v1` stats files share a byte-identical visit order (verified across teacher / nr1 / dagger / esv2 / esv3), so those are paired-comparable. A stats file produced on the 38-episode `EVAL` file has a **different** order and is *not* comparable episode-for-episode to a `train36_v1` file, even for positions 0–35 whose episode content is byte-identical.

---

## 2. How the learned LL is switched on

**Mechanism:** `hierarchical_policy.py:96` does `cls = eval(skill_config.skill_name)`, so changing one string swaps the class in the same `backoff` slot. The HL/LL interface is unchanged: same slot index 0, same 2‑D `base_velocity` output, same `hl_action_names: ["backoff","wait","go_to_goal"]`.

**Two ways, both valid:**
1. **Preferred (single hydra override, no yaml edit)** — on the production `social_nav_hierarchical_overfit_v2.yaml` stack, which already carries `lidar_scan`:
   ```
   RULE_EXTRA="$SK.skill_name=LearnedYieldSkill \
   +$SK.skill_data.model_path=<pth> +$SK.skill_data.lidar_max_range=3.0"
   ```
   The control arm is the identical command with `RULE_EXTRA` removed.
2. `--config-name=social_nav/social_nav_hierarchical_v2_llyield.yaml` (lines 17‑21). **Its `model_path` default at `:20` is stale** (`/habitat-lab/hrl_pipeline/ll_yield_bc.pth`) and `LearnedYieldSkill.__init__` `torch.load`s it eagerly, so this config crashes today unless `LL_MODEL` is set. It is the July-era route, superseded by (1).

`LL_MODEL` env wins over `skill_data.model_path` (`social_nav_skills.py:384-385`) — that is how the ES driver injects candidates without touching yaml.

**Observations it consumes** (`required_obs_keys`, `social_nav_skills.py:414`): `lidar_scan`, `social_nav_policy_state`, `backoff_waypoint_delta`. Sensor prerequisites are already satisfied by `hssd_spot_human_social_nav_twoagent_v2.yaml`: `lidar_scan` lab sensor (`:10`), `agent_0_lidar_scan` in `gym.obs_keys` (`:99`), `include_door_dir: true` (`:41`), `include_branch: true` (`:46`), `include_approach_speed: true` (`:54`).

**Privileged inputs it drops relative to the scripted skill:**

| Input | Scripted `ClearCorridorYieldSkill` | `LearnedYieldSkill` |
|---|---|---|
| `backoff_waypoint_delta[0:2]` = pocket-planner waypoint (x,z) | drives the reverse turn-then-go (`:281-302`) | **dropped** |
| `backoff_waypoint_delta[2]` = geodesic distance remaining | drives arrival/stop and (without `hold_at_target`) termination | **dropped** |
| `backoff_waypoint_delta[3:5]` = door direction | used only to face the door while holding | **kept**, as input dims 20‑21 |
| `backoff_waypoint_delta[5]` = fallback branch flag | — | dropped |
| `localization_sensor[3]` = world yaw | required (`:233`) | **dropped** |
| `social_nav_policy_state[0]` | explicit `human_stop_dist=1.0` freeze branch (`:268-274`) | kept as an input dim, but **no explicit freeze branch** |
| `lidar_scan` | not used | **added** (16 dims) |

**Honest caveat:** the learned skill still *reads* `backoff_waypoint_delta` for the door direction, so the privileged pocket planner still runs every step and must still be configured. Only its *control* output is discarded. This is "planner distilled out of the control loop", not "planner removed from the system". And if `wp` is absent or narrower than 5 the door dims silently fall back to zeros (`:437-443`) — no warning.

---

## 3. Which HL checkpoint it must be paired with

**`weights/nr1_ck14_hl.pth`** (NR1 @ ~1.5 M steps, the pre-registered short-budget point; 4,356 trainable HL params, `hidden_dim 32`, `num_rnn_layers 1`).

Reason, and it is a hard constraint rather than a preference: **`ll_yield_es_v3.pth` was *selected against* that exact frozen HL.** Recovered from `outputs/2026-08-16/08-32-53/.hydra/{overrides,hydra}.yaml` plus 110 agreeing hydra dirs, ES round 2 ran with `RULE_MODE=learned`, `config_name=social_nav_hierarchical_overfit_v2.yaml`, and `+…pretrained_hl_weights=…/nr1_ck14_hl.pth` — **not** the driver defaults (`rule_yield` + `…_v2_llyield.yaml`), which its own docstring still advertises. A repro using the *stated* defaults gives center fitness **0.114** instead of 0.827.

Consequences:
- The 258 ES parameters encode a response to `nr1_ck14`'s specific yield-trigger timing. Pairing `es_v3` with a different HL is an unmeasured configuration.
- Conversely, `ll_yield_es_v3`'s train36 numbers are **not HL-independent** — the LL was tuned against the HL it is scored with. The BC and DAgger checkpoints carry no such coupling (they were trained under the `rule_markov` teacher HL) and are the honest ones to use with any HL.
- The config must supply `hidden_dim: 32` / `num_rnn_layers: 1` (`overfit_v2.yaml:58-59`) or the checkpoint will not load into the HL head.

---

## 4. Exact input-dimension breakdown (22)

Assembled at `social_nav_skills.py:430-444`; duplicated by hand in `ll/bc_ll_yield.py:40-49` and `ll/ll_mse_units.py:40-49`.

| dims | source | raw meaning | normalization | notes |
|---|---|---|---|---|
| 0–15 | `lidar_scan` (`social_nav_sensors.py:1715`) | 16 navmesh ray ranges, ray 0 = robot forward, CCW, `forward=(cos yaw, −sin yaw)` | `/ lidar_max_range` (3.0) → [0,1] | sensor `num_rays 16 / max_range 3.0 / ray_step 0.1`; the skill's `lidar_max_range` must match the sensor or the scale is silently wrong |
| 16 | `social_nav_policy_state[0]` | `human_dist` (m) | `min(d, 6.0)/6.0` | measured min in the corpus 1.0153 m |
| 17 | `[1]` | `human_bearing` (rad, 0 = ahead) | `/π` | |
| 18 | `[2]` | `human_rel_heading` (rad) | `/π` | |
| 19 | `[3]` | `human_rel_speed` (m/s, unsigned, ego-polluted) | `clip(·, ±1.5)/1.5` | dims 4–6 (`goal_dist`, `goal_bearing`, `human_approach_speed`) are **not** used |
| 20–21 | `backoff_waypoint_delta[3:5]` | door direction, world (x,z) | divided by its own norm → unit vector; zeros if `‖·‖ < 1e-6` or the sensor is < 5 wide | the only surviving privileged channel |

Head: `Linear(22,128)-ReLU-Linear(128,128)-ReLU-Linear(128,2)-Tanh` → written into the `base_velocity` slot located by `find_action_range` (`:86-88`), `[lin, ang]`.

Action scaling (`actions.py`): `clip(lin,−1,1) × longitudinal_lin_speed = 10.0`, `clip(ang,−1,1) × ang_speed = **4.0**` (v2 override, `hssd_..._twoagent_v2.yaml:67`; the structured default is 10.0). Integration window `1/ctrl_freq = 1/120 s` (`ac_freq_ratio: 1`).

Two scale pathologies worth knowing before retuning anything:
- **`ll/ll_mse_units.py:16` hardcodes `ANG_SPEED = 10.0`.** Every "PHYSICAL … rad/s / °/s" figure derived from it (including `docs/LL_TRAINING_CURVES.md §1.6`: "0.6117 rad/s = 35.05 °/s") is **2.5× too large**; the correct value is 0.2447 rad/s ≈ 14.0 °/s. The linear figures (×10.0) are correct.
- **The tanh head is badly matched to the label scale.** `lin` targets span only `[−0.10, 0.00]` — 5 % of the tanh range — while `ang` saturates at ±0.35 for 25–39 % of rows. MSE weights both dims equally, so measured `MSE(ang)` is 123× `MSE(lin)`: `lin` is effectively unsupervised.

`FlatNavSkill` (`:689`) is the same net at `in_dim=24` (+ goal polar). It is a dead branch: `flat_bc.pth` scores 0.000 success with every episode hitting the cap, and its training data is gone. It inherits `LearnedYieldSkill.__init__`, so `LL_MODEL` silently overrides its `model_path` too.

---

## 5. Stale hardcoded paths (file:line → old → new)

All "old" targets verified non-existent today. Nothing was modified.

**Fatal (silent wrong behaviour or crash):**

| file:line | old | new |
|---|---|---|
| `ll/es_ll_finetune.py:32` | `/habitat-lab/hrl_pipeline/scripted_hl_eval.py` | `/habitat-lab/hrl_pipeline/hl/scripted_hl_eval.py` — **every candidate returns the `-1.0` sentinel (`:66-67`) and the ES saves the unmodified warm start** |
| `ll/es_ll_finetune.py:33` | `.../hrl_pipeline/ll_yield_bc.pth` | `.../hrl_pipeline/weights/ll_yield_bc.pth` (and the run actually used `weights/ll_yield_dagger1.pth`) |
| `ll/bc_ll_yield.py:18` (`LL_DATA` default) | `.../hrl_pipeline/ll_demos.jsonl` | `.../hrl_pipeline/demos/ll_demos_v2_clean.jsonl` (the July `ll_demos.jsonl` is deleted and lacks `t`/`env`, so `verify_ll_demos.py:28` would assert on it anyway) |
| `ll/bc_ll_yield.py:19` (`LL_OUT` default) | `.../hrl_pipeline/ll_yield_bc.pth` | `.../hrl_pipeline/weights/ll_yield_bc.pth` — the old path now *creates* a stray file at the reorganised root instead of overwriting the artifact |
| `…/config/social_nav/social_nav_hierarchical_v2_llyield.yaml:20` | `/habitat-lab/hrl_pipeline/ll_yield_bc.pth` | `/habitat-lab/hrl_pipeline/weights/ll_yield_bc.pth` — **`torch.load` at `social_nav_skills.py:395` runs in `__init__`, so this config raises today** |
| `habitat-baselines/.../skills/social_nav_skills.py:385` (`model_path` fallback) | `/habitat-lab/hrl_pipeline/ll_yield_bc.pth` | `/habitat-lab/hrl_pipeline/weights/ll_yield_bc.pth` |
| `data_tools/build_stableset.py:14,16` | `{ROOT}/hrl_pipeline/TEACHER_CERT3.csv` | `{ROOT}/hrl_pipeline/results/TEACHER_CERT3.csv` — the LL demo set cannot be rebuilt without this |

**Breaks on run:**

| file:line | old | new |
|---|---|---|
| `ll/ll_dataset_stats.py:8` `D` | `/habitat-lab/hrl_pipeline` | `/habitat-lab/hrl_pipeline/demos` |
| `ll/ll_dataset_stats.py:12,107` | `ll_train_r1.jsonl` | **file deleted** — regenerate via the `cat` in step 6 |
| `ll/ll_mse_units.py:115` `D` | `/habitat-lab/hrl_pipeline` | split: jsonl → `demos/`, pth → `weights/` |
| `ll/ll_mse_units.py:117,119,122,123,135` | `_tmp_bc_v2.pth`, `_tmp_bc_dagger1.pth` | **deleted** (throwaway repro checkpoints) |
| `ll/ll_mse_units.py:118` | `ll_train_r1.jsonl` | **deleted** |
| `ll/ll_mse_units.py:122` | `ll_yield_bc_v2.pth` | `weights/ll_yield_bc.pth` (verified equal: MSE 0.001784 on corpus A and 0.020698 on the DAgger states, exactly the historical `bc_v2` numbers) |
| `ll/es_ll_finetune.py:79` (`--out` default) | `.../hrl_pipeline/ll_yield_es.pth` | `.../hrl_pipeline/weights/ll_yield_es.pth` |
| `ll/es_ll_finetune.py:18` (docstring) | `python hrl_pipeline/es_ll_finetune.py` | `python hrl_pipeline/ll/es_ll_finetune.py` |
| `ll/es_ll_finetune.py:78,91` (defaults) | `bcsolid6_v2.json.gz`, `..._v2_llyield.yaml` | the run used `es14.json.gz` + `..._overfit_v2.yaml` |
| `ll/bc_ll_yield.py:3,9` (docstring) | `hrl_pipeline/ll_demos.jsonl`, `hrl_pipeline/ll_yield_bc.pth` | `demos/…`, `weights/…` |
| `hl/scripted_hl_eval.py:20,46` (`RULE_LOG` default) | `/habitat-lab/hrl_pipeline/hl_decisions_{MODE}.jsonl` | `.../hrl_pipeline/demos/…` (dir still exists → writes silently pollute the root; also append-mode) |
| `hl/scripted_hl_eval.py:48` (`VIDEO_DIR`) | `hrl_pipeline/video_{MODE}` | `hrl_pipeline/videos/{MODE}` |
| `ll/es_ll_finetune.py:105` (`mkdtemp dir=`) | `/habitat-lab/hrl_pipeline` | `/habitat-lab/hrl_pipeline/_archive/dirs/` or a scratch root (works, but re-pollutes the reorganised tree) |
| `data_tools/build_final_datasets.py:26,29` | `{SD}/TEACHER_CERT3.csv` | `{SD}/results/TEACHER_CERT3.csv` |
| `data_tools/build_final_datasets.py:83,88` | `data/social_nav_episode_{train32_up3,holdout2}.json.gz` | `data/archive_20260817/…` — both asserts fail today |
| `data_tools/classify_teacher.py:30,46,47,48,97,118` | `{SD}/stats_teacher_cert3.json`, `stats_teacher_wait_t36.json`, `stats_teacher_wait_bc.json`, `TEACHER_CERT3.csv` | all under `{SD}/results/` |
| `data_tools/verify_demo_jsonl.py:19,22,59` | `{SD}/TEACHER_CERT3.csv` | `{SD}/results/TEACHER_CERT3.csv` |
| `data_tools/build_quickset.py:21,23`, `build_fullset.py:15,17` | `{SD}/TEACHER_CERT3.csv` | `{SD}/results/TEACHER_CERT3.csv` |
| `data_tools/build_train36.py:20,138` | `{SD}/TRAIN36.csv` | `{SD}/results/TRAIN36.csv` |
| `data_tools/certify_solvability.sh:12,19` | `SD=/habitat-lab/hrl_pipeline`; `python hrl_pipeline/scripted_hl_eval.py` | `python hrl_pipeline/hl/scripted_hl_eval.py`; outputs → `runs/` |
| `diag/by_group.py:22,24` | `{SD}/TEACHER_CERT3.csv` | `{SD}/results/TEACHER_CERT3.csv` |
| `diag/analyze_yield_geometry.py:35,45` | `{SD}/stats_trace_{tag}.json` | `{SD}/results/stats_trace_{tag}.json` |
| `hl/bc_pretrain_hl.py:34,36` | `{SD}/hl_decisions_rule_markov.jsonl`, `{SD}/bc_markov_hl.pth` | `_archive/data/…`, `weights/…` |

**Archived drivers (still the only record of these experiments):**

| file:line | old | new |
|---|---|---|
| `_archive/scripts/_ll_variance.sh:15` | `SD=/habitat-lab/hrl_pipeline` | subdir-qualified per file below |
| `:19` `WD` | `$SD/es_ll_jl5ncpsk` | `$SD/_archive/dirs/es_ll_jl5ncpsk` (root‑700; use `docker exec -u root`) |
| `:28` | `$SD/stats_var_$1.json`, `$SD/nr1_ck14_hl.pth` | `$SD/results/stats_var_$1.json`, `$SD/weights/nr1_ck14_hl.pth` |
| `:29,55` | `python hrl_pipeline/{scripted_hl_eval,es_ll_finetune}.py` | `hrl_pipeline/hl/…`, `hrl_pipeline/ll/…` |
| `:29,31,44,46,59,61` | `$SD/_var_$1.log`, `$SD/_center_recon.log`, `$SD/_es_seed$S.log` | `$SD/runs/…` (shrunk `.txt`) or `_archive/logs_raw/…` |
| `:35,43,47,54,57,62` | `$SD/ll_yield_es_v3.pth`, `$SD/reconstruct_es_center.py`, `$SD/ll_yield_{dagger1,center_r2,es_seed$S}.pth`, `data/…es14.json.gz` | `weights/…`, `ll/reconstruct_es_center.py`; **`ll_yield_center_r2.pth` and `ll_yield_es_seed{1,2}.pth` were deleted** (per `docs/CLEANUP_PLAN.md:116`); `es14` → `data/archive_20260817/` |
| `:65-71` | `$SD/eval_summary.py`, `$SD/stats_*.json` | `$SD/diag/eval_summary.py`, `$SD/results/stats_*.json` |
| `_archive/scripts/_eval_ll.sh:15` | `python -u hrl_pipeline/scripted_hl_eval.py` | `hrl_pipeline/hl/scripted_hl_eval.py` |
| `_eval_ll.sh:12,28` | `..._v2_llyield.yaml`; `$SD/hl_from_msd20.pth` | superseded by the `RULE_EXTRA`-on-`overfit_v2` pattern; **`hl_from_msd20.pth` deleted** — this script cannot run |

Also stale (documentation, not paths): `docs/LL_PLAN.md:12,15,57` cite `social_nav_skills.py:357 / :226 / :460-480`; the real lines are `:363 / :229 / :474-498`.

The reorg script's own header (`_archive/scripts/_reorganize.sh:8-9`) promises "absolute paths inside kept scripts are rewritten" — **there is no such step in the file.** That is the single root cause of every row above.

---

## 6. Measured cost of the learned LL vs the scripted one

All numbers recomputed from `results/*.json`. Protocol: `overfit_v2` config, seed 7, `evals_per_ep=3`, cap 1200, `num_environments=1`. "stable" = all 3 evals succeed. Classes from `results/TEACHER_CERT3.csv`.

### 6a. LL-only, teacher HL held fixed (`rule_markov`) — the cleanest isolation

| | scripted LL | DAgger LL |
|---|---|---|
| stable-success (28) | **28/28**, 0 collision-episodes, 840 steps | **25/28**, 2 collision-episodes, 849 steps |
| stable-failure (4) | 0/4 | 0/4 |
| flaky (4) | 0/4 | 0/4 |
| rollout success | 90/108 | 84/108 |
| human collisions | 13/108 | 23/108 |
| paired Δsteps | — | **+10** |
| episodes lost | — | 8, 9, 18 |
| episodes gained | — | **none** |

Files: `stats_teacher_cert3.json` / `stats_ll36_teachhl.json` (identical visit order — verified). **Under the teacher HL the learned LL is pure loss: −3 stable episodes, +10 human collisions, +10 steps, zero capability gain.** No ES-v3-under-teacher-HL run exists; that is a gap.

### 6b. LL-only, HL = `nr1_ck14` (all five columns share one visit order — verified)

| | scripted | DAgger | ES v2 | **ES v3** |
|---|---|---|---|---|
| stable-success (28) | **25/28, coll 0**, 803 st | 23/28, coll 3, 822 st | 18/28, coll 6, 838 st | **24/28, coll 2**, 842 st |
| capability on stable-failure (4) | 1/4 `[20]` | 1/4 `[2]` | 3/4 `[2,3,25]` | **3/4 `[2,3,20]`** |
| flaky (4) | 1/4 | 1/4 | 3/4 | 2/4 |
| holdout2 (2, zero-leak) | **1/2, coll 1**, 722 st | — | 1/2, coll 1, 943 st | **1/2, coll 1**, 828 st |
| rollout success (108) | 87 | 82 | 83 | **90** |
| human collisions (108) | **16** | 22 | 19 | **13** |
| robot–scene collisions (108) | 6,336 | 7,819 | 6,210 | **8,397 (+33 %)** |
| paired Δsteps vs scripted | — | +15 | +24 | **+35** |
| stable episodes lost vs scripted | — | 13, 23, 27 | 8,9,10,11,12,16,26,30 | **13, 14** |

**Honest bottom line for ES v3, the shipped LL:** it costs **1 stable-success episode (25→24), 2 collision-tainted episodes (0→2), +35 steps on every jointly-solved episode, and +33 % robot–scene grinding**, and buys **2 capability episodes the scripted LL cannot solve (ids 2 and 3)**. Of those, **id 2 is inside the ES selection set `es14`; id 3 is not** — id 3 is the only genuinely off-selection capability gain in the entire LL line. Aggregate rollout success and human collisions come out slightly *better* than the scripted control (90 vs 87, 13 vs 16), so under a rollout-mean lens the LL looks free; under the stable-success/retention lens it is not.

### 6c. The measurement is barely larger than the noise floor

Two independent evaluations of the **same** `es_v3` weights, same HL, same seed 7, same config, differing only in which episode file was used (36-ep `train36_v1`+`holdout2` vs 38-ep `EVAL`):

| | train36 files | EVAL file |
|---|---|---|
| stable-success | 24/28 | 23/28 |
| collision-episodes | 2 | 4 |
| paired Δsteps vs scripted | +35 | +36 |
| stable episodes lost | 13, 14 | 8, 11, 14 |

**35 of 36 episodes differ in step count; 5 flip their verdict (8, 11, 13, 14, 17).** Root cause verified: `iterator_options.shuffle=True` + `group_by_scene=True` (structured defaults, unoverridden) make visit order a function of the episode-file contents; the two runs visited the same 36 byte-identical episodes in different orders. Same-file re-runs *are* bit-reproducible (`docs/LL_TRAINING_CURVES.md §5.1`: sd 0.000000 over 5 repeats on `es14`). So: **an effect of ±1 stable episode and ±2 collision episodes is exactly the size of the visit-order artefact.** Consequences —
- `_archive/data/_ll_variance_summary.txt` compares 老师 (train36 files) against `es_v3`/ties/center/seeds (EVAL file). **That column-to-column comparison is confounded**; only the EVAL-file arms are comparable to each other (es_v3 23/28 coll4, tie candidates 24/22/23, discarded center 24/28 coll3, seeded replications 21/28 coll4 ×2).
- `docs/LL_TRAINING_CURVES.md §4.2`'s "24/28, 2 collisions, 3 capability" is reproducible from `stats_esv3_train36_v1.json` + `stats_esv3_holdout2.json` (§4.4's failure to reproduce it used a different subsetting: train36 ids 8–35 rather than the `stable-success` class).
- `docs/LL_TRAINING_CURVES.md §5.3` attributes re-realization spread to `dist.sample()` under "the eval default `deterministic=False`". That mechanism is **not active**: `social_nav_hierarchical_v2.yaml:15` and `overfit_v2.yaml:45` both set `eval.deterministic: true`, which `habitat_evaluator.py:183-189` forwards through `hierarchical_policy.py:542` to `dist.mode()`. The spread is real; the stated cause is wrong.

### 6d. For completeness — July-era weights on the current stack

`stats_ll0_ll_yield_bc.json` 20/36, 16/36 collisions; `stats_ll0_ll_yield_es_ms4.json` 21/36, 15/36 (1 eval each). Both checkpoints have since been deleted; the July `ll_yield_bc.pth` was **overwritten in place on 2026‑08‑15**, so any pre-August command naming it refers to weights that no longer exist.

---

## Gaps that block a clean re-run

1. `ll/es_ll_finetune.py:32` must be fixed before any ES run (silent total failure otherwise).
2. `data_tools/build_stableset.py` + `build_final_datasets.py` need the `results/` prefix on `TEACHER_CERT3.csv`, and `build_final_datasets.py:83,88` need `data/archive_20260817/` for their two asserts.
3. `stableok` / `l0set` / `es8` / `es14` / `holdout2` must be copied back from `data/archive_20260817/` (or rebuilt with the index lists in §0).
4. `demos/ll_train_r1.jsonl` and the raw pre-filter `LL_LOG` are gone; only the filtered `ll_demos_v2_clean.jsonl` survives, so step 2 cannot be re-verified against its own input.
5. No `es_v3` + teacher-HL run exists — the ES-tuned LL has never been measured under an HL other than the one it was selected against.