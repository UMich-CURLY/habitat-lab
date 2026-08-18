# hrl_pipeline

Hierarchical social navigation: a rule-based **expert** produces demonstrations,
BC clones them, PPO fine-tunes. The high level (HL) picks one of three skills
once a second; the low level (LL) executes it.

Everything runs **inside the container**, where this directory is
`/habitat-lab/hrl_pipeline`:

```bash
DEX() { docker exec -u root wxinyuan bash -lc ". activate habitat && cd /habitat-lab && $1"; }
```

---

## The nine datasets

`data/` holds exactly nine episode files and nothing else; every other episode file
lives in `data/archive_20260817/`. Full manifest, per-episode index and label
criteria: **`data/DATASETS.md`**.

| file | eps | scenes | what |
|---|---|---|---|
| `data/socialnav_ALL_labeled.json.gz` | 64 | 26 | catalogue of every configuration, with labels + suite membership. Not for training |
| `data/socialnav_big_train.json.gz` | 21 | 19 | main suite |
| `data/socialnav_big_val.json.gz` | 5 | 3 | pick the checkpoint here |
| `data/socialnav_big_eval.json.gz` | 5 | 4 | touch once, report this |
| `data/socialnav_small_train.json.gz` | 12 | 7 | quick suite, for the 1–2 h loop |
| `data/socialnav_small_val.json.gz` | 3 | 2 | |
| `data/socialnav_small_eval.json.gz` | 3 | 2 | |
| `data/socialnav_legacy_hl_train.json.gz` | 40 | 23 | what the EXISTING checkpoints trained on (was `_TRAIN`) |
| `data/socialnav_legacy_hl_eval.json.gz` | 38 | 25 | what they were evaluated on (was `_EVAL`) |

`big_train / big_val / big_eval` share **no scene**, and each `small_*` split draws
only from the scenes of the matching `big_*` split — `build_datasets.py` asserts both.

⚠️ **The legacy pair does not have that property.** 33 of its 38 eval episodes are
byte-identical to a train episode and all 23 train scenes reappear in eval, so every
number ever measured on it is a **training-set number**. It is kept verbatim only so
the existing checkpoints stay reproducible; new work uses the big/small suites.

```bash
DEX 'python hrl_pipeline/data_tools/build_datasets.py'          # plan, writes nothing
DEX 'python hrl_pipeline/data_tools/build_datasets.py --build'  # rewrite all nine (needs -u root)
DEX 'python hrl_pipeline/data_tools/dataset_provenance.py'      # leakage check on any pair
```

---

## Directory map

```
hrl_pipeline/
  hl/          the expert + HL training            ← start here
  ll/          low-level yield-skill distillation
  data_tools/  dataset building, certification, leakage checks
  scene_tools/ scene rendering + the hand-clicking tool that authors episodes
  diag/        evaluation → tensorboard, curves, reward decomposition
  docs/        reference documents (see below)
  weights/     21 checkpoints
  results/     per-episode eval stats (json) + csv tables
  demos/       demonstration corpora (jsonl) — BC trains on these
  runs/        shrunk training logs (metric + traceback lines only)
  tb/          raw tensorboard archive, 30 dirs / 163 sidebar entries
  tb_clean/    ← point tensorboard HERE. 23 curated entries, built from tb/
  checkpoints/ raw PPO checkpoint folders, one per training run
  videos/      rendered rollouts
  clicking/    output of the click workflow (rendered pages, review images)
  _archive/    everything not part of the final pipeline (old scripts, ES workdirs…)
```

---

## HL pipeline

The expert itself is **`hl/scripted_hl_eval.py`**. It monkey-patches
`SocialNavNeuralHighLevelPolicy.get_next_skill`, overrides the network's choice with a
rule, and logs every decision. `RULE_MODE` selects the rule; the final one is
`rule_markov`.

### The rule (`hl/scripted_hl_eval.py`, `MARKOV_PARAMS`)

State = the previous HL action (`prev_action ∈ {backoff, wait}` ⇒ "yielding").

```
not yielding:                                        # walking
    trigger = d < 3.0 ∧ |bearing| < 1.75
              ∧ |rel_heading| > π/2 ∧ approach > 0.15
    → backoff if trigger else go_to_goal             # tight, fires immediately

yielding:                                            # already giving way
    still = (d < 3.5 ∧ approach > 0.05)
            ∨ (d < 1.0 ∧ still retreating)           # the gate lifts once we hold
    → go_to_goal   if ¬still for 2 consecutive ticks # loose, needs confirmation
    → wait         if we have backed off ≥ 6 ticks
    → backoff      otherwise
```

Every term is a function of the **7-dim observation + previous action** only — no
privileged information — so the student can reproduce the teacher's state. `approach`
is the human's absolute closing speed (dim 7), which is what makes a parked human
read 0 regardless of how the robot moves.

### Steps

```bash
# 1. certify a dataset (which scripted mode solves each episode)
DEX 'DS=big_train; for m in always_go rule_wait rule_waitgo smart_wait rule_markov rule_yield; do
  RULE_MODE=$m RULE_EVALS=1 STEP_CAP=1200 \
    RULE_CONFIG=social_nav/social_nav_hierarchical_overfit_v2.yaml \
    RULE_DATASET=/habitat-lab/data/socialnav_$DS.json.gz \
    RULE_LOG=/habitat-lab/hrl_pipeline/results/hl_dec_${m}_$DS.jsonl \
    EVAL_STATS=/habitat-lab/hrl_pipeline/results/cert_${DS}_${m}.json \
    python hrl_pipeline/hl/scripted_hl_eval.py > hrl_pipeline/runs/_cert_${DS}_${m}.log 2>&1
done'
DEX 'cd hrl_pipeline/results && python ../data_tools/certify_merge.py big_train'  # -> cert_big_train.csv

# 2. collect teacher demonstrations (only episodes the teacher solves cleanly)
DEX 'rm -f hrl_pipeline/demos/hl_demos.jsonl && \
  RULE_MODE=rule_markov RULE_EVALS=1 STEP_CAP=1200 \
  RULE_CONFIG=social_nav/social_nav_hierarchical_overfit_v2.yaml \
  RULE_DATASET=/habitat-lab/data/socialnav_big_train.json.gz \
  RULE_LOG=/habitat-lab/hrl_pipeline/demos/hl_demos.jsonl \
  EVAL_STATS=/habitat-lab/hrl_pipeline/results/stats_teacher_big_train.json \
  python hrl_pipeline/hl/scripted_hl_eval.py > hrl_pipeline/runs/_teacher_big_train.log 2>&1'

# 3. behaviour cloning
DEX 'LD_LIBRARY_PATH=/opt/conda/envs/habitat/lib \
  BC_JSONL=/habitat-lab/hrl_pipeline/demos/hl_demos.jsonl \
  BC_OUT=/habitat-lab/hrl_pipeline/weights/bc_new_hl.pth \
  python hrl_pipeline/hl/bc_pretrain_hl.py'            # accept: acc ≥ 0.98

# 4. PPO fine-tune. The recipe IS the config — pass only what is per-run.
DEX 'SPLIT_SCENES=1 DISABLE_CUDNN=1 python -u -m habitat_baselines.run \
  --config-name=social_nav/social_nav_hrl_final.yaml \
  habitat.dataset.data_path=/habitat-lab/data/socialnav_big_train.json.gz \
  +habitat_baselines.rl.policy.agent_0.hierarchical_policy.high_level_policy.pretrained_hl_weights=/habitat-lab/hrl_pipeline/weights/bc_new_hl.pth \
  habitat_baselines.checkpoint_folder=hrl_pipeline/checkpoints/new \
  habitat_baselines.tensorboard_dir=hrl_pipeline/tb/new > hrl_pipeline/runs/_ppo_new.log 2>&1'

# 5. pick the checkpoint ON THE VAL SPLIT, then evaluate the winner once on eval
#    Do NOT hard-code an index: ckpt.14 was the historical pick and the numbers
#    later showed ckpt.9 dominating it. That is what socialnav_big_val is for.
#    (loading a multi-agent ckpt directly is a SILENT no-op — always go through this)
DEX 'python hrl_pipeline/hl/_extract_hl_map.py hrl_pipeline/checkpoints/new/ckpt.<N>.pth \
       hrl_pipeline/weights/new_hl.pth'
DEX 'python -u -m habitat_baselines.run \
  --config-name=social_nav/social_nav_hrl_final.yaml \
  habitat_baselines.evaluate=True \
  +habitat_baselines.rl.policy.agent_0.hierarchical_policy.high_level_policy.pretrained_hl_weights=/habitat-lab/hrl_pipeline/weights/new_hl.pth \
  habitat.dataset.data_path=/habitat-lab/data/socialnav_big_eval.json.gz \
  habitat_baselines.num_environments=1 "habitat_baselines.eval.video_option=[]" \
  habitat_baselines.eval.episode_stats_path=/habitat-lab/hrl_pipeline/results/stats_new.json'
#    should_load_ckpt=False, deterministic=True and evals_per_ep=3 are already in
#    the config; the eval now runs under the SAME reward the training used.

# 6. publish to tensorboard, then re-curate
DEX 'python hrl_pipeline/diag/eval_to_tb.py'                          # -> tb/eval_summary
DEX 'python hrl_pipeline/diag/logs_to_tb.py name=path/to/train.log'   # -> tb/curves
DEX 'python hrl_pipeline/diag/tb_curate.py'   # tb/ -> tb_clean/ ; add the new run to
                                              # TRAIN_RUNS / EVAL_RUNS first or it
                                              # will not appear in the clean board
```

**Why those PPO flags** (each fixed a diagnosed failure — full story in
`docs/PARAM_AUDIT.md` and the reward section below):

| flag | fixes |
|---|---|
| `critic_lr=3e-2` | the value head cannot reach ±40 returns at the actor's lr; it never fits, so advantages have no baseline |
| `critic_warmup_calls=120` | BC only warms the actor; a random critic destroys it in ~50 updates |
| `entropy_coef=0` | with noisy advantages the entropy bonus is the only consistent gradient and melts the cloned policy |
| `num_steps=1024, num_mini_batch=1` | the HL only acts every 30 env steps → the default rollout gives ~17 samples per gradient step |
| `safe_dis_min=0` | the proximity penalty accumulated per step, welding "shorter episode = higher return" into the objective |
| `total_num_steps=1.5e6` | a budget, **not** an optimum. On the 108-key eval the run is already past its best by 1.0 M (0.8148 → 0.8056 → 0.7963 → 0.7593 at 1.0/1.5/2.0/3.0 M) and the training curve peaks near 190 k. Run the budget, then pick the checkpoint on `socialnav_big_val` |

---

## LL pipeline

Three skills, all hand-written and parameter-free:
`ClearCorridorYieldSkill` (backoff, **uses privileged navmesh planning**),
`WaitSkill` (hold, face the goal), `GoToGoalSkill` (geodesic drive, **no human
avoidance at all** — yielding is exclusively the HL's job).

The LL line distills the scripted yield into an observation-only policy
(16-beam lidar + local features), so the deployed stack has no privileged inputs:

```
demo collection (LL_LOG=…)  →  verify_ll_demos.py  →  bc_ll_yield.py
                                    →  DAgger  →  es_ll_finetune.py
```

Measured cost of switching the scripted yield for the learned one: success drops
from ~89 % to 56-75 % with 8-16 collisions (`results/stats_ll*.json`). Details and
per-generation curves in `docs/LL_TRAINING_CURVES.md`.

---

## Checkpoints (`weights/`)

| file | what it is |
|---|---|
| `nr1_ck9_hl.pth` ⭐ | best evaluated point of the 3 M run (1.0 M). On the identical 108 keys it **dominates `ck14`** on all three metrics — 0.8148/0.1389/0.0463 vs 0.8056/0.1481/0.0463 |
| `nr1_ck14 / ck19 / final` | 1.5 / 2.0 / 3.0 M points of the **same run** — the degradation curve (0.8148 → 0.8056 → 0.7963 → 0.7593) |
| `nr1_s200 / s300` | same recipe, different seeds. **`s200` is the best HL result anywhere: 0.8333/0.1204/0.0463, exactly tying the teacher.** `s300` is 0.8056 — the seed spread is as large as the whole PPO effect |
| `bc_stable_hl.pth` | the BC starting point of every HL result (**its training log is lost**) |
| `mechB_hl / mechC_hl` | three-arm ablation: B = upsampling only, C = + DAPG. Only B grew a new capability; C's demo anchor suppressed it (`docs/MECH_CHECK.md`) |
| `ll_yield_es_v3.pth` ⭐ | LL main result |
| `ll_yield_dagger1 / ll_yield_bc` | LL warm starts |
| `b2_final / g3_final / bc_train36_hl` | the diagnostic line that produced the PPO recipe. **Trained on train36_v1, which overlaps EVAL by 95 % — process evidence only, not a result** |
| `flat_bc.pth` | July flat (non-hierarchical) baseline |

---

## Where to look for what

| question | file |
|---|---|
| full pipeline reference (older, longer) | `docs/PIPELINE_REFERENCE.md` |
| every threshold in the stack, classified | `docs/PARAM_AUDIT.md` |
| dataset construction and provenance | **`data/DATASETS.md`** (`docs/DATASETS_FINAL.md` is the superseded TRAIN/EVAL-era version) |
| HL main result | `docs/NR1_REPORT.md` |
| does a demo anchor help or hurt? | `docs/MECH_CHECK.md` |
| LL training curves and ES details | `docs/LL_TRAINING_CURVES.md` |
| known traps (silent no-ops, eval protocol) | `docs/FRAMEWORKS_HOWTO.md` |
| raw audit reports (2026-08-17) | `docs/_audit_raw/` |

## Tensorboard

```bash
DEX 'tensorboard --logdir /habitat-lab/hrl_pipeline/tb_clean --host 0.0.0.0 --port 6006'
```

**Use `tb_clean/`, not `tb/`.** `tb/` is the raw archive — 30 dirs but **163 entries**
in the sidebar and 48–61 tags per run. `tb_clean/` is 23 entries and ~20 tags,
rebuilt from `tb/` by `diag/tb_curate.py`; nothing is deleted, so widen the
whitelist in that script and re-run if you need a dropped curve back.

### Why `tb/` is unreadable

| cause | detail |
|---|---|
| `eval_summary` fans out | one sub-run per stats file → **125 sidebar entries** on its own |
| eval dirs were appended to for months | `overfit_ep70_v2` holds **807** event files, `social_nav_hrl` **246**, all in one dir, all at step 0 — TensorBoard merges them into one unreadable run |
| three tag namespaces for the same quantity | training writes `metrics/*` + `reward`; habitat's eval mode writes `eval_metrics/*` + `eval_reward/average_reward`; `diag/eval_to_tb.py` writes `eval/*` + `reward_{success,collision,timeout}/*` |
| tag bloat | of 48–61 tags per run, ~35 are `perf/*` profiler timings, `articulated_agent_force.*`, scene-collision counters and inherited `social_nav_stats.*` (three of them misspelled upstream: `frist_ecnounter_steps`) |

### What is in `tb_clean/`

| entry | source dir | what it is |
|---|---|---|
| `HL_1_main_nr1_3M` ⭐ | `nr1` | the 3 M run. Rollout success peaks near 190 k (0.669 smoothed) and falls to 0.489 by 3 M — this is the stochastic policy on the upsampled training set, so its absolute value is **not** comparable to the `EVAL_*` numbers |
| `HL_2_seed200`, `HL_3_seed300` | `nr1_s200/s300` | same recipe, other seeds — reproducibility |
| `HL_4_ablation_B_upsample` | `mechB` | upsampling only — the arm that grew a new capability |
| `HL_5_ablation_C_dapg` | `mechC` | + DAPG — the demo anchor suppressed it |
| `HL_6_probe_lr_half` | `lrhalf` | lr 2.5e-5 probe |
| `HL_7_recipe_b2`, `HL_8_recipe_g3`, `HL_9_recipe_train36` | `b2`,`g3`,`train36` | the diagnostic line that produced the PPO recipe |
| `LL_1_bc_and_es` | `curves` | BC / DAgger / ES curves behind `ll_yield_es_v3.pth` |
| `EVAL_0_ref_teacher_n108` | `teacher__cert3` | the teacher, drawn as a **flat line** across the whole x-range so "did we pass it" is readable without arithmetic. 0.8333 / 0.1204 |
| `EVAL_1_nr1_over_training_n108` ⭐ | `mechBC` + `nr1__u122/183/244/366` | **the whole HL story in one plot.** `eval_to_tb.py` writes every stats file at step 0; these are re-keyed to the training step they came from (`u<N>` × 8192), with BC at step 0 because it *is* this run's initialisation. Reads: 0.8056 → **0.8148 @1.0M** → 0.8056 → 0.7963 → 0.7593. Rises ~1 point over BC, then gives it all back, and never reaches the teacher line |
| `EVAL_2_…_holdout_n6` | `nr1__ck*_holdout` | the same four models on the zero-leak pair. Kept for honesty, but n=6 can only take the values k/6 — it reads 0.8333 at every checkpoint and resolves nothing |
| `EVAL_3…9_*` | `eval_summary/*` | seeds, both ablation arms, teacher/BC references, the LL result, the two recipe finals. The `_n108` / `_n6` suffix is the eval-set size — never compare across it |

### What is *not* in it

`b1`, `g1/g2/g4/g5`, `train36b…e`, `prod1`, `bcft_ep70`, `cleandev`, `cleandev13`,
`multiscene_v2/_d20` — diagnostic runs with no kept checkpoint; and
`overfit_ep70_v2`, `social_nav_hrl`, `social_nav_overfit`, `flat_e2e` — habitat
eval-mode dirs whose steps are all 0. Plus 110 of the 125 `eval_summary` sub-runs,
which certify scripted modes on retired datasets. All still in `tb/`.

### The curves worth looking at

| group | tags |
|---|---|
| outcome | `metrics/social_nav_to_pos_success`, `metrics/did_collide`, `metrics/num_steps`, `reward` |
| safety | `metrics/min_agent_clearance`, `metrics/human_delay` |
| behaviour | `metrics/social_nav_stats.yield_ratio`, `.backup_ratio` |
| reward decomposition | `metrics/social_nav_reward_breakdown.*` (7) |
| optimisation | `learner/agent_0_{value_loss,action_loss,value_pred_mean,dist_entropy,grad_norm}` |
| eval scorecard | `eval/{success,collision,timeout}_rate`, `eval/mean_steps*`, `reward_{success,collision,timeout}/*` |

## Known traps

1. **Loading a multi-agent checkpoint for eval is a silent no-op.** Always extract
   with `hl/_extract_hl_map.py` and pass `pretrained_hl_weights` +
   `eval.should_load_ckpt=False`.
2. **Eval stats keys are `scene|episode_id|eval_idx`.** Aggregating by `episode_id`
   silently drops all but the last repeat — a real bug that once made
   `evals_per_ep=3` runs look like `n=36` instead of `n=108`.
3. ~~**Evaluate under the training reward.**~~ **Fixed 2026-08-17.** This used to bite
   because `social_nav_hierarchical_overfit_v2.yaml` carried its own frozen phase-1
   values (`success 30`, `collide 5`, `safe_dis_min 1.5`) and every caller had to
   override them by hand. It now *inherits* `social_nav_hrl_final.yaml`, so training,
   evaluation and teacher certification all read one reward. Keep it that way: put
   recipe changes in `social_nav_hrl_final.yaml`, never on a command line.
4. **`RULE_LOG` appends.** `rm -f` the target first or demos from several runs merge.
5. **Training-window metrics are noise** (±15 reward, ±6 pp success measured while the
   actor was frozen). Only deterministic evals count.

---

## Authoring new episodes (`scene_tools/`)

The scene renderers and the hand-clicking tool. They import each other, so they
live in one directory.

```bash
# 1. generate door candidates from a raw scene set
DEX 'python hrl_pipeline/scene_tools/gen_v3_datasets.py …'

# 2. (optional) render browsing thumbnails
DEX 'python hrl_pipeline/scene_tools/render_scene_menu.py data/archive_20260817/social_nav_episode_doorcand_v4.json.gz'

# 3. build the click page, open it in a browser, click 4 points per door, Export
DEX 'python hrl_pipeline/scene_tools/make_click_page.py \
       --src data/archive_20260817/social_nav_episode_doorcand_v4.json.gz \
       --out hrl_pipeline/clicking/click_new.html --half 6.0'
#    -> save the downloaded clicks.json into hrl_pipeline/results/

# 4. turn the clicks into a runnable dataset. It goes into the ARCHIVE, not data/:
#    data/ holds only the nine built files, and raw click batches are pool SOURCES.
DEX 'python hrl_pipeline/scene_tools/build_manual_episodes.py \
       hrl_pipeline/results/<your>.json \
       data/archive_20260817/social_nav_episode_<name>.json.gz'

# 5. certify it (which scripted mode solves each episode) -- see the HL section
# 6. fold it into the pool: add its stem to build_pool_v1.py, then rebuild the nine
DEX 'python hrl_pipeline/data_tools/build_datasets.py --build'
```

`click_server.py` + `live_run.sh` are the live variant: click a door and get a
single runnable episode back immediately.

Past click sessions and their artifacts are in `clicking/`: `click30.html` /
`click.html` (the generated pages), `review/` + `review0805/` (one image per
authored door), `live_requests/` (raw live-click payloads, including one iteration
that was discarded and exists nowhere else), and the audit CSVs. The clicked
coordinates themselves are `results/clicks.json` (17 doors) and
`results/0805.json` (30 doors).

`habitat-lab/scripts/` is upstream Habitat code, **except `scripts/tk/`** — 8 project
scripts (`run_oracle_social_nav.py`, `get_topdown_map.py`, plotting helpers) that
predate this pipeline and are not called by anything in `hrl_pipeline/`.
