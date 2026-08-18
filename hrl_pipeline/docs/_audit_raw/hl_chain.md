Verified everything against the working tree and against a live container (`wxinyuan`, up 2 days, repo mounted at `/habitat-lab`, new subdirs visible). Smoke-tested the two HL scripts at their new paths, and reproduced the group table from archived stats.

---

# THE HL PIPELINE (copy-pasteable, as it must be run today)

## Step 0 — prelude (run once per host shell)

```bash
# One wrapper that also defines every shorthand INSIDE the container.
DEX() { docker exec -u root wxinyuan bash -c '. activate habitat && cd /habitat-lab \
&& SD=/habitat-lab/hrl_pipeline \
&& W=$SD/weights && DM=$SD/demos && RES=$SD/results && RUNS=$SD/runs \
&& V2=social_nav/social_nav_hierarchical_overfit_v2.yaml \
&& HL=habitat_baselines.rl.policy.agent_0.hierarchical_policy.high_level_policy \
&& SK=habitat_baselines.rl.policy.agent_0.hierarchical_policy.defined_skills.backoff \
&& R=habitat.task.measurements.social_nav_reward && '"$1"; }
```

**REQUIRED PATCH before Steps 1, 2, 3-verify, 8** — 11 scripts hardcode `SD=/habitat-lab/hrl_pipeline` for files that now live in `results/`. Confirmed live failure:
`FileNotFoundError: '/habitat-lab/hrl_pipeline/TEACHER_CERT3.csv'` (`diag/by_group.py:24`), and `eval_to_tb.py` globs 0 files.

```bash
DEX 'cd $SD \
 && sed -i "s|SD = f\"{ROOT}/hrl_pipeline\"|SD = f\"{ROOT}/hrl_pipeline/results\"|" \
      data_tools/build_final_datasets.py data_tools/build_fullset.py \
      data_tools/build_quickset.py data_tools/build_stableset.py \
      data_tools/classify_teacher.py data_tools/verify_demo_jsonl.py \
      diag/by_group.py diag/analyze_yield_geometry.py \
 && sed -i "s|SD = \"/habitat-lab/hrl_pipeline\"|SD = \"/habitat-lab/hrl_pipeline/results\"|" \
      data_tools/build_pool_v1.py data_tools/build_train36.py diag/eval_to_tb.py \
 && grep -n "^SD" data_tools/*.py diag/by_group.py diag/eval_to_tb.py'
```
Acceptance: every printed `SD` line ends in `/hrl_pipeline/results`.
(`data_tools/certify_solvability.sh` needs 3 edits — see Step 2; I give the inline loop instead so you never need it.)

---

## Step 1 — dataset build

**(a) Purpose:** produce/confirm the only two datasets the whole project uses — `TRAIN` (40 entries = 28 stable-success + 4 stable-failure ×3) and `EVAL` (38 = train36_v1 positions 0-35 in original order + 2 zero-leak holdout).

**(b) Normal case — the files already exist; just verify:**
```bash
DEX 'python hrl_pipeline/data_tools/dataset_provenance.py'
```

**(b′) Full rebuild** (only if `TEACHER_CERT3.csv` changed). `build_final_datasets.py:83,88` verify against `train32_up3` and `holdout2`, which were moved to `data/archive_20260817/` — restore them first or the asserts crash:
```bash
DEX 'cp data/archive_20260817/social_nav_episode_train32_up3.json.gz \
        data/archive_20260817/social_nav_episode_holdout2.json.gz data/ \
     && python hrl_pipeline/data_tools/build_final_datasets.py'
```
Upstream of that (only when re-labelling from scratch), in this exact order — `verify_demo_jsonl.py` REWRITES `TEACHER_CERT3.csv`, and re-running `classify_teacher.py` afterwards erases the downgrade:
`build_pool_v1.py` → `build_train36.py` → teacher cert3 eval → `classify_teacher.py` → `build_stableset.py` → demos → `verify_demo_jsonl.py` → `build_final_datasets.py`.

**(c) Inputs → outputs:**
`hrl_pipeline/results/TEACHER_CERT3.csv` + `data/social_nav_episode_train36_v1.json.gz` + `..._manual0805.json.gz` → `data/social_nav_episode_TRAIN.json.gz`, `data/social_nav_episode_EVAL.json.gz`

**(d) Acceptance:** `build_final_datasets.py` must print
`VERIFIED: TRAIN == train32_up3; EVAL[0:36] == train36_v1 in order; EVAL[36:38] == holdout2`
and `TRAIN 40 entries = 28 stable-success + 4x3 stable-failure (32 unique episodes), 23 scenes`.
Provenance check (verified output today): `scenes TRAIN=23 EVAL=25 shared=23`, `episodes identical in both : 30`.
Caveat I measured: unique fingerprints are 30/35, not 32/38 — `dataset_provenance.py:28-36` collides episodes whose `info.robot_goal/human_start/human_goal` are absent. Treat it as a smoke test, not a leakage proof.
Class counts (verified): `Counter({'stable-success': 28, 'stable-failure': 4, 'flaky': 4})`, stable-failure ids `2,3,20,25`, flaky ids `0,1,4,29`.

---

## Step 2 — solvability certification

**(a) Purpose:** run all six scripted modes on a candidate dataset; an episode is certified solvable if any mode succeeds, and *which* mode succeeds names the behaviour PPO must discover (`rule_markov`→SWEET, any-other→SOLVABLE, none→UNCERT).

**(b) Command** (inline loop with every path explicit — avoids the append-mode trap in `certify_solvability.sh`):
```bash
DEX 'DS=manual0805; for m in always_go rule_wait rule_waitgo smart_wait rule_markov rule_yield; do
  rm -f $RES/hl_decisions_${m}_cert_${DS}.jsonl
  RULE_MODE=$m RULE_EVALS=1 STEP_CAP=1200 RULE_CONFIG=$V2 \
    RULE_DATASET=/habitat-lab/data/social_nav_episode_${DS}.json.gz \
    RULE_LOG=$RES/hl_decisions_${m}_cert_${DS}.jsonl \
    EVAL_STATS=$RES/cert_${DS}_${m}.json \
    python hrl_pipeline/hl/scripted_hl_eval.py > $RUNS/_cert_${DS}_${m}.log 2>&1
  echo "done $m rc=$?"
done'
DEX 'cd $RES && python hrl_pipeline/data_tools/certify_merge.py manual0805'
```
`certify_merge.py:33` reads bare relative `cert_<ds>_<mode>.json`, so the `cd $RES` is mandatory.

**(c)** `data/social_nav_episode_<DS>.json.gz` → `results/cert_<DS>_<mode>.json` ×6, `results/hl_decisions_<mode>_cert_<DS>.jsonl` ×6, `runs/_cert_<DS>_<mode>.log` ×6 → `results/cert_<DS>.csv`

**(d) Acceptance:** `certify_merge.py` prints `{'SWEET': n1, 'SOLVABLE': n2, 'UNCERT': n3} -> cert_manual0805.csv`, and `n1+n2+n3` equals the episode count. Each per-mode log must contain `Average episode social_nav_to_pos_success:`; a mode whose log lacks it crashed and its column is silently `-1`.

---

## Step 3 — teacher demo collection

**(a) Purpose:** record one `rule_markov` HL decision per 30-step control interval on the 28 episodes the teacher passes cleanly — the BC/DAPG corpus.

**(b) Commands.** `stableok` was archived; rebuild it. I verified `TRAIN[0:28]` is byte-for-byte the same episodes in the same order as the archived `stableok` (train36 ids `5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,21,22,23,24,26,27,28,30,31,32,33,34,35`):
```bash
# 3a. rebuild the demo dataset (either line works)
DEX 'python hrl_pipeline/data_tools/make_subset.py TRAIN stableok $(seq 0 27)'
#   canonical alternative: DEX 'python hrl_pipeline/data_tools/build_stableset.py'

# 3b. collect (RULE_LOG is APPEND mode -> rm -f first)
DEX 'rm -f $DM/hl_decisions_stable.jsonl && \
  RULE_MODE=rule_markov RULE_EVALS=1 STEP_CAP=1200 RULE_CONFIG=$V2 \
  RULE_DATASET=/habitat-lab/data/social_nav_episode_stableok.json.gz \
  RULE_LOG=$DM/hl_decisions_stable.jsonl \
  EVAL_STATS=$RES/stats_demo_pass.json \
  python hrl_pipeline/hl/scripted_hl_eval.py > $RUNS/_teacher_stable_demo.log 2>&1; \
  grep -E "Average episode (social_nav_to_pos_success|did_collide|num_steps):" $RUNS/_teacher_stable_demo.log'

# 3c. verify the demo pass; any demo-failing episode is downgraded to flaky in TEACHER_CERT3.csv
DEX 'python hrl_pipeline/data_tools/verify_demo_jsonl.py \
      $DM/hl_decisions_stable.jsonl $RES/stats_demo_pass.json \
      $DM/hl_decisions_dapg_success.jsonl'
```

**CORRECTION to `docs/PIPELINE_REFERENCE.md:955`**, which says `RULE_EVALS=3` for this step. It was **1**: the archived `results/stats_demo_pass.json` has exactly **28** entries all keyed `|1` (1-based eval counter), and `verify_demo_jsonl.py:38` asserts `len(segs) == len(stats) == 28`. `RULE_EVALS=3` would produce 84 segments and abort.

**(c)** `data/social_nav_episode_stableok.json.gz` → `demos/hl_decisions_stable.jsonl`, `results/stats_demo_pass.json`, `runs/_teacher_stable_demo.log` → `demos/hl_decisions_dapg_success.jsonl`

**(d) Acceptance** (all reproduced from the archived artifacts):
- log: `Average episode social_nav_to_pos_success: 1.0000`, `Average episode did_collide: 0.0000`, `Average episode num_steps: 847.7143`
- `wc -l demos/hl_decisions_stable.jsonl` = **849**
- `verify_demo_jsonl.py` prints `demo pass: 28 episodes, downgraded to flaky: none` and `kept 849 decisions (28 episodes)`
- output is byte-identical to input (md5 `3454b5a6145861f3fcf0b5b82500e0b5` on the reference corpus)

---

## Step 4 — BC pretrain

**(a) Purpose:** clone the teacher into the *same* modules the neural HL uses (1-layer GRU(32) + `CategoricalNet(32,3)`), so PPO fine-tunes from a policy that already yields instead of exploring go→yield→go from scratch.

**(b) Command** (`LD_LIBRARY_PATH` is required — verified):
```bash
DEX 'LD_LIBRARY_PATH=/opt/conda/envs/habitat/lib \
     BC_JSONL=$DM/hl_decisions_dapg_success.jsonl \
     BC_OUT=$W/bc_stable_hl.pth \
     BC_EPOCHS=1000 \
     python hrl_pipeline/hl/bc_pretrain_hl.py 2>&1 | grep -E "input|episodes|epoch|saved"'
```

**(c)** `demos/hl_decisions_dapg_success.jsonl` → `weights/bc_stable_hl.pth` (`{"state_encoder":…, "policy":…}` only; no critic)

**(d) Acceptance** — I ran this today at the new path (3-epoch probe) and got the first two lines verbatim:
```
input: approach=True prev_action=True dim=10
episodes=28 decisions=849 action_dist={2: 597, 0: 181, 1: 71}
epoch  999  loss/dec=…  acc≈0.985
saved -> /habitat-lab/hrl_pipeline/weights/bc_stable_hl.pth
```
`dim=10` is the load-bearing number: 6 feat + `approach` + one-hot3(`prev_action`), matching `input_feature_dim: 7` + `use_prev_action: true`. If it prints `dim=6` the demos came from a non-v2 config and the warm start will fail with a shape error at `social_nav_neural_policy.py:234`.

---

## Step 5 — PPO fine-tune (the NR1 recipe)

**(a) Purpose:** fine-tune the cloned actor with HRLPPO under the layer-2/3 reward, no DAPG, warm critic first.

**(b) Command:**
```bash
DEX 'export SPLIT_SCENES=1 DISABLE_CUDNN=1 && \
 python -u -m habitat_baselines.run --config-name=$V2 \
   habitat.dataset.data_path=/habitat-lab/data/social_nav_episode_TRAIN.json.gz \
   +$HL.pretrained_hl_weights=$W/bc_stable_hl.pth \
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
   > $RUNS/_ppo_nr1.log 2>&1; echo NR1_TRAIN_EXIT=$?'
```
`TRAIN.json.gz` replaces `train32_up3` (archived; content-identical, asserted by `build_final_datasets.py:85`). Multi-seed replication = the same command + `habitat.seed={200,300}` + `total_num_steps=1.5e6 num_checkpoints=15` and `checkpoint_folder/tensorboard_dir` suffixed `_s$SEED`.

**(c)** `data/social_nav_episode_TRAIN.json.gz` + `weights/bc_stable_hl.pth` → `checkpoints_nr1/ckpt.{0..29}.pth` + `latest.pth`, `checkpoints_nr1/rollout_counts.json`, `tb_nr1/`, `runs/_ppo_nr1.log`

**(d) Acceptance** (all four verified in the archived raw log / TB):
- `cuDNN disabled (DISABLE_CUDNN=1)` — `_ppo_nr1.log:6`
- `Number of params to train: 4356` — `:114`
- `critic_lr=0.03: 2 critic tensors at 0.03, 6 at 5e-05` — `:115`. If this line is absent, `critic_lr` did not take and the run is invalid.
- **warm start actually loaded**: the first logged window (`update: 10`) already shows `social_nav_to_pos_success: 0.681`. A random init sits near 0. Do **not** grep for `Loaded BC-pretrained HL weights` — that `baselines_logger.info` line does **not** appear in the run logs, and a missing-weights load is only a *warning* (`social_nav_neural_policy.py:252-255`), i.e. silently random.
- `checkpoints_nr1/rollout_counts.json` exists and is keyed `teacher_class|train36_id` (proves rollout-mix tracking is live; it self-disables permanently if episodes lack `info.pool.teacher_class`).
- TB `metrics/rollout_fail_frac` mean **0.157** (measured over 367 updates; last value 0.230). This is the *measured* dilution of the ×3 upsampling under `SPLIT_SCENES=1`.
- Guard before any eval:
```bash
DEX '[ -f checkpoints_nr1/latest.pth ] && echo OK || echo TRAINING_INCOMPLETE_ABORTING_EVALS'
```
- Health probe: `DEX 'python hrl_pipeline/hl/check_train_health.py tb_nr1 $RUNS/_ppo_nr1.log 60'`. On the reference 3M run this correctly returns `ALERT OPTIMIZATION_FAILURE succ 0.62->0.49 reward 25.2->13.7` — expected, and exactly why the headline is the pre-registered **1.5M** point, not the endpoint.

---

## Step 6 — checkpoint extraction

**(a) Purpose:** strip the multi-agent RL checkpoint down to `{state_encoder, policy}` in the `pretrained_hl_weights` format, so eval is decoupled from the (random-ish) critic and runs deterministic-argmax.

**(b) Command:**
```bash
DEX 'for ck in 9 14 19; do
       python hrl_pipeline/hl/_extract_hl_map.py checkpoints_nr1/ckpt.$ck.pth $W/nr1_ck${ck}_hl.pth
     done; python hrl_pipeline/hl/_extract_hl_map.py checkpoints_nr1/latest.pth $W/nr1_final_hl.pth'
```
Readout map (verified: 8192 frames/update, 367 updates ≈ 3.006M): `ckpt.N` ≙ (N+1)×100k steps → **ckpt.9 = 1.0M (u122), ckpt.14 = 1.5M (u183), ckpt.19 = 2.0M (u244), latest = 3.0M (u366)**.

**(c)** `checkpoints_nr1/{ckpt.9,ckpt.14,ckpt.19,latest}.pth` → `weights/nr1_{ck9,ck14,ck19,final}_hl.pth`

**(d) Acceptance** — I ran this today at the new path:
```
saved /habitat-lab/hrl_pipeline/weights/nr1_ck14_hl.pth: state_encoder=4 policy=2 head_shape= (3, 32)
```
`state_encoder=4` (`rnn.weight_ih_l0 (96,10)`, `weight_hh_l0 (96,32)`, 2 biases), `policy=2`, `head_shape=(3,32)`. The critic is deliberately dropped — such a file cannot resume PPO.

---

## Step 7 — deterministic eval

**(a) Purpose:** deterministic-argmax, 3-eval-per-episode readout of the extracted HL weights on the full evaluation set, with the (random) critic irrelevant.

**(b) Command** (repeat per tag `ck9→u122 / ck14→u183 / ck19→u244 / final→u366`):
```bash
DEX 'TAG=u183; CK=$W/nr1_ck14_hl.pth; \
 python -u -m habitat_baselines.run --config-name=$V2 \
   habitat_baselines.evaluate=True habitat_baselines.eval.should_load_ckpt=False \
   +$HL.pretrained_hl_weights=$CK \
   habitat_baselines.num_environments=1 habitat_baselines.eval.evals_per_ep=3 \
   habitat.dataset.data_path=/habitat-lab/data/social_nav_episode_EVAL.json.gz \
   habitat.seed=7 habitat.environment.max_episode_steps=1200 \
   "habitat_baselines.eval.video_option=[]" \
   habitat_baselines.eval.episode_stats_path=$RES/stats_nr1_$TAG.json \
   > $RUNS/_eval_nr1_$TAG.log 2>&1; echo EVAL_EXIT=$?; \
 grep -E "Average episode (social_nav_to_pos_success|did_collide|num_steps):" $RUNS/_eval_nr1_$TAG.log'
```
**Caveat:** `EVAL` is 38 episodes (36 + 2 holdout), so the printed headline averages over 114 evals and is **not** directly comparable to the historical 36-episode headline. For an exact historical comparison run `data_path=…train36_v1.json.gz` (still at `data/` top level) and the holdout separately. `episode_stats_path` must be non-empty or **no stats file is written at all** (`habitat_evaluator.py:500-503`).

Analysis (this is the one tool with **no** stale paths and no CSV dependency — it reads `EVAL.json.gz`'s own `info.split`):
```bash
DEX 'python hrl_pipeline/diag/eval_summary.py \
       TEACHER=$RES/stats_teacher_cert3.json \
       NR1_1.5M=$RES/stats_nr1_u183.json,$RES/stats_nr1_ck14_holdout.json'
```
A single `EVAL`-run stats file also works as one argument (positions 0-37 map straight onto `SPLIT`).

**(c)** `weights/nr1_*_hl.pth` + `data/social_nav_episode_EVAL.json.gz` → `results/stats_nr1_<tag>.json` (108 or 114 entries), `runs/_eval_nr1_<tag>.log`

**(d) Acceptance** — reference numbers (all re-read from `results/` today; `Average episode …` lines from `runs/_eval_nr1_*.txt`):

| tag | ckpt | steps | success | did_collide | mean num_steps | entries |
|---|---|---|---|---|---|---|
| teacher (rule_markov) | — | — | 0.8333 | 0.1204 | 804.95 | 108 |
| u122 | ckpt.9 | 1.0M | 0.8148 | 0.1389 | 816.11 | 108 |
| **u183** | **ckpt.14** | **1.5M** | **0.8056** | **0.1481** | **775.47** | 108 |
| u244 | ckpt.19 | 2.0M | 0.7963 | 0.1481 | 773.79 | 108 |
| u366 | latest | 3.0M | 0.7593 | 0.1574 | 768.99 | 108 |
| u366_holdout | latest | 3.0M | 0.8333 | 0.1667 | 701.17 | 6 |
| seed 200 | 1.5M | — | 0.8333 | 0.1204 | 783.63 | 108 |
| seed 300 | 1.5M | — | 0.8056 | 0.1296 | 790.72 | 108 |

And the `eval_summary.py` table I regenerated today (this is the deliverable readout):
```
group                n               TEACHER              NR1_1.5M
stable-success      28      28/28 coll0 840步      25/28 coll0 803步
  ├ efficient       13      13/13 coll0 729步      12/13 coll0 729步
  ├ inefficient     14      14/14 coll0 921步      13/14 coll0 871步
  └ near_cap         1       1/1 coll0 1149步          0/1 coll0 0步
stable-failure       4          0/4 coll3 0步        1/4 coll3 922步
flaky                4          0/4 coll3 0步        1/4 coll3 960步
holdout              2                     -        1/2 coll1 722步
  NR1_1.5M   stable-success   n=25  Δ-26 步
  NR1_1.5M   丢掉的稳定成功: [17, 18, 35] | 攻克的稳定失败: [20]
```
Sanity floor: a silently-random policy (typo'd weights path) evaluates at ~0.0-0.1 success; anything ≥0.75 means the warm start loaded.

---

## Step 8 — tensorboard publish

**(a) Purpose:** put the numbers that actually decide things (per-episode deterministic evals, and the BC/DAgger/ES curves that only ever went to stdout) into TensorBoard alongside the PPO curves.

**(b) Commands** (needs the Step-0 patch — `eval_to_tb.py:157-159` globs `$SD/stats_*.json`, which today matches nothing):
```bash
DEX 'python hrl_pipeline/diag/eval_to_tb.py'                       # add --keep to append
DEX 'python hrl_pipeline/diag/logs_to_tb.py bc_hl=$RUNS/_bc_hl.log'
DEX 'ls /habitat-lab/tb_eval_summary | head; ls /habitat-lab/tb_curves'
```

**(c)** `results/{stats_*,bd_*,cert_*}.json` → `/habitat-lab/tb_eval_summary/<group>__<run>/` (+ a `summary/all_evals` markdown text tab); stdout logs → `/habitat-lab/tb_curves/<name>/`

**(d) Acceptance:** `published <N> eval runs -> /habitat-lab/tb_eval_summary` with `N ≥ 100` (there are 137 files in `results/`), followed by per-run `name n=… succ=…% coll=…` lines whose `nr1__u183` row reads `n=108 succ=81%`. `N = 0` means the patch was not applied.

---

# A. Every env var `hl/scripted_hl_eval.py` honours

| Var | Line | Default | Meaning |
|---|---|---|---|
| `RULE_MODE` | `:33` | `rule_backoff` | Teacher: `always_go`, `rule_backoff`, `rule_wait`, `rule_waitgo`, `rule_yield`, `rule_markov` (production), `smart_wait`, `always_backoff`, `learned` (pass-through: logs the network's own choice, no override, `:340-341`) |
| `RULE_DIST` | `:34` | `"3.0"` | `DIST_T`, the conflict distance. **Also binds `MARKOV_PARAMS["trig_dist"]` at import** (`:65`) while `rel_dist=3.5`/`rel_min_dist=1.0` stay fixed — `RULE_DIST>3.5` makes the trigger wider than the release band |
| `RULE_DATASET` | `:35-37` | `/habitat-lab/data/social_nav_episode_overfit_switch.json.gz` — **now archived, default is dead** | `habitat.dataset.data_path` (`:436`); also the source of per-episode door geometry for `rule_yield` (`:371`) |
| `RULE_EVALS` | `:38` | `"8"` | `habitat_baselines.eval.evals_per_ep` (`:435`). Production: 1 for certification/demos, 3 for stability certificates |
| `RULE_VIDEO` | `:39` | `"0"` | `"1"` → `eval.video_option=['disk']` (`:432`) |
| `RULE_VIDEO_DIR` | `:48` | `hrl_pipeline/video_<MODE>` | `habitat_baselines.video_dir` (`:433`) |
| `RULE_CONFIG` | `:40-42` | `social_nav/social_nav_hierarchical_overfit.yaml` | `--config-name` (`:428`). **`rule_markov` raises `rule_markov needs the 7-dim policy state` under this default** (`:110-114`); every real run passes `..._overfit_v2.yaml` |
| `EVAL_STATS` | `:43` | `""` (off) | → `habitat_baselines.eval.episode_stats_path` (`:439-442`). Empty ⇒ **no stats file is written at all** |
| `RULE_LOG` | `:45-47` | `/habitat-lab/hrl_pipeline/hl_decisions_<MODE>.jsonl` | Per-HL-decision jsonl. **Opened in APPEND mode** (`:313`) — always `rm -f` first |
| `RULE_TRACE` | `:56` | `""` (off) | Path → additionally patches `HierarchicalPolicy.act` for one row per env STEP (loc, feat, active skill index) (`:373-421`); joins the decision log via `num_steps` |
| `DEBUG_PREV` | `:49` | `"0"` | Prints harness `prev_actions` vs the teacher's own `_prev_hl` at each decision (`:331-337`) |
| `STEP_CAP` | `:443-448` | `""` (off) | → `habitat.environment.max_episode_steps`. Caps via the TASK length, deliberately *not* the evaluator's forced reset (which slices long episodes and skips others). Every production run passes 1200 |
| `RULE_EXTRA` | `:449-453` | `""` | Whitespace-split extra hydra overrides appended to `sys.argv`. This is how `RULE_MODE=learned` gets `+…high_level_policy.pretrained_hl_weights=…`, and how the learned LL is swapped in |

Hard-coded, **not** overridable (`:426-438`): `habitat_baselines.evaluate=True`, `eval.should_load_ckpt=False`, `eval_ckpt_path_dir=checkpoints_social_nav_hrl/latest.pth`, `num_environments=1`, `habitat.seed=7`.

Not read by the harness but affecting the same runs: `SPLIT_SCENES` (default `"1"`, `habitat_env_factory.py:49`), `DISABLE_CUDNN` (`ppo_trainer.py:99-101`), `LL_MODEL`/`LL_LOG`/`LL_DAGGER` (`social_nav_skills.py:229`, `:384-394`), `LD_LIBRARY_PATH=/opt/conda/envs/habitat/lib`, `HAB` (used by the `data_tools`/`diag` scripts, not by the harness).

---

# B. `rule_markov` teacher — pseudo-code with every threshold

Source of truth: `hl/scripted_hl_eval.py:64-147`. Actions: `0=backoff(yield)`, `1=wait`, `2=go_to_goal`. Called once per `control_interval=30` env steps (≈1 s @30 Hz); "tick" = one HL decision.

```
CONSTANTS (MARKOV_PARAMS, scripted_hl_eval.py:64-98)
  trig_dist        = DIST_T = 3.0      # bound to RULE_DIST at IMPORT time
  trig_bearing     = 1.75  rad
  trig_rel_heading = pi/2  rad
  trig_approach    = 0.15  m/s
  trig_ticks       = 1                 # no debounce: trigger fires immediately
  rel_dist         = 3.5   m           # WIDER than trigger -> the hysteresis band
  rel_approach     = 0.05  m/s
  rel_min_dist     = 1.0   m           # retreat phase ONLY
  rel_ticks        = 2                 # release needs 2 consecutive clear decisions
  wait_after_ticks = 6                 # retreat/hold split, counted in DECISIONS

PER-ENV STATE (all reset when masks[i]==0, i.e. episode boundary, :314-320)
  trig = 0 ; rel = 0 ; boff = 0 ; prev_hl = -1

OBSERVATION  feat = social_nav_policy_state, MUST be 7-dim else RuntimeError (:110-114)
  d        = feat[0]   human_dist
  b        = feat[1]   human_bearing        (robot frame)
  rh       = feat[2]   human_rel_heading    (|rh| > pi/2  <=>  human faces us)
  approach = feat[6]   human_approach_speed (EMA alpha 0.2; ego-motion INVARIANT --
                       a parked human reads ~0 no matter how the robot moves.
                       feat[3] human_rel_speed is unsigned + ego-polluted and is
                       NOT used by this teacher.)

DECIDE(feat):
  yielding := (prev_hl in {0, 1})            # regime switch reads prev action ALONE

  # ---------- NOT YIELDING: trigger ----------
  if not yielding:
      trig_now := (d < 3.0) and (|b| < 1.75) and (|rh| > pi/2) and (approach > 0.15)
      trig = trig + 1 if trig_now else 0
      rel  = 0 ; boff = 0
      return 0 if trig >= 1 else 2           # backoff on the very first conflict tick

  # ---------- YIELDING: release / phase split ----------
  # "still in conflict" -- EGO-INVARIANT terms only. bearing was tried and REMOVED:
  # the yield skill spins the robot while reversing, swinging the human's bearing
  # out of any release cone and firing release mid-transit (collided cd17 ep0-7).
  still := ( (d < 3.5) and (approach > 0.05) )
        or ( (d < 1.0) and (boff < 6) )      # rel_min_dist applies ONLY while retreating:
                                             # in the wait phase nothing moves, d freezes
                                             # (measured 0.99 for 900+ steps) and the gate
                                             # could never clear.
  rel  = rel + 1 if not still else 0
  trig = 0
  if rel >= 2:   return 2                    # corridor clear -> resume
  if boff >= 6:  return 1                    # retreat done -> hold in place, face the goal
  boff = boff + 1
  return 0                                   # keep retreating
```

Notes that matter:
- The **backoff phase is 7 decisions, not 6**: the triggering decision returns 0 with `boff` reset to 0 (`:126-129`); `boff` only increments on subsequent yielding decisions (`:146`, checked `:144`). `wait` therefore starts on the 8th decision.
- Both `backoff` **and** `wait` enter the release branch (`yielding = prev_hl in (0,1)`, `:118`), so the whole hysteresis is reconstructible from the prev-action one-hot the student receives.
- The split is by retreat **duration**, not distance. A distance split (`wait` whenever `d<1.5`) deadlocked: stopping near the human can leave the robot standing *in* its path, the firm human freezes against it (0.8 m block gate), `d` never grows, and the release predicate never fires — 2 of 3 sweet episodes timed out (`:91-96`).
- `_rule_action` (`:210-264`) — the `human_rel_speed>0.1` motion gate, `smart_wait`'s `>0.15`, `rule_yield`'s door-pass latch (0.3 m past the door line, robot's side) and 3-tick recede counter — is **dead code for `rule_markov`**.
- `_prev_hl[i] = a` is written for **all** modes including `learned` (`:351`), so the `prev_action` column of a learned trace is meaningful too.
- Replay check: `hl/analyze_hl_decisions.py:33-67` re-implements this exactly; on a healthy corpus `(obs,prev_action)-conditioned markov replay: disagree 0/849 (0.00%)`.

---

# C. Stale hardcoded paths

## C1 — BROKEN (crashes, or silently finds nothing)

| File:line | Old path | Should now be |
|---|---|---|
| `data_tools/certify_solvability.sh:19` | `python hrl_pipeline/scripted_hl_eval.py` | `python hrl_pipeline/hl/scripted_hl_eval.py` |
| `data_tools/certify_solvability.sh:12` | `SD=/habitat-lab/hrl_pipeline` | `SD=/habitat-lab/hrl_pipeline/results` (and `:19` log → `…/runs/`) |
| `data_tools/certify_solvability.sh:15,20` | `$SD/hl_decisions_$mode.jsonl` (rm/mv) | mismatches the harness's `RULE_LOG` default once `SD` moves → set `RULE_LOG` explicitly, else append-mode contamination |
| `data_tools/build_pool_v1.py:27` | `SD = "/habitat-lab/hrl_pipeline"` | `…/hrl_pipeline/results` — feeds 17 `cert_*`/`recheck_*`/`stats_*` reads (`:33,:43`), `0805.json` (`:111`), `POOL_v1.csv` write (`:242`) |
| `data_tools/build_train36.py:20` | `SD = "/habitat-lab/hrl_pipeline"` | `…/results` (writes `TRAIN36.csv`, `:138`) |
| `data_tools/classify_teacher.py:30` | `SD = f"{ROOT}/hrl_pipeline"` | `…/results` (reads `stats_teacher_cert3/wait_t36/wait_bc.json` `:46-48`; writes `TEACHER_CERT3.csv` `:97`) |
| `data_tools/build_stableset.py:14` | `SD = f"{ROOT}/hrl_pipeline"` | `…/results` (`:16`) |
| `data_tools/build_fullset.py:15` | `SD = f"{ROOT}/hrl_pipeline"` | `…/results` (`:17`) |
| `data_tools/build_quickset.py:21` | `SD = f"{ROOT}/hrl_pipeline"` | `…/results` (`:23` read, `:86` `QUICK<N>.csv` write) |
| `data_tools/verify_demo_jsonl.py:19` | `SD = f"{ROOT}/hrl_pipeline"` | `…/results` (`:22` read, `:59` in-place rewrite) |
| `data_tools/build_final_datasets.py:26` | `SD = f"{ROOT}/hrl_pipeline"` | `…/results` (`:29`) |
| `data_tools/certify_merge.py:33` | bare relative `cert_{ds}_{m}.json` | must `cd hrl_pipeline/results` first (missing file → silent `-1` column, not an error) |
| `diag/by_group.py:22` | `SD = f"{ROOT}/hrl_pipeline"` | `…/results` — **verified `FileNotFoundError` today** |
| `diag/eval_to_tb.py:28` | `SD = "/habitat-lab/hrl_pipeline"` | `…/results` — **verified globs 0 files today** (`:157-159`) |
| `diag/analyze_yield_geometry.py:35` | `SD = f"{ROOT}/hrl_pipeline"` | `…/results` for `stats_trace_<tag>.json` (`:45`); `trace_<tag>.jsonl` (`:47`) **no longer exists anywhere** — regenerate with `RULE_TRACE` |
| `hl/scripted_hl_eval.py:35-37` | default `RULE_DATASET=…/social_nav_episode_overfit_switch.json.gz` | moved to `data/archive_20260817/` — default now dead (always overridden) |
| `hl/bc_pretrain_hl.py:34` | default `BC_JSONL=…/hrl_pipeline/hl_decisions_rule_markov.jsonl` | `…/hrl_pipeline/demos/hl_decisions_rule_markov_all.jsonl` (the old file is in `_archive/data/`) |
| `ll/es_ll_finetune.py:32` | `HARNESS = "…/hrl_pipeline/scripted_hl_eval.py"` | `…/hrl_pipeline/hl/scripted_hl_eval.py` — LL ES is broken until fixed |
| `ll/es_ll_finetune.py:33` | `BASE = "…/hrl_pipeline/ll_yield_bc.pth"` | `…/hrl_pipeline/weights/ll_yield_bc.pth` |
| `ll/ll_dataset_stats.py:8` | `D = "/habitat-lab/hrl_pipeline"` | `…/hrl_pipeline/demos` |
| `ll/ll_mse_units.py:115` | `D = "/habitat-lab/hrl_pipeline"` | `…/hrl_pipeline/demos` |
| `ll/bc_ll_yield.py:18` | default `LL_DATA=…/hrl_pipeline/ll_demos.jsonl` | `…/hrl_pipeline/demos/ll_demos_v2_clean.jsonl` (old name never existed post-reorg) |

## C2 — Dataset paths stale for a different reason (moved to `data/archive_20260817/`)

| File:line | Old dataset | Fix |
|---|---|---|
| `data_tools/build_final_datasets.py:83` | `data/social_nav_episode_train32_up3.json.gz` | restore from archive, or use `data/social_nav_episode_TRAIN.json.gz` |
| `data_tools/build_final_datasets.py:88` | `data/social_nav_episode_holdout2.json.gz` | restore from archive, or `EVAL[36:38]` |
| `_archive/scripts/_formal_run.sh:35`, `_seed_runs.sh:21` | `…_train32_up3.json.gz` | `…_TRAIN.json.gz` |
| `_archive/scripts/_formal_run.sh:85` | `…_holdout2.json.gz` | `EVAL` positions 36-37 |
| PIPELINE_REFERENCE call-chain (b)1 | `…_stableok.json.gz` | rebuild (`build_stableset.py`) or `make_subset.py TRAIN stableok 0..27` |

## C3 — COSMETIC (still runs; writes into the old flat `hrl_pipeline/`)

| File:line | Old | Should be |
|---|---|---|
| `hl/scripted_hl_eval.py:46` | `RULE_LOG` default `…/hrl_pipeline/hl_decisions_<MODE>.jsonl` | `…/hrl_pipeline/demos/…` |
| `hl/scripted_hl_eval.py:48` | `RULE_VIDEO_DIR` default `hrl_pipeline/video_<MODE>` | `hrl_pipeline/videos/<MODE>` |
| `hl/bc_pretrain_hl.py:36` | `BC_OUT` default `…/hrl_pipeline/bc_markov_hl.pth` | `…/hrl_pipeline/weights/…` |
| `ll/bc_ll_yield.py:19` | `LL_OUT` default `…/ll_yield_bc.pth` | `…/weights/ll_yield_bc.pth` |
| `ll/es_ll_finetune.py:79` | `--out` default `…/ll_yield_es.pth` | `…/weights/ll_yield_es.pth` |
| `ll/es_ll_finetune.py:105` | `tempfile.mkdtemp(dir="/habitat-lab/hrl_pipeline")` | a `runs/` or `_archive/dirs/` subdir |

## C4 — Stale docstring/usage lines (no runtime effect, but they are what people copy)

`hl/scripted_hl_eval.py:20` · `data_tools/build_pool_v1.py:18` · `data_tools/build_train36.py:9` · `data_tools/classify_teacher.py:19` · `data_tools/dataset_provenance.py:13` · `diag/eval_to_tb.py:15` · `diag/logs_to_tb.py:13` · `diag/analyze_yield_geometry.py:4` · `ll/bc_ll_yield.py:3,9` · `ll/es_ll_finetune.py:18` — all say `hrl_pipeline/<name>.py` or `hrl_pipeline/<file>`; all now need the subdirectory.

Clean (no stale paths): `hl/_extract_hl_map.py`, `hl/analyze_hl_decisions.py`, `hl/probe_wait_prob.py`, `hl/check_train_health.py`, `data_tools/make_subset.py`, `diag/eval_summary.py`, `diag/reward_table.py`, `diag/dump_losses.py`, `diag/dump_reward.py`.

---

# D. Config keys that select the final recipe

`HL = habitat_baselines.rl.policy.agent_0.hierarchical_policy.high_level_policy`, `R = habitat.task.measurements.social_nav_reward`. `+` = key absent from the schema, so Hydra needs the append form.

## D1 — warm start / critic

| Key | Value | Form | Default | Why |
|---|---|---|---|---|
| `+$HL.pretrained_hl_weights` | `$W/bc_stable_hl.pth` | `+` | `""` (`social_nav_neural_policy.py:228`) | Loads only `state_encoder`+`policy`; critic stays random. **A missing file is a warning, not an error** (`:252-255`) |
| `+$HL.critic_warmup_calls` | `120` | `+` | `0` (`:171-172`) | Counted in `evaluate_actions` **calls**, not updates: `ppo_epoch=2 × num_mini_batch=1` = 2 calls/update ⇒ 120 = **60 updates**. Features are detached during warmup so the shared GRU stays frozen |
| `habitat_baselines.rl.ppo.critic_lr` | `3e-2` | plain | `0.0` = single param group (`default_structured_configs.py:308`; impl `ppo.py:53,130-146`) | Timescale fix: an orthogonal-init value head needs O(1e5) Adam steps to reach ±40 returns at lr 5e-5 |
| `+$HL.bc_anchor_coef` | **not set** (0.0) | — | `0.0` (`:182-183`) | Legacy L2-SP trust region; off in every current run |
| `+$HL.critic_output_scale` | **not set** (1.0) | — | `1.0` (`:196`) | Diagnostic probe; off |
| `+$HL.dapg_*` | **not set** | — | `dapg_coef 0.0`, `dapg_decay_calls 240`, `dapg_batch_eps 8` (`:208-212`) | **Dropped from the formal round**: the gradient probe measured cosine(g_dapg, g_ppo) median **−0.669** = real suppression |
| `map_obs_key` | `""` | — | `""` (`:121`) ⇒ `_map_dim=0` | Map branch implemented but inert |

## D2 — PPO

| Key | Value | Default it overrides |
|---|---|---|
| `rl.ppo.num_steps` | **1024** | 256 (`social_nav_hierarchical.yaml:86`) → 8192 frames/update at 8 envs |
| `rl.ppo.num_mini_batch` | **1** | 4 (`:80`) → exactly 2 `evaluate_actions` calls/update |
| `rl.ppo.entropy_coef` | **0.0** | 0.004 (`social_nav_hierarchical_overfit_v2.yaml:53`) / 0.001 base |
| `rl.ppo.value_loss_coef` | **0.05** | 0.5 (`social_nav_hierarchical.yaml:81`) |
| `rl.ppo.lr` | 5e-5 | 2.5e-4 (`:83`) |
| `rl.ppo.max_grad_norm` | 2.0 | 10.0 (overfit_v2 `:52`) / 0.5 base |
| `rl.ppo.gamma` | 0.99 | 0.99 (`:88`) — passed explicitly for the record |
| `rl.ppo.ppo_epoch` / `clip_param` / `tau` / `use_gae` | 2 / 0.2 / 0.95 / true | from yaml, not overridden |
| `rl.ppo.hidden_size` | 32 | must equal `high_level_policy.hidden_dim` (overfit_v2 `:50`, `:58`) |
| `rl.ppo.use_normalized_advantage` | True | overfit_v2 `:51` |
| `updater_name` / `rollout_storage_name` | `HRLPPO` / `HrlRolloutStorage` | supplies `loss_mask`; advantages normalized over VALID slots only |
| `total_num_steps` / `num_checkpoints` | 3.0e6 / 30 (seeds: 1.5e6 / 15) | ckpt every 100k |

## D3 — reward overrides (the three new terms default to **0.0** — omit them and you are running the old reward)

| Key | Value | Schema default | Note |
|---|---|---|---|
| `habitat.task.success_reward` | **50.0** | 30.0 (overfit_v2 `:22`) | applied OUTSIDE the measure (`habitat/core/environments.py:79-80`), so it never appears in `comp_sums` |
| `$R.collide_penalty` | **30.0** | 5.0 yaml / 1.0 schema | |
| `$R.safe_dis_min` | **0.0** | 1.5 | fully disables the proximity penalty; the yield potential still fires via `yield_dis=3.0` |
| `$R.eff_success_reward` | **20.0** | **0.0 (OFF)** (`habitat-lab/…/default_structured_configs.py:1495`) | terminal-only, `× max(0, 1 − steps/eff_step_cap)` |
| `$R.eff_step_cap` | 1200.0 | 1200.0 (`:1496`) | **training cap is 1500** (overfit_v2 `:17`), so a training success at 1201-1500 steps earns zero efficiency bonus while still paying slack |
| `$R.corridor_potential_coef` | **3.0** | **0.0 (OFF)** (`:1502`) | potential-based (telescopes); `corridor_safe_clear=1.0` (`:1503`) from the yield-geometry diagnosis |
| `$R.release_bonus` | **3.0** | **0.0 (OFF)** (`:1507`) | once-per-episode, gated on `release_mpd_min=0.75` (`:1508`), `release_hold_steps=30`, `release_robot_speed=0.008` |
| from yaml, not overridden | `slack_reward −0.01`, `end_on_collide true`, `yield_dis 3.0`, `backoff_reward 3.0`, `goal_progress_reward 2.0` | overfit_v2 `:23,:26,:35,:36,:34` | |
| `habitat.environment.max_episode_steps` | 1500 train (yaml `:17`) / **1200 eval** (CLI) | | |

## D4 — environment variables that are part of the recipe

`SPLIT_SCENES=1` (formal/seed) — 0 makes every env cycle the full episode set but OOMs on the 23-scene set (~18 GB by u60); the resulting upsampling dilution is *measured* as `metrics/rollout_fail_frac` (0.157), not assumed away. `DISABLE_CUDNN=1` — the 10→32 GRU gains nothing from cuDNN and its workspace was the first OOM casualty.

---

# E. Discrepancies found while verifying (report these upstream)

1. **`docs/PIPELINE_REFERENCE.md:955` says `RULE_EVALS=3` for the demo pass. It was `RULE_EVALS=1`.** `results/stats_demo_pass.json` has 28 entries, all keyed `|1`; `verify_demo_jsonl.py:38` asserts 28. `RULE_EVALS=3` aborts the chain.
2. **The `Loaded BC-pretrained HL weights from …` line is not in any run log** — it does not appear in `_archive/logs_raw/_ppo_nr1.log` (3.08M lines) or the eval logs. Do not use it as an acceptance check; use `Number of params to train: 4356` + `critic_lr=0.03: …` + first-window success ≈0.68.
3. **`check_train_health.py` correctly flags the reference 3M run**: `ALERT OPTIMIZATION_FAILURE succ 0.62->0.49 reward 25.2->13.7`. The 1.5M pre-registered point (ckpt.14/u183) is the headline; the endpoint degrades. Every eval number above confirms the monotone slide 0.815 → 0.806 → 0.796 → 0.759.
4. **`data_tools/build_final_datasets.py` can no longer re-run unmodified** — its own verification block reads `train32_up3` and `holdout2`, both moved to `data/archive_20260817/`.
5. **`diag/eval_summary.py` supersedes `diag/by_group.py`** for HL readouts: it reads the split out of `EVAL.json.gz`'s `info.split`, needs no `TEACHER_CERT3.csv`, has no stale paths, and handles the holdout rows. `by_group.py` additionally mis-keys any stats file that is not the full 36-episode `train36_v1`.