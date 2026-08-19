Verification complete. Here is the corrections list.

---

# ADVERSARIAL VERIFICATION — `ll_chain` report

## A. WRONG claims (priority)

**A1. §6c "5 flip their verdict (8, 11, 13, 14, 17)" — WRONG. Exactly 3 flip: 8, 11, 13.**
Directly computed from `results/stats_esv3_train36_v1.json` vs `results/stats_var_es_v3.json` (both `es_v3`, same HL, seed 7):
- train36-file non-stable set = **{13, 14, 17, 35}**
- EVAL-file non-stable set = **{8, 11, 14, 17, 35}**
- symmetric difference = **{8, 11, 13}** — 3 episodes, over all 36 episodes (not just the ss class).

Episodes **14 and 17 fail in BOTH runs**, so naming them as "flips" is false, and the count 5 is inflated by 67 %. Episode 35 (which the report never mentions) fails in both too. The neighbouring sentence "35 of 36 episodes differ in step count" **is** correct (only ep 35 has an identical step vector). This weakens the report's own headline conclusion: the visit-order artefact moves **3** verdicts, not 5.

**A2. §6c "The spread is real; the stated cause is wrong" — UNSUPPORTED, and self-contradictory.**
The report correctly kills `LL_TRAINING_CURVES.md §5.3`'s stated mechanism (`eval.deterministic: true` at `social_nav_hierarchical_v2.yaml:15` / `overfit_v2.yaml:45` → `habitat_evaluator.py:183-189` → `social_nav_neural_policy.py:731-732 dist.mode()`; the `dist.sample()` branch at `:734` is dead). But it then asserts the spread itself is real. The only on-disk evidence says the opposite:
```
_archive/dirs/_repro_v3/log_x_win{1..5}.txt  → 1.0000 / 0.0000 / 839.7857  (5/5 identical)
_archive/dirs/_repro_v3/log_x_ctr{1..3}.txt  → 0.9286 / 0.0714 / 789.0714  (3/3 identical)
```
i.e. sd 0.000000 — the same evidence the report itself cites two bullets earlier. §5.3's numbers `0.929673 / 0.929095 / 0.824232` appear **nowhere on disk except the doc text** (`grep -rl` over all of `hrl_pipeline/` hits only `LL_TRAINING_CURVES.md`). Correct verdict: mechanism wrong **and** the spread is unreproducible from artifacts.

**A3. §6c understates the `_ll_variance_summary.txt` confound.** The report says the 老师 column is confounded by *episode file*. It is also a **different HL**: `_archive/scripts/_ll_variance.sh:66` sets 老师 = `stats_teacher_cert3.json` + `stats_holdout2_teacher.json` (rule_markov teacher HL), while every other column runs `RULE_MODE=learned` + `nr1_ck14_hl.pth` (`:25-28`). HL is the larger of the two confounds and goes unmentioned.

**A4. §5 header "All 'old' targets verified non-existent today" — false for at least three of its own rows.**
- `/habitat-lab/data/social_nav_episode_bcsolid6_v2.json.gz` (`es_ll_finetune.py:78` default) **EXISTS** (4778 B, Jul 17).
- `/habitat-lab/hrl_pipeline` (`es_ll_finetune.py:105` mkdtemp `dir=`) exists — the report's own row says it "works".
- `scripted_hl_eval.py:46` RULE_LOG dir exists — the report's own row says "dir still exists".

**A5. §1 step 1 vs step 5 use the same phrase "at the ±0.35 floor" with two different definitions.** Measured on `demos/ll_demos_v2_clean.jsonl` / `demos/ll_dagger_r1.jsonl`:

| statistic | A (demos) | B (DAgger) |
|---|---|---|
| `abs(ang) == 0.35` exactly | 0.1029 | 0.1987 |
| `abs(ang) >= 0.35` | 0.2486 | 0.3913 |
| `abs(ang) > 0.35` | 0.1457 | 0.1926 |
| `abs(ang) >= 0.9999` | 0.0089 | 0.0147 |

Step 1's "24.9 % at the ±0.35 floor" is the `>=` figure; step 5's "10.3 %→19.9 %" is the `==` figure. Both derivable, but a reader comparing 24.9 % against 10.3 % concludes the corpora differ when they don't. Related: **§4's "`ang` saturates at ±0.35 for 25–39 % of rows" is a misnomer** — 0.35 is the teacher's `min_turn_rate` floor (`social_nav_skills.py:495-496`), not tanh saturation; genuine saturation (`|ang|≈1.0`) is 0.9 % / 1.5 % of rows, and 14.6 % / 19.3 % of rows sit strictly *above* 0.35.

**A6. §7 `‖W_last‖ = 3.2043` — WRONG, it is 3.204668.** (`‖Δ_last‖ = 0.5827616` ✓, ratio 0.181848 → 18.2 % ✓.)

**A7. §3 "4,356 trainable HL params" is the harness print, not the checkpoint.** `weights/nr1_ck14_hl.pth` holds `{state_encoder, policy}` = **4,323** params (rnn 960+3072+96+96, policy linear 96+3). 4,356 = 4,323 + the 33-param critic head, which the checkpoint does not contain.

## B. Commands that would fail / do damage if run today

**B1. §8, first command — it OVERWRITES the artifact §6a depends on.** It writes `EVAL_STATS=$SD/results/stats_ll36_teachhl.json` while loading `model_path=…/ll_yield_es_v3.pth`. But §6a's table uses that exact file as the **DAgger-LL-under-teacher-HL** arm (verified: 25/28 stable, collision-eps `{8,9}`, 849 own-stable steps, 84/108, 23 collisions — exactly §6a's DAgger column). Running the runbook as written destroys the only DAgger/teacher-HL evidence in the report and silently relabels it as es_v3. Needs a new stats path (e.g. `stats_esv3_teachhl.json`) — which is also the report's own Gap #5.

**B2. §1 step 6's `cat` regeneration line names a file that does not exist.** `$SD/demos/ll_demos_v3_clean.jsonl` is not on disk; only `demos/ll_demos_v2_clean.jsonl` (15,490 rows) and `demos/ll_dagger_r1.jsonl` (15,460) exist. Gap #4 says "regenerate it with the `cat` above" — the `cat` above is the v3-naming one and will fail.

**B3. §1 step 7's "mandatory first check" glob is stale for the historical run.** `$SD/es_ll_*/log_center.txt` matches nothing; the round-2 workdir is `$SD/_archive/dirs/es_ll_jl5ncpsk`, mode **drwx------ root** (`docker exec -u root` required). Both greps do pass there — verified `Skills: {0: LearnedYieldSkill(` at line 51 and `Number of params to train: 4356` at line 17.

## C. Stale paths the report MISSED

| file:line | stale value | why it breaks |
|---|---|---|
| `diag/eval_to_tb.py:28` | `SD = "/habitat-lab/hrl_pipeline"` | every `stats_*.json` / `bd_*.json` in its `ALIASES` table (`:36-45`) now lives in `results/`; the tool silently emits an empty TB run |
| `data_tools/build_pool_v1.py:27` (+ join at `:33`) | `SD = "/habitat-lab/hrl_pipeline"` | `stats(path)` returns `{}` for every missing file (`:34-35`) — it degrades silently, no error |
| `data_tools/dataset_provenance.py:13`, `diag/logs_to_tb.py:13`, `diag/eval_to_tb.py:15`, `data_tools/build_train36.py:9`, `data_tools/build_pool_v1.py:18`, `data_tools/classify_teacher.py:19`, `diag/analyze_yield_geometry.py:4` | `hrl_pipeline/<f>` in usage strings | the report lists docstring-only staleness for `ll/` files but omits these |

Line-number nits in the report's own table:
- `ll/ll_mse_units.py:123` is `f"{D}/ll_yield_dagger1.pth"`, **not** a deleted `_tmp_*` file — correct fix is `weights/ll_yield_dagger1.pth`, not "deleted".
- `_archive/scripts/_reorganize.sh` — the "paths are rewritten" promise is at **:9-10**, not `:8-9`. (The substantive claim is right: `grep -n "sed\|rewrit\|absolute"` over all 73 lines returns only the comment.)
- `docs/LL_PLAN.md` — the `:460-480` citation is on **:58**, not `:57`. Also, LL_PLAN:57-58's claimed bug ("`_teacher_action` 缺 human_stop_dist=1.0 冻结分支") **is fixed** in current code (`social_nav_skills.py:479-480`); that is more stale than the line numbers.
- `hierarchical_policy.py:542` (§6c) is `self._call_high_level,`; `deterministic` is passed at **:543**.
- §1 step 2 "the assert at `:28`" conflates two asserts: `t` presence is `:28`, `t <= cum[-1]` is `:30`.

## D. MISSING coverage

**D1. §6b tabulates only *losses* vs the scripted control, never *gains within the stable-success class*.** Measured gains over `stats_nr1_u183.json`: DAgger **+ep17**, ES v2 **+ep18**, ES v3 **+ep18**. This is what makes 25 − 2 = 24 arithmetically consistent, and it means ES v3 buys a third stable-success episode the scripted LL fails under `nr1_ck14` — omitted from the "honest bottom line", which counts only capability ids 2 and 3.

**D2. §6b's "all five columns share one visit order — verified" is not possible as stated.** The five columns include `holdout2`, whose stats come from separate 2-episode files. Verified: the five *train36* files (`teacher_cert3`, `ll36_teachhl`, `ll36_nr1hl`, `nr1_u183`, `esv2_t36`, `esv3_t36`) do share byte-identical key order.

**D3. §6a omits robot–scene collisions** (the metric §6b leans on): teacher-HL scripted 5,676 vs DAgger 5,759 — i.e. the +33 % scene-grinding effect is an `nr1_ck14`-only phenomenon, which strengthens the report's own HL-coupling argument and is left on the table.

**D4. `train36_v1` contains a content-duplicate pair (episodes `1` and `3` are byte-identical apart from `episode_id`/`info`).** Nothing in the report mentions it; it makes any episode-content-based index mapping ambiguous for those two ids and is worth a line given §0 rebuilds subsets by index.

## E. Confirmed (spot list, all re-derived)

Source/lines: `social_nav_skills.py` `:229` LL_LOG read, `:307-326` write + `:325` append-mode, `:233`/`:414` required_obs_keys, `:268-274` freeze, `:281-302` reverse, `:384-385` LL_MODEL precedence + stale default, `:395` eager `torch.load`, `:430-444` 22-D assembly, `:437-443` door zero-fallback, `:453-471` DAgger log, `:474-498` `_teacher_action`, `:689` FlatNavSkill; `hierarchical_policy.py:96` `eval(skill_name)`, `:362` `num_steps` (never reset ⇒ run-global); `find_action_range` at `:86-88`; `scripted_hl_eval.py:434` `num_environments=1`, `:437` seed 7, `:451` RULE_EXTRA; `es_ll_finetune.py` `:32/:33/:79/:105` stale, `:53` env copy without RULE_EXTRA, `:66-67` −1.0 sentinel, `:136` vs `:145` center-vs-best; `bc_ll_yield.py:9` "norm" docstring vs `:77` no-norm; `v2_llyield.yaml:17-21` + stale `:20`; `twoagent_v2.yaml:10/41/46/54/67/99`; `overfit_v2.yaml:45/58-59`; `LidarScanSensor` at `social_nav_sensors.py:1715` (16 / 3.0 / 0.1); `SocialNavPolicyStateSensor` dims 4-6 unused; `did_collide` = agent–agent only (`multi_agent_sensors.py:16-24`); `ll_mse_units.py:16 ANG_SPEED=10.0` with a comment citing the *tk* yaml ⇒ the 2.5× correction (0.6117→0.2447 rad/s ≈ 14.0 °/s) is right, `LIN_SPEED=10.0` is right (`ac_freq_ratio: 1` inherited from `hssd_spot_human_social_nav_twoagent.yaml:211`); all 26 spot-checked `file:line` citations in the data_tools/diag/hl rows land on the claimed text.

Data: 15,490 / 15,460 / 30,950 rows; lin ∈ [−0.099999, 0]; |ang| mean 0.1816→0.2338; |ang|<0.05 26.5→19.2 %; freeze rows 0/0; 2,403 shared `t`, 2,402 differing, max |Δang| 1.4055, max Δhuman_dist 1.2958; min human_dist 1.0153; ang var 0.0707→0.0837; predict-mean 0.035445→0.042670; BC curves 0.0368/0.0377→0.0018/0.0019 and 0.0025/0.0027; val MSE 0.001886 → 0.020698 (11.0×), |err ang| max 1.283423, MSE(ang)/MSE(lin)=123; demo run 84/84, 0 coll, 840.7976 steps, stats 84/84 clean; DAgger run 83/84, 0 coll; L0 probe table exact including ep20 timeout at cap 1200.

ES: gen0 0.827101 → g4_c11 0.930018, never beaten; four tied-perfect at 839.79/847.36/851.29/855.71; winner logs 984 (ep10) and 483 (ep9) scene collisions at 14/14 0-coll; `es_v3` ≡ `g4_c11` in **all six** tensors, `net.0`/`net.2` bit-identical to `dagger1`; 258 params. Provenance: `outputs/2026-08-16/08-32-53/.hydra` shows `overfit_v2.yaml` + `+…pretrained_hl_weights=…nr1_ck14_hl.pth` + `model_path=…ll_yield_dagger1.pth` + `es14`, and 111 hydra dirs (not 110+1 — 111 total) reference `es_ll_jl5ncpsk`, **all** with `nr1_ck14`. The "stated defaults give 0.114" claim reproduces exactly: `_repro_v3/log_dup_{A,B}.txt` → 0.3571/0.3571/779.6429 ⇒ fitness 0.1136.

Datasets: `stableok/l0set/es8/es14/holdout2` absent from `data/`, present in `data/archive_20260817/`; index lists `[5..35]`, `[2,20,25,21,26]`, `[2,13,23,27,21,26,20,5]`, `[…,8,9,10,12,16,30]` all reproduce by non-info episode equality; `stableok == TRAIN[0:28]` elementwise; `stableok` == the 28 `stable-success` rows of `results/TEACHER_CERT3.csv`; `holdout2` ∉ train36 and ∉ TRAIN, = EVAL positions 36-37 ⇒ zero-leak confirmed; EVAL positions 0-35 are content-identical to train36 ids 0-35.

§6a/§6b numbers: every rollout-success, human-collision, scene-collision, capability, flaky, stable-count, loss-list, holdout2 and paired-Δstep figure reproduces. Two aggregation conventions are in play and both are correct but undeclared — §6b step counts are means over *that arm's own* 3/3 episodes within the stable-success class (803.5 / 822.6 / 838.2 / 842.1), and holdout2 step counts are means over the single solved episode (722.3 / 943.3 / 828.7); paired Δ is per-*episode* on jointly-3/3 episodes (+9.8, +14.9, +23.6, +35.3, +36.0). Worth stating, since the per-*eval* pairing gives different numbers (+11.1, +20.0, +19.6, +33.9, +39.4).