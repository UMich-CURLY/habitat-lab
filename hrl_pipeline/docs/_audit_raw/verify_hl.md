# Adversarial verification — corrections list

## A. WRONG claims

**A1. §0 — "all 36 `Average window` keys exist as `metrics/<k>`" is false for 1 of 36.**
`reward` is published as the **top-level** tag `reward`, not `metrics/reward`. Verified: `EventAccumulator('/habitat-lab/tb_nr1')` → `'metrics/reward' in tags == False`; `reward` exists with n=367. Correct statement: 35 of 36 map to `metrics/<k>`; `reward` maps to `reward`.

**A2. §0 — "values match to 6 dp" is unverifiable/false.**
The stdout prints 3 decimals (`did_collide: 0.202`). TB holds `0.20212766528129578`. They agree to the log's **3 dp**, which is all that can be checked. (`runs/_ppo_nr1.txt` line 2 vs TB step 81920.)

**A3. §0 — TB-only tag count undercounts by 3.**
Report: "12 `learner/*` + 7 `perf/*` tags that never reach stdout." Also TB-only: `metrics/rollout_fail_frac`, `metrics/rollout_frac_stable-failure`, `metrics/rollout_frac_stable-success` — `grep -c rollout_fail_frac _ppo_nr1.log` = **0**. Correct count: **22** TB-only tags (12+7+3), out of 58.

**A4. §0 — the "+4 h" offset is stated against the wrong reference.**
TB `wall_time` for `tb_nr1` step 8192 = `2026-08-15 06:54:15 UTC`; the log's first `update: 10` = `2026-08-15 07:02:24`. Container is UTC (`docker exec wxinyuan date` → `UTC`), host is EDT. **Log timestamps equal TB wall-clock**; the +4 h is versus **host `ls` mtimes**. Pairings still hold, but the stated fact is wrong.

**A5. §2.7 — "`_ppo_mechA.log` (3 updates before it died)" is wrong; it is 30 updates.**
`grep -o "update: [0-9]*" _ppo_mechA.log | tail -3` → `update: 10 / 20 / 30`. Three *log lines* at `log_interval=10`. (`_ppo_dryrun.log` really is 31 updates — that one logs every update, `update: 1…31`.)

**A6. §1 row 14 — `ll/ll_mse_units.py:122-123` does not rename anything.**
Those lines are a **bitwise identity check**, preceded by the comment `# cross-check: does the retrained ckpt match the production one bitwise?`:
```python
for new, prod in ((f"{D}/_tmp_bc_v2.pth", f"{D}/ll_yield_bc_v2.pth"),
                  (f"{D}/_tmp_bc_dagger1.pth", f"{D}/ll_yield_dagger1.pth")):
```
`ll_yield_dagger1.pth` already existed; this script never writes it. The provenance claim for row 14 is unsupported. (Related: `ll_yield_bc_v2.pth`, `_tmp_bc_v2.pth`, `_tmp_bc_dagger1.pth`, `ll_demos_v2_clean.jsonl`, `ll_train_r1.jsonl`, `ll_dagger_r1.jsonl` exist **nowhere** on disk — `ll_mse_units.py` is entirely non-runnable today, not just "stale defaults at :122-123".)

**A7. §2.6(b) — "present in all 4 `_es_*.log`" is false for the `torch seed` line.**
`[es] torch seed N` appears **only** in `_es_seed1.log` and `_es_seed2.log`. `_es_v2.log` and `_es_v3.log` have only `workdir` / `perturbing` / `done` (3 lines — which is consistent with row 15's "3 header lines", so §1 and §2.6(b) contradict each other).

**A8. §2.2 — "(`cert3_` must precede `cert_` in that tuple.)" is false.**
`"cert3_m0805_always_go".startswith("cert_")` is `False`; the loop `break`s on first match, so ordering is irrelevant. Harmless, but stated as a requirement.

**A9. §3 — "142 of 168 logs" contradicts its own itemization (143).**
17 PPO + 2 tb_curves + 123 eval + 1 `_cert_dev20…` = **143**, and 143 + 25 must-keep = 168. The 5.51 GB figure is right (total raw = 5,615,155,241 B; must-keep = 105,109,290 B for `_ppo_mechA`+`_ppo_dryrun` plus ~0.86 MB).

**A10. §3 — "~106 MB → ~62 MB" is arithmetically impossible.**
The 6 named keepers are `_ppo_mechA` (59,657,491 B) + `_ppo_dryrun` (45,451,799 B) + `_ep0video` (142,281) + `_uncert3_video` (628,247) + `_ppo_mechB` (2,780) + `_ppo_g3b` (2,728) = **105.9 MB**. 106 MB → 106 MB. The follow-on "`_archive/logs_raw/` can go to zero if §2.7 is also done" is also wrong — `_ep0video` and `_uncert3_video` have no json by the report's own §0.

**A11. §0 vs §3 — mutually inconsistent eval-log accounting.**
§0: "130 of 132 eval logs matched exactly; only `_ep0video` and `_uncert3_video` have no json." §3: 123 value-matched, plus `_cert_dev20_parked_rule_waitgo` produced no json. My independent match (mean success / did_collide / num_steps of `runs/*.txt` vs mean over every `results/*.json`) gives **132 eval logs, 129 matched, 3 unmatched**: `_ep0video.txt`, `_uncert3_video.txt`, and `_chain_bd.txt` (driver log, no `num_steps` line). "130 of 132" is not reproducible.

**A12. §1 row 11 — line citation off by two.** The `bc_pretrain_hl.py | grep` block in `_archive/scripts/_hl_train_stage1.sh` is at **lines 32-35**, not 30-34.

**A13. §2.6(f) — `w.add_text("source", fname, 0)` is at `eval_to_tb.py:173`, not :172.** Line 172 is `w.add_scalar(tag, float(val), 0)`.

**A14. §1 "TB dirs with no surviving log" — `tb_social_nav_overfit` is misfiled.**
It has **zero training curves**: 27 scalars, all `eval_metrics/*` + `eval_reward/*`, 3 points, all at step 0, across 3 event files. It belongs with `tb_overfit_ep70_v2` / `tb_social_nav_hrl` as a default-`tensorboard_dir` eval dump, not with `tb_bcft_ep70` / `tb_cleandev*` / `tb_multiscene_*` (which each hold 48 real `learner/*`+`metrics/*` training tags).

**A15. §1 — "all unnamed and colliding at step 0" overstates `tb_overfit_ep70_v2`.**
475 of 480 points are at step 0; 5 are at steps 278528 / 370688. `tb_social_nav_hrl` (225/225) and `tb_social_nav_overfit` (3/3) *are* fully step-0.

---

## B. Internal contradictions / claims that break the recommendation

**B1. §3 lists `_es_v3.log` and `_bc_ll_dagger1_full.log` as "deletable outright" while §1 row 15 and §2.6(b)(c) say those exact files hold unique content.**
Verified: `runs/_es_v3.txt` keeps the 11 `[es] gen` lines but **drops** `workdir=`, `perturbing 2 tensors, 258 params; init=bc sigma=0.03`, and the `done. best fitness=… -> …ll_yield_es_v3.pth` line (the KEEP regex has `saved ->`, not `->`). `runs/_bc_ll_dagger1_full.txt` **drops** `demos: 30950  act lin[min,max]=…`. They cannot be both "deletable" and the sole source for §2.6(b)/(c).

**B2. §3 must-keep row for `_chain_g5.log` — the HL-BC curve is NOT log-only.**
`runs/_chain_g5.txt` already contains all 11 `epoch … loss/dec=… acc=…` lines plus `saved -> …bc_train36_wait_hl.pth` (the KEEP regex has `epoch `). What is raw-only in that log is the `input: approach=… dim=10` line, the `episodes=30 decisions=915 action_dist=…` line, the `probe_wait_prob` output, and the `+ python -u -m habitat_baselines.run …` echo. §2.5 can be satisfied from the 28-line `runs/_chain_g5.txt`, not the 7.7 KB log.

**B3. §2.5 — the curve does not belong to any kept checkpoint.**
`_chain_g5.log:132` → `saved -> /habitat-lab/hrl_pipeline/bc_train36_wait_hl.pth`. That file is **not** among the 16 in `weights/` (which has `bc_train36_hl.pth` and `bc_stable_hl.pth`). So "the only HL-BC learning curve the project has" is for a third, discarded BC model — it does not fill the row-11/row-12 gap.

**B4. §2.6(f) — "the only machine-readable log → stats-json provenance link" that "the shrink drops" is wrong for 6 files.**
123 raw logs contain `Wrote per-episode eval stats to …`; **6 survive the shrink**: `runs/_ppo_ckpt1_eval.txt`, `_prod1_ckpt6_eval.txt`, `_prod1_ckpt29_eval.txt`, `_ppoD_ckpt7_eval.txt`, `_ppoD_ckpt8_eval.txt`, `_ppoE_ckpt7_eval.txt` (kept by the `ckpt` alternative in the KEEP regex).

---

## C. Proposed fixes that would fail or misbehave if run today

**C1. §2.2's widened glob creates a silent run-name collision the report does not mention.**
With `("stats_","bd_","cert3_","cert_","recheck_")`, both `cert3_manual_rule_markov.json` and `cert_manual_rule_markov.json` → `manual__rule_markov`. Two `SummaryWriter`s open the same logdir → two event files in one run, duplicate step-0 scalars. Verified by replaying `run_name()` over all 126 eval-shaped jsons: exactly one collision.

**C2. §2.6(e)'s routing rule `re.search(r"tensorboard_dir=(\S+)", cmd)` silently drops 12 of 23 CLI echoes.**
Per-file `+ python -u -m habitat_baselines.run` counts vs. those carrying `tensorboard_dir=`:
`_chain_bd` 2/0, `_formal_evals` 5/0, `_seed_runs_driver` 4/2, `_mech_check_C` 3/1, `_mech_check_B2` 2/1, the rest 1/1. Eval invocations have no `tensorboard_dir` — there is no target dir for them.

**C3. §2.3's ES commands work, but seed1/seed2 produce a flat dead line, not a "variance result" with structure.**
All four `_es_*.log` do have 11 `[es] gen` lines (report is right that `RE_ES` parses them). But `_es_seed1/2` are `gen0 center fitness=-1.000 {}` + 10 × `fits: -1.000 … | best=-1.000 {} sigma=…` — `ast.literal_eval("{}")` yields no `es/best_success` / `es/best_collide` tags at all. Expect 3 tags (`es/best_fitness`, `es/pop_*`, `es/sigma`), all constant −1.

**C4. §2.6(a)'s `RE_RECON` misses the output line.** `_center_recon.log`'s last line is `reconstructed final center -> /habitat-lab/hrl_pipeline/ll_yield_center_r2.pth` — the only record of that artifact's provenance, and dropped by the shrink (`saved ->` doesn't match `center ->`).

**C5. Confirmed load-bearing, as reported:** with `SD = "/habitat-lab/hrl_pipeline"`, `ls /habitat-lab/hrl_pipeline/stats_*.json` inside the container → `No such file or directory`. `main()` `shutil.rmtree(OUT)` (line 154) runs *before* the glob (156-160), so re-running `eval_to_tb.py` today destroys all 111 published runs and republishes zero. This is the report's single most important finding and it is correct.

---

## D. MISSING — stale paths §4 does not list

§4 cites only per-file defaults; it misses every **directory constant**, which is the same bug class:

- `/home/xinyuan/habicrowd/Simulator/habitat-lab/hrl_pipeline/data_tools/build_pool_v1.py:27` — `SD = "/habitat-lab/hrl_pipeline"`; also `:18` writes `POOL_v1.csv` (now `results/`)
- `.../data_tools/build_train36.py:20` — `SD = "/habitat-lab/hrl_pipeline"`; `:9` `TRAIN36.csv` (now `results/`)
- `.../data_tools/classify_teacher.py:19` — `TEACHER_CERT3.csv` (now `results/`)
- `.../data_tools/certify_solvability.sh:12` — `SD=/habitat-lab/hrl_pipeline` (§4 cites only `:19`)
- `.../ll/ll_dataset_stats.py:8` — `D = "/habitat-lab/hrl_pipeline"`
- `.../ll/ll_mse_units.py:115` — `D = "/habitat-lab/hrl_pipeline"` (§4 cites only `:122-123`)
- `.../ll/es_ll_finetune.py:18` and `.../data_tools/dataset_provenance.py:13` — usage strings still `hrl_pipeline/<x>.py` (§4 flagged the same class of defect for `eval_to_tb.py:15` / `logs_to_tb.py:13` but stopped there)

Also missing: `diag/` holds two scripts the audit never mentions — `make_video_index.py` and `_probe_summary.py`.

---

## E. Confirmed (spot-checked, correct)

- 31 `tb_*` dirs, of which `tb_view` (7 symlinks) + `tb_focus` (6 symlinks) are pointers → **29 real** ✓
- `tb_nr1`: 58 tags, 367 pts, steps 8192→3006464 ✓ · `tb_mechB` 55 tags/127 pts/→1040384 ✓ · `tb_mechC` 61 tags/184 pts with exactly 6 `learner/agent_0_*dapg*` ✓ · `tb_b2` 318 pts/→651264 ✓ · `tb_nr1_s200`/`s300`/`tb_g3` 184 pts ✓ · `tb_curves/ll_yield_es_v3` 7 tags/11 pts ✓ · `tb_curves/ll_yield_dagger1` 2 tags/9 pts/0→399 ✓ · `tb_flat_e2e` 30 `eval_metrics/*`, 1 pt ✓ · `tb_overfit_ep70_v2` 807 event files, `tb_social_nav_hrl` 246 ✓
- All 17 named PPO logs pair cleanly to a tb dir (log max-update vs TB point count: b1 250/258, b2 310/318, g1-g3 180/184, g4/g5/nr1 360/367, prod1 1460/1465, train36 370/371, b 710/712, c 400/405, d 240/249, e 260/264, mechC 180/184) ✓
- Total raw: **168 logs / 5,615,155,241 B / 31,468,817 lines**; shrink kept **5,931** lines ✓ (all three exact)
- Exactly 5 logs have no `runs/*.txt`: `_center_recon`, `_mech_check_{run,B,BA,AB}` ✓
- Exactly 12 logs carry `^+ python -u -m habitat_baselines.run` — the report's list is exact ✓
- §2.1: 7 `stats_var_*` jsons (mtimes 05:12–08:21 on 08-17) vs `tb_eval_summary` mtime 04:56; no `var__*` run dirs exist ✓
- §2.2: exactly 8 eval-shaped jsons unmatched by the current glob — the report's list is exact ✓. 126 eval-shaped jsons total, 118 globbed, 111 published → **15 unpublished** ✓
- `_archive/scripts/_reorganize.sh:55` is the KEEP regex ✓ · `_dapg_dryrun.sh:18` is `rm -rf checkpoints_dryrun tb_dryrun video_dryrun` ✓ · `docs/CLEANUP_PLAN.md:56` is the `bc_stable_hl.pth … **训练日志已丢失**` row ✓ · `_archive/scripts/bc_flat.py:58` is the `--out` default ✓
- No log anywhere mentions `ll_yield_bc.pth`; the only `saved ->` targets across all 168 logs are `_tmp_bc_v2.pth`, `_tmp_bc_v2_rep.pth`, `_tmp_bc_dagger1.pth`, `bc_train36_wait_hl.pth` ✓ (row 13 correct)
- `_bc_ll_v2_full.log` / `_bc_ll_v2_repeat.log` curves are byte-identical ✓
- `_ppo_mechB.log` is a `FileExistsError` retry echoing the full override list ✓; `MECH_B_EXIT=1` at `_mech_check_B2.log:91` ✓
- §4 line numbers verified exact: `eval_to_tb.py:15,28`, `logs_to_tb.py:13`, `es_ll_finetune.py:32,33,79`, `bc_ll_yield.py:18,19`, `bc_pretrain_hl.py:34,36`, `scripted_hl_eval.py:46,48`, `certify_solvability.sh:19` ✓
- No log references `tb_bcft_ep70` / `tb_cleandev` / `tb_cleandev13` / `tb_multiscene_v2` / `tb_multiscene_d20` / `tb_social_nav_overfit` / `tb_dryrun`; `_mech_check_run.log` and `_mech_check_AB.log` do reference the now-deleted `tb_mechA` ✓