Verified against the working tree and the live `wxinyuan` container. Findings, priority-ordered.

---

# A. WRONG / UNSUPPORTED

**A1. The Step-6 checkpoint→step readout map is wrong for 3 of its 4 tags.** Read from each checkpoint's own `extra_state["step"]` (`/habitat-lab/checkpoints_nr1/*.pth`):

| ckpt | actual step | actual update | report says |
|---|---|---|---|
| ckpt.0 | 8,192 | u1 | ("(N+1)×100k" ⇒ 100k) |
| ckpt.9 | 966,656 | **u118** | 1.0M, u122 |
| ckpt.14 | 1,499,136 | u183 | 1.5M, u183 ✓ |
| ckpt.19 | 2,031,616 | **u248** | 2.0M, u244 |
| latest | 2,990,080 | **u365** | 3.0M, u366 |

Real law: `ckpt.N` = `8192 + N×106,496` steps = `u(1 + 13N)`. Only the headline `ckpt.14 = u183` is right. `latest.pth` is bit-identical to `ckpt.28` (I compared the HL tensors) — there is no 3.0M policy; the "3M endpoint" is 2.99M.

**A2. `checkpoints_nr1/ckpt.{0..29}.pth` (Step 5c) is wrong — only `ckpt.0 … ckpt.28` exist (29 files).** `num_checkpoints=30` did not emit a ckpt.29. `docs/NR1_REPORT.md` (assets line) already says "checkpoints_nr1(29 ckpt + latest)".

**A3. Step 2's merge command fails as written.** `DEX 'cd $RES && python hrl_pipeline/data_tools/certify_merge.py manual0805'` — after `cd $RES` the *relative script path* no longer resolves. Verified live: `ls: cannot access 'hrl_pipeline/data_tools/certify_merge.py': No such file or directory`. Fix: `cd $RES && python $SD/data_tools/certify_merge.py manual0805`.

**A4. Step 1's "caveat" has the right numbers and the wrong cause — and buries a real defect.** Report: "30/35 … `dataset_provenance.py:28-36` collides episodes whose `info.robot_goal/human_start/human_goal` are absent." **Zero** TRAIN and **zero** EVAL episodes lack `info.robot_goal` (measured). The collisions are genuine duplicate geometry inside `train36_v1`:
- train36 ids **27 and 31** are the *same* episode (scene `104348361_171513414`), both labelled `stable-success` — both are in TRAIN and both in EVAL.
- train36 ids **1, 2, 3** are one geometry (scene `103997541_171030615`) split across classes: id1=`flaky`, id2 and id3=`stable-failure`. So the "4 stable-failure episodes" are **3 distinct geometries**, and TRAIN's 12 upsampled entries cover 3, not 4.

This survives a much stricter fingerprint (adding `start_rotation` + `pool.click_id`): still 30/35. So it is *not* fingerprint coarseness and *not* "a smoke test, not a leakage proof" — it is a duplicate-episode + inconsistent-label finding that belongs in Section E.

**A5. `demos/hl_decisions_stable.jsonl` and `demos/hl_decisions_dapg_success.jsonl` do not exist.** `hrl_pipeline/demos/` holds exactly three files: `hl_decisions_rule_markov_all.jsonl`, `ll_dagger_r1.jsonl`, `ll_demos_v2_clean.jsonl`. The reference corpora are at `/home/xinyuan/habicrowd/Simulator/habitat-lab/hrl_pipeline/_archive/data/hl_decisions_{stable,dapg_success}.jsonl` (both 849 lines, both md5 `3454b5a6145861f3fcf0b5b82500e0b5` — the byte-identity claim is correct, the path is not). Consequences: Step 3(d)'s `wc -l demos/hl_decisions_stable.jsonl` fails, and Step 4's `BC_JSONL=$DM/hl_decisions_dapg_success.jsonl` fails on any shell that hasn't just re-run Step 3.

**A6. Step 8's `logs_to_tb.py bc_hl=$RUNS/_bc_hl.log` fails — that file does not exist anywhere in the repo** (`find` over the whole tree: nothing matching `*bc_hl*`), and `runs/` holds only `.txt`. The only surviving log carrying HL-BC `loss/dec` lines is `hrl_pipeline/runs/_chain_g5.txt` (raw: `_archive/logs_raw/_chain_g5.log`). Confirming the gap: `/habitat-lab/tb_curves/` contains only `ll_yield_dagger1` and `ll_yield_es_v3` — the HL BC curve was never published.

**A7. Step 8's acceptance count is wrong.** `diag/eval_to_tb.py:156-159` globs `stats_*.json | bd_*.json | cert_*.json` = **118** files in `results/`, not 137 (137 is the whole dir, incl. 7 `.csv`, `0805.json`, `clicks.json`). The glob also **silently skips 6 `cert3_*.json` and 2 `recheck_*.json`** that are real eval stats — a coverage gap the report doesn't mention. The last publish produced **111** run dirs in `/habitat-lab/tb_eval_summary/` (112 entries = 111 dirs + 1 root event file).

**A8. Two "I ran this today at the new path" claims are partly doctored.** `weights/bc_stable_hl.pth` mtime `2026-08-13 18:45`; `weights/nr1_ck14_hl.pth` mtime `2026-08-15 08:25` — neither was rewritten today, so the `saved -> …/weights/bc_stable_hl.pth` and `saved /habitat-lab/hrl_pipeline/weights/nr1_ck14_hl.pth: …` lines in the acceptance blocks were not produced by the claimed runs. I re-ran both (to `/tmp`) and the *substance* holds: `input: approach=True prev_action=True dim=10`, `episodes=28 decisions=849 action_dist={2: 597, 0: 181, 1: 71}`, `saved …: state_encoder=4 policy=2 head_shape= (3, 32)` with `rnn.weight_ih_l0 (96,10)`, and `weights/nr1_ck14_hl.pth` is bit-identical to a fresh extraction from `ckpt.14.pth`.

**A9. "TRAIN[0:28] is byte-for-byte the same episodes … as the archived stableok" is over-claimed.** Raw JSON is *not* identical (TRAIN carries `info.split` and renumbered `episode_id`); it is identical after stripping those two keys, same order, same 28 ids. Likewise "content-identical to `train32_up3`, asserted by `build_final_datasets.py:85`" — that assert compares only `(scene_id, start_position)` tuples, nothing else.

**A10. `hl/bc_pretrain_hl.py:34`'s proposed new default is a data change, not a path fix.** C1 says the default should become `demos/hl_decisions_rule_markov_all.jsonl` "(the old file is in `_archive/data/`)". They are different corpora: `demos/hl_decisions_rule_markov_all.jsonl` = 2,882 rows, md5 `b4e88ee336981a8492ec73bb8b03fd5d`; `_archive/data/hl_decisions_rule_markov.jsonl` = 14,374 rows, md5 `8ac3a360d2cb3c3546be43b31074d3fe`.

**A11. Step 0's acceptance grep misses a file it patched.** The sed patches 8 f-string files including `diag/analyze_yield_geometry.py`, but `grep -n "^SD" data_tools/*.py diag/by_group.py diag/eval_to_tb.py` never inspects it.

**A12. `hl/check_train_health.py` never reads its `<train_log>` argument** (assigned at `:13`, unused thereafter). Step 5's health probe would print the identical ALERT with a nonexistent log path, so it is not the gate the report implies. (The output itself reproduces verbatim: `ALERT OPTIMIZATION_FAILURE succ 0.62->0.49 reward 25.2->13.7`.)

**A13. D3, `$R.safe_dis_min` row: "Schema default 1.5" is wrong.** The schema default is **1.0** (`habitat-lab/habitat/config/default_structured_configs.py:1465`, class `SocialNavReward` at `:1459`); 1.5 is the *yaml* value (`social_nav_hierarchical_overfit_v2.yaml:37`). The adjacent `collide_penalty` row ("5.0 yaml / 1.0 schema", schema at `:1531`) is correct.

**A14. E3's "the headline is the pre-registered 1.5M point" conflicts with the script it's derived from.** `_archive/scripts/_formal_run.sh:9-12` states the pre-registration as "the **3M endpoint is the primary policy**". The 1.5M framing is supported only by `docs/NR1_REPORT.md` ("主结果口径:1.5M 预注册点(既定短预算协议,非事后选点)"). Defensible, but the report should cite the doc and note the two records disagree.

**A15. E1 is right but overstated as a "CORRECTION".** `PIPELINE_REFERENCE.md:1187` already lists "The exact invocation of the `stableok` HL demo pass" under **Unresolved / not determined**. Also `RULE_EVALS=3` appears at `:935, :1035, :1053, :1065, :1106` too — the report cites only `:955`.

---

# B. MISSING

- **The Step-7 reference table was produced on `train36_v1`, not `EVAL`.** `_archive/scripts/_formal_run.sh:63-73, 81-85` and `_seed_runs.sh:44-52` use `data_path=…social_nav_episode_train36_v1.json.gz` (+ `holdout2` separately). The report flags non-comparability but never states the historical `data_path`, so "repeat the Step 7 command per tag" does **not** reproduce the table.
- **`build_final_datasets.py` writes TRAIN and EVAL at lines 72 and 78, *before* the verification block at 81-89.** Running it without restoring `train32_up3`/`holdout2` **overwrites the two production datasets and then crashes** — worse than "the asserts crash".
- **Three undocumented holdout artifacts.** `results/stats_nr1_{ck9,ck14,ck19}_holdout.json` (6 entries each; succ 0.8333 / coll 0.1667 all three; mean steps 798.33 / 655.50 / 730.83) with logs `runs/_eval_nr1_ck{9,14,19}_ho.txt`. `_formal_run.sh` only produced `u366_holdout`; the other three have no recorded command. The report's table lists only `u366_holdout` yet its Step-7 analysis command consumes `stats_nr1_ck14_holdout.json`.
- **Four scripts never classified as stale or clean** (all four are clean — no hardcoded `hrl_pipeline` paths): `diag/make_video_index.py`, `diag/_probe_summary.py`, `ll/verify_ll_demos.py`, `ll/reconstruct_es_center.py`. `diag/` has 10 files, not the 8 the report enumerates.
- `hrl_pipeline/videos/` already exists (containing `expert_wait_teacher_ep579/`); C3 proposes `hrl_pipeline/videos/<MODE>` without noting this.
- `SK=` in the Step-0 `DEX` prelude is defined and never used anywhere in the report.

---

# C. CONFIRMED (brief)

- **Section A**: every env-var line number and default in `hl/scripted_hl_eval.py` checked individually — `:33,:34,:35-37,:38,:39,:40-42,:43,:45-47,:48,:49,:56,:313,:340-341,:351,:373-421,:426-438,:439-442,:443-448,:449-453` all correct, including the "APPEND mode" and "hardcoded, not overridable" claims. `control_interval: 30` (`social_nav_hierarchical.yaml:121`) with `ctrl_freq 120 / ac_freq_ratio 4` = 30 Hz ⇒ "≈1 s per tick" ✓.
- **Section B**: pseudo-code matches `:64-147` exactly, including the 7-decision backoff trace and the `still` predicate. Ran `analyze_hl_decisions.py` on the archived corpus: `(obs,prev_action)-conditioned markov replay: disagree 0/849 (0.00%)`, `memoryless-legacy 226/849 (26.6%)`, `episodes=28`.
- Demo pass: `847.7143 / 0.0000 / 1.0000`, 849 rows, `stats_demo_pass.json` = 28 entries all keyed `|1` (so `RULE_EVALS=1` is proven).
- `_archive/logs_raw/_ppo_nr1.log` (3,077,863 lines): `:6` cuDNN line, `:114` `Number of params to train: 4356`, `:115` `critic_lr=0.03: 2 critic tensors at 0.03, 6 at 5e-05`, `Loaded BC-pretrained HL weights` count = **0**, first window `update: 10` → `social_nav_to_pos_success: 0.681`.
- `tb_nr1`: 367 points, `metrics/rollout_fail_frac` mean **0.1565**, last **0.2297**, last step 3,006,464.
- Every number in the Step 7(d) table recomputed from `results/stats_*.json` and cross-checked against `runs/_eval_nr1_*.txt` — all exact. The `eval_summary.py` group table reproduces character-for-character (I ran it).
- `TEACHER_CERT3.csv`: 28/4/4, sf `[2,3,20,25]`, flaky `[0,1,4,29]`. `dataset_provenance.py` live output: `scenes TRAIN=23 EVAL=25 shared=23`, `episodes identical in both : 30`.
- `by_group.py` `FileNotFoundError` reproduced live at `:24` (SD at `:22`); `eval_to_tb.py:28` glob matches 0 files today.
- Every C1/C3/C4 line number spot-checked and correct, including `certify_solvability.sh:12/15/19/20`, `certify_merge.py:33`, and the claim that `trace_<tag>.jsonl` exists nowhere (only `_archive/data/hl_dec_trace_*.jsonl`).
- Every D1–D4 source reference verified: `social_nav_neural_policy.py` `:121,:171-172,:182-183,:195-196,:208-212,:228,:234,:249-255`; `social_nav_hierarchical.yaml:78-90`; `overfit_v2.yaml:17,22,23,26,27,34,35,36,37,50,51,52,53,58`; reward schema `:1495,1496,1502,1503,1507,1508,1509,1512`; `critic_lr` default at baselines `default_structured_configs.py:308` + `ppo.py:53,130-146`; `environments.py:79-80`; `habitat_evaluator.py:500-503`; `habitat_env_factory.py:49`; `ppo_trainer.py:99-101`; `social_nav_skills.py:229,384,394`. `checkpoints_nr1/rollout_counts.json` exists and is keyed `teacher_class|train36_id`. Seed recipe `1.5e6 / 15` confirmed at `_seed_runs.sh:32`.