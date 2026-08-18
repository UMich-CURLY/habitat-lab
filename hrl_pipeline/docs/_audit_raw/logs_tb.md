# Log → TensorBoard coverage audit

## 0. Facts established (all verified, not inferred)

- `tb_view/*` and `tb_focus/*` are **symlinks**, not data (`ls -la` on both dirs). Real TB data = 29 dirs.
- PPO logs are **strictly a subset** of TB. Verified on `tb_nr1` vs `runs/_ppo_nr1.txt`: all 36 `Average window` keys exist as `metrics/<k>`, values match to 6 dp, and TB has 367 points vs 36 log lines (`log_interval=10`) plus 12 `learner/*` + 7 `perf/*` tags that never reach stdout.
- The eval logs' `Average episode <k>` block is reproducible from `results/stats_*.json`. I matched every `runs/*.txt` by (success, collide, num_steps) against every json in `results/`: **130 of 132 eval logs matched exactly**; only `_ep0video` and `_uncert3_video` (1- and 3-episode video renders) have no json.
- Log timestamps are container-local, **+4 h vs TB wall-clock**. All PPO log→tb-dir pairings below were confirmed by that offset plus `update:` count × `log_interval`.
- Total raw: 168 logs / 5.62 GB / 31.47 M lines. The shrink kept **5 931 lines**; residue classification shows the discarded 31.47 M lines are 99.9 % Magnum/RVO renderer spam (`MeshTools::compile()`, `Instantiating render asset ... incompatible light setup`, `In the humanoid`).

---

## 1. Checkpoint → log → tb dir

| # | checkpoint (`hrl_pipeline/weights/`) | training log (`_archive/logs_raw/`) | tb dir | verdict |
|---|---|---|---|---|
| 1 | `nr1_ck9_hl.pth` (u122≈1.0 M) | `_ppo_nr1.log` (546 MB) | `tb_nr1` steps 8192→3006464, 367 pts, 58 tags | **fully covered** |
| 2 | `nr1_ck14_hl.pth` ⭐ (u183≈1.5 M) | `_ppo_nr1.log` | `tb_nr1` | **fully covered** |
| 3 | `nr1_ck19_hl.pth` (u244≈2.0 M) | `_ppo_nr1.log` | `tb_nr1` | **fully covered** |
| 4 | `nr1_final_hl.pth` (u366=3.0 M) | `_ppo_nr1.log` | `tb_nr1` | **fully covered** |
| 5 | `nr1_s200_hl.pth` | `_ppo_nr1_s200.log` (273 MB) | `tb_nr1_s200` →1507328, 184 pts | **fully covered** |
| 6 | `nr1_s300_hl.pth` | `_ppo_nr1_s300.log` (273 MB) | `tb_nr1_s300` →1507328, 184 pts | **fully covered** |
| 7 | `mechC_hl.pth` | `_ppo_mechC.log` (273 MB) | `tb_mechC` →1507328, 61 tags (incl. 6 `learner/agent_0_*dapg*`) | **fully covered** |
| 8 | `mechB_hl.pth` | **lost** — `_ppo_mechB.log` was overwritten on 08-14 13:56 by a failed retry (`FileExistsError`) | `tb_mechB` →1040384, 127 pts (run died at u127 of 183; driver printed `MECH_B_EXIT=1`) | **TB is the only record**; the log that exists documents the *failed retry*, not the run |
| 9 | `g3_final_hl.pth` | `_ppo_g3.log` (275 MB) | `tb_g3` →1507328, 184 pts | **fully covered** |
| 10 | `b2_final_hl.pth` | `_ppo_b2.log` (119 MB) | `tb_b2` →651264, 318 pts | **fully covered** |
| 11 | `bc_train36_hl.pth` | **none** — `_hl_train_stage1.sh:30-34` pipes `bc_pretrain_hl.py` through `grep` to the *driver's* stdout, and that driver run was never redirected to a file | none | **BC loss/acc curve lost entirely** (no log, no TB) |
| 12 | `bc_stable_hl.pth` (BC start of every HL result) | **none** (`docs/CLEANUP_PLAN.md:56` already says 训练日志已丢失) | none | **BC curve lost entirely** |
| 13 | `ll_yield_bc.pth` | **none** — no log contains `saved -> .../ll_yield_bc.pth` (grepped all 168) | none | **lost**; the 3 surviving LL-BC logs trained `_tmp_bc_v2/_tmp_bc_v2_rep/_tmp_bc_dagger1` |
| 14 | `ll_yield_dagger1.pth` | `_bc_ll_dagger1_full.log` (512 B; renamed from `_tmp_bc_dagger1.pth` by `ll/ll_mse_units.py:122-123`) | `tb_curves/ll_yield_dagger1` — `train/train_mse`, `train/val_mse`, 9 pts, steps 0→399 | **fully covered** |
| 15 | `ll_yield_es_v3.pth` ⭐ | `_es_v3.log` (2.1 KB) | `tb_curves/ll_yield_es_v3` — 7 tags (`es/best_fitness`, `best_success`, `best_collide`, `pop_min/mean/max`, `sigma`), 11 pts | **covered except 3 header lines** (see §2.6) |
| 16 | `flat_bc.pth` | **none** (`_archive/scripts/bc_flat.py:58`; July-era) | `tb_flat_e2e` holds only its *eval* (30 `eval_metrics/*`, 1 point) | **BC curve lost**; eval covered |

TB dirs with **no** surviving log at all (July-era, predate log-keeping): `tb_bcft_ep70`, `tb_cleandev`, `tb_cleandev13`, `tb_multiscene_v2`, `tb_multiscene_d20`, `tb_social_nav_overfit`.
Logs with **no** TB dir (dir deleted): `_ppo_mechA.log` (59 MB, `tb_mechA` gone), `_ppo_dryrun.log` (45 MB, `_dapg_dryrun.sh:18` does `rm -rf tb_dryrun`).
`tb_overfit_ep70_v2` (807 event files) and `tb_social_nav_hrl` (246) are the **default** `tensorboard_dir` — every eval that didn't override it dumped `eval_metrics/*` there, all unnamed and colliding at step 0. Unusable as coverage; treat as scratch.

---

## 2. What still needs publishing to TensorBoard

### 2.1 `stats_var_*.json` — 7 LL-variance evals, missing from `tb_eval_summary`
`tb_eval_summary` was written 08-17 04:56; the variance evals landed 05:12–08:21. Missing runs: `var__es_v3`, `var__center_r2`, `var__seed1`, `var__seed2`, `var__tie_g7_c14`, `var__tie_g8_c0`, `var__tie_g10_c10`.
**Fix:** no new regex — just re-run `eval_to_tb.py`. But first fix `diag/eval_to_tb.py:28`, which is stale after the reorg:
```python
SD = "/habitat-lab/hrl_pipeline/results"   # was "/habitat-lab/hrl_pipeline"
```
Without this the glob at `eval_to_tb.py:157-159` returns zero files and the rebuild wipes `tb_eval_summary`.

### 2.2 8 eval jsons the glob never matched
`cert3_m0805_{always_go,rule_markov,rule_wait,rule_waitgo,rule_yield}.json`, `cert3_manual_rule_markov.json`, `recheck_cd17fam_yield.json`, `recheck_m0805_markov.json` — all verified eval-stats-shaped.
**Fix:** widen `eval_to_tb.py:157-159` and the prefix strip at `run_name()`:
```python
set(glob.glob(f"{SD}/stats_*.json")) | set(glob.glob(f"{SD}/bd_*.json"))
| set(glob.glob(f"{SD}/cert_*.json")) | set(glob.glob(f"{SD}/cert3_*.json"))
| set(glob.glob(f"{SD}/recheck_*.json"))
# and in run_name():  for p in ("stats_", "bd_", "cert3_", "cert_", "recheck_"):
```
(`cert3_` must precede `cert_` in that tuple.)

### 2.3 ES runs missing from `tb_curves` — `logs_to_tb.py` handles these as-is
`_es_v2.log`, `_es_seed1.log`, `_es_seed2.log`. `RE_ES` (logs_to_tb.py:24-27) already parses every `[es] gen…` line in them; `_es_seed1/2` legitimately produce `best=-1.000 {}` (the seeded arms all failed — that *is* the variance result and is worth a curve).
```
python hrl_pipeline/diag/logs_to_tb.py \
  ll_yield_es_v2=..._es_v2.log ll_es_seed1=..._es_seed1.log ll_es_seed2=..._es_seed2.log
```

### 2.4 LL-BC MSE curves missing — `logs_to_tb.py` handles these as-is
`_bc_ll_v2_full.log`, `_bc_ll_v2_repeat.log` (`epoch N train_mse= val_mse=` × 9). `RE_EPOCH`+`RE_KV` (lines 28-29) already match. The repeat pair is the determinism evidence (byte-identical curves).

### 2.5 HL-BC accuracy curve — **needs no new regex, but needs to be run**
`_chain_g5.log` is the *only* surviving HL BC curve anywhere:
`epoch 0 loss/dec=1.0985 acc=0.458` … `epoch 999 loss/dec=0.0360 acc=0.988` (11 points, `bc_train36_wait_hl.pth`, 30 episodes / 915 decisions).
`RE_EPOCH`+`RE_KV` already parse it → `train/loss_dec`, `train/acc`. Publish as e.g. `bc_hl_wait_train36`. Since #11/#12's curves are gone, this is the only HL-BC learning curve the project has.

### 2.6 Things `logs_to_tb.py` **cannot** parse — new rules needed

**(a) ES center-reconstruction curve** — `_center_recon.log`, format not recognised, and it is one of the 5 logs the shrink dropped entirely (no `runs/*.txt`):
```
  gen 1: 16 cands, top4 fitness [0.7227, 0.7227, 0.6447, 0.6113], center moved 0.2165
```
```python
RE_RECON = re.compile(
    r"^\s*gen\s*(\d+):\s*(\d+) cands, top4 fitness \[([\d.,\s-]+)\], center moved ([\d.]+)")
# -> recon/pop_size, recon/top1..top4 (or recon/top4_mean), recon/center_moved  @ step=gen
```

**(b) ES run header / hyperparameters** — present in all 4 `_es_*.log`, dropped by both the shrink and the parser:
```
[es] torch seed 1
[es] perturbing 2 tensors, 258 params; init=bc sigma=0.03
[es] workdir=/habitat-lab/hrl_pipeline/es_ll_jl5ncpsk
[es] done. best fitness=0.930 {'success': 1.0, 'collide': 0.0} -> .../ll_yield_es_v3.pth
```
```python
RE_ES_CFG  = re.compile(r"\[es\] perturbing (\d+) tensors, (\d+) params; init=(\S+) sigma=([\d.]+)")
RE_ES_SEED = re.compile(r"\[es\] torch seed (\d+)")
RE_ES_DONE = re.compile(r"\[es\] done\. best fitness=(-?[\d.]+) (\{[^}]*\}) -> (\S+)")
# -> hparams/n_params, hparams/sigma0, hparams/seed as scalars at step 0
#    + w.add_text("config", …)  and  w.add_text("output_checkpoint", …)
```

**(c) LL-BC dataset size** — first line of every `_bc_ll_*.log`, dropped:
```
demos: 30950  act lin[min,max]=[-0.10,0.00] ang=[-1.00,1.00]
```
```python
RE_LLBC = re.compile(r"^demos:\s*(\d+)\s+act lin\[min,max\]=\[(-?[\d.]+),(-?[\d.]+)\] ang=\[(-?[\d.]+),(-?[\d.]+)\]")
# -> data/n_demos, data/lin_min, data/lin_max, data/ang_min, data/ang_max @ step 0
```

**(d) HL-BC dataset line** — `_chain_g5.log`, dropped:
```
input: approach=True prev_action=True dim=10
episodes=30 decisions=915 action_dist={2: 647, 0: 195, 1: 73}
```
```python
RE_HLBC = re.compile(r"^episodes=(\d+) decisions=(\d+) action_dist=(\{[^}]*\})")
# -> data/n_episodes, data/n_decisions, data/action_count_{k} (ast.literal_eval the dict)
```

**(e) PPO config echo** — the exact hyperparameter override list for every kept HL checkpoint exists **only** as `+ python -u -m habitat_baselines.run …` lines in 12 driver logs (`_mech_check_{run,B,BA,AB,B2,C}.log`, `_formal_run_driver.log`, `_formal_evals.log`, `_seed_runs_driver.log`, `_chain_bd.log`, `_chain_g5.log`, `_g2_chain.log`). The shrink's KEEP regex (`_archive/scripts/_reorganize.sh:55`) does not match `+ python`, so 4 of those logs have no `runs/*.txt` at all. This is not a scalar — publish as text into the matching tb dir:
```python
RE_CMD = re.compile(r"^\+ python -u -m habitat_baselines\.run (.*)$")
# tb dir = re.search(r"tensorboard_dir=(\S+)", cmd)  ->  w.add_text("config/cli", cmd, 0)
```
Same rule recovers configs of *crashed* runs from `Error executing job with overrides: [...]` (already in `runs/_ppo_mechB.txt`, `runs/_ppo_g3b.txt`).

**(f) Policy/skill architecture echo** — every eval log prints, and the shrink drops:
```
Skills: {0: ClearCorridorYieldSkill(), 1: WaitSkill(), 2: GoToGoalSkill()}
Name to idx: {'backoff': 0, 'wait': 1, 'go_to_goal': 2}
Number of params to train: N / Agent number of parameters: N
Wrote per-episode eval stats to /habitat-lab/hrl_pipeline/stats_nr1_u366.json
```
The last line is the only machine-readable **log → stats-json provenance link**; my value-based matching had to substitute for it. Worth capturing as `w.add_text("source_log", …)` alongside the existing `w.add_text("source", fname)` at `eval_to_tb.py:172`.

**(g) `by_group.py` paired-step tables** in `_mech_check_B2.log` / `_mech_check_C.log` — already duplicated to `_archive/data/_by_group_{mech,nr1,seeds}.txt`, so **no action needed**; just don't delete those `.txt`.

### 2.7 Two orphan TB-less PPO runs
`_ppo_mechA.log` (3 updates before it died) and `_ppo_dryrun.log` (31 updates, the DAPG dry run) have no TB dir. `runs/_ppo_mechA.txt` / `runs/_ppo_dryrun.txt` hold every metric line. If you want them in TB, a small `Average window size: N k: v  k: v` parser would do it — `logs_to_tb.py` has no rule for that shape today:
```python
RE_WINDOW = re.compile(r"Average window size:\s*(\d+)\s+(.*)$")
RE_WKV    = re.compile(r"([A-Za-z_][A-Za-z_0-9.]*):\s*(-?[\d.]+)")
RE_UPDATE = re.compile(r"update:\s*(\d+)\s+fps:\s*([\d.]+)")
# step = update * num_envs * num_steps ; tags metrics/<k> to match the trainer's own naming
```
Low value — both are dead ends — but it is the only way to get them into TB at all.

---

## 3. Raw logs deletable outright

**142 of 168 logs = 5.51 GB of the 5.62 GB.** Every one is either (a) a PPO run whose `tb_<name>/` holds the same metrics at 10× resolution, or (b) an eval whose per-episode `results/*.json` reproduces its aggregate exactly.

- **17 PPO logs → TB verified:** `_ppo_b1 _ppo_b2 _ppo_g1 _ppo_g2 _ppo_g3 _ppo_g4 _ppo_g5 _ppo_mechC _ppo_nr1 _ppo_nr1_s200 _ppo_nr1_s300 _ppo_prod1 _ppo_train36_run1_destroyed (→tb_train36) _ppo_train36b _ppo_train36c _ppo_train36d _ppo_train36e`
- **2 already in `tb_curves`:** `_es_v3 _bc_ll_dagger1_full`
- **123 eval logs → `results/*.json` value-matched:** the `_cert_* _cert3_* _recheck_* _eval_nr1_* _var_* _ll* _trace_* _teacher_* _holdout2_* _l0_* _esv2_* _esv3_* _bd_* _prod1_* _ppoD/E_*` families, plus `_b2_final_eval _g3_final_eval _bc_train36_eval _bc_cd17 _bc_ms4 _learned_g3 _expertvid36 _manual_markov_x3 _train36bc_demos _smoke_markov _regress_rule_yield _wait579 _waitB579 _waitC579 _ms4_rule_markov _parked_* _ep0x3* _ep70_rule_markov _solid6_rule_markov _cd17_*`
- `_cert_dev20_parked_rule_waitgo.log` (3.6 MB): crashed, produced no json; its entire signal (2 lines) is already in `runs/_cert_dev20_parked_rule_waitgo.txt`.

⚠️ **Delete the eval logs only after §2.1+§2.2 are applied and `eval_to_tb.py` re-run** — right now 15 eval results exist *only* as json, published nowhere.

### Must NOT delete (25 logs, 106 MB) until §2 is done
| log | log-only content |
|---|---|
| `_center_recon.log` | recon fitness/center-drift curve; **no `runs/*.txt` either** |
| `_mech_check_{run,B,BA,AB}.log` | full PPO CLI override echo; **no `runs/*.txt`** (4 files) |
| `_mech_check_{B2,C}.log` | CLI echo + `by_group` tables + `_extract_hl_map` shapes |
| `_chain_g5.log` | only surviving HL-BC loss/acc curve + `probe_wait_prob` output + G5 CLI |
| `_chain_bd.log`, `_g2_chain.log`, `_formal_run_driver.log`, `_formal_evals.log`, `_seed_runs_driver.log`, `_ll_variance_driver.log` | CLI echoes + `_extract_hl_map.py` provenance lines |
| `_es_v2.log`, `_es_seed1.log`, `_es_seed2.log` | ES generations + seed/sigma header |
| `_bc_ll_v2_full.log`, `_bc_ll_v2_repeat.log` | LL-BC MSE curves + `demos:` counts |
| `_ppo_mechA.log`, `_ppo_dryrun.log` | metrics with no TB dir |
| `_ppo_mechB.log`, `_ppo_g3b.log` | `FileExistsError` + override list (already mirrored in `runs/*.txt` — raw itself is droppable) |
| `_ep0video.log`, `_uncert3_video.log` | only evals with no `stats_*.json` |

After §2.6(a–e) are implemented and run, all 25 collapse to **6 keepers** (`_ppo_mechA`, `_ppo_dryrun`, `_ep0video`, `_uncert3_video`, and the two mirrored crash logs), ~106 MB → ~62 MB, and `_archive/logs_raw/` can go to zero if §2.7 is also done.

---

## 4. Stale paths found along the way

- `diag/eval_to_tb.py:28` — `SD = "/habitat-lab/hrl_pipeline"` → must be `.../hrl_pipeline/results`. **This one is load-bearing: re-running the script as-is deletes `tb_eval_summary` and republishes nothing** (`main()` does `shutil.rmtree(OUT)` before the glob).
- `diag/eval_to_tb.py:15` and `diag/logs_to_tb.py:13` — usage strings say `hrl_pipeline/<x>.py`, now `hrl_pipeline/diag/<x>.py`.
- `ll/es_ll_finetune.py:32-33` — `HARNESS = "/habitat-lab/hrl_pipeline/scripted_hl_eval.py"` (now `hl/`), `BASE = "/habitat-lab/hrl_pipeline/ll_yield_bc.pth"` (now `weights/`); `:79` `--out` default likewise.
- `ll/bc_ll_yield.py:18-19`, `hl/bc_pretrain_hl.py:34,36`, `ll/ll_mse_units.py:122-123` — default in/out paths still at the flat root (`weights/`, `demos/`).
- `hl/scripted_hl_eval.py:46,48` — `RULE_LOG` / `VIDEO_DIR` defaults at flat root.
- `data_tools/certify_solvability.sh:19` — `python hrl_pipeline/scripted_hl_eval.py` → `hrl_pipeline/hl/`.