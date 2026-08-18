## CORRECTIONS

### WRONG — factual

**1. `archive/logs/es_ll_epqcdrqq` DOES have `stats_*.json`.** Report: "`.pth`-only with **no `stats_*.json`** so `reconstruct_es_center.py` cannot use it." Actual: 99 files = 33 `.pth` + 33 `log_*.txt` + **33 `stats_*.json`** (`stats_center.json`, `stats_g1c0.json` …). `ll/reconstruct_es_center.py:56-58` needs exactly `g{g}_c{i}.pth` + `stats_g{g}c{i}.json` — both present. This is the same "irreplaceable" property the report used to justify keeping the other two workdirs, applied inconsistently to justify a 32 MB delete.

**2. The `click30.html` grep counts are fabricated.** Report: `"pts"` ×0, `"clicks"` ×0, `"png"` ×0. Actual in `_archive/dirs/manual/click30.html`: **`pts` ×160, `clicks` ×4, `png` ×156**, `robot_start` ×3. The conclusion (clicks not persisted) happens to be right, but for a reason the report never states: click state lives in `localStorage` under `KEY = "manual_clicks_state_v1"`. The `pts`/`png` hits are base64 noise plus `"img": "data:image/png;base64,`; the `clicks` hits are the Export button and `a.download = "clicks.json"`.

**3. "`manual/` holds no unique human input" is false.** `_archive/dirs/manual/_live_r1.req.json` carries `"pts": [[-10.61,-3.336],[-7.102,-1.518],[-7.817,-1.697],[-9.97,-2.054]]` for `src_idx 112` — a hand-clicked 4-point set that does **not** match the shipped `results/0805.json` entry for idx 112 (`[[-10.584,-3.274],[-7.181,-1.684],[-7.566,-1.765],[-10.312,-0.938]]`, which is `_live_r2`). r3/r4/r5 do match. So one discarded click iteration exists nowhere else. The report misclassified `_live_r*.*` as "eval logs / probes"; the `.req.json` are click payloads. Also unaddressed: `manual/pool.json.gz` (21,499 B, md5 `d85a3bfce7…`) has **no md5 twin anywhere under `data/`**, and `singleep_done.txt` is never mentioned.

**4. The `tb/` CSVs are NOT regenerable, and not by `logs_to_tb.py`.** Two errors:
- `diag/logs_to_tb.py:1-14` parses *stdout training logs* and writes to `/habitat-lab/tb_curves` — it never exports CSVs from tfevents.
- The source run dirs for most CSVs are **gone**: `tb_social_nav_skills_rvo` (22 CSVs), `tb_curriculum_stage0`, `tb_curriculum_stage1` do not exist anywhere (`find /habitat-lab -name … -type d` → empty). `tb_social_nav_hrl` exists but contains none of `1782239228/1782239296/1782239358`; `tb_social_nav_overfit` holds 3 August event files, no `run1`…`run8`. So ~35 of the 37 CSVs are the *only* record of their runs, exactly like the 6 tfevents.

**5. `tb/` file breakdown wrong.** Report: "22 exported CSVs + `manifest.csv` + `peaks_report.txt` + 6 tfevents" = 30, while also claiming 46 files. Actual 46 = **37 CSVs** (22 `skills_rvo` + 3 `hrl` + 12 `ckpt0708__*`) + `manifest.csv` + `peaks_report.txt` + `ckpt0708_summary.txt` (omitted entirely) + 6 tfevents.

**6. `CLEANUP_PLAN.md:114-120` does not cover the 71 July eval JSONs.** Lines 114-116 are the C2 *checkpoint* table (`_tmp_bc_v2.pth`, `ll_yield_*`), 117 is blank, 118-120 is the C3 header + `hl_decisions_learned.jsonl`/`trace_*`. The "~40 早期评测" note is at **125-126**.

**7. `CLEANUP_PLAN.md:117` does not mark the unfiltered LL corpora.** Line 117 is blank. The real text is **121-122** and it names `ll_train_r1.jsonl`, `ll_demos.jsonl`, `ll_demos_v2.jsonl` — **not** `ll_all.jsonl` / `ll_dagger.jsonl`.

**8. `ll_dagger_r1.jsonl` is not the filtered successor of `ll_dagger.jsonl`.** `archive/logs/ll_dagger.jsonl` = 11,183 lines (Jul 18); `demos/ll_dagger_r1.jsonl` = 15,460 lines (Aug 16). Larger and a month later — a different run, not a `_clean` version. The report's stated redundancy for that file is unsupported. (`ll_all.jsonl` 26,535 → `ll_demos_v2_clean.jsonl` 15,490 is at least directionally plausible; still unverified.)

**9. "six tiny label files (~40 KB total)" — actual 7,665 B (~7.5 KB), and only five are distinct.** `manual_audit.csv` and `manual_audit_fixed.csv` are **byte-identical** (both md5 `d3e6b862989970e5905295982df56e09`, 1171 B).

**10. `results/POOL_v1.csv` is 95 rows, not 96** (96 lines incl. header). It also does **not** carry `AGsu`/`RYsu`/`verdict` — its columns are `pool_id,label,source,source_idx,click_id,note`, with a 7-label taxonomy (`sweet/challenge/dead/fail_B_wait/fail_C_dither/trivial_eff/trivial_zero`) vs `pool_verdicts.jsonl`'s 3 (`SWEET/trivial/unsolv`). So "verdicts … re-derived in `POOL_v1.csv`" is overstated: the AGsu/RYsu counts survive nowhere in `results/`.

**11. `outputs/` date span wrong.** Report: "`2026-07-02` … `2026-08-16`". Actual: **`2026-01-30` … `2026-08-17`**, 47 date dirs.

**12. `outputs/` byte figures.** 850 `run.log` (not 963) totalling **7.76 GB**, not 7.85. All 3,002 non-`run.log` files total **32.6 MB actual bytes** — the "44 MB of `.hydra`" is `du` block-rounding on 2,889 tiny files, not real payload. (963 run dirs / 963 `.hydra` ✓, `outputs/2026-08-15/06-52-57/run.log` = 529,137,664 B ✓.)

**13. Total is 1066 MB, not 1072.** Per-dir sizes are otherwise correct. The keep/delete split is also off: the 3 "keep" dirs are 531 MB (539 with `tb/`), leaving 527 MB deletable, not 535.

**14. `map_check` mention set incomplete.** Report: "only `docs_old/INDEX.md:30` and `_reorganize.sh` mention it." **`docs/CLEANUP_PLAN.md:137`** also names it. (No `.py` producer ✓.)

**15. Off-by-N line citations.** `PIPELINE_REFERENCE.md` cites `hrl_pipeline/0805.json` at **:911**, not :913 (:913 is `--review-dir hrl_pipeline/manual/review`). `make_click_page.py` returns `{}` at **:37**, not :36. `_es_seed1/2` dirs are 322 files = **161** `.pth` + **161** `.txt` (center included in both), not "160 + 160 + center".

**16. Script paths are ambiguous/misleading.** `make_click_page.py`, `build_manual_episodes.py`, `render_scene_*.py`, `scene_layout.py`, `render_topdown.py` are at **`/habitat-lab/scripts/`** (repo root) — not `hrl_pipeline/_archive/scripts/`, where the report's bare `scripts/` prefix reads as pointing, given it prefixes `_archive/` elsewhere. `_archive/scripts/` contains 23 files, none of them these.

### COMMANDS THAT WOULD FAIL TODAY (missed by the report)

- **The recovered round-2 invocation is itself unrunnable.** `outputs/2026-08-16/08-32-53/.hydra/overrides.yaml` line 10 → `/habitat-lab/hrl_pipeline/es_ll_jl5ncpsk/stats_center.json` (now `_archive/dirs/`), line 13 → `/habitat-lab/hrl_pipeline/ll_yield_dagger1.pth`, line 15 → `/habitat-lab/hrl_pipeline/nr1_ck14_hl.pth` (both now `weights/`). The report calls these dirs load-bearing for reproducibility but never checks that all three paths in them are dead.
- `_archive/scripts/_repro2.sh:18` also hardcodes `/habitat-lab/hrl_pipeline/ll_yield_dagger1.pth` and `/habitat-lab/hrl_pipeline/nr1_ck14_hl.pth`. Report flags only `:6` and `:21` — fixing those two still fails.
- `_repro2.sh` regenerates only **16 of 28** `_repro_v3` files (`log_x_{win1-5,ctr1-3}` + matching stats). The other 12 (`log_center`, `log_g4c11`, `log_dup_A/B`, `log_r2_center`, `log_r2_g4c11`, `stats_repro_*`, `stats_dup_*`, `stats_r2_*`) have no surviving producer in `_archive/scripts/`.
- `PIPELINE_REFERENCE.md:902-903` (`hrl_pipeline/stats_ag.json`, `stats_ry.json`, `hrl_pipeline/doorcand_v3_audit.csv`), `:913` (`--review-dir hrl_pipeline/manual/review`), `:915` (`bash hrl_pipeline/certify_solvability.sh` → now `data_tools/`) are all equally stale; the report lists only 907/913.
- `docs/LL_TRAINING_CURVES.md:738-739` points at `/habitat-lab/hrl_pipeline/es_ll_{zji8yz5q,jl5ncpsk}` (moved), `:734-735` at `hrl_pipeline/_es_v2.log`/`_es_v3.log` (now `_archive/logs_raw/`), `:744` names `ll_yield_bc_v2.pth` which does not exist (`weights/` has `ll_yield_bc.pth`), `:740-741` names `hrl_pipeline/ll_demos_v2_clean.jsonl` (now `demos/`). Given the report's own topic, these should have been caught.

### MISSING

- **`CLEANUP_PLAN.md:137` directly contradicts the report**: it lumps `es_ll_*/` into 删 at "~9 MB". Actual `es_ll_*` = 558 MB (60× off), and the report recommends keeping 530 MB of it. The conflict with the project's own written plan is never surfaced.
- `manual/probes/` (34 MB, 155 files = 77 `.log` + 71 `.json` + 7 `.json.gz`) is dismissed unverified. The 7 `.json.gz` probe datasets have **no md5 twin in `data/`**.
- No check that the 6 `tb/` tfevents are actually loadable before recommending keep (they are — 47–48 scalar tags each, 23–245 points).

### CONFIRMED

22 dirs; all per-dir sizes as listed. `es_ll_jl5ncpsk` 339 MB/483 files and `es_ll_zji8yz5q` 191 MB/483, each 161×{`.pth`, `log_gNcN.txt`, `stats_gNcN.json`}; `_es_v3.log`/`_es_v2.log` have no seed line, `_es_seed1/2.log` do (`[es] torch seed 1/2`) and are all-`-1.000` with 0 `.json`. `weights/ll_yield_es_v3.pth` `net.4.{weight,bias}` **is** `torch.equal` to `es_ll_jl5ncpsk/g4_c11.pth` (and not to `g3_c9.pth`); `ll_yield_es_v2.pth` exists nowhere in the repo — both keeps are justified. The 6 tfevents are unique repo-wide. `video_manual_uncert` ≡ `video_rule_markov` (3 md5 matches; `_verify_and_mine.sh:31` is the `cp`). `videos/expert_wait_teacher_ep579` holds the 3 `reward=0.01` mp4s (`_reorganize.sh:64`). `review/`↔`clicks.json` 17:17 and `review0805/`↔`0805.json` 30:30, ids matching modulo an `NN_` prefix; both built `.json.gz` present. `doorcand_v3_audit.csv` 78 rows, column-swap claim correct. `expert_videos_train36` 38 files, `INDEX.md` "31/36, collisions 0", both regen inputs present. `archive/` 193 files / stats 71 / weights 7 (`maphl2_hl.pth` 30 MB) / frames 12 PNG / `multiscene_d20_train.log` 53 MB. `figures/` 5, `__pycache__/` 9 mixed cp38/39, `scene_rgb` 58 / `scene_menu` 58 / `scene_tiled` 1 with differing md5s. Stale paths confirmed at `hrl_pipeline/scene_tools/make_click_page.py:32`, `hrl_pipeline/scene_tools/render_scene_menu.py:64`, `hl/scripted_hl_eval.py:48`, `ll/es_ll_finetune.py:105`, `_ll_variance.sh:19`, `PIPELINE_REFERENCE.md:422`.