# `_archive/dirs/` triage — 22 directories, 1072 MB total

**Bottom line: 18 dirs / ~535 MB are safely deletable. 3 dirs / ~537 MB must be kept (2 ES workdirs + 5 orphan scripts), plus `tb/` (6.7 MB of unique tfevents).**

`outputs/` is **not** under `_archive/dirs/` — it is `/home/xinyuan/habicrowd/Simulator/habitat-lab/outputs` (7.9 GB); judged separately at the end.

---

## IRREPLACEABLE — keep

### `es_ll_jl5ncpsk/` — 339 MB, 483 files (161 × {`g{g}_c{i}.pth`, `log_*.txt`, `stats_*.json`})
ES round-2 workdir, created by `hrl_pipeline/ll/es_ll_finetune.py:105` (`tempfile.mkdtemp(prefix="es_ll_", dir="/habitat-lab/hrl_pipeline")`).
**Not regenerable.** `_archive/logs_raw/_es_v3.log:1` has no `[es] torch seed` line — the run predates `--seed` being passed, so perturbations came from an unseeded global RNG (`es_ll_finetune.py:93-97` comment says exactly this). Re-running produces a different landscape.
It is the sole backing evidence for `docs/LL_TRAINING_CURVES.md` §2.1–2.2 (2254 episode-runs, "log ↔ recomputed fitness exact match, 0 mismatches") and the shipped `weights/ll_yield_es_v3.pth` is byte-identical to `g4_c11.pth`'s last layer (`LL_TRAINING_CURVES.md:745`). It is also the only input `hrl_pipeline/ll/reconstruct_es_center.py` can run on (it needs `g{g}_c{i}.pth` + `stats_g{g}c{i}.json` pairs). **Keep.**

### `es_ll_zji8yz5q/` — 191 MB, 483 files
ES round-1 workdir (`_archive/logs_raw/_es_v2.log:1`), likewise unseeded.
**Extra reason to keep:** `ll_yield_es_v2.pth` **no longer exists anywhere** — I searched host + container (`find /habitat-lab -name "ll_yield*"` returns only `ll_yield_bc.pth`, `ll_yield_dagger1.pth`, `ll_yield_es_v3.pth`). Per `LL_TRAINING_CURVES.md:744`, `es_ll_zji8yz5q/g3_c9.pth` is now the **only surviving copy** of the round-1 ES result. **Keep.**

### `datasets/` — 1.2 MB, 14 files
Contains 5 Python analysis scripts that exist **nowhere else in the repo** (`find . -name geometry_audit.py` → only this path):
`analyze_geometry.py`, `analyze_refuge.py`, `analyze_backoff_target.py`, `geometry_addendum.py`, `geometry_audit.py`, plus their outputs (`big_dataset_geometry.json`, `geometry.csv`, `summary.txt`).
Source code, not output. **Keep** (or move the 5 `.py` into `data_tools/` and delete the rest, ~1.1 MB).

### `tb/` — 6.7 MB, 46 files
22 exported CSVs + `manifest.csv` + `peaks_report.txt` (regenerable via `diag/logs_to_tb.py`), **plus 6 raw `events.out.tfevents.*` in `ctrl_scalar/ flat_ms4/ joint_full/ joint_ra1/ maphl/ maphl2/`**. Those six run names do not exist under any root-level `tb_*` dir — they are the only record of the July flat/joint/map-HL experiments. **Keep** (6.7 MB is nothing); at minimum keep the 6 subdirs.

---

## REGENERABLE — delete

### `manual/` — 121 MB, 258 files → **DELETE** (verified redundant)
This was the main question. **It holds no unique human input.**

| what | verdict |
|---|---|
| `click30.html` (32 MB) | Output of `hrl_pipeline/scene_tools/make_click_page.py`. I grepped it: `"pts"` ×0, `"clicks"` ×0, `"png"` ×0, `robot_start` ×3 (all in the JS at `make_click_page.py:142+`). The payload is only base64 renders + audit verdicts (`make_click_page.py:110-133`). **The human's clicks were never stored in it** — the Export button writes them out (`make_click_page.py:324-325`). Regenerable. |
| `click.html` (2 MB) | Same, v3 vintage. |
| `review/` (17 PNG) | Exactly 1:1 with the 17 entries of `results/clicks.json`. Output of `build_manual_episodes.py --review-dir`. |
| `review0805/` (30 PNG) | Exactly 1:1 with the 30 entries of `results/0805.json` (I diffed the id lists — perfect match). Same generator. |
| `probes/` (34 MB), `_live_r1..r5.*`, `vid_*.mp4`, `frames_*.png`, `stats_*.json`, `_gen30.log` | Eval logs / rollout videos / live-build probes. |

**The irreplaceable hand-clicked coordinates are already safe** in
`/home/xinyuan/habicrowd/Simulator/habitat-lab/hrl_pipeline/results/clicks.json` (17 doors, `source_dataset: doorcand_v3`) and
`/home/xinyuan/habicrowd/Simulator/habitat-lab/hrl_pipeline/results/0805.json` (30 doors, `source_dataset: doorcand_v4`),
and the built products `data/social_nav_episode_manual.json.gz` + `data/social_nav_episode_manual0805.json.gz` both exist. Regenerate the whole dir with `PIPELINE_REFERENCE.md:905-913` (`make_click_page.py … --half 6.0`, then `build_manual_episodes.py results/0805.json … --review-dir <dir>`).

**One caveat before you `rm`:** six tiny label files (~40 KB total) are *not* byte-identical to anything in `results/` —
`manual_audit.csv`, `manual_audit_fixed.csv`, `audit78_fixed.csv`, `pool_verdicts.jsonl`, `pool_done.txt`, `singleep_ry.jsonl`.
Their content is superseded (`audit78_fixed.csv` = same 78 rows as `results/doorcand_v3_audit.csv` with `click_id`/`note`/`dt_arrival` swapped in for `yield_det`/`hgoal_clr`; `pool_verdicts.jsonl` verdicts are mechanically derived from AGsu/RYsu and re-derived in `results/POOL_v1.csv`, 96 rows, which carries `click_id` + `note`). Cheapest safe move: `cp` those six into `results/` first, then delete the 121 MB.

### `archive/` — 137 MB, 193 files → **DELETE**
Nested older archive: `frames/` (12 PNG strips), `logs/` (incl. a 53 MB `multiscene_d20_train.log` and a **third** ES workdir `es_ll_epqcdrqq`, 32 MB, `.pth`-only with **no `stats_*.json`** so `reconstruct_es_center.py` cannot use it), `stats/` (71 superseded July eval JSONs, all named in `CLEANUP_PLAN.md:114-120` as deletable), `weights/` (7 July HL checkpoints — `bc_multiscene*`, `maphl2_hl.pth` 30 MB, `ra1_hl.pth`, `flat_es.pth`; all superseded by `weights/nr1_*`/`b2_final`/`g3_final`).
`logs/ll_all.jsonl` + `ll_dagger.jsonl` are the **unfiltered** LL corpora; the `_clean` successors ship in `demos/ll_demos_v2_clean.jsonl` and `demos/ll_dagger_r1.jsonl`. `CLEANUP_PLAN.md:117` explicitly marks the unfiltered ones for deletion.

### `expert_videos_train36/` — 90 MB, 38 files → **DELETE**
36 renamed mp4s + `INDEX.md` + `index.csv`, produced by `scripted_hl_eval.py` with `RULE_VIDEO=1 RULE_VIDEO_DIR=hrl_pipeline/expert_videos_train36` on `train36_v1` (see `runs/_expertvid36.txt`), then renamed/indexed by `diag/make_video_index.py` (which renames in place via `os.replace`, `PIPELINE_REFERENCE.md:422`).
Both regeneration inputs survive: `results/stats_expertvid36.json` and `_archive/data/hl_decisions_expertvid36.jsonl`. The numeric content of `INDEX.md` (31/36 success, 0 collisions, per-episode switch/branch counts) is fully recoverable from those two. `CLEANUP_PLAN.md:131` already says 删(可随时重渲).

### `scene_rgb/` (66 MB) + `scene_menu/` (2.9 MB) + `scene_tiled/` (1.7 MB) → **DELETE**
Top-down scene renders, 58 / 58 / 1 PNGs. Not duplicates of each other (md5 of `102344022.png` differs across the two dirs), but each is pure output:
`hrl_pipeline/scene_tools/render_scene_rgb_topdown.py <dataset.json.gz> <out_dir>`, `hrl_pipeline/scene_tools/render_scene_menu.py <dataset.json.gz> <out_dir>` (default hardcoded `/habitat-lab/hrl_pipeline/scene_menu` at line 64), `hrl_pipeline/scene_tools/render_scene_tiled.py <dataset.json.gz> <out_dir>`.
These were the raw material for the clicker page; clicking is done and exported.

### `video_diag/` — 33 MB, 12 files → **DELETE**
Hand-curated *selection* of diagnostic mp4s (`SWEET_idx36/45/47`, `V5FAIL_col`, `V6FAIL_timeout`, `ep70fix_rule_yield`, `BC_ep70_success`, …). The curation is human, but every clip is a re-renderable rollout and each conclusion is written up in `docs/`. Renaming is the only human bit.

### `video_manual_uncert/` — 7.0 MB → **DELETE (exact duplicate)**
All 3 mp4s are **byte-identical** to `video_rule_markov/` (md5: `1c19442b…`, `3570143c…`, `9f262d74…`). `_archive/scripts/_verify_and_mine.sh:31` literally does `cp -f $SD/video_rule_markov/*.mp4 $SD/video_manual_uncert/`.

### `video_rule_markov/` (7.0 MB), `video_wait579/` (6.4 MB), `video_waitB579/` (6.6 MB) → **DELETE**
3 mp4s each, raw evaluator output (`scripted_hl_eval.py:48` `VIDEO_DIR = os.environ.get("RULE_VIDEO_DIR", "hrl_pipeline/video_" + MODE)`; see `runs/_wait579.txt`). Their stats survive as `results/stats_wait579.json` / `stats_waitB579.json`. The keeper — waitC, the final teacher's three clean episodes — was already moved out by `_reorganize.sh:64` to `hrl_pipeline/videos/expert_wait_teacher_ep579/` (verified present, 3 mp4s, all `reward=0.01`).

### `video_rule_yield/` — 4 KB, **0 files** → **DELETE** (empty)

### `_repro_v3/` — 27 MB, 28 files → **DELETE**
Today's (Aug 17) ES reproduction sweep, produced by `_archive/scripts/_repro2.sh:6` (`OUT=/habitat-lab/hrl_pipeline/_repro_v3`). 14 multi-MB `log_*.txt` + 14 `stats_*.json`. Conclusions are written into `docs/LL_TRAINING_CURVES.md` §2.3 (the "center fitness 0.114 vs 0.827" finding). Re-runnable: `bash hrl_pipeline/_archive/scripts/_repro2.sh` (fix the `$SD/es_ll_jl5ncpsk` path at line 21 first).

### `map_check/` — 100 KB, 24 files (`map_00025.png` … `map_00600.png`) → **DELETE**
Top-down map frames sampled every 25 steps. **No producer survives** — `grep -rn "map_check" --include=*.py .` returns nothing; only `docs_old/INDEX.md:30` and `_reorganize.sh` mention it. Strictly speaking non-regenerable, but it is 24 throwaway debug frames from a one-off map-observation sanity check, entirely superseded by `docs/` and by `hl/_extract_hl_map.py`'s map-branch work. Not worth keeping.

### `figures/` — 868 KB, 5 files → **DELETE, or keep (it's cheap)**
`layout_ep70.png`, `layout_ep70_furniture.png`, `layout_ep70fix.png` ← `hrl_pipeline/scene_tools/scene_layout.py <dataset.json.gz> <episode_id> <out.png>`; `topdown_dev20.png`, `topdown_doorcand78.png` ← `hrl_pipeline/scene_tools/render_topdown.py <dataset.json.gz> <scene_types.csv> <out.png>`. Fully regenerable. `CLEANUP_PLAN.md:137` flags it as the one to keep if you have paper-figure needs — at 868 KB I'd just keep it.

### `es_ll_0wn47pql/` (14 MB) + `es_ll_3ggr2gbr/` (14 MB) → **DELETE (dead runs)**
Seed-1 and seed-2 ES replicates. **Both are total harness failures:** every generation logs `-1.000` sentinels (`_es_seed1.log` / `_es_seed2.log`, gen0 through gen10, `best=-1.000 {}`). Consistently, neither dir contains a single `stats_*.json` (322 files = 160 `.pth` + 160 `.txt` + center only), so `reconstruct_es_center.py` cannot read them either. And unlike the two keepers, these two **were seeded** (`[es] torch seed 1` / `2` at line 1) — so they are exactly reproducible if ever wanted. Zero information content.

### `__pycache__/` — 64 KB, 9 `.pyc` → **DELETE**
Stale bytecode for scripts that have since moved into `hl/ ll/ data_tools/ diag/` (mixed cpython-38 and cpython-39). Regenerated on next import.

---

## Stale paths that block the "just regenerate it" claims

Fix these before relying on any regeneration command above:

- `hrl_pipeline/scene_tools/make_click_page.py:32` — `AUDIT_CSV = "hrl_pipeline/doorcand_v3_audit.csv"`; the file now lives at `hrl_pipeline/results/doorcand_v3_audit.csv`. It fails soft (`audit_by_idx()` returns `{}` at line 36), so a regenerated click page would silently lose every `verdict` badge.
- `hrl_pipeline/hl/scripted_hl_eval.py:48` — `VIDEO_DIR = os.environ.get("RULE_VIDEO_DIR", "hrl_pipeline/video_" + MODE)`; the default still litters `hrl_pipeline/` root instead of `hrl_pipeline/videos/`.
- `hrl_pipeline/ll/es_ll_finetune.py:105` — `tempfile.mkdtemp(prefix="es_ll_", dir="/habitat-lab/hrl_pipeline")`; any new ES run drops another root-owned `es_ll_*` at the top level.
- `hrl_pipeline/docs/PIPELINE_REFERENCE.md:907` and `:913` — cite `hrl_pipeline/manual/click30.html` and `hrl_pipeline/0805.json`; actual locations are `_archive/dirs/manual/click30.html` and `hrl_pipeline/results/0805.json`.
- `hrl_pipeline/_archive/scripts/_repro2.sh:6,21` and `_ll_variance.sh:19` — reference `/habitat-lab/hrl_pipeline/es_ll_jl5ncpsk` and `$SD/_repro_v3` at the old top level.

---

## `outputs/` (repo root, 7.9 GB) — **PARTIAL keep**

`/home/xinyuan/habicrowd/Simulator/habitat-lab/outputs` — 963 Hydra run dirs across `2026-07-02` … `2026-08-16`.

- **Keep the 963 `.hydra/` subtrees — 44 MB total.** `docs/LL_TRAINING_CURVES.md:220-231` reconstructs the entire round-2 ES invocation *solely* from `outputs/2026-08-16/08-32-53/.hydra/{overrides,hydra}.yaml` "plus 110 more agreeing hydra dirs". This is the only record that round 2 used `RULE_MODE=learned` + `social_nav_hierarchical_overfit_v2.yaml` + `nr1_ck14_hl.pth` rather than the driver defaults — the doc warns that re-running with the *stated* facts yields fitness 0.114 instead of 0.827. Losing these makes the headline LL result unreproducible.
- **Delete the `run.log` files — that's essentially all 7.85 GB.** Five single logs are 265–529 MB each (e.g. `outputs/2026-08-15/06-52-57/run.log` = 529,137,664 B). The metric/traceback lines already survive in the 163 shrunk `hrl_pipeline/runs/*.txt`.

Suggested: `find outputs -name run.log -delete` (frees ~7.8 GB), keep the rest.