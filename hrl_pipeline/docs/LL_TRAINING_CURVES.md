# LL Yield Policy — What "Training Progress" Actually Looks Like

*Definitive account of the low-level (LL) yield policy's training history. Written 2026-08-17.*
*Every number here was recovered from the on-disk artifacts (training logs, ES workdirs, stats JSONs) and cross-checked
against the driver source. Where something cannot be determined from the data, it says so.*

---

## 0. Read this first: there is no reward curve

**The LL policy was never trained with reinforcement learning.** It went through three stages, none of which is
policy-gradient RL:

| # | Stage | Kind of learning | Objective optimized | Artifact |
|---|-------|------------------|---------------------|----------|
| 1 | **BC** | supervised regression | MSE to scripted-teacher actions on teacher-visited states | `ll_yield_bc_v2.pth` |
| 2 | **DAgger r1** | supervised regression | MSE to scripted-teacher **relabels** on **student**-visited states | `ll_yield_dagger1.pth` |
| 3 | **ES** | black-box search, last layer only (258 params) | closed-loop fitness, evaluated per candidate | `ll_yield_es_v2.pth`, `ll_yield_es_v3.pth` |

Consequences for the question "what did reward / success look like over training?":

- **Stages 1–2 have no reward and no success metric during training at all.** They have an MSE curve. Success is only
  observable by stopping and running a closed-loop probe.
- **A reward-like quantity exists only in stage 3**, and it is not a per-timestep return. It is a scalar *fitness*
  computed once per candidate weight vector by running full episodes:

  ```
  fitness = mean_success − 0.5 · mean_collide − 0.1 · mean(num_steps) / 1200
  ```
  (`hrl_pipeline/es_ll_finetune.py`, `eval_candidate`, lines 68–71.)

- Therefore success-vs-progress exists at exactly **two granularities**, and nothing finer:
  - **(a) within ES** — per generation, across the 16 candidates;
  - **(b) across stages** — the discrete closed-loop probes run between BC / DAgger / ES rounds.

Anything that looks like a smooth PPO-style reward-vs-timestep curve for this policy would be manufactured. There is none.

Architecture throughout: `22 → 128 → 128 → 2` MLP (16 lidar + 4 human-polar + 2 door-direction → `[lin, ang]`, `Tanh`
output). ES perturbs only `net.4.weight` (2×128) + `net.4.bias` (2) = **258 parameters**.

---

## 1. Stages 1–2: the supervised curves (the only smooth curves that exist)

Trainer: `hrl_pipeline/bc_ll_yield.py`. 400 epochs, Adam **constant lr 1e-3** (no scheduler, no weight decay, no
clipping, no early stopping), batch 4096, plain MSE over both action dims jointly, `torch.manual_seed(0)`, CPU-only,
fully deterministic. Print cadence is `epoch % 50 == 0 or epoch == 399` → **9 points per run. That is the entire curve
that exists on disk**; there is no finer-grained data without editing the trainer.

### 1.1 BC (dataset A = demos only) → reproduces `ll_yield_bc_v2.pth`

Source: `hrl_pipeline/ll_demos_v2_clean.jsonl`, 15,490 rows.

| epoch | 0 | 50 | 100 | 150 | 200 | 250 | 300 | 350 | 399 |
|---|---|---|---|---|---|---|---|---|---|
| **train MSE** | 0.0368 | 0.0056 | 0.0037 | 0.0031 | 0.0028 | 0.0023 | 0.0023 | 0.0020 | **0.0018** |
| **val MSE** | 0.0377 | 0.0059 | 0.0037 | 0.0031 | 0.0030 | 0.0025 | 0.0024 | 0.0022 | **0.0019** |

### 1.2 BC + DAgger r1 (dataset B = demos ∪ DAgger) → reproduces `ll_yield_dagger1.pth`

Source: `hrl_pipeline/ll_train_r1.jsonl`, 30,950 rows (verified to be exactly A followed by `ll_dagger_r1.jsonl`,
15,490 + 15,460, elementwise-identical actions).

| epoch | 0 | 50 | 100 | 150 | 200 | 250 | 300 | 350 | 399 |
|---|---|---|---|---|---|---|---|---|---|
| **train MSE** | 0.0405 | 0.0074 | 0.0056 | 0.0040 | 0.0036 | 0.0031 | 0.0028 | 0.0026 | **0.0025** |
| **val MSE** | 0.0413 | 0.0076 | 0.0060 | 0.0045 | 0.0040 | 0.0034 | 0.0031 | 0.0029 | **0.0027** |

**These are the original curves, not reconstructions.** Both re-runs reproduce the historical finals exactly, and the
freshly written throwaway checkpoints are **bitwise identical** to the production weights (`max|Δ| = 0.000e+00` for
`_tmp_bc_v2.pth` vs `ll_yield_bc_v2.pth`, and for `_tmp_bc_dagger1.pth` vs `ll_yield_dagger1.pth`). A third independent
re-run of A printed the identical 9-point curve. Production artifacts were never touched.

**Both curves are still descending at epoch 399** (A went 0.0020 → 0.0018 over the final 49 epochs) under a constant lr
with no early stopping. 400 epochs is an arbitrary stop, not convergence.

### 1.3 B's higher val MSE does NOT mean the DAgger model is worse

Cross-dataset val-MSE comparison between these two runs is **meaningless**. B is a strictly harder regression target:

| | dataset A | dataset B |
|---|---|---|
| rows | 15,490 | 30,950 |
| angular target variance | 0.070704 | 0.083733 |
| predict-train-mean baseline val MSE | 0.035445 | 0.042670 |
| achieved val MSE | 0.0019 | 0.0027 |
| implied val R² (angular) | 0.946 | — |

Only closed-loop probes can rank these two checkpoints, and they do (§4).

### 1.4 What DAgger actually changed in the data distribution

The common assumption — "DAgger added freeze/stop rows" — is **definitively false**. Measured:

| statistic | demos (A) | DAgger r1 | change |
|---|---|---|---|
| freeze rows (`\|lin\|<1e-3` **and** `\|ang\|<1e-3`) | **0** | **0** | none — the reverse-safety freeze never fired in either half |
| min human distance | 1.0153 m | 1.0559 m | 0.00% of rows under 1.0 m in either |
| `lin == 0` exactly | 2008 (12.96%) | 1570 (10.16%) | **fewer**, not more |
| lin mean / std | −0.051125 / 0.033776 | −0.051492 / 0.032566 | essentially unchanged |
| lin support | [−0.099999, 0.000000] | [−0.100000, 0.000000] | **never positive in either** — this skill only reverses or holds |
| **`\|ang\|` mean** | **0.181598** | **0.233771** | **+28.7%** |
| **ang variance** | **0.070704** | **0.096421** | **+36.4%** |
| `\|ang\| ≥ 0.35` (min_turn_rate floor) | 3851 (24.86%) | 6049 (39.13%) | +14.3 pts |
| exactly **on** the 0.35 floor | 1594 (10.29%) | 3073 (19.88%) | **nearly doubled** |
| full saturation `\|ang\| ≥ 0.999` | 138 (0.89%) | 227 (1.47%) | +0.58 pts |
| near-straight `\|ang\| < 0.05` | 4101 (26.48%) | 2966 (19.18%) | **−7.3 pts** |
| ang p5 / p95 | −0.350000 / +0.513848 | −0.418925 / +0.540192 | both tails widened |

**Read plainly: DAgger taught recovery-from-heading-error, not stopping.** The BC student drifts into headings the
teacher considers badly misaligned, and the teacher's relabel is "turn hard, at or beyond the floored rate". A quarter
of the near-straight mass was converted into hard turns.

**Direct evidence of state drift.** Only 2,403 of the 15,460 DAgger rows share a `t` value with any demo row (`t` is a
global env-step counter; student rollouts desynchronize almost immediately). On those 2,403 overlapping steps, **2,402
have a different action**: `|Δang|` mean 0.158042 / p50 0.073119 / p95 0.511669 / **max 1.405488**; `|Δlin|` mean
0.019964 / max 0.089226. The human-distance *observation itself* differs by mean 0.318 m / p95 0.861 m / max 1.296 m.
The student is genuinely standing somewhere else, not merely commanding something else.

*Structural note:* demo rows carry the privileged planner waypoint `wp` and a `branch` flag; DAgger rows carry neither.
Neither field is read by the trainer, so this does not affect training — but it makes branch-level attribution
impossible for the DAgger half. `env` is 0 for every row in every file, so contiguous `t` is the only usable segment key.

### 1.5 The open-loop → closed-loop gap, measured

This is the single most important number in the supervised half.

| evaluation of `ll_yield_bc_v2` weights | MSE (both dims) | MSE (angular) | `\|err ang\|` p95 | `\|err ang\|` max |
|---|---|---|---|---|
| own held-out rows (teacher-visited states) | **0.001886** | 0.003741 | 0.1171 | 0.5157 |
| the 15,460 states its **own rollout** visited | **0.020698** | 0.041270 | 0.4556 | **1.2834** |
| **degradation** | **11.0×** | **11.0×** | 3.9× | 2.5× |

A 1.28 error in command units is 12.8 rad/s of angular velocity — larger than a full-scale command, i.e. the student
turning the wrong way at full rate. **That 11× factor is the compounding-error gap, measured rather than assumed.**

### 1.6 Physical units of "MSE 0.002", and why low MSE ≠ competence

Command scaling from `hssd_spot_human_social_nav_tk.yaml`: `longitudinal_lin_speed = 10.0`, `ang_speed = 10.0`, applied
as `clip(cmd, −1, 1) * speed`. The headline 0.0019 is a mean over both dims and is **almost entirely angular**:

| | dataset A val | dataset B val |
|---|---|---|
| MSE(lin) | 0.0000304 | 0.0000344 |
| MSE(ang) | 0.003741 (**123× the linear term**) | 0.005424 |
| RMSE(lin) | 0.005517 cmd = 0.0552 m/s | — |
| RMSE(ang) | 0.061167 cmd = **0.6117 rad/s = 35.05 °/s** | 0.7365 rad/s = **42.20 °/s** |
| per env step @ 120 Hz | 0.46 mm, **0.292 °/step** | 0.352 °/step |

*Caveat: the per-step figures assume `ctrl_freq = 120 Hz` (the config default, `ac_freq_ratio: 1`); the effective
integration window was not verified by running the sim. The m/s and rad/s figures are exact and assumption-free.*

Yield segments run ~200 steps. Per-step error is negligible **only if** it is zero-mean and independent — it is
neither, because the residual concentrates at the teacher's branch switches (reverse-and-drive vs in-place turn vs
hold-and-face-door), precisely the moments that determine where the robot ends up. If the sign persisted across a
segment, A's residual integrates to **58° of heading error and 9.2 cm** of position error; B's to **70° and 9.8 cm**.
In a narrow-door yield those magnitudes decide clearance versus collision.

Three independent, measured reasons low MSE does not imply closed-loop competence:

1. **The val split is random ROWS, not held-out trajectories.** `torch.randperm(n)`, first `n//10` is val. Consecutive
   steps of the same 200-step trajectory land on both sides, so val measures *interpolation within a seen trajectory*.
2. **The metric is open-loop one-step regression on teacher-visited states** — and the same weights degrade **11.0×**
   on their own rollout states (§1.5).
3. **R² = 0.946 on angular val.** The net leaves ~5% of angular variance unexplained, concentrated at branch
   boundaries — exactly the behaviourally decisive 5%.

---

## 2. Stage 3: ES — the only place a "reward" exists

### 2.1 The algorithm as actually implemented

Read from `hrl_pipeline/es_ll_finetune.py` lines 100–141. Both rounds used identical hyperparameters:
`pop = 16`, `gens = 10`, **`RULE_EVALS = 1`**, `sigma = 0.03`, `sigma_decay = 0.95`, `topk = 4`, last-layer-only
(258 params), warm start `ll_yield_dagger1.pth`, step cap 1200.

Four properties of this implementation matter for every number below:

**(a) The saved weights are the BEST-EVER SINGLE CANDIDATE, not the final center.** Line 139 saves `best_vec`, which
line 133 sets only when a generation's top candidate strictly exceeds `best_fit`. `best_fit`/`best_vec` are *seeded*
with the gen-0 unperturbed center (lines 114–115), so the center competes on the same elitist ladder — had no candidate
ever beaten it, `--out` would have received the unmodified warm start.

**(b) The final center is computed and silently discarded.** Line 130 computes generation 10's top-4 mean, line 135
decays sigma, the loop exits, line 139 saves `best_vec` instead. **The final center is never saved and never
evaluated.**

**(c) The center update is a plain unweighted elitist mean.** `center = torch.stack([top-4 vectors]).mean(0)` — a
(μ=4, λ=16) elitist-mean ES. No fitness weighting, no rank shaping, no learning rate, no covariance or per-coordinate
adaptation, no momentum, no antithetic/mirrored sampling. **The previous center is not a member of the averaged set**,
so the center can regress with no guard.

**(d) The sigma printed in the log is already the next generation's.** `sigma *= 0.95` runs *after* scoring, so
generation *g* samples with `sigma_g = 0.03 · 0.95^(g−1)`:

| gen | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|
| sigma **used** | 0.030000 | 0.028500 | 0.027075 | 0.025721 | 0.024435 | 0.023213 | 0.022053 | 0.020950 | 0.019902 | 0.018907 |
| sigma **printed** | 0.0285 | 0.0271 | 0.0257 | 0.0244 | 0.0232 | 0.0221 | 0.0210 | 0.0199 | 0.0189 | 0.0180 |

Expected perturbation norm `sigma·√258` falls 0.4819 → 0.3037 against a warm-start last-layer weight norm of
`‖W‖ = 3.20467`, i.e. **15.0% → 9.5% relative**. Total anneal over the whole run is only `0.95^10 = 0.5987` — sigma
never fell below 60% of its start.

### 2.2 Verification status

| check | round 1 | round 2 |
|---|---|---|
| workdir | `es_ll_zji8yz5q` | `es_ll_jl5ncpsk` |
| log | `_es_v2.log` | `_es_v3.log` |
| files present | 161 stats + 161 logs + 161 .pth (483) | 161 stats + 161 logs + 161 .pth (483) |
| episode-runs | 1288 (8 × 161), none missing | 2254 (14 × 161), none missing |
| ↳ candidate-only subset used for the φ / marginal stats below | 1280 (8 × 160) | 2240 (14 × 160) |
| harness crashes (−1.0 sentinel) | **zero** | **zero** |
| log ↔ recomputed fitness | **exact match, all 10 gens / 160 candidates, 0 mismatches** | **exact match, max discrepancy 0.000** |
| shipped weights identified by byte equality | `ll_yield_es_v2.pth` last layer ≡ `g3_c9.pth` | `ll_yield_es_v3.pth` last layer ≡ `g4_c11.pth` |
| `net.0`/`net.2` vs `ll_yield_dagger1.pth` | identical (last-layer-only confirmed) | identical (last-layer-only confirmed) |

### 2.3 IMPORTANT run-facts correction for round 2

**Round 2 did NOT use the scripted teacher HL, and did NOT use the driver's default config.** Recovered from
`outputs/2026-08-16/08-32-53/.hydra/{overrides,hydra}.yaml` (whose `episode_stats_path` is
`es_ll_jl5ncpsk/stats_center.json`, definitively the ES center eval) plus 110 more agreeing hydra dirs:

| | driver default (what was assumed) | round 2 actual (recovered) |
|---|---|---|
| `RULE_MODE` | `rule_yield` | **`learned`** (pure pass-through; the harness only logs the policy's own choice) |
| config | `social_nav_hierarchical_v2_llyield.yaml` | **`social_nav_hierarchical_overfit_v2.yaml`** |
| HL | scripted teacher | **frozen neural HL** `nr1_ck14_hl.pth`, 4356 trainable params (vs 601,604 under the llyield config — the tell that exposed the mismatch) |
| success_reward / collide_penalty | 10.0 / 1.0 | **30.0 / 5.0** |
| safe_dis_min / yield_dis / backoff / slack | 1.0 / 1.0 / 2.0 / −0.1 | **1.5 / 3.0 / 3.0 / −0.01** |

Anyone re-running round 2 must use the recovered invocation. A first repro attempt with the *stated* facts produced
center fitness **0.114** instead of 0.827.

### 2.4 Round 1 — per-generation fitness (8-episode set: train36 ids 2, 13, 23, 27, 21, 26, 20, 5)

Gen-0 center (= the DAgger warm start) fitness = **0.746083**, success 0.875 (7/8), collide 0.125.

| gen | best | mean | median | min | #beat center | best succ | best coll | best steps |
|---|---|---|---|---|---|---|---|---|
| 1 | 0.747812 | 0.468507 | 0.498495 | 0.069833 | 1 | 0.875 | 0.125 | 776.25 |
| 2 | 0.746344 | 0.470146 | 0.378990 | 0.016583 | 2 | 0.875 | 0.125 | 793.88 |
| **3** | **0.932885** ← record | 0.508889 | 0.499870 | 0.009146 | 4 | **1.000** | **0.000** | 805.38 |
| 4 | 0.929688 | 0.511919 | 0.406807 | 0.196865 | 4 | 1.000 | 0.000 | 843.75 |
| 5 | 0.931583 | 0.537211 | 0.568677 | 0.198406 | 6 | 1.000 | 0.000 | 821.00 |
| 6 | 0.930229 | 0.594010 | 0.683057 | 0.203021 | 7 | 1.000 | 0.000 | 837.25 |
| 7 | 0.931583 | 0.567829 | 0.594385 | 0.200146 | 3 | 1.000 | 0.000 | 821.00 |
| 8 | 0.932083 | **0.717833** | 0.748396 | 0.018365 | **13** | 1.000 | 0.000 | 815.00 |
| 9 | 0.931312 | 0.640321 | 0.683036 | 0.374854 | 8 | 1.000 | 0.000 | 824.25 |
| 10 | 0.932500 | 0.640755 | 0.621365 | 0.200187 | 6 | 1.000 | 0.000 | 810.00 |

Population-level success/collision, averaged over all 16 candidates × 8 episodes:

| gen | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|
| mean solve rate | 0.6641 | 0.6797 | 0.7031 | 0.7031 | 0.7266 | 0.7656 | 0.7422 | 0.8516 | 0.7969 | 0.7969 |
| mean collide rate | 0.2656 | 0.2969 | 0.2656 | 0.2578 | 0.2578 | 0.2188 | 0.2266 | 0.1406 | 0.1875 | 0.1875 |

**Winner: `g3c9`** (fitness 0.932885, 8/8 success, 0 collide, mean 805.375 steps). Per-episode steps:
ep2 697, ep13 939, ep23 649, ep27 795, ep21 606, ep26 946, ep20 956, ep5 855.

### 2.5 Round 2 — per-generation fitness (14-episode set: the round-1 eight + repairs 8, 9, 10, 12, 16, 30)

Fitness set verified: `social_nav_episode_es14.json.gz` positions 0–7 ≡ the round-1 `es8` set; positions 8–13 are the
six new repair episodes. Gen-0 center fitness = **0.827101**, success 0.928571 (13/14), collide 0.071429.

| gen | best | mean | median | min | #beat center | best succ | best coll | best steps |
|---|---|---|---|---|---|---|---|---|
| 1 | 0.722708 | 0.406904 | 0.400295 | 0.020351 | **0** | 0.857 | 0.143 | 756.07 |
| 2 | 0.825363 | 0.621953 | 0.614000 | 0.514696 | **0** | 0.929 | 0.071 | 809.93 |
| 3 | 0.828190 | 0.549219 | 0.577324 | **−0.216720** | 1 | 0.929 | 0.071 | 776.00 |
| **4** | **0.930018** ← record | 0.559897 | 0.544250 | 0.096845 | 1 | **1.000** | **0.000** | 839.79 |
| 5 | 0.825643 | 0.600249 | 0.631673 | 0.233250 | **0** | 0.929 | 0.071 | 806.57 |
| 6 | 0.851321 | 0.566570 | 0.595324 | 0.096923 | 1 | 0.929 | 0.000 | 927.00 |
| 7 | 0.928690 | 0.666978 | 0.645039 | 0.465923 | 1 | 1.000 | 0.000 | 855.71 |
| 8 | 0.929387 | 0.657040 | 0.683318 | 0.410619 | 1 | 1.000 | 0.000 | 847.36 |
| 9 | 0.854405 | 0.680077 | 0.748777 | 0.336387 | 1 | 0.929 | 0.000 | 890.00 |
| 10 | 0.929060 | **0.739077** | **0.786440** | 0.513137 | 3 | 1.000 | 0.000 | 851.29 |

Population-level success/collision, averaged over all 16 candidates × 14 episodes:

| gen | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|
| mean solve rate | 0.6295 | 0.7768 | 0.7277 | 0.7321 | 0.7589 | 0.7366 | 0.8036 | 0.8036 | 0.8170 | **0.8616** |
| mean collide rate | 0.3259 | 0.1786 | 0.2321 | 0.2188 | 0.1875 | 0.2098 | 0.1339 | 0.1607 | 0.1384 | **0.1071** |

Center drift from `c_0` (last-layer L2, against `‖W‖ = 3.20467`):

| gen | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|
| ‖center − c₀‖ | 0.2165 | 0.3394 | 0.3942 | 0.4535 | 0.5085 | 0.5409 | 0.5700 | 0.6152 | 0.6144 | 0.6239 |
| ‖center − winner‖ | — | — | — | 0.3738 | 0.4189 | 0.4825 | 0.5172 | 0.5519 | 0.5570 | **0.5743** |

**The center drifts steadily AWAY from the saved weights after gen 4** — the artifact and the search's own best estimate
of a good region diverge.

**Winner: `g4c11`** (fitness 0.930018, 14/14 success, 0 collide, mean 839.7857 steps). `‖out − base‖` on the last layer
= 0.58276 (max|Δ| = 0.11911, rms = 0.03628) — an **18.2% relative displacement**. Per-episode steps: ep2 779, ep13 1032,
ep23 713, ep27 809, ep21 791, ep26 916, ep20 963, ep5 881, ep8 744, ep9 803, ep10 983, ep12 780, ep16 696, ep30 867.

### 2.6 Both rounds plateaued, and for the same structural reason

| | round 1 | round 2 |
|---|---|---|
| record set at | **gen 3** (0.932885) | **gen 4** (0.930018) |
| further evaluations that beat it | **0 of 112** | **0 of 96** |
| candidates reaching the max *outcome* | **15 of 160 (9.4%)** — 8/8 succ, 0 coll | **4 of 160 (2.5%)** — 14/14 succ, 0 coll |
| their fitness spread | 0.006583 | 0.001328 |
| what decided the winner among them | mean steps only: 805.38 (g3c9) … 884.38 (g6c6) | mean steps only: 839.79 … 855.71 (15.9 steps out of ~845) |

Once a candidate hits perfect success with zero collisions, the **only remaining gradient is the step term, total weight
0.1**. To beat round 2's record a perfect candidate needed `mean_steps < 839.78`. Three later candidates reached the
identical perfect outcome and lost purely on speed (g8c0 847.36, g10c10 851.29, g7c14 855.71).

**In outcome space, neither search made any progress after its record generation.** Which of the tied candidates became
the shipped artifact was decided by a step-count tiebreak, not by any behavioural difference the metric can see.

---

## 3. The contested frontier: per-episode dynamics

### 3.1 No episode is ever always-solved or always-broken — in either round

This is unusual and important: **the entire fitness set is search frontier, with no stable anchor.**

**Round 1** solve rate over all 160 candidates (contestedness ranking, hardest first) — even the easiest episode is only
147/160 = 91.9%, and **not one episode reaches a 1.0 solve rate in any single generation** (per-gen max is 15/16):

| ep | 20 | 5 | 13 | 27 | 26 | 2 | 23 | 21 |
|---|---|---|---|---|---|---|---|---|
| solved / 160 | 93 | 97 | 105 | 120 | 126 | 130 | 133 | 147 |
| rate | **0.581** | 0.606 | 0.656 | 0.750 | 0.788 | 0.813 | 0.831 | 0.919 |

**Round 2** solve rate over all 160 candidates:

| ep | 20 | 26 | 13 | 27 | 8 | 10 | 5 | 12 | 9 | 2 | 23 | 21 | 16 | 30 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| rate | **0.2812** | **0.4000** | 0.6750 | 0.7125 | 0.7438 | 0.7500 | 0.8063 | 0.8562 | 0.8625 | 0.8875 | 0.8938 | 0.9375 | 0.9437 | 0.9563 |

*Caveat on reading per-generation curves: with only 16 candidates, a solve-rate estimate has binomial sd
`√(p(1−p)/16) = 0.125` at p = 0.5, so single-generation wiggles of ±3 candidates are inside noise. Only ep20's
persistently low band (0.125–0.4375) and ep26's (0.25–0.5625) are robust in round 2.*

### 3.2 Per-generation solve rate, round 1 (rows = episodes, columns = generations)

| ep | g1 | g2 | g3 | g4 | g5 | g6 | g7 | g8 | g9 | g10 | Δ |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 2 | .688 | .625 | .750 | .750 | .750 | .938 | .875 | .938 | .875 | .938 | +.250 |
| 13 | .438 | .688 | .500 | .500 | .688 | .625 | .813 | .813 | .750 | .750 | +.313 |
| 23 | .938 | .750 | .938 | .875 | .875 | .938 | **.500** | .875 | .750 | .875 | −.063 |
| 27 | .688 | .688 | .625 | .688 | .688 | .688 | .875 | .938 | .813 | .813 | +.125 |
| 21 | .938 | .813 | .938 | .938 | .938 | .938 | .875 | .938 | .938 | .938 | .000 |
| 26 | .563 | .688 | .750 | .688 | .875 | .875 | .750 | .938 | .875 | .875 | +.313 |
| **20** | **.250** | .563 | .375 | .563 | .500 | .500 | **.875** | .750 | .688 | .750 | **+.500** |
| **5** | **.813** | .625 | .750 | .625 | .500 | .625 | **.375** | .625 | .688 | .438 | **−.375** |

### 3.3 Per-generation solve rate, round 2

| ep | g1 | g2 | g3 | g4 | g5 | g6 | g7 | g8 | g9 | g10 | Δ |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 2 | .563 | .875 | 1.00 | .813 | 1.00 | .938 | .875 | .938 | 1.00 | .875 | +.313 |
| 13 | .688 | .313 | .625 | .688 | .813 | .563 | .563 | .938 | .813 | .750 | +.063 |
| 23 | .750 | 1.00 | .750 | .938 | .938 | .875 | 1.00 | .813 | .875 | 1.00 | +.250 |
| 27 | .625 | .313 | .813 | .625 | .688 | .625 | .750 | .938 | .938 | .813 | +.188 |
| 21 | .688 | 1.00 | .875 | .938 | .938 | 1.00 | .938 | 1.00 | 1.00 | 1.00 | +.313 |
| 26 | .500 | .375 | .563 | .313 | .250 | .250 | .313 | .500 | .375 | .563 | +.063 |
| **20** | .375 | .188 | .313 | .188 | .188 | .375 | .125 | .313 | .313 | .438 | +.063 |
| 5 | .688 | .938 | .688 | .750 | .750 | .813 | 1.00 | .750 | .813 | .875 | +.188 |
| 8 | .563 | 1.00 | .750 | .750 | .625 | .625 | .938 | .563 | .688 | .938 | **+.375** |
| 9 | .625 | 1.00 | .813 | .813 | .875 | .875 | 1.00 | .813 | .875 | .938 | +.313 |
| 10 | .438 | .875 | .625 | .688 | .688 | .750 | .875 | .813 | .813 | .938 | **+.500** |
| 12 | .813 | 1.00 | .625 | .875 | .875 | .750 | .875 | .875 | .938 | .938 | +.125 |
| 16 | .750 | 1.00 | .813 | .938 | 1.00 | .938 | 1.00 | 1.00 | 1.00 | 1.00 | +.250 |
| 30 | .750 | 1.00 | .938 | .938 | 1.00 | .938 | 1.00 | 1.00 | 1.00 | 1.00 | +.250 |

**All 14 episodes improved from gen 1 to gen 10.** That is the population-level signal, and it lives entirely in the
discarded center.

### 3.4 The tradeoff structure — ep20 is the axis of conflict in both rounds

**Round 1, strongest pair: ep20 vs ep5.** Generation-level solve-rate correlation **r = −0.767**; candidate-level φ over
all 160 candidates **−0.503** — the largest-magnitude negative at *both* levels, so this is a genuine policy-level
conflict, not drift co-movement. ep20 climbs .250 → .875 (g7) → .750 (g10) while ep5 falls .813 → .375 (g7) → .438.
**ep20 is precisely the episode the gen-0 center failed, so the ES traded away an already-solved episode to buy the one
it was selecting on.** Second real pair: ep23 vs ep20 (gen −0.696, cand −0.349) — ep23 collapses to .500 exactly at g7
where ep20 peaks at .875. Weaker, generation-level-only pairs (small candidate-level φ → treat as drift co-movement):
13 vs 5 (−0.697 / −0.233), 13 vs 23 (−0.603 / −0.115), 23 vs 27 (−0.503 / −0.029), 27 vs 5 (−0.472 / −0.140).

**Round 2, the same axis, now visible as a two-cluster split.** Conditioning all 160 candidates on ep20 (45 solve it,
115 do not):

| ep | 8 | 5 | 10 | 23 | 9 | 16 | 30 | | 27 | 26 | 2 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| solve rate given ep20 **solved** | .489 | .556 | .511 | .733 | .711 | .844 | .889 | | .844 | .511 | .956 |
| solve rate given ep20 **failed** | .843 | .904 | .843 | .957 | .922 | .983 | .983 | | .661 | .357 | .861 |
| Δ | **−.355** | **−.349** | **−.332** | −.223 | −.211 | −.138 | −.094 | | +.184 | +.155 | +.095 |

**This is not a candidate-quality confound:** ep20-solvers have a *lower* mean total success (10.444/14) than
ep20-failers (10.809/14). Within-candidate φ over 160 samples: ep20–ep5 −0.397, ep20–ep8 −0.365, ep20–ep10 −0.345,
ep20–ep23 −0.326, ep20–ep9 −0.275. (Independent HL sampling noise can only dilute correlations toward zero, never
manufacture negative ones, so these are structural.)

**The axis is exactly the round-1 vs round-2 conflict the 14-episode set was built to resolve:** cluster A {20, 27, 26,
2} — all round-1 originals — against cluster B {5, 8, 9, 10, 16, 30}, of which 8/9/10/16/30 are five of the six round-2
repair episodes. Joint counts: only **17/45 ep20-solvers (37.8%)** also solve {5, 8, 10}, vs **85/115 ep20-failers
(73.9%)**. The winner `g4c11` is one of only 4/160 candidates that got everything at once.

Second round-2 pair — **ep27 vs ep8/ep10/ep5, coupled to ep13.** Candidates solving ep27 (n=114) vs not (n=46):
ep8 .649 vs .978 (−.329), ep10 .667 vs .957 (−.290), ep5 .737 vs .978 (−.241), but **ep13 .816 vs .326 (+.490)**.
Generation-to-generation *change* correlations (9 deltas) confirm the opposing motion: ep13↔ep5 −0.866, ep27↔ep10
−0.858, ep23↔ep27 −0.856, ep13↔ep9 −0.834, ep13↔ep8 −0.826, ep27↔ep16 −0.799, ep23↔ep20 −0.751; against co-movers
ep5↔ep8 +0.930, ep9↔ep10 +0.925, ep5↔ep9 +0.919, ep23↔ep5 +0.908.

### 3.5 Failure modes are not homogeneous

**ep20 is a pure collision episode and carries the whole result.** In round 2, all 115 of its failures are collisions
(zero timeouts); mean 512.5 steps when it collides vs **1024.8 steps when solved** — solving it means waiting roughly
twice as long. It is the *only* episode the gen-0 center fails and the *only* episode whose outcome differs between
center and winner, i.e. **100% of the +0.102917 advantage**.

**ep26 is bimodal, not a narrow timeout.** 96 of its 160 round-2 rollouts fail and 95 of those are step-cap timeouts
(only 1 collision — the lowest of any episode). Its `num_steps` histogram is `{600s:1, 800s:14, 900s:42, 1000s:7,
1100s:1, 1200(cap):95}` — **exactly one rollout finished in [1100, 1200)**. Candidates either get through in ~900 steps
or stall permanently; **raising STEP_CAP would not convert these into successes.** Same pattern in round 1: ep26's 34
failures are 32 timeouts + 2 collisions, so its collide rate is 0.0 in 9 of 10 generations.

On the other 6 round-1 episodes, failure ⟺ collide with perfect fidelity (`success + collide` is 0/160 on every one),
which is why round-1 fitness collapses to a near-integer count of episodes solved.

---

## 4. Stage-wise closed-loop progression (the only cross-stage "success curve" that exists)

All closed-loop numbers: deterministic harness, `evals_per_ep = 3`, seed 7, step cap 1200, **3/3 = stable success**.

### 4.1 The L0 probe set (5 diagnostic episodes)

| LL policy | id2 (collision family) | id20 (deadlock) | id25 (collision) | id21 (fast success) | id26 (slow success) |
|---|---|---|---|---|---|
| scripted (teacher's own) | 0/3, 3 coll | 0/3, timeout | 0/3, 3 coll | 3/3 | 3/3 |
| **pure BC** | 0/3, 3 coll | **3/3 (!)** | 0/3, 3 coll | **1/3, 2 coll ← REGRESSION** | 3/3 |
| **DAgger** | 0/3, 3 coll | 0/3, 3 coll | 0/3, 3 coll | **3/3 (fixed)** | 3/3 |

This is the clearest small-scale picture of what each supervised stage bought and cost: **BC accidentally solved the
deadlock the teacher cannot solve (id20) but broke a working episode (id21); DAgger repaired id21 and lost id20.** The
three hard collision episodes were untouched by either.

### 4.2 Full train36 (stable-success out of 28 / capability out of 4; the teacher solves 0 of the 4)

| system | stable / 28 | collide | capability / 4 | which capability | paired steps vs teacher |
|---|---|---|---|---|---|
| teacher HL + scripted LL (reference) | **28** | 0 | **0** | — | 0 (baseline) |
| NR1 HL + scripted LL | 25 | 0 | 1 | id20 | −27 |
| NR1 HL + **DAgger LL** (the ES warm start) | 23 | 3 | 1 | id2 | −8 |
| NR1 HL + **ES round 1** (`ll_yield_es_v2`) | **18** | **6** | 3 | id2, id3, id25 | — |
| NR1 HL + **ES round 2** (`ll_yield_es_v3`) | **24** | 2 | 3 | id2, id3, id20 | — |

Reference: teacher HL + scripted LL on full train36 = succ 0.833 / collide 0.120 avg. ES round 2 = succ 0.833 /
collide 0.120 avg — matching the scripted teacher on aggregate while adding 3 capability wins the teacher cannot get.

For context, the July-era LL weights re-evaluated on the current stack: BC 0.556 succ / 0.444 collide; ES 0.583 / 0.417.

### 4.3 The stage-wise curve, stated plainly

```
scripted LL   28/28 stable, 0 capability     (the ceiling the teacher defines, and its blind spot)
      ↓ BC (open-loop imitation)
pure BC       ~ solves id20 the teacher can't, but regresses id21   [probe-set evidence only]
      ↓ DAgger r1 (relabel on student states)
DAgger LL     23/28 stable, 3 collide, 1 capability      ← the ES warm start
      ↓ ES round 1 (8-episode fitness set)
ES v2         18/28 stable, 6 collide, 3 capability      ← WORSE than its own warm start
      ↓ ES round 2 (14-episode fitness set, 6 repair episodes added)
ES v3         24/28 stable, 2 collide, 3 capability      ← recovers the loss, keeps the capability wins
```

**Round 1's ES made the policy worse on train36 than the checkpoint it started from** (18/28 with 6 collisions vs 23/28
with 3), while reporting a perfect 0.933 fitness. That single fact is the strongest available evidence about what the
ES fitness number means.

### 4.4 Verification caveat on the round-2 headline

**The "24/28 stable-success" figure could not be reproduced from the artifacts on disk.** Only two post-run evaluations
of `ll_yield_es_v3.pth` exist: `stats_esv3_train36_v1.json` (36 episodes) and `stats_esv3_holdout2.json` (2 episodes).
The former gives stable(all-3-evals) success **29/36 overall**, **22/28** if restricted to train36 ids 8–35, **23/28**
at a ≥2/3 threshold, and 90/108 raw rollouts. No 28-episode evaluation was found. The 24/28 in §4.2 is carried from the
established results table and **is not independently verifiable from the artifacts**.

The **"3 capability wins" does check out**, and is worth stating precisely — along with the two regressions of equal
size that the headline omits:

| held-out episode | warm start (DAgger) | ES v3 |
|---|---|---|
| ep4 | 0/3 | **3/3** |
| ep3 | 2/3 | **3/3** |
| ep18 | 2/3 | **3/3** |
| ep14 | 3/3 | **0/3** |
| ep17 | 3/3 | **1/3** |
| **held-out raw total** | **49/66** | **49/66 — exactly unchanged** |

---

## 5. THE CENTRAL QUESTION: did the ES search actually learn, or did it sample a lucky candidate?

**Both. And they are different objects — which is the whole finding.**

> **The search DID learn: in both rounds the population distribution improved with a strong, consistent trend.
> But the learning was DISCARDED. What was shipped is a single lucky candidate drawn from an early, worse center —
> not the improved center, which was computed on the last iteration and thrown away without ever being evaluated.**

### 5.1 Evidence the search learned — the population distribution, not the best

The right statistic is the *distribution*, because candidates are drawn around the center; if the population improves,
the center improved. Linear fit over the 10 generations:

| statistic | round 1 slope/gen | round 1 Pearson r | round 1 g1 → g10 | round 2 slope/gen | round 2 Pearson r | round 2 g1 → g10 |
|---|---|---|---|---|---|---|
| **population mean fitness** | +0.024307 | **+0.896** | 0.4685 → 0.6408 (+0.1722) | +0.025595 | **+0.841** | 0.4069 → 0.7391 (**+0.3322**) |
| population median fitness | +0.031236 | +0.776 | 0.4985 → 0.6214 | +0.031604 | **+0.884** | 0.4003 → 0.7864 (+0.3861) |
| population **min** fitness | +0.022677 | +0.584 | 0.0698 → 0.2002 | +0.044209 | +0.548 | 0.0204 → 0.5131 |
| **population mean solve rate** | +0.017661 | **+0.900** | 0.6641 → 0.7969 | +0.017830 | **+0.844** | 0.6295 → **0.8616** |
| **population mean collide rate** | −0.013494 | **−0.860** | 0.2656 → 0.1875 | −0.017208 | **−0.834** | 0.3259 → **0.1071** |
| #candidates beating gen-0 center | +0.788 | +0.687 | 1 → 6 | +0.212 | +0.734 | 0 → 3 |

First-half vs second-half means (a coarser, assumption-free version of the same claim):

| | round 1 | round 2 |
|---|---|---|
| mean fitness, g1–5 → g6–10 | 0.4993 → 0.6322 (**+0.1328**) | 0.5476 → 0.6619 (**+0.1143**) |
| mean solve rate, g1–5 → g6–10 | 0.6953 → 0.7906 (**+0.0953**) | 0.7250 → 0.8045 (**+0.0795**) |

The population mean rises, the median rises, the *worst* candidate rises, the mean solve rate rises ~13–23 points, and
the mean collide rate falls by 30–67% relative. In round 2 **all 14 episodes improved from g1 to g10** (ep10 +0.500,
ep8 +0.375, ep2/ep21/ep9 +0.313 each), and the count of candidates above fitness 0.8 went **0/16 → 8/16**. This is not
what a stationary random search looks like. **The elitist-mean center update was doing real hill-climbing.**

**And in round 2 we know this is not measurement noise, because there is none.** The harness was proven
**bit-deterministic given the weights**: `habitat_baselines/run.py` lines 50–52 seed `random`/`numpy`/`torch` from
`config.habitat.seed`, every candidate ran in its own process with `habitat.seed = 7`, so all 161 evaluations share one
common-random-number realization. Verified by re-running the recovered invocation: **5 independent fresh processes on
`g4_c11.pth` all returned fitness = 0.930018, 14/14, 0/14, 839.7857 steps — sd 0.000000, spread 0.000000**; 3
independent processes on `center.pth` all returned 0.827101, 13/14, 1/14, 789.071 — sd 0.000000. The `[HL input #N]`
traces are byte-identical to the original ES logs. **The ES ranking of the 160 candidates is exact.**

### 5.2 Evidence the shipped artifact is a lucky draw

**Round 1 — the null hypothesis is not merely unrejected, it fits to three digits.** Assume each candidate's 8 outcomes
are independent Bernoulli draws at that generation's own marginal solve rates (i.e. *no candidate is genuinely better
than a typical draw from its own cloud*):

| candidates solving k of 8 | k=0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|---|
| **observed** | 0 | 0 | 0 | 5 | 18 | 36 | 38 | 48 | **15** |
| **pooled-null expectation** | 0.0 | 0.0 | 0.5 | 3.5 | 14.4 | 35.4 | 51.6 | 40.9 | **14.93** |

**Expected number of perfect 8/8 candidates = 14.93. Observed = 15.** `P(at least one 8/8 among 160 draws) = 0.999999`
— **the existence of a "perfect" candidate was guaranteed by the sampling procedure alone.** Round 1 contains zero
evidence that any candidate outperforms its generation's mean perturbation.

The winner's advantage decomposes to **exactly one episode**: `0.932885 − 0.746083 = 0.186802 = 1.5/8`, i.e. train36
ep20 flipping from fail+collide to success. Nothing else changed. And ep20 is the *noisiest* episode in the set —
solved by 93/160 candidates (58.1%), and by 6–14 of 16 within any single generation. **Solving it is near a coin flip
over the perturbation cloud, not a discovered capability.**

The tie-break was a pure lottery: 15 candidates achieved identical 8/8 + 0 collide; their entire fitness spread is
0.006583 (3.5% of one episode flip) and comes **100% from mean steps** (805.38 for g3c9 vs 884.38 for g6c6). **The
shipped weights were selected on a 79-step average difference measured once.** The first `best_fit` update (g1c4,
+0.001729 over the center) had *identical* success and collide and won on a 20.75-step difference.

**Round 1's fitness-vs-holdout discrepancy is decisive.** Fitness set: 8/8 = 100%. Full train36: **18/28 = 64.3%**. The
mean marginal solve rate across all 160 candidates on the fitness set is **74.3%** — so the winner's 100% is a tail
draw, not a level. And 18/28 with 6 collisions is *worse than the DAgger warm start it replaced* (23/28, 3 collisions).
**A search that reports a perfect score and delivers a regression on the holdout has selected on noise.**

The root cause is a design flaw, not the ES algorithm: **only one episode of headroom existed.** The gen-0 center
already solved 7 of 8; its single failure was ep20. Maximum achievable measured improvement was one flip ≈ 0.1875, and
that is exactly what was "achieved". The 8-episode set had almost no signal to give and no capacity to distinguish a
real improvement from a coin flip.

### 5.3 How round 2 differed — and what it changes

Round 2 fixed the measurement problem and the answer changed character.

| | round 1 | round 2 |
|---|---|---|
| fitness set | 8 episodes | **14** (the 8 + 6 repair episodes) |
| headroom (episodes the center failed) | 1 | 1 (ep20) — *still only one* |
| perfect candidates observed vs null | **15 vs 14.93 expected → no evidence** | **4 vs 1.93 expected → 2.07×, weak evidence in the right direction** |
| eval determinism | not tested | **proven: sd 0.000000 over 5 repeats** |
| train36 outcome vs warm start | **18/28 vs 23/28 — regression** | **24/28 vs 23/28 — improvement** |

**But the round-2 gain does not generalize off the selection set.** Under an identical independent 3-evals-per-episode
protocol on train36 with the same NR1 HL (`stats_ll36_nr1hl.json` → `stats_esv3_train36_v1.json`):

| | warm start (DAgger) | ES v3 | Δ |
|---|---|---|---|
| overall stable (all-3) success | 25/36 | 29/36 | +4 |
| overall raw rollouts | 82/108 | 90/108 | +8 |
| overall collisions | 22/108 | 13/108 | −9 |
| **on the 14 fitness-set episodes** — stable | 10/14 | **13/14** | **+3** |
| **on the 14 fitness-set episodes** — raw | 33/42 | **41/42** | **+8** |
| **on the 22 held-out episodes** — stable | 15/22 | 16/22 | +1 |
| **on the 22 held-out episodes** — raw | 49/66 | **49/66** | **+0 — exactly unchanged** |

**Every one of the +8 raw successes landed inside the fitness set.** Held-out detail: gains ep4 0/3→3/3, ep3 2/3→3/3,
ep18 2/3→3/3, against regressions ep14 3/3→0/3 and ep17 3/3→1/3 — **net exactly zero.**

**And the reported round-2 fitness value is optimistic.** Because the eval is one fixed RNG realization of a genuinely
stochastic loop (the neural HL samples: `social_nav_neural_policy.py` line 734, `skill_actions = dist.sample()` under
the eval default `deterministic=False`), re-scoring the *same* weights on fresh realizations gives 0.929673 / 0.929095 /
**0.824232** (14/14, 14/14, **13/14**). Spread **0.105441 — larger than the entire winner-vs-center advantage of
0.102917.** Mean 0.894333, so **the selection pass overstates the winner by +0.035685** (best-of-160 maximum-selection
bias against a frozen realization).

Structural evidence that round 2 was also working the razor's edge rather than finding a general fix: only 45/160
candidates solve ep20 at all, and **ep20-solvers have a lower mean total success (10.444/14) than ep20-failers
(10.809/14)** — the direction that fixes ep20 actively costs elsewhere.

### 5.4 Verdict

| claim | round 1 | round 2 |
|---|---|---|
| **The search learned** (population distribution improved) | **YES** — mean fitness r = +0.896, mean solve +13.3 pts, mean collide −7.8 pts | **YES** — mean fitness r = +0.841, mean solve +23.2 pts, mean collide −21.9 pts, all 14 episodes up |
| **The shipped weights embody that learning** | **NO** — g3c9, drawn from generation 3's center; 7 later generations discarded | **NO** — g4c11, drawn from generation 4's center; center drifts to ‖·‖ = 0.5743 away by g10, then discarded |
| **The reported fitness is a real capability level** | **NO** — 100% on 8 episodes vs 64.3% on train36; perfect candidates exactly as numerous as the null predicts (15 vs 14.93) | **NO** — the value is upward-biased by +0.0357; re-realization spread (0.105) exceeds the advantage (0.103) |
| **The improvement over the warm start is real** | **NO** — defensible point estimate of real improvement is **0**, and the train36 result is a regression (18/28 vs 23/28) | **PARTIALLY** — real and reproducible **on the selection set** (raw 33/42 → 41/42), **exactly zero off it** (49/66 → 49/66) |

**Bottom line.** The ES was a working optimizer pointed at a broken objective and wired to save the wrong thing.
Essentially 100% of round 1's headline 0.1868 gain is one ~58%-probability coin flip plus 160-draw maximum-selection
bias. Round 2's gain is genuine optimization — the harness is deterministic, so the ranking is exact — but what it
optimized was *14 specific episodes under one specific RNG realization*, and the measured transfer to 22 held-out
episodes is exactly zero. In both rounds the artifact that shipped is a single lucky draw, and in both rounds the thing
that actually improved — the center — was computed on the final iteration and thrown away without ever being scored.

---

## 6. Metric defects that corrupt every number above

These are not caveats about interpretation; they are defects in the measured quantity.

**(a) `did_collide` does not track actual collisions.**
- **Round 1: 361 of 1288 episode-runs (28.0%)** log `robot_collisions.total_collisions > 0` while `did_collide == 0.0`.
  Worst case: `g5c12` on ep26 logs **482** robot-scene collisions and is scored collision-free. **The winner itself**
  logs 177 scene collisions on ep2 and 58 on ep27 while scoring `did_collide = 0.0` on both.
- **Round 2: 1202 of 2240 rollouts (53.7%)** had `robot_scene_colls > 0`, totalling **287,101 scene-collision events**,
  none of which touch `did_collide` or the fitness. The winner grinds **984** scene collisions on ep10, 483 on ep9, 159
  on ep2, 139 on ep21, 63 on ep27 — while being scored a clean 14/14 with 0 collisions.

**"Zero collisions" in these runs means zero robot–human collisions only. A policy can grind along walls at zero fitness
cost.** This is a plausible independent route to the train36 generalisation gap, separate from the noise-selection issue.

**(b) The collide term is degenerate and rewards stalling.** Across all 2240 round-2 rollouts there are **zero** with
`success = 1` and `did_collide = 1` — collision strictly implies failure. So `−0.5·mean_collide` never trades against
success; it only makes a collision-failure worth `0.5/14 = 0.0357` less than a timeout-failure. **Converting a collision
into a stall is worth about +0.0315 fitness versus +0.1071 for converting it into a real success — i.e. freezing
captures ~29% of the reward of solving, with none of the difficulty.** Of 527 total failures, 424 (80.5%) are collisions
and 103 (19.5%) are timeouts (95 of those on ep26 alone).

**(c) Step-count pressure is effectively absent — except as a tiebreak, where it decides everything.** The steps term
caps at 0.1 total while one episode's success is worth 0.0714 (0.1071 if it converts a collision), so **one extra
episode solved is worth roughly 857 extra steps of mean episode length**. Consistent with this, the best candidate got
*slower* as the run progressed (center 789.07 → g4 839.79 → g6 927.00 → g9 890.00 → g10 851.29), and **the winner is
slower than the warm start it replaced.** Yet once the outcome saturates, the step term is the *only* remaining signal
and it selects the artifact.

**(d) The elite mean averages in a clearly worse vector every generation.** Round 1's gen-10 top-4 were 0.932500 /
0.931354 / 0.928906 / **0.750833** — spread 0.181667. The 4th elite solves one fewer episode (a full 0.1875 step below)
yet enters the mean with equal weight 0.25. Same pattern in most generations — a direct consequence of `topk = 4` out
of `pop = 16` with no fitness weighting. Round 2's top-4 spread at the record generation was 0.104655.

**(e) The center has no elitism and demonstrably regressed.** In round 2, **0 of 16 candidates beat the gen-0 center in
generations 1, 2 and 5**, yet the center still moved to the mean of that generation's best 4 losers. Only **9 of 160
candidates (5.6%)** ever beat the gen-0 center at all; only 4/160 exceeded its 13/14 success count. **The sigma = 0.03
kick is large enough (15% of ‖W_last‖) that the modal perturbation is a regression, so most of the ES budget was spent
recovering from its own noise.**

**(f) Extreme chaotic sensitivity to tiny weight changes.** Perturbations are only 9.5–15.0% of the last-layer weight
norm, yet within a *single* generation candidates span 8/8 solved to 3/8 solved (round 1). Even at the smallest sigma
(g10, 0.018907) round 1's worst candidate scores 0.200187, and the per-generation minimum never rises above 0.375 in
any generation. **A 0.95 decay over 10 generations (1.59× total shrink) is nowhere near enough to make selection
reliable.**

---

## 7. Honest caveats

1. **One evaluation per candidate.** `RULE_EVALS = 1`. In round 2 this is defensible because the harness is
   bit-deterministic (proven, sd = 0.000000), but it means the fitness is one fixed RNG realization of a stochastic
   loop; the same weights score 0.929673 / 0.929095 / 0.824232 across fresh realizations. **In round 1, re-evaluation
   variance cannot be estimated from the data at all** — no weight vector was ever evaluated twice, and the center was
   scored once. That is stated rather than guessed. Round-1 determinism was **not** directly tested; round 1 used
   `RULE_MODE = rule_yield` (a scripted, non-sampling HL) so it is plausibly at least as deterministic, but this is an
   inference, not a measurement.
2. **The center was measured exactly once in each run** (gen 0). The log's `best=` field is the best-ever *candidate*,
   not the center. **Nothing in either run establishes that the ES center ever improved in fitness terms** — the
   population-distribution evidence in §5.1 is the only handle, and it is indirect.
3. **The final center was never saved and never evaluated** in either round. Round 1's generation 10 and round 2's
   generation 10 contributed 16 evaluations each and nothing to the artifact.
4. **Tiny fitness sets with almost no headroom.** 8 episodes (round 1) and 14 (round 2); in *both* cases the warm start
   already solved all but one, so maximum measurable improvement was one episode flip. Round 1's 0.1875 ceiling is
   exactly what it "achieved".
5. **Single seed throughout.** `habitat.seed = 7` for every ES evaluation; `torch.manual_seed(0)` for BC/DAgger. No
   seed sweep exists for any stage. The ES perturbation draws are also unseeded relative to each other in any recorded
   way — the search cannot be replayed candidate-for-candidate.
6. **No held-out set inside the ES loop.** Generalization was only measured *after* both runs finished, which is why
   round 1's regression was not caught until 160 evaluations had been spent.
7. **The BC/DAgger val split is random rows, not held-out episodes**, so BC val MSE is a leaky interpolation metric
   (§1.6). Cross-dataset val-MSE comparison between the two supervised runs is meaningless.
8. **The BC/DAgger curves have 9 points each.** That is the full resolution that exists on disk; a finer curve would
   require editing the trainer and re-running (not done, to avoid touching production reproduction).
9. **400 epochs is an arbitrary stop, not convergence** — both curves were still descending under a constant lr.
10. **Round 2's stated run facts were wrong on disk** and had to be recovered from hydra records (§2.3). A repro using
    the *stated* facts gives center fitness 0.114 instead of 0.827. Anyone reproducing must use the recovered
    invocation. The **round-1 run facts were not independently recovered from hydra**; they are taken from the driver
    defaults and the log, so the round-1 HL mode and config are less firmly established than round 2's.
11. **The "24/28 stable + 3 capability wins" headline for ES v3 could not be reproduced** from the artifacts; the only
    matching evaluation on disk gives 22/28 strict (23/28 at ≥2/3), 29/36 overall (§4.4). The 3 capability wins are
    real but are exactly cancelled by two unreported regressions of equal size.
12. **Aggregate closed-loop numbers hide the collision-metric defect** (§6a). Any "0 collisions" claim in this document
    means zero robot–human collisions and says nothing about scene contact.
13. **Comparisons across HL policies are confounded.** The stage table in §4.2 varies the LL while holding the NR1 HL
    fixed, but the round-2 ES *selected* against that same NR1 HL, so ES v3's train36 numbers are not HL-independent.
14. **`env` is 0 for every row in every demo file**, so episodes cannot be separated in the supervised data; contiguous
    `t` is the only usable segment key, and DAgger rows lack the `wp`/`branch` fields that would allow branch-level
    attribution.

---

## 8. Source artifacts

**Training drivers:** `hrl_pipeline/bc_ll_yield.py`, `hrl_pipeline/es_ll_finetune.py` (neither modified).

**Logs:** `hrl_pipeline/_es_v2.log` (ES r1), `hrl_pipeline/_es_v3.log` (ES r2), `hrl_pipeline/_bc_ll_v2_full.log`,
`hrl_pipeline/_bc_ll_dagger1_full.log`, `hrl_pipeline/_bc_ll_v2_repeat.log` (determinism repeat).

**ES workdirs (root-700; read via `docker exec -u root`):** `/habitat-lab/hrl_pipeline/es_ll_zji8yz5q` (r1, 483 files),
`/habitat-lab/hrl_pipeline/es_ll_jl5ncpsk` (r2, 483 files).

**Datasets:** `hrl_pipeline/ll_demos_v2_clean.jsonl` (15,490), `hrl_pipeline/ll_dagger_r1.jsonl` (15,460),
`hrl_pipeline/ll_train_r1.jsonl` (30,950), `data/social_nav_episode_es8.json.gz`, `data/social_nav_episode_es14.json.gz`.

**Weights:** `ll_yield_bc_v2.pth`, `ll_yield_dagger1.pth`, `ll_yield_es_v2.pth` (≡ `es_ll_zji8yz5q/g3_c9.pth` last
layer), `ll_yield_es_v3.pth` (≡ `es_ll_jl5ncpsk/g4_c11.pth` last layer).

**Post-run evaluations:** `hrl_pipeline/stats_ll36_nr1hl.json` (warm-start baseline),
`hrl_pipeline/stats_esv3_train36_v1.json`, `hrl_pipeline/stats_esv3_holdout2.json`.

**Supporting analysis output:** `hrl_pipeline/_ll_dataset_stats.txt`, `hrl_pipeline/_ll_mse_units.txt`,
`hrl_pipeline/ll_dataset_stats.py`, `hrl_pipeline/ll_mse_units.py`. Round-2 hydra recovery from
`outputs/2026-08-16/08-32-53/.hydra/`.
