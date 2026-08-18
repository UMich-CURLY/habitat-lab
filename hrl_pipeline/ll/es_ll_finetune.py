"""ES fine-tuning driver for the learned yield LL (route 1: RL beyond BC parity).

Black-box policy search on the tiny 22->128->128->2 MLP: perturb weights,
evaluate each candidate in the REAL closed loop via the existing harness
(scripted teacher HL + LearnedYieldSkill; candidate injected via the LL_MODEL
env var), select on episode outcomes. Zero new sim infrastructure.

fitness = mean_success - 0.5*mean_collide - 0.1*mean(steps/step_cap)

Modes:
  --init bc      start from an existing BC checkpoint (fine-tune; default)
  --init random  start from random weights (pure-RL-from-scratch ablation)
  --full         perturb ALL weights (sep-ES); default = last layer only (258p)
  --smoke        pop 2 x 1 gen x STEP_CAP 300 pipeline check

Run inside the container:
  cd /habitat-lab && . activate habitat && \
  python hrl_pipeline/es_ll_finetune.py --episodes-gz data/social_nav_episode_bcsolid6_v2.json.gz \
      --pop 16 --gens 15 --workers 8 --evals 1
Never run concurrently with PPO training (GPU contention).
"""
import argparse
import json
import os
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch

HARNESS = "/habitat-lab/hrl_pipeline/hl/scripted_hl_eval.py"
BASE = "/habitat-lab/hrl_pipeline/weights/ll_yield_bc.pth"
LAST_KEYS = ("net.4.weight", "net.4.bias")   # final Linear before Tanh


def flat_of(sd, keys):
    return torch.cat([sd[k].flatten() for k in keys])


def load_into(sd, keys, vec):
    out = {k: v.clone() for k, v in sd.items()}
    i = 0
    for k in keys:
        n = sd[k].numel()
        out[k] = vec[i:i + n].view_as(sd[k])
        i += n
    return out


def eval_candidate(pth_path, args, tag):
    stats = os.path.join(args.workdir, f"stats_{tag}.json")
    env = dict(os.environ)
    env.update(
        LL_MODEL=pth_path,
        RULE_MODE=args.rule_mode, RULE_DIST=str(args.rule_dist),
        RULE_CONFIG=args.rule_config,
        RULE_DATASET=args.episodes_gz,
        RULE_EVALS=str(args.evals), STEP_CAP=str(args.step_cap),
        EVAL_STATS=stats, MAGNUM_LOG="quiet", HABITAT_SIM_LOG="quiet",
    )
    log = os.path.join(args.workdir, f"log_{tag}.txt")
    with open(log, "w") as lf:
        r = subprocess.run(["python", HARNESS], env=env, stdout=lf,
                           stderr=subprocess.STDOUT, cwd="/habitat-lab")
    if r.returncode != 0 or not os.path.exists(stats):
        return -1.0, {}
    d = json.load(open(stats))
    su = np.mean([v["social_nav_to_pos_success"] for v in d.values()])
    co = np.mean([v["did_collide"] for v in d.values()])
    st = np.mean([v["num_steps"] for v in d.values()]) / args.step_cap
    return float(su - 0.5 * co - 0.1 * st), {"success": float(su), "collide": float(co)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=BASE)
    ap.add_argument("--episodes-gz",
                    default="/habitat-lab/data/archive_20260817/social_nav_episode_bcsolid6_v2.json.gz")
    ap.add_argument("--out", default="/habitat-lab/hrl_pipeline/weights/ll_yield_es.pth")
    ap.add_argument("--pop", type=int, default=16)
    ap.add_argument("--gens", type=int, default=15)
    ap.add_argument("--evals", type=int, default=1)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--sigma", type=float, default=0.03)
    ap.add_argument("--sigma-decay", type=float, default=0.95)
    ap.add_argument("--topk", type=int, default=4)
    ap.add_argument("--step-cap", type=int, default=1200)
    ap.add_argument("--rule-dist", type=float, default=2.0)
    ap.add_argument("--rule-mode", default="rule_yield")
    ap.add_argument("--rule-config",
                    default="social_nav/social_nav_hierarchical_v2_llyield.yaml")
    ap.add_argument("--full", action="store_true", help="perturb all weights (default: last layer)")
    ap.add_argument("--init", choices=["bc", "random"], default="bc")
    ap.add_argument("--smoke", action="store_true")
    # Without this the perturbations come from an unseeded global RNG, so no ES
    # run can be reproduced and repeat runs cannot be compared as replicates.
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()
    if args.seed is not None:
        torch.manual_seed(args.seed)
        print(f"[es] torch seed {args.seed}")
    if args.smoke:
        args.pop, args.gens, args.evals, args.step_cap = 2, 1, 1, 300

    args.workdir = tempfile.mkdtemp(prefix="es_ll_", dir="/habitat-lab/hrl_pipeline")
    print(f"[es] workdir={args.workdir}")

    base = torch.load(args.base, map_location="cpu")
    sd = base["model"]
    if args.init == "random":
        sd = {k: torch.randn_like(v) * 0.1 for k, v in sd.items()}
    keys = list(sd.keys()) if args.full else [k for k in LAST_KEYS if k in sd]
    center = flat_of(sd, keys)
    print(f"[es] perturbing {len(keys)} tensors, {center.numel()} params; "
          f"init={args.init} sigma={args.sigma}")

    # generation 0 reference: the unperturbed center
    ref_pth = os.path.join(args.workdir, "center.pth")
    torch.save({"model": load_into(sd, keys, center), "in_dim": base.get("in_dim", 22)}, ref_pth)
    best_fit, best_info = eval_candidate(ref_pth, args, "center")
    best_vec = center.clone()
    print(f"[es] gen0 center fitness={best_fit:.3f} {best_info}")

    sigma = args.sigma
    for g in range(1, args.gens + 1):
        cands = []
        for i in range(args.pop):
            vec = center + torch.randn_like(center) * sigma
            pth = os.path.join(args.workdir, f"g{g}_c{i}.pth")
            torch.save({"model": load_into(sd, keys, vec), "in_dim": base.get("in_dim", 22)}, pth)
            cands.append((i, vec, pth))
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            fits = list(ex.map(lambda c: eval_candidate(c[2], args, f"g{g}c{c[0]}"), cands))
        scored = sorted(zip(cands, fits), key=lambda t: -t[1][0])
        top = scored[: args.topk]
        center = torch.stack([c[1] for c, _ in top]).mean(0)
        if top[0][1][0] > best_fit:
            best_fit = top[0][1][0]
            best_vec = top[0][0][1].clone()
            best_info = top[0][1][1]
        sigma *= args.sigma_decay
        line = " ".join(f"{f[0]:.3f}" for _, f in scored)
        print(f"[es] gen{g} fits: {line} | best={best_fit:.3f} {best_info} sigma={sigma:.4f}", flush=True)

    torch.save({"model": load_into(sd, keys, best_vec), "in_dim": base.get("in_dim", 22)}, args.out)
    print(f"[es] done. best fitness={best_fit:.3f} {best_info} -> {args.out}")


if __name__ == "__main__":
    main()
