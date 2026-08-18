"""Turn stdout training logs into TensorBoard event files.

Only the PPO trainer writes events; BC / DAgger / evolution-search training all
just print to stdout, so their curves are invisible in TB. This parses those
logs and republishes them under `tb_curves/<name>/`.

Recognised line formats:
  [es] gen0 center fitness=0.827 {'success': 0.93, 'collide': 0.07}
  [es] gen3 fits: ... | best=0.828 {'success': ..., 'collide': ...} sigma=0.0257
  epoch  50  train_mse=0.0074  val_mse=0.0076
  epoch 100  loss/dec=0.1370  acc=0.948

Usage: python hrl_pipeline/logs_to_tb.py <name>=<logfile> [<name>=<logfile> ...]
"""
import ast
import os
import re
import sys

from torch.utils.tensorboard import SummaryWriter

OUT = "/habitat-lab/hrl_pipeline/tb/curves"

RE_ES = re.compile(
    r"\[es\] gen(\d+).*?(?:center fitness|best)=(-?[\d.]+)\s*(\{[^}]*\})"
    r"(?:.*?sigma=([\d.]+))?"
)
RE_EPOCH = re.compile(r"^epoch\s+(\d+)\s+(.*)$")
RE_KV = re.compile(r"([A-Za-z_/]+)=(-?[\d.]+)")
# ES run header / footer: hyperparameters and the produced checkpoint. These live
# ONLY in the log -- neither the trainer nor the ES loop writes them anywhere else.
RE_ES_CFG = re.compile(
    r"\[es\] perturbing (\d+) tensors, (\d+) params; init=(\S+) sigma=([\d.]+)")
RE_ES_SEED = re.compile(r"\[es\] torch seed (\d+)")
RE_ES_DONE = re.compile(r"\[es\] done\. best fitness=(-?[\d.]+) (\{[^}]*\}) -> (\S+)")
# ES center reconstruction (reconstruct_es_center.py) -- its own line shape.
RE_RECON = re.compile(
    r"^\s*gen\s*(\d+):\s*(\d+) cands, top4 fitness \[([\d.,\s-]+)\], center moved ([\d.]+)")
# Dataset-size lines, dropped by every other parser.
RE_LLBC = re.compile(
    r"^demos:\s*(\d+)\s+act lin\[min,max\]=\[(-?[\d.]+),(-?[\d.]+)\] ang=\[(-?[\d.]+),(-?[\d.]+)\]")
RE_HLBC = re.compile(r"^episodes=(\d+) decisions=(\d+) action_dist=(\{[^}]*\})")
# The full CLI of a training run: the ONLY record of the exact hyperparameter
# overrides for several kept checkpoints (driver logs echo it under `set -x`).
RE_CMD = re.compile(r"^\+ python -u -m habitat_baselines\.run (.*)$")


def convert(name, path):
    w = SummaryWriter(os.path.join(OUT, name))
    n = 0
    for line in open(path, errors="ignore"):
        m = RE_ES.search(line)
        if m:
            gen, fit, blob, sigma = m.groups()
            w.add_scalar("es/best_fitness", float(fit), int(gen))
            try:
                d = ast.literal_eval(blob)
                for k, v in d.items():
                    w.add_scalar(f"es/best_{k}", float(v), int(gen))
            except Exception:
                pass
            if sigma:
                w.add_scalar("es/sigma", float(sigma), int(gen))
            # spread of the population this generation
            fits = re.search(r"fits:\s*([-\d.\s]+)\|", line)
            if fits:
                vals = [float(x) for x in fits.group(1).split()]
                w.add_scalar("es/pop_mean", sum(vals) / len(vals), int(gen))
                w.add_scalar("es/pop_max", max(vals), int(gen))
                w.add_scalar("es/pop_min", min(vals), int(gen))
            n += 1
            continue
        m = RE_ES_CFG.search(line)
        if m:
            w.add_scalar("hparams/n_tensors", float(m.group(1)), 0)
            w.add_scalar("hparams/n_params", float(m.group(2)), 0)
            w.add_scalar("hparams/sigma0", float(m.group(4)), 0)
            w.add_text("config/es", line.strip(), 0)
            n += 1
            continue
        m = RE_ES_SEED.search(line)
        if m:
            w.add_scalar("hparams/seed", float(m.group(1)), 0)
            n += 1
            continue
        m = RE_ES_DONE.search(line)
        if m:
            w.add_scalar("es/final_fitness", float(m.group(1)), 0)
            w.add_text("output_checkpoint", m.group(3), 0)
            n += 1
            continue
        m = RE_RECON.match(line)
        if m:
            gen, pop, tops, moved = m.groups()
            w.add_scalar("recon/pop_size", float(pop), int(gen))
            w.add_scalar("recon/center_moved", float(moved), int(gen))
            for i, v in enumerate(
                [float(x) for x in tops.replace(" ", "").split(",") if x], 1
            ):
                w.add_scalar(f"recon/top{i}", v, int(gen))
            n += 1
            continue
        m = RE_LLBC.match(line)
        if m:
            for k, v in zip(
                ("n_demos", "lin_min", "lin_max", "ang_min", "ang_max"), m.groups()
            ):
                w.add_scalar(f"data/{k}", float(v), 0)
            n += 1
            continue
        m = RE_HLBC.match(line)
        if m:
            w.add_scalar("data/n_episodes", float(m.group(1)), 0)
            w.add_scalar("data/n_decisions", float(m.group(2)), 0)
            try:
                for k, v in ast.literal_eval(m.group(3)).items():
                    w.add_scalar(f"data/action_count_{k}", float(v), 0)
            except Exception:
                pass
            n += 1
            continue
        m = RE_CMD.match(line.strip())
        if m:
            w.add_text("config/cli", m.group(1), 0)
            n += 1
            continue
        m = RE_EPOCH.match(line.strip())
        if m:
            ep, rest = m.groups()
            for k, v in RE_KV.findall(rest):
                w.add_scalar(f"train/{k.replace('/', '_')}", float(v), int(ep))
            n += 1
    w.close()
    print(f"{name:28} <- {os.path.basename(path)}   ({n} points)")


if __name__ == "__main__":
    for arg in sys.argv[1:]:
        name, _, path = arg.partition("=")
        convert(name, path)
    print(f"\n-> {OUT}")
