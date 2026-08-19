"""Decompose the BC val MSE per action dimension and convert to physical units.

Reproduces the exact train/val split of bc_ll_yield.py (torch.manual_seed(0)
before net construction and before the randperm), then reports per-dimension
MSE/RMSE and the equivalent physical error given BaseVelAction scaling.
"""
import json
import os

import numpy as np
import torch
import torch.nn as nn

MAX_RANGE = 3.0
LIN_SPEED = 10.0   # longitudinal_lin_speed (m/s) that clip(lin,-1,1) multiplies
ANG_SPEED = 10.0   # ang_speed (rad/s) that clip(ang,-1,1) multiplies
# habitat.simulator.ctrl_freq default = 120.0 (not overridden in
# hssd_spot_human_social_nav_tk.yaml, which sets ac_freq_ratio: 1), so
# BaseVelAction integrates one command for 1/120 s.
CTRL_HZ = 120.0


class YieldNet(nn.Module):
    def __init__(self, in_dim=22, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 2), nn.Tanh(),
        )

    def forward(self, x):
        return self.net(x)


def load(jsonl):
    X, Y = [], []
    for line in open(jsonl):
        r = json.loads(line)
        lidar = np.array(r["lidar"], np.float32) / MAX_RANGE
        feat = np.array(r["feat"], np.float32)
        feat[0] = min(feat[0], 6.0) / 6.0
        feat[1] /= np.pi
        feat[2] /= np.pi
        feat[3] = np.clip(feat[3], -1.5, 1.5) / 1.5
        door = np.array(r["door"], np.float32)
        dn = np.linalg.norm(door)
        door = door / dn if dn > 1e-6 else door
        X.append(np.concatenate([lidar, feat, door]))
        Y.append(np.array(r["act"], np.float32))
    return np.stack(X), np.stack(Y)


def report(tag, jsonl, ckpt):
    X, Y = load(jsonl)
    torch.manual_seed(0)
    net = YieldNet(X.shape[1])
    sd = torch.load(ckpt, map_location="cpu")
    net.load_state_dict(sd["model"])
    net.eval()
    tx, ty = torch.tensor(X), torch.tensor(Y)
    n = len(tx)
    idx = torch.randperm(n)          # NOTE: consumes RNG exactly as trainer does
    val = idx[: n // 10]
    trn = idx[n // 10:]

    print(f"\n===== {tag} =====")
    print(f"ckpt {ckpt}   rows {n}  val {len(val)}  train {len(trn)}")
    with torch.no_grad():
        for split_name, split in (("train", trn), ("val", val)):
            p = net(tx[split]).numpy()
            y = ty[split].numpy()
            err = p - y
            mse_all = float((err ** 2).mean())
            mse_lin = float((err[:, 0] ** 2).mean())
            mse_ang = float((err[:, 1] ** 2).mean())
            print(f"  {split_name}: MSE(both)={mse_all:.6f}  "
                  f"MSE(lin)={mse_lin:.6f}  MSE(ang)={mse_ang:.6f}")
            print(f"     RMSE(lin)={np.sqrt(mse_lin):.6f} (normalized units)  "
                  f"RMSE(ang)={np.sqrt(mse_ang):.6f}")
            print(f"     MAE(lin)={np.abs(err[:,0]).mean():.6f}  "
                  f"MAE(ang)={np.abs(err[:,1]).mean():.6f}")
            print(f"     |err lin| p50={np.percentile(np.abs(err[:,0]),50):.6f} "
                  f"p95={np.percentile(np.abs(err[:,0]),95):.6f} "
                  f"max={np.abs(err[:,0]).max():.6f}")
            print(f"     |err ang| p50={np.percentile(np.abs(err[:,1]),50):.6f} "
                  f"p95={np.percentile(np.abs(err[:,1]),95):.6f} "
                  f"max={np.abs(err[:,1]).max():.6f}")
            # variance-explained vs. predict-the-mean baseline
            for k, nm in ((0, "lin"), (1, "ang")):
                var = float(y[:, k].var())
                mse = float((err[:, k] ** 2).mean())
                print(f"     R^2({nm}) = 1 - {mse:.6f}/{var:.6f} = "
                      f"{1 - mse/var:+.4f}")
            # physical units
            print(f"     PHYSICAL: RMSE(lin) -> {np.sqrt(mse_lin)*LIN_SPEED:.4f} m/s "
                  f"(cmd space x{LIN_SPEED:g});  per 1/{CTRL_HZ:g}s step "
                  f"{np.sqrt(mse_lin)*LIN_SPEED/CTRL_HZ*100:.3f} cm")
            print(f"     PHYSICAL: RMSE(ang) -> {np.sqrt(mse_ang)*ANG_SPEED:.4f} rad/s "
                  f"= {np.degrees(np.sqrt(mse_ang)*ANG_SPEED):.2f} deg/s;  per step "
                  f"{np.degrees(np.sqrt(mse_ang)*ANG_SPEED/CTRL_HZ):.3f} deg")
            # drift if the per-step error were systematic over a 200-step segment
            print(f"     if biased-constant over a 200-step segment: "
                  f"lin drift {np.sqrt(mse_lin)*LIN_SPEED/CTRL_HZ*200:.3f} m, "
                  f"ang drift {np.degrees(np.sqrt(mse_ang)*ANG_SPEED/CTRL_HZ*200):.1f} deg")

    # what does a naive baseline score?
    with torch.no_grad():
        ymean = ty[trn].mean(0)
        base = float(((ty[val] - ymean) ** 2).mean())
    print(f"  predict-train-mean baseline val MSE = {base:.6f}")


def main():
    D = "/habitat-lab/hrl_pipeline/weights"
    report("A: demos-only (BC)", os.path.join(D, "ll_demos_v2_clean.jsonl"),
           os.path.join(D, "_tmp_bc_v2.pth"))
    report("B: demos+DAgger", os.path.join(D, "ll_train_r1.jsonl"),
           os.path.join(D, "_tmp_bc_dagger1.pth"))

    # cross-check: does the retrained ckpt match the production one bitwise?
    for new, prod in ((f"{D}/_tmp_bc_v2.pth", f"{D}/ll_yield_bc_v2.pth"),
                      (f"{D}/_tmp_bc_dagger1.pth", f"{D}/ll_yield_dagger1.pth")):
        a = torch.load(new, map_location="cpu")["model"]
        b = torch.load(prod, map_location="cpu")["model"]
        same = all(torch.equal(a[k], b[k]) for k in a)
        maxd = max(float((a[k] - b[k]).abs().max()) for k in a)
        print(f"\nweights {os.path.basename(new)} vs {os.path.basename(prod)}: "
              f"identical={same}  max|delta|={maxd:.3e}")

    # cross-eval: run the BC-only student on the DAgger states
    Xa, Ya = load(os.path.join(D, "ll_demos_v2_clean.jsonl"))
    Xb, Yb = load(os.path.join(D, "ll_dagger_r1.jsonl"))
    net = YieldNet(Xa.shape[1])
    net.load_state_dict(torch.load(f"{D}/_tmp_bc_v2.pth", map_location="cpu")["model"])
    net.eval()
    with torch.no_grad():
        e = net(torch.tensor(Xb)).numpy() - Yb
    print(f"\n===== BC-only student evaluated on the DAgger-visited states =====")
    print(f"  rows {len(Xb)}  MSE(both)={float((e**2).mean()):.6f}  "
          f"MSE(lin)={float((e[:,0]**2).mean()):.6f}  "
          f"MSE(ang)={float((e[:,1]**2).mean()):.6f}")
    print(f"  (compare: its own in-distribution val MSE from the run log)")
    print(f"  |err ang| p50={np.percentile(np.abs(e[:,1]),50):.6f} "
          f"p95={np.percentile(np.abs(e[:,1]),95):.6f} "
          f"max={np.abs(e[:,1]).max():.6f}")


if __name__ == "__main__":
    main()
