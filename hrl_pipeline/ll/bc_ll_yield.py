"""Distill the scripted yield LL into an observation-only policy (L3).

Input rows (hrl_pipeline/ll_demos.jsonl, one per env step of the scripted
ClearCorridorYieldSkill): lidar[16] + human polar feat[4] + door dir[2] -> act
[lin, ang]. The privileged pocket planner is thereby baked into a reactive net
that only sees deployment-legal observations.

Net: MLP 22 -> 128 -> 128 -> 2 (tanh head; lin/ang are already in [-1, 1]).
Output: hrl_pipeline/ll_yield_bc.pth  {"model": state_dict, "norm": {...}}
"""
import json

import numpy as np
import torch
import torch.nn as nn

import os
JSONL = os.environ.get("LL_DATA", "/habitat-lab/hrl_pipeline/demos/ll_demos_v2_clean.jsonl")
OUT = os.environ.get("LL_OUT", "/habitat-lab/hrl_pipeline/weights/ll_yield_bc.pth")
MAX_RANGE = 3.0


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


def load():
    X, Y = [], []
    for line in open(JSONL):
        r = json.loads(line)
        lidar = np.array(r["lidar"], np.float32) / MAX_RANGE     # [0,1]
        feat = np.array(r["feat"], np.float32)
        feat[0] = min(feat[0], 6.0) / 6.0                        # dist
        feat[1] /= np.pi                                         # bearing
        feat[2] /= np.pi                                         # rel heading
        feat[3] = np.clip(feat[3], -1.5, 1.5) / 1.5              # rel speed
        door = np.array(r["door"], np.float32)
        dn = np.linalg.norm(door)
        door = door / dn if dn > 1e-6 else door                  # unit dir
        X.append(np.concatenate([lidar, feat, door]))
        Y.append(np.array(r["act"], np.float32))
    return np.stack(X), np.stack(Y)


def main():
    X, Y = load()
    print(f"demos: {len(X)}  act lin[min,max]=[{Y[:,0].min():.2f},{Y[:,0].max():.2f}] "
          f"ang=[{Y[:,1].min():.2f},{Y[:,1].max():.2f}]")
    torch.manual_seed(0)
    net = YieldNet(X.shape[1])
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    tx, ty = torch.tensor(X), torch.tensor(Y)
    n = len(tx)
    idx = torch.randperm(n)
    val = idx[: n // 10]
    trn = idx[n // 10:]
    for epoch in range(400):
        perm = trn[torch.randperm(len(trn))]
        for i in range(0, len(perm), 4096):
            b = perm[i:i + 4096]
            loss = ((net(tx[b]) - ty[b]) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        if epoch % 50 == 0 or epoch == 399:
            with torch.no_grad():
                vl = ((net(tx[val]) - ty[val]) ** 2).mean().item()
                tl = ((net(tx[trn]) - ty[trn]) ** 2).mean().item()
            print(f"epoch {epoch:3d}  train_mse={tl:.4f}  val_mse={vl:.4f}")
    torch.save({"model": net.state_dict(), "in_dim": X.shape[1]}, OUT)
    print("saved ->", OUT)


if __name__ == "__main__":
    main()
