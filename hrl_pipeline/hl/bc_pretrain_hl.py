"""BC-pretrain the SocialNav HL policy on scripted-teacher decisions.

PPO-from-scratch on ep70 converges to an avoid-timeout local optimum (never
discovers the go->yield->go sequence). This behaviour-clones the scripted
teacher into the SAME modules the neural HL uses (GRU state encoder
+ CategoricalNet head), so PPO can fine-tune from a policy that already yields.

Input:  a hl_decisions_*.jsonl ({feat[6], action, mask, ...}; rule_markov logs
        additionally carry "approach" and "prev_action", which are appended to
        the input exactly like the online policy does: [feat6, approach,
        onehot3(prev_action)] -> width 10, matching input_feature_dim=7 +
        use_prev_action=true. prev_action=-1 (episode start) -> zero one-hot.)
Output: {"state_encoder":..., "policy":...} for pretrained_hl_weights.

Env overrides: BC_JSONL, BC_OUT.
Run in the container with LD_LIBRARY_PATH=/opt/conda/envs/habitat/lib.
"""
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

from habitat_baselines.rl.models.rnn_state_encoder import build_rnn_state_encoder
from habitat_baselines.utils.common import CategoricalNet

# v2 hl_action_names: index 0=backoff(yield), 1=wait, 2=go_to_goal.
ACT2IDX = {"backoff": 0, "wait": 1, "go_to_goal": 2}
N_ACTIONS = 3
HIDDEN = 32
JSONL = os.environ.get(
    "BC_JSONL", "/habitat-lab/hrl_pipeline/demos/hl_decisions_rule_markov_all.jsonl"
)
OUT = os.environ.get("BC_OUT", "/habitat-lab/hrl_pipeline/weights/bc_markov_hl.pth")


def _row_input(d, use_approach, use_prev):
    x = list(d["feat"][:6])
    if use_approach:
        x.append(float(d["approach"]))
    if use_prev:
        onehot = [0.0] * N_ACTIONS
        pa = int(d["prev_action"])
        if 0 <= pa < N_ACTIONS:
            onehot[pa] = 1.0
        x += onehot
    return x


def load_episodes(path):
    """Split the decision log into episodes (mask==0 marks a new episode).
    Input width auto-follows the log: +approach and +prev_action one-hot when
    those (additive) keys are present."""
    rows = [json.loads(line) for line in open(path)]
    use_approach = all("approach" in d for d in rows)
    use_prev = all("prev_action" in d for d in rows)
    eps, cur = [], []
    for d in rows:
        if float(d["mask"]) == 0.0 and cur:
            eps.append(cur)
            cur = []
        cur.append((_row_input(d, use_approach, use_prev), ACT2IDX[d["action"]]))
    if cur:
        eps.append(cur)
    in_dim = 6 + (1 if use_approach else 0) + (N_ACTIONS if use_prev else 0)
    print(f"input: approach={use_approach} prev_action={use_prev} dim={in_dim}")
    return eps, in_dim


def main():
    eps, in_dim = load_episodes(JSONL)
    n_dec = sum(len(e) for e in eps)
    from collections import Counter
    dist = Counter(a for e in eps for _, a in e)
    print(f"episodes={len(eps)} decisions={n_dec} action_dist={dict(dist)}")
    if len(eps) < 3 or n_dec < 50:
        print("too few decisions for BC", file=sys.stderr)

    torch.manual_seed(0)
    se = build_rnn_state_encoder(in_dim, HIDDEN, rnn_type="GRU", num_layers=1)
    head = CategoricalNet(HIDDEN, N_ACTIONS)
    opt = torch.optim.Adam(list(se.parameters()) + list(head.parameters()), lr=1e-3)
    # Inverse-frequency class weights: backoff is ~20% of decisions but the
    # safety-critical class -- unweighted CE biases the argmax toward go at the
    # trigger boundary, which closed-loop turns into drive-into-human.
    counts = torch.tensor(
        [max(dist.get(a, 0), 1) for a in range(N_ACTIONS)], dtype=torch.float32
    )
    class_w = counts.sum() / (N_ACTIONS * counts)

    # Pre-pack episodes as tensors.
    packed = [
        (
            torch.tensor([f for f, _ in e], dtype=torch.float32),   # [T,in_dim]
            torch.tensor([a for _, a in e], dtype=torch.long),      # [T]
        )
        for e in eps
    ]

    EPOCHS = int(os.environ.get("BC_EPOCHS", "1000"))
    for epoch in range(EPOCHS):
        if epoch == EPOCHS // 2:
            for g in opt.param_groups:
                g["lr"] = 2e-4
        opt.zero_grad()
        total_loss, total_correct, total = 0.0, 0, 0
        for feats, acts in packed:
            T = feats.shape[0]
            h0 = torch.zeros(1, 1, HIDDEN)
            out, _ = se.rnn(feats.view(T, 1, in_dim), h0)     # [T,1,H]
            logits = head.linear(out.squeeze(1))              # [T,3]
            loss = F.cross_entropy(
                logits, acts, weight=class_w, reduction="sum"
            )
            loss.backward()
            total_loss += float(loss.item())
            total_correct += int((logits.argmax(-1) == acts).sum().item())
            total += T
        opt.step()
        if epoch % 100 == 0 or epoch == EPOCHS - 1:
            print(f"epoch {epoch:4d}  loss/dec={total_loss/total:.4f}  "
                  f"acc={total_correct/total:.3f}")

    torch.save({"state_encoder": se.state_dict(), "policy": head.state_dict()}, OUT)
    print("saved ->", OUT)


if __name__ == "__main__":
    main()
