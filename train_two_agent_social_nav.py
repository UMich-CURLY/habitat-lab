#!/usr/bin/env python3
"""
Quick trainer for TwoAgentSocialNavTask-v0.

This script runs a lightweight actor-critic (A2C-style) on the
`hssd_spot_human_social_nav_twoagent.yaml` config to train agent_0's
`agent_0_base_velocity` continuous action to reach its goal while avoiding
agent_1. It's intentionally minimal and meant as a smoke test / baseline.

Notes:
- The observation dict can contain images; this script will flatten only
  numeric array observations and skip high-dim image tensors to keep the
  network small. For a real experiment, build a vision encoder instead.
- Training in Habitat can be slow. Use small `--max-steps` to smoke-test.

Usage:
  python scripts/train_two_agent_social_nav.py --cfg habitat/config/benchmark/multi_agent/hssd_spot_human_social_nav_twoagent.yaml --max-steps 2000

"""
import argparse
import os
import sys
import time
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "habitat-lab"))

import habitat


def obs_to_vec(obs_dict, skip_image_threshold=3):
    """Flatten numeric entries from the observation dict into a 1D numpy array.
    Skip entries with ndim >= skip_image_threshold (likely images).
    """
    parts = []
    for k, v in obs_dict.items():
        try:
            arr = np.array(v)
        except Exception:
            # skip non-numeric
            continue
        if arr.dtype.type is np.str_ or arr.dtype.type is np.object_:
            continue
        # skip large image-like tensors (height,width,channels)
        if getattr(arr, "ndim", 0) >= skip_image_threshold:
            continue
        parts.append(arr.ravel().astype(np.float32))
    if len(parts) == 0:
        return np.zeros(1, dtype=np.float32)
    return np.concatenate(parts, axis=0)


class ActorCritic(nn.Module):
    def __init__(self, input_dim, action_dim=2, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.actor_mean = nn.Linear(hidden, action_dim)
        # independent log std parameter
        self.log_std = nn.Parameter(torch.zeros(action_dim))
        self.critic = nn.Linear(hidden, 1)

    def forward(self, x):
        h = self.net(x)
        mean = self.actor_mean(h)
        std = torch.exp(self.log_std)
        value = self.critic(h).squeeze(-1)
        return mean, std, value


def make_action_from_output(a_mean, a_std):
    # sample gaussian and tanh-clamp to [-1,1]
    noise = torch.randn_like(a_mean)
    action = a_mean + noise * a_std
    action = torch.tanh(action)
    return action.detach().cpu().numpy()


def train(cfg_path, dataset_override=None, max_steps=5000, lr=1e-3, gamma=0.99):
    cfg = habitat.get_config(cfg_path)
    if dataset_override is not None:
        cfg.dataset.data_path = dataset_override

    print("Config dataset:", cfg.dataset.data_path)

    env = habitat.Env(cfg=cfg)

    obs = env.reset()
    vec = obs_to_vec(obs)
    obs_dim = vec.shape[0]
    action_dim = 2  # longitudinal, angular (BaseVelNonCylinderAction)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ActorCritic(obs_dim, action_dim).to(device)
    optimz = optim.Adam(model.parameters(), lr=lr)

    state = vec
    ep_rewards = deque(maxlen=50)

    # simple on-policy loop: collect trajectory until batch_steps then update
    batch_states = []
    batch_actions = []
    batch_logps = []
    batch_returns = []
    batch_values = []

    step_count = 0
    episode_reward = 0.0
    episode_len = 0
    start_time = time.time()

    while step_count < max_steps:
        state_t = torch.from_numpy(state).float().to(device)
        mean, std, value = model(state_t.unsqueeze(0))
        mean = mean.squeeze(0)
        std = std

        action = make_action_from_output(mean, std)

        # build habitat action
        action_args = {"agent_0_base_vel": action}
        act = {"action": "agent_0_base_velocity", "action_args": action_args}

        next_obs = env.step(act)
        reward = float(env.task.measurements.measures.get("social_nav_reward", None).get_metric() if "social_nav_reward" in env.task.measurements.measures else 0.0)
        # fallback: try env.get_metrics
        try:
            metrics = env.get_metrics()
            if isinstance(metrics, dict) and "social_nav_reward" in metrics:
                reward = float(metrics["social_nav_reward"])
        except Exception:
            pass

        next_state = obs_to_vec(next_obs)

        batch_states.append(state)
        batch_actions.append(action)
        # compute log prob under gaussian
        a_mean = mean.detach().cpu()
        a_std = std.detach().cpu()
        var = (a_std ** 2)
        logp = -0.5 * (((torch.from_numpy(action) - a_mean) ** 2) / var).sum().item()
        batch_logps.append(logp)
        batch_values.append(value.item())
        batch_returns.append(reward)

        episode_reward += reward
        episode_len += 1
        step_count += 1

        if step_count % 50 == 0:
            print(f"step {step_count}  recent ep rew {np.mean(list(ep_rewards)) if len(ep_rewards)>0 else episode_reward:.3f}  episode_len {episode_len}")

        # end of episode handling
        if env.episode_over:
            ep_rewards.append(episode_reward)
            obs = env.reset()
            state = obs_to_vec(obs)
            episode_reward = 0.0
            episode_len = 0
            continue
        else:
            state = next_state

        # simple update every 256 steps
        if len(batch_states) >= 256:
            returns = []
            G = 0.0
            for r in reversed(batch_returns):
                G = r + gamma * G
                returns.insert(0, G)
            returns = torch.tensor(returns, dtype=torch.float32).to(device)
            states = torch.tensor(np.array(batch_states), dtype=torch.float32).to(device)
            actions = torch.tensor(np.array(batch_actions), dtype=torch.float32).to(device)
            values = torch.tensor(batch_values, dtype=torch.float32).to(device)
            logps = torch.tensor(batch_logps, dtype=torch.float32).to(device)

            # compute current values and logps
            means, stds, vals = model(states)
            # compute gaussian log probs
            var = stds ** 2
            dist_logps = -0.5 * (((actions - means) ** 2) / var).sum(dim=1)
            advantages = returns - vals.detach()

            actor_loss = -(dist_logps * advantages).mean()
            critic_loss = 0.5 * ((returns - vals) ** 2).mean()
            loss = actor_loss + critic_loss * 0.5 - 0.001 * dist_logps.mean()

            optimz.zero_grad()
            loss.backward()
            optimz.step()

            batch_states = []
            batch_actions = []
            batch_logps = []
            batch_returns = []
            batch_values = []

    env.close()
    print("Training finished in", time.time() - start_time)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--cfg", default="habitat/config/benchmark/multi_agent/hssd_spot_human_social_nav_twoagent.yaml")
    p.add_argument("--dataset", default=None)
    p.add_argument("--max-steps", type=int, default=2000)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args.cfg, dataset_override=args.dataset, max_steps=args.max_steps)
