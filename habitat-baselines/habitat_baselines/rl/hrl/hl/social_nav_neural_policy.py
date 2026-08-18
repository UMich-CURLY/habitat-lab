"""
Hierarchical policy for social navigation with Spot robot.

This module implements a neural high-level policy for a Spot robot that must 
navigate from one side of a door to the other while interacting with a human.
The policy learns to choose from three discrete actions:
  - "go_to_wait_pos": navigate to a waiting position near the human
  - "wait": remain at current position
  - "go_to_goal": navigate directly to the goal

Input features (6-D, all in the robot's local frame; from the
`social_nav_policy_state` sensor):
  - human_dist, human_bearing: polar position of the human w.r.t. the robot
  - human_rel_heading: human heading relative to the robot heading
  - human_rel_speed: speed of the human relative to the robot
  - goal_dist, goal_bearing: polar position of the nav goal w.r.t. the robot
"""

import logging
from itertools import chain
from typing import Any, Dict, List, Optional

import gym.spaces as spaces
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from habitat.tasks.rearrange.multi_task.pddl_domain import PddlProblem
from habitat_baselines.common.logging import baselines_logger
from habitat_baselines.rl.hrl.hl.high_level_policy import HighLevelPolicy
from habitat_baselines.rl.models.rnn_state_encoder import (
    build_rnn_state_encoder,
)
from habitat_baselines.rl.ppo.policy import (
    CriticHead,
    PolicyActionData,
    get_aux_modules,
)
from habitat_baselines.utils.common import CategoricalNet


class SocialNavNeuralHighLevelPolicy(HighLevelPolicy):
    """
    A neural high-level policy for social navigation tasks.
    
    This policy uses relative human state features (position, velocity, orientation)
    and robot orientation to select from discrete high-level actions:
      0: go_to_wait_pos  (navigate to a safe distance from human)
      1: wait            (stay in place)
      2: go_to_goal      (navigate to goal destination)
    
    The policy learns when it's safe to move toward the goal vs. when to wait
    or move to a safer position near the human.
    """

    def __init__(
        self,
        config,
        pddl_problem: PddlProblem,
        num_envs: int,
        skill_name_to_idx: Dict[str, int],
        observation_space: spaces.Space,
        action_space,
        aux_loss_config,
        agent_name: Optional[str] = None,
        pddl_action_name_to_skill_name=None,
    ):
        super().__init__(
            config,
            pddl_problem,
            num_envs,
            skill_name_to_idx,
            observation_space,
            action_space,
            aux_loss_config,
            agent_name,
            pddl_action_name_to_skill_name,
        )

        # Define the discrete high-level action space. Names are config-driven
        # (default reproduces the legacy backoff/wait/go_to_goal mapping); v2
        # uses "yield" for index 0 to match ClearCorridorYieldSkill in the yaml.
        hl_action_names = list(
            getattr(config, "hl_action_names", ["backoff", "wait", "go_to_goal"])
        )
        self._skill_to_action = {
            name: idx for idx, name in enumerate(hl_action_names)
        }
        self._action_to_skill = {v: k for k, v in self._skill_to_action.items()}
        self._n_actions = len(self._skill_to_action)

        # Feature dimension of the social_nav_policy_state sensor (6 legacy,
        # 7 with include_approach_speed). With use_prev_action the previous HL
        # choice is appended as a one-hot BEFORE the GRU, so the recurrent
        # state can integrate the action history (this is what makes the
        # teacher's hysteresis tick counters recoverable by the student).
        self._input_feature_dim = int(
            getattr(self._config, "input_feature_dim", 6)
        )
        self._use_prev_action = bool(
            getattr(self._config, "use_prev_action", False)
        )
        self._hidden_size = getattr(self._config, "hidden_dim", 256)

        # Build the neural network
        self._state_encoder = build_rnn_state_encoder(
            self._input_feature_dim
            + (self._n_actions if self._use_prev_action else 0),
            self._hidden_size,
            rnn_type=getattr(self._config, "rnn_type", "GRU"),
            num_layers=getattr(self._config, "num_rnn_layers", 2),
        )

        # Optional egocentric top-down map branch (route 3 / S7-lite).
        # POST-GRU FUSION: the 6-D->hidden GRU is untouched (so scalar-BC
        # weights load verbatim); the map feature is concatenated AFTER the
        # recurrent encoder and feeds the heads only. Map columns of both
        # heads are ZERO-INITIALISED, so at init the policy is functionally
        # identical to the scalar policy and the proven BC+PPO recipe applies.
        self._map_key = str(getattr(config, "map_obs_key", "") or "")
        self._map_dim = 0
        obs_sp = getattr(observation_space, "spaces", {}) or {}
        map_space_key = None
        for cand in (self._map_key, f"agent_0_{self._map_key}"):
            if cand and cand in obs_sp:
                map_space_key = cand
                break
        if self._map_key and map_space_key is not None:
            from gym import spaces as _spaces
            from habitat_baselines.rl.ddppo.policy import resnet as _resnet
            from habitat_baselines.rl.ddppo.policy.resnet_policy import (
                ResNetEncoder,
            )

            self._map_space_key = map_space_key
            self._map_encoder = ResNetEncoder(
                _spaces.Dict({map_space_key: obs_sp[map_space_key]}),
                baseplanes=32,
                ngroups=16,
                make_backbone=getattr(_resnet, "resnet18"),
            )
            self._map_dim = int(getattr(config, "map_feature_dim", 64))
            self._map_fc = nn.Sequential(
                nn.Flatten(),
                nn.Linear(
                    int(np.prod(self._map_encoder.output_shape)), self._map_dim
                ),
                nn.ReLU(),
            )

        head_in = self._hidden_size + self._map_dim
        # Policy head (outputs action logits)
        self._policy = CategoricalNet(head_in, self._n_actions)

        # Value head (critic)
        self._critic = CriticHead(head_in)
        if self._map_dim:
            with torch.no_grad():
                self._policy.linear.weight[:, self._hidden_size:].zero_()
                self._critic.fc.weight[:, self._hidden_size:].zero_()

        # Aux modules (e.g., for auxiliary losses)
        self.aux_modules = get_aux_modules(aux_loss_config, action_space, self)
        self._debug_call_count = 0
        # Critic warmup: for the first N evaluate_actions calls the actor is
        # held still (detached) while the critic head fits returns. The critic
        # is deliberately random after a BC warm start; without this, the
        # normalized advantages it produces destroy the cloned actor within a
        # few dozen updates (train36: success 0.84 -> 0.25 by update 300).
        self._critic_warmup_calls = int(
            getattr(self._config, "critic_warmup_calls", 0)
        )
        self._eval_calls = 0
        # BC anchor (L2-SP): penalize actor drift from the BC warm start.
        # PPO's per-decision credit assignment rewards trimming yield steps
        # while the (later) collision cost lands on other decisions, so the
        # policy slides monotonically toward collide-short episodes (train36
        # runs c+d followed the same slide under different reward scales). A
        # weight-space trust region keeps the actor near the cloned optimum
        # while still letting small local improvements through.
        self._bc_anchor_coef = float(
            getattr(self._config, "bc_anchor_coef", 0.0)
        )
        # Critic output scale (DIAGNOSTIC probe, default 1.0 = off). CriticHead
        # is orthogonal-init'd (|w|=1, bias=0), so with hidden activations of
        # magnitude ~1.7 its reachable output at init is ~±1.7 -- while returns
        # here are ~±40. Closing that gap requires growing |w| ~20x, and Adam
        # moves each weight by ~lr per step, so it needs O(1e5) gradient steps
        # (observed: V_pred crawled 0 -> 6.7 over 11.7k steps). Scaling the head
        # output puts the target inside the init-reachable range, so the critic
        # only has to ROTATE, not grow. If this makes the critic fit, the
        # bottleneck is value SCALE (fix properly with a critic-specific lr or
        # value/return normalization), not the observation's information.
        self._critic_scale = float(
            getattr(self._config, "critic_output_scale", 1.0)
        )
        # Captured lazily on the first evaluate_actions call, i.e. AFTER the
        # pretrained BC weights below have been loaded.
        self._anchor_ref = None
        # DAPG: success-only demonstration anchor. Unlike the L2-SP weight
        # anchor above, this penalizes drift in ACTION space: a weak, linearly
        # decaying CE term on teacher-success demos joins the PPO loss through
        # the aux channel. Skipped during critic warmup (its independent
        # forward would bypass the features.detach() actor freeze). The decay
        # clock counts evaluate_actions calls, same convention as
        # critic_warmup_calls, starting when warmup ends.
        self._dapg_coef = float(getattr(self._config, "dapg_coef", 0.0))
        self._dapg_decay_calls = int(
            getattr(self._config, "dapg_decay_calls", 240)
        )
        self._dapg_batch_eps = int(getattr(self._config, "dapg_batch_eps", 8))
        self._dapg_eps: List[Any] = []
        self._dapg_class_w = None
        dapg_path = str(getattr(self._config, "dapg_demo_path", "") or "")
        if dapg_path and self._dapg_coef > 0:
            if self._map_dim:
                raise ValueError(
                    "DAPG demos carry no map observation; it only supports "
                    "the scalar policy (disable map_obs_key or dapg_demo_path)"
                )
            self._load_dapg_demos(dapg_path)

        # Optional BC warm-start: load a state_encoder+policy state_dict cloned
        # from the rule_yield teacher, so PPO fine-tunes from a policy that
        # already yields instead of exploring go->yield->go from scratch. The
        # critic is left random (PPO learns it). Default "" = no warm-start.
        pretrained = str(getattr(self._config, "pretrained_hl_weights", "") or "")
        if pretrained:
            import os

            if os.path.exists(pretrained):
                d = torch.load(pretrained, map_location="cpu")
                self._state_encoder.load_state_dict(d["state_encoder"])
                pol_w = d["policy"]["linear.weight"]
                if self._map_dim and pol_w.shape[1] == self._hidden_size:
                    # Scalar checkpoint -> map-augmented head: copy the scalar
                    # columns, keep the zero-initialised map columns.
                    with torch.no_grad():
                        self._policy.linear.weight[:, : self._hidden_size].copy_(pol_w)
                        self._policy.linear.bias.copy_(d["policy"]["linear.bias"])
                else:
                    # Width matches this head (scalar->scalar, or a
                    # map-augmented checkpoint) -> load verbatim.
                    self._policy.load_state_dict(d["policy"])
                if self._map_dim and "map_encoder" in d:
                    self._map_encoder.load_state_dict(d["map_encoder"])
                    self._map_fc.load_state_dict(d["map_fc"])
                baselines_logger.info(
                    f"Loaded BC-pretrained HL weights from {pretrained}"
                )
            else:
                baselines_logger.warning(
                    f"pretrained_hl_weights not found: {pretrained}"
                )

        baselines_logger.info(
            f"SocialNavNeuralHighLevelPolicy initialized with {self._n_actions} actions: "
            f"{list(self._action_to_skill.values())} "
            f"(0=backoff, 1=wait, 2=go_to_goal)"
        )

    @property
    def should_load_agent_state(self) -> bool:
        """Always load and save agent state for this learned policy."""
        return True

    @property
    def policy_action_space(self) -> spaces.Space:
        """
        The stored action is the discrete skill choice, held as a single float
        in a width-1 Box. We use a float Box (rather than ``spaces.Discrete``) so
        the rollout action buffer is float and concatenates cleanly with the
        partner agent's continuous action buffer in the multi-agent storage.
        ``evaluate_actions`` casts the stored value back to a long index.
        """
        return spaces.Box(
            low=0.0,
            high=float(self._n_actions - 1),
            shape=(1,),
            dtype=np.float32,
        )

    @property
    def num_recurrent_layers(self) -> int:
        """Return number of RNN layers."""
        return self._state_encoder.num_recurrent_layers

    @property
    def recurrent_hidden_size(self) -> int:
        """Return RNN hidden state size."""
        return self._hidden_size

    @property
    def hidden_state_shape(self):
        """Return the shape of the hidden state (num_layers, hidden_size)."""
        return (self.num_recurrent_layers, self.recurrent_hidden_size)

    @property
    def hidden_state_shape_lens(self):
        """Return list of dimensions for hidden state shape."""
        return [self.recurrent_hidden_size]

    def parameters(self):
        """Yield all learnable parameters."""
        mods = [
            self._state_encoder.parameters(),
            self._policy.parameters(),
            self._critic.parameters(),
            self.aux_modules.parameters(),
        ]
        if self._map_dim:
            mods += [self._map_encoder.parameters(), self._map_fc.parameters()]
        return chain(*mods)

    def get_policy_components(self) -> List[nn.Module]:
        """The torch modules that make up the HL policy (used for PPO)."""
        return [self]

    def to(self, device):
        """Move all modules to device."""
        self._device = device
        self._state_encoder.to(device)
        self._policy.to(device)
        self._critic.to(device)
        self.aux_modules.to(device)
        if self._map_dim:
            self._map_encoder.to(device)
            self._map_fc.to(device)
        return self

    def _map_features(self, observations):
        """Encode the egocentric top-down map (or None when the branch is off)."""
        if self._map_dim == 0:
            return None
        m = observations.get(self._map_space_key)
        if m is None:
            m = observations.get(self._map_key)
        if m is None:
            raise KeyError(
                f"map_obs_key '{self._map_key}' not found in observations "
                f"(keys={sorted(observations.keys())})"
            )
        dev = next(self._map_encoder.parameters()).device
        x = (m.float() / 255.0).to(dev)
        feat = self._map_encoder({self._map_space_key: x})
        return self._map_fc(feat)

    def _load_dapg_demos(self, path: str):
        """Parse an hl_decisions jsonl into per-episode (inputs, actions).

        Row format matches the online input exactly: [feat6, approach,
        onehot3(prev_action)]; mask==0 marks an episode start. Every row must
        carry approach+prev_action (a mixed legacy log would silently change
        the input width, so it is an error here, not a fallback).
        """
        import json

        rows = [json.loads(line) for line in open(path)]
        if not rows:
            raise ValueError(f"dapg_demo_path is empty: {path}")
        if not all("approach" in d and "prev_action" in d for d in rows):
            raise ValueError(
                f"dapg demos must all carry approach+prev_action: {path}"
            )
        in_dim = self._input_feature_dim + (
            self._n_actions if self._use_prev_action else 0
        )
        if in_dim != 6 + 1 + self._n_actions:
            raise ValueError(
                f"DAPG expects the 7+prev_action input (dim 10), policy has {in_dim}"
            )
        eps, cur = [], []
        for d in rows:
            if float(d["mask"]) == 0.0 and cur:
                eps.append(cur)
                cur = []
            x = list(d["feat"][:6]) + [float(d["approach"])]
            onehot = [0.0] * self._n_actions
            pa = int(d["prev_action"])
            if 0 <= pa < self._n_actions:
                onehot[pa] = 1.0
            cur.append((x + onehot, self._skill_to_action[d["action"]]))
        if cur:
            eps.append(cur)
        self._dapg_eps = [
            (
                torch.tensor([f for f, _ in e], dtype=torch.float32),
                torch.tensor([a for _, a in e], dtype=torch.long),
            )
            for e in eps
        ]
        counts = torch.zeros(self._n_actions)
        for _, acts in self._dapg_eps:
            counts += torch.bincount(acts, minlength=self._n_actions).float()
        counts = counts.clamp(min=1.0)
        # Same inverse-frequency weighting as bc_pretrain_hl.py: backoff/wait
        # are rare but safety-critical; unweighted CE under-anchors them.
        self._dapg_class_w = counts.sum() / (self._n_actions * counts)
        baselines_logger.info(
            f"DAPG demos: {len(self._dapg_eps)} episodes, "
            f"{int(counts.sum())} decisions, action counts {counts.tolist()}, "
            f"coef={self._dapg_coef} decay_calls={self._dapg_decay_calls} "
            f"batch_eps={self._dapg_batch_eps}"
        )

    def forward(
        self,
        observations: Dict[str, torch.Tensor],
        rnn_hidden_states: torch.Tensor,
        masks: torch.Tensor,
        rnn_build_seq_info=None,
        prev_actions: Optional[torch.Tensor] = None,
    ):
        """
        Encode the social-nav features through the recurrent state encoder.

        Used for both single-step value queries and sequence-based
        `evaluate_actions` during the PPO update. ``rnn_hidden_states`` is in the
        ``[batch, num_layers, hidden]`` layout expected by ``RNNStateEncoder``.
        """
        features = self._extract_social_nav_features(
            observations, prev_actions, masks
        )
        features = features.to(rnn_hidden_states.device)
        features, rnn_hidden_states = self._state_encoder(
            features, rnn_hidden_states, masks, rnn_build_seq_info
        )
        mf = self._map_features(observations)
        if mf is not None:
            features = torch.cat([features, mf.to(features.device)], dim=-1)
        return features, rnn_hidden_states

    def get_value(self, observations, rnn_hidden_states, prev_actions, masks):
        """Critic value estimate for the bootstrap at the end of a rollout."""
        features, _ = self.forward(
            observations, rnn_hidden_states, masks, prev_actions=prev_actions
        )
        return self._critic(features) * self._critic_scale

    def evaluate_actions(
        self,
        observations,
        rnn_hidden_states,
        prev_actions,
        masks,
        action,
        rnn_build_seq_info,
    ):
        """
        Re-evaluate the stored high-level skill choices under the current policy
        for the PPO update.

        :param action: The stored discrete skill indices, shape [T*N, 1].
        :returns: (value, action_log_probs, entropy, rnn_hidden_states, aux_losses)
        """
        features, rnn_hidden_states = self.forward(
            observations,
            rnn_hidden_states,
            masks,
            rnn_build_seq_info,
            prev_actions=prev_actions,
        )
        self._eval_calls += 1
        if self._eval_calls <= self._critic_warmup_calls:
            # Warmup: actor terms detached (zero policy gradient); the value
            # loss reaches only the critic head (features detached, so the
            # shared GRU encoder stays frozen too).
            features = features.detach()
            distribution = self._policy(features)
            value = self._critic(features) * self._critic_scale
            action_log_probs = distribution.log_probs(action.long()).detach()
            distribution_entropy = distribution.entropy().detach()
        else:
            distribution = self._policy(features)
            value = self._critic(features) * self._critic_scale
            # The skill choice is stored as a float (width-1 Box); cast back
            # to a discrete index for the categorical log-prob.
            action_log_probs = distribution.log_probs(action.long())
            distribution_entropy = distribution.entropy()
        aux_loss_res = {
            k: v(features, observations) for k, v in self.aux_modules.items()
        }
        if self._bc_anchor_coef > 0:
            if self._anchor_ref is None:
                self._anchor_ref = {
                    k: v.detach().clone()
                    for k, v in list(self._state_encoder.named_parameters())
                    + list(self._policy.named_parameters())
                }
            drift = sum(
                (p - self._anchor_ref[k].to(p.device)).pow(2).sum()
                for k, p in list(self._state_encoder.named_parameters())
                + list(self._policy.named_parameters())
            )
            aux_loss_res["bc_anchor"] = {
                "loss": self._bc_anchor_coef * drift
            }
        if self._dapg_eps and self._eval_calls > self._critic_warmup_calls:
            t = self._eval_calls - self._critic_warmup_calls
            lam = self._dapg_coef * max(
                0.0, 1.0 - (t - 1) / self._dapg_decay_calls
            )
            if lam > 0.0:
                dev = next(self._policy.parameters()).device
                if self._dapg_eps[0][0].device != dev:
                    self._dapg_eps = [
                        (f.to(dev), a.to(dev)) for f, a in self._dapg_eps
                    ]
                    self._dapg_class_w = self._dapg_class_w.to(dev)
                n = len(self._dapg_eps)
                k = min(self._dapg_batch_eps, n)
                # Deterministic round-robin over demo episodes: every episode
                # recurs with the same cadence, no RNG state touched.
                idxs = [((t - 1) * k + j) % n for j in range(k)]
                logits_l, acts_l = [], []
                n_layers = self._state_encoder.rnn.num_layers
                for i in idxs:
                    feats, acts = self._dapg_eps[i]
                    h0 = torch.zeros(
                        n_layers, 1, self._hidden_size, device=dev
                    )
                    out, _ = self._state_encoder.rnn(feats.unsqueeze(1), h0)
                    logits_l.append(self._policy.linear(out.squeeze(1)))
                    acts_l.append(acts)
                logits = torch.cat(logits_l)
                acts = torch.cat(acts_l)
                ce = F.cross_entropy(logits, acts, weight=self._dapg_class_w)
                with torch.no_grad():
                    acc = (logits.argmax(-1) == acts).float().mean()
                aux_loss_res["dapg"] = {
                    "loss": lam * ce,
                    "coef": torch.tensor(lam, device=dev),
                    "ce": ce.detach(),
                    "acc": acc,
                }
        return (
            value,
            action_log_probs,
            distribution_entropy,
            rnn_hidden_states,
            aux_loss_res,
        )

    def get_termination(
        self,
        observations: Dict[str, torch.Tensor],
        rnn_hidden_states: torch.Tensor,
        prev_actions: torch.Tensor,
        masks: torch.Tensor,
        cur_skills: np.ndarray,
        log_info: List[Dict[str, Any]],
    ) -> torch.BoolTensor:
        """
        Determine if the current skill should terminate.

        For social nav, we check if a termination sensor is available.
        Otherwise, fall back to parent implementation.
        """
        termination_obs_name = getattr(
            self._config, "termination_obs_name", None
        )
        if termination_obs_name is not None and termination_obs_name in observations:
            return (observations[termination_obs_name] > 0.0).view(-1).cpu()

        # Fall back to parent implementation (typically checks PDDL success)
        return super().get_termination(
            observations, rnn_hidden_states, prev_actions, masks, cur_skills, log_info
        )

    def _extract_social_nav_features(
        self,
        observations: Dict[str, torch.Tensor],
        prev_actions: Optional[torch.Tensor] = None,
        masks: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Read the robot-frame feature vector from the SocialNavPolicyState
        sensor (``social_nav_policy_state`` / ``agent_0_social_nav_policy_state``):

            [human_dist, human_bearing, human_rel_heading, human_rel_speed,
             goal_dist, goal_bearing (, human_approach_speed)]

        All quantities are already expressed in the robot's local frame (see
        ``SocialNavPolicyStateSensor``), so the policy input is invariant to the
        robot's absolute world pose. With ``use_prev_action`` the previous HL
        choice is appended as a masked one-hot (all-zero on episode starts,
        matching the jsonl convention prev_action=-1).
        """
        device = list(observations.values())[0].device

        feat = None
        for k in ("agent_0_social_nav_policy_state", "social_nav_policy_state"):
            if k in observations:
                v = observations[k]
                feat = (
                    torch.from_numpy(v).float().to(device)
                    if isinstance(v, np.ndarray)
                    else v.float().to(device)
                )
                break

        if feat is None:
            raise KeyError(
                "SocialNavNeuralHighLevelPolicy requires the "
                "'social_nav_policy_state' observation but it was not found. "
                f"Available observation keys: {list(observations.keys())}"
            )

        social_nav_features = feat.view(feat.shape[0], -1)

        if social_nav_features.shape[1] != self._input_feature_dim:
            raise ValueError(
                f"Expected {self._input_feature_dim} features, got "
                f"{social_nav_features.shape[1]}."
            )

        self._debug_call_count += 1
        if self._debug_call_count <= 5 or self._debug_call_count % 50 == 0:
            f = social_nav_features[0].tolist()
            extra = f" approach={f[6]:.3f}" if len(f) > 6 else ""
            print(
                f"[HL input #{self._debug_call_count}] "
                f"human_dist={f[0]:.3f} human_bearing={f[1]:.3f} "
                f"human_rel_heading={f[2]:.3f} human_rel_speed={f[3]:.3f} "
                f"goal_dist={f[4]:.3f} goal_bearing={f[5]:.3f}" + extra,
                flush=True,
            )

        if self._use_prev_action:
            if prev_actions is None or masks is None:
                raise ValueError(
                    "use_prev_action=True requires prev_actions and masks"
                )
            onehot = torch.nn.functional.one_hot(
                prev_actions.to(social_nav_features.device)
                .long()
                .clamp(0, self._n_actions - 1)
                .view(-1),
                num_classes=self._n_actions,
            ).float() * masks.to(social_nav_features.device).float().view(-1, 1)
            social_nav_features = torch.cat(
                [social_nav_features, onehot], dim=-1
            )

        return social_nav_features

    def get_next_skill(
        self,
        observations: Dict[str, torch.Tensor],
        rnn_hidden_states: torch.Tensor,
        prev_actions: torch.Tensor,
        masks: torch.Tensor,
        should_choose_new_skill: torch.BoolTensor,
        deterministic: bool = False,
        log_info: Optional[List[Dict[str, Any]]] = None,
    ) -> tuple:
        """
        Compute the next skill (high-level action) to execute.

        :param observations: Dictionary of observations.
        :param rnn_hidden_states: RNN hidden states of shape (batch_size, num_layers, hidden_size).
        :param prev_actions: Previous actions.
        :param masks: Episode masks (0 = episode done, 1 = ongoing).
        :param should_choose_new_skill: Which batch elements should get a new skill.
        :param deterministic: If True, use argmax; otherwise sample.
        :param log_info: List to append logging info to.

        :return: Tuple of (new_skills, new_skill_args, should_terminate_episode, policy_info)
        """
        if log_info is None:
            log_info = []

        batch_size = masks.shape[0]

        # Extract social nav features
        social_nav_features = self._extract_social_nav_features(
            observations, prev_actions, masks
        )
        social_nav_features = social_nav_features.to(rnn_hidden_states.device)

        # Forward pass through RNN and policy
        # Manually step through RNN to avoid permute issues with the state encoder
        # rnn_hidden_states shape: [num_layers, batch, hidden]
        # social_nav_features shape: [batch, features]
        
        # Reshape hidden states for RNN: [num_layers, batch, hidden] -> tuple of [batch, hidden] per layer
        rnn_input = social_nav_features.unsqueeze(0)  # [1, batch, features]

        # Reset the hidden state on episode boundaries (masks == 0). This mirrors
        # RNNStateEncoder.single_forward so that the act-time recurrence matches
        # the sequence recurrence used in evaluate_actions (keeps the PPO
        # importance ratio ~1 on the first epoch).
        masks_dev = masks.to(rnn_hidden_states.device)
        reset_hidden = torch.where(
            masks_dev.view(1, -1, 1).bool(),
            rnn_hidden_states,
            rnn_hidden_states.new_zeros(()),
        )

        # Step RNN
        rnn_output, new_hidden_states = self._state_encoder.rnn(
            rnn_input,
            reset_hidden.contiguous(),
        )

        # Only envs that (re)plan this step advance their recurrent state:
        # get_next_skill runs whenever ANY env replans, but evaluate_actions
        # replays the GRU over each env's own decision points only.
        keep = should_choose_new_skill.view(1, -1, 1).to(
            device=new_hidden_states.device, dtype=torch.bool
        )
        new_hidden_states = torch.where(keep, new_hidden_states, reset_hidden)

        # rnn_output: [1, batch, hidden]
        actor_hidden_states = rnn_output.squeeze(0)  # [batch, hidden]
        new_rnn_hidden_states = new_hidden_states  # [num_layers, batch, hidden]
        
        actor_hidden = actor_hidden_states
        mf = self._map_features(observations)
        if mf is not None:
            actor_hidden = torch.cat(
                [actor_hidden, mf.to(actor_hidden.device)], dim=-1
            )

        # Compute policy output
        # CategoricalNet returns a distribution directly, not logits
        dist = self._policy(actor_hidden)

        # CustomFixedCategorical.mode()/sample() both return shape [batch, 1].
        if deterministic:
            skill_actions = dist.mode()
        else:
            skill_actions = dist.sample()

        # Value estimate (critic) and log-prob of the chosen skill, both [batch, 1].
        value_estimate = (
            self._critic(actor_hidden) * self._critic_scale
        ).detach()
        action_log_probs = dist.log_probs(skill_actions).detach()

        # Map to skill indices (create on CPU since discrete indices don't need GPU)
        new_skills = torch.zeros(batch_size, dtype=torch.int32, device="cpu")
        new_skill_args = [None] * batch_size

        for i in range(batch_size):
            if should_choose_new_skill[i]:
                action_idx = skill_actions[i].item()
                skill_name = self._action_to_skill[action_idx]
                new_skills[i] = self._skill_name_to_idx.get(skill_name, 0)
                new_skill_args[i] = {}  # No arguments for discrete skills

                # Log the action choice
                if log_info and i < len(log_info):
                    log_info[i]["hl_action"] = skill_name
                    log_info[i]["hl_action_prob"] = dist.probs[i, action_idx].item()
                    log_info[i]["hl_value"] = value_estimate[i].item()

        # Check for termination (PDDL success)
        should_terminate_episode = torch.zeros(batch_size, dtype=torch.bool)
        if "pddl_success" in observations:
            pddl_success = observations["pddl_success"]
            if isinstance(pddl_success, np.ndarray):
                pddl_success = torch.from_numpy(pddl_success).bool()
            should_terminate_episode = pddl_success.view(-1)

        # Package the high-level learning signal. `actions` are the discrete
        # skill choices [batch, 1] that PPO will re-evaluate; `rnn_hidden_states`
        # is in [num_layers, batch, hidden] layout (the caller transposes it back
        # to [batch, num_layers, hidden] for storage).
        policy_info = PolicyActionData(
            actions=skill_actions,
            rnn_hidden_states=new_rnn_hidden_states,
            values=value_estimate,
            action_log_probs=action_log_probs,
            policy_info=log_info if log_info else None,
        )

        return new_skills, new_skill_args, should_terminate_episode, policy_info

    @classmethod
    def from_config(cls, config, observation_space, action_space, **kwargs):
        """
        Create SocialNavNeuralHighLevelPolicy from config.
        Accepts extra kwargs for compatibility with policy wrappers.
        """
        return cls(
            config=config,
            observation_space=observation_space,
            action_space=action_space,
        )
