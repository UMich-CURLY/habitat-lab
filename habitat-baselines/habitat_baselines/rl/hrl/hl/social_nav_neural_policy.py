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

        # Define the discrete high-level action space
        self._skill_to_action = {
            "backoff": 0,
            "wait": 1,
            "go_to_goal": 2,
        }
        self._action_to_skill = {v: k for k, v in self._skill_to_action.items()}
        self._n_actions = len(self._skill_to_action)

        # Feature dimension for social nav input
        # Input: [hpx, hpy, velx, vely, thetar, thetah]
        self._input_feature_dim = 6
        self._hidden_size = getattr(self._config, "hidden_dim", 256)

        # Build the neural network
        self._state_encoder = build_rnn_state_encoder(
            self._input_feature_dim,
            self._hidden_size,
            rnn_type=getattr(self._config, "rnn_type", "GRU"),
            num_layers=getattr(self._config, "num_rnn_layers", 2),
        )

        # Policy head (outputs action logits)
        self._policy = CategoricalNet(self._hidden_size, self._n_actions)

        # Value head (critic)
        self._critic = CriticHead(self._hidden_size)

        # Aux modules (e.g., for auxiliary losses)
        self.aux_modules = get_aux_modules(aux_loss_config, action_space, self)
        self._debug_call_count = 0

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
        return chain(
            self._state_encoder.parameters(),
            self._policy.parameters(),
            self._critic.parameters(),
            self.aux_modules.parameters(),
        )

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
        return self

    def forward(
        self,
        observations: Dict[str, torch.Tensor],
        rnn_hidden_states: torch.Tensor,
        masks: torch.Tensor,
        rnn_build_seq_info=None,
    ):
        """
        Encode the 6-D social-nav features through the recurrent state encoder.

        Used for both single-step value queries and sequence-based
        `evaluate_actions` during the PPO update. ``rnn_hidden_states`` is in the
        ``[batch, num_layers, hidden]`` layout expected by ``RNNStateEncoder``.
        """
        features = self._extract_social_nav_features(observations)
        features = features.to(rnn_hidden_states.device)
        return self._state_encoder(
            features, rnn_hidden_states, masks, rnn_build_seq_info
        )

    def get_value(self, observations, rnn_hidden_states, prev_actions, masks):
        """Critic value estimate for the bootstrap at the end of a rollout."""
        features, _ = self.forward(observations, rnn_hidden_states, masks)
        return self._critic(features)

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
            observations, rnn_hidden_states, masks, rnn_build_seq_info
        )
        distribution = self._policy(features)
        value = self._critic(features)
        # The skill choice is stored as a float (width-1 Box); cast back to a
        # discrete index for the categorical log-prob.
        action_log_probs = distribution.log_probs(action.long())
        distribution_entropy = distribution.entropy()
        aux_loss_res = {
            k: v(features, observations) for k, v in self.aux_modules.items()
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
        self, observations: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Read the 6-D robot-frame feature vector from the SocialNavPolicyState
        sensor (``social_nav_policy_state`` / ``agent_0_social_nav_policy_state``):

            [human_dist, human_bearing, human_rel_heading, human_rel_speed,
             goal_dist, goal_bearing]

        All quantities are already expressed in the robot's local frame (see
        ``SocialNavPolicyStateSensor``), so the policy input is invariant to the
        robot's absolute world pose.
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
            print(
                f"[HL input #{self._debug_call_count}] "
                f"human_dist={f[0]:.3f} human_bearing={f[1]:.3f} "
                f"human_rel_heading={f[2]:.3f} human_rel_speed={f[3]:.3f} "
                f"goal_dist={f[4]:.3f} goal_bearing={f[5]:.3f}",
                flush=True,
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
        social_nav_features = self._extract_social_nav_features(observations)
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
        
        # rnn_output: [1, batch, hidden]
        actor_hidden_states = rnn_output.squeeze(0)  # [batch, hidden]
        new_rnn_hidden_states = new_hidden_states  # [num_layers, batch, hidden]
        
        actor_hidden = actor_hidden_states

        # Compute policy output
        # CategoricalNet returns a distribution directly, not logits
        dist = self._policy(actor_hidden)

        # CustomFixedCategorical.mode()/sample() both return shape [batch, 1].
        if deterministic:
            skill_actions = dist.mode()
        else:
            skill_actions = dist.sample()

        # Value estimate (critic) and log-prob of the chosen skill, both [batch, 1].
        value_estimate = self._critic(actor_hidden).detach()
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

        # Rule-based override: force backoff if human is too close or approaching
        for i in range(batch_size):
            if should_choose_new_skill[i]:

                # Get relative position and velocity
                dist_to_human = None
                vel_towards_human = None

                # Calculate distance to human
                for key in ["human_relative_position", "human_pos_relative"]:
                    if key in observations:
                        hpos = observations[key]
                        if isinstance(hpos, np.ndarray):
                            hpos = torch.from_numpy(hpos).float()
                        # Get distance from relative position [hpx, hpy]
                        if hpos.shape[1] >= 2:
                            dist = torch.sqrt(hpos[i, 0]**2 + hpos[i, 1]**2).item()
                            dist_to_human = dist
                        break

                # Calculate relative velocity (dot product with relative position to get closing speed)
                for key in ["human_relative_velocity", "human_vel_relative"]:
                    if key in observations:
                        hvel = observations[key]
                        if isinstance(hvel, np.ndarray):
                            hvel = torch.from_numpy(hvel).float()
                        # Get velocity components [velx, vely]
                        if hvel.shape[1] >= 2:
                            velx = hvel[i, 0].item()
                            vely = hvel[i, 1].item()
                            # Get relative position to compute dot product
                            for pos_key in ["human_relative_position", "human_pos_relative"]:
                                if pos_key in observations:
                                    hpos = observations[pos_key]
                                    if isinstance(hpos, np.ndarray):
                                        hpos = torch.from_numpy(hpos).float()
                                    if hpos.shape[1] >= 2:
                                        hpx = hpos[i, 0].item()
                                        hpy = hpos[i, 1].item()
                                        dist_sq = hpx**2 + hpy**2
                                        if dist_sq > 0:
                                            # Closing speed = (velocity dot relative_position) / distance
                                            vel_towards_human = (velx * hpx + vely * hpy) / (dist_sq**0.5)
                                    break
                        break

                # Apply rules: force backoff if:
                # 1. Human is within 0.5m, OR
                # 2. Human is approaching at > 0.5 m/s closing speed
                should_backoff = False
                if dist_to_human is not None and dist_to_human < 0.5:
                    should_backoff = True
                elif vel_towards_human is not None and vel_towards_human > 0.5:
                    should_backoff = True

                if should_backoff:
                    # Force backoff skill (action 0 = backoff)
                    new_skills[i] = self._skill_name_to_idx.get("backoff", 0)
                    new_skill_args[i] = {}
                    if log_info and i < len(log_info):
                        log_info[i]["hl_action"] = "backoff (rule-based)"
                        log_info[i]["rule_dist_to_human"] = dist_to_human if dist_to_human is not None else -1
                        log_info[i]["rule_closing_speed"] = vel_towards_human if vel_towards_human is not None else -1

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
