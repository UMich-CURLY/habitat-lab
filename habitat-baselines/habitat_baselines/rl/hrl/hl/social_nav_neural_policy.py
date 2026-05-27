"""
Hierarchical policy for social navigation with Spot robot.

This module implements a neural high-level policy for a Spot robot that must 
navigate from one side of a door to the other while interacting with a human.
The policy learns to choose from three discrete actions:
  - "go_to_wait_pos": navigate to a waiting position near the human
  - "wait": remain at current position
  - "go_to_goal": navigate directly to the goal

Input features: {hpx, hpy, velx, vely, thetar, thetah}
  - hpx, hpy: relative position of human w.r.t. robot
  - velx, vely: relative velocity of human w.r.t. robot
  - thetar: robot's orientation
  - thetah: human's orientation
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
        """The policy's action space is discrete over high-level skills."""
        return spaces.Discrete(self._n_actions)

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
        )

    def to(self, device):
        """Move all modules to device."""
        self._state_encoder.to(device)
        self._policy.to(device)
        self._critic.to(device)

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
        Extract the 6-D feature vector from observations:
        [hpx, hpy, velx, vely, thetar, thetah]

        Expects observations to contain keys with these features or
        computable from robot/human state observations.

        :param observations: Dictionary of observations from the environment.
        :return: Tensor of shape (batch_size, 6) with social nav features.
        """
        batch_size = None
        features = []

        # Extract human relative position (hpx, hpy)
        for key in ["human_relative_position", "human_pos_relative"]:
            if key in observations:
                hpos = observations[key]
                if isinstance(hpos, np.ndarray):
                    hpos = torch.from_numpy(hpos).float()
                if batch_size is None:
                    batch_size = hpos.shape[0]
                # Take only x, y components
                features.extend([hpos[:, 0:1], hpos[:, 1:2]])
                break
        else:
            # Fallback: zeros if not available
            if batch_size is None:
                batch_size = list(observations.values())[0].shape[0]
            features.append(torch.zeros(batch_size, 1, device=list(observations.values())[0].device))
            features.append(torch.zeros(batch_size, 1, device=list(observations.values())[0].device))

        # Extract relative velocity (velx, vely)
        for key in ["human_relative_velocity", "human_vel_relative"]:
            if key in observations:
                hvel = observations[key]
                if isinstance(hvel, np.ndarray):
                    hvel = torch.from_numpy(hvel).float()
                features.extend([hvel[:, 0:1], hvel[:, 1:2]])
                break
        else:
            # Fallback: zeros
            features.append(torch.zeros(batch_size, 1, device=features[0].device))
            features.append(torch.zeros(batch_size, 1, device=features[0].device))

        # Extract robot orientation (thetar)
        for key in ["robot_orientation", "agent_0_orientation"]:
            if key in observations:
                thetar = observations[key]
                if isinstance(thetar, np.ndarray):
                    thetar = torch.from_numpy(thetar).float()
                # Keep only yaw (typically last component or index 2 for XYZ Euler)
                if thetar.shape[-1] >= 3:
                    thetar = thetar[:, 2:3]
                elif thetar.shape[-1] == 1:
                    thetar = thetar[:, 0:1]
                features.append(thetar)
                break
        else:
            features.append(torch.zeros(batch_size, 1, device=features[0].device))

        # Extract human orientation (thetah)
        for key in ["human_orientation", "agent_1_orientation"]:
            if key in observations:
                thetah = observations[key]
                if isinstance(thetah, np.ndarray):
                    thetah = torch.from_numpy(thetah).float()
                # Keep only yaw
                if thetah.shape[-1] >= 3:
                    thetah = thetah[:, 2:3]
                elif thetah.shape[-1] == 1:
                    thetah = thetah[:, 0:1]
                features.append(thetah)
                break
        else:
            features.append(torch.zeros(batch_size, 1, device=features[0].device))

        # Concatenate all features
        social_nav_features = torch.cat(features, dim=1)  # Shape: (batch_size, 6)

        if social_nav_features.shape[1] != self._input_feature_dim:
            raise ValueError(
                f"Expected {self._input_feature_dim} features, got {social_nav_features.shape[1]}. "
                f"Available observation keys: {list(observations.keys())}"
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
        
        # Step RNN
        rnn_output, new_hidden_states = self._state_encoder.rnn(
            rnn_input, 
            rnn_hidden_states.contiguous()
        )
        
        # rnn_output: [1, batch, hidden]
        actor_hidden_states = rnn_output.squeeze(0)  # [batch, hidden]
        new_rnn_hidden_states = new_hidden_states  # [num_layers, batch, hidden]
        
        actor_hidden = actor_hidden_states

        # Compute policy output
        # CategoricalNet returns a distribution directly, not logits
        dist = self._policy(actor_hidden)

        if deterministic:
            skill_actions = dist.probs.argmax(dim=-1)
        else:
            skill_actions = dist.sample()

        # Compute value estimate for logging
        value_estimate = self._critic(actor_hidden).detach()

        # Compute action log probs
        action_log_probs = dist.log_prob(skill_actions).detach()

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

        # Create policy info
        policy_info = PolicyActionData(
            actions=skill_actions.unsqueeze(-1),
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
