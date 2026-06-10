# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os.path as osp
from dataclasses import dataclass

import gym.spaces as spaces
import torch

from habitat.core.spaces import ActionSpace
from habitat.tasks.rearrange.rearrange_sensors import (
    HasFinishedOracleNavSensor,
    IsHoldingSensor,
)
from habitat_baselines.common.logging import baselines_logger
from habitat_baselines.rl.hrl.skills.nn_skill import NnSkillPolicy
from habitat_baselines.rl.hrl.utils import find_action_range
from habitat_baselines.rl.ppo.policy import PolicyActionData


class OracleNavPolicy(NnSkillPolicy):
    @dataclass
    class OracleNavActionArgs:
        """
        :property action_idx: The index of the oracle action we want to execute
        """

        action_idx: int

    def __init__(
        self,
        wrap_policy,
        config,
        action_space,
        filtered_obs_space,
        filtered_action_space,
        batch_size,
        pddl_domain_path,
        pddl_task_path,
        task_config,
    ):
        super().__init__(
            wrap_policy,
            config,
            action_space,
            filtered_obs_space,
            filtered_action_space,
            batch_size,
        )
        # Handle both Dict and Box action spaces
        from gym import spaces
        if isinstance(action_space, spaces.Box):
            # For Box action space (flattened from Dict), find oracle_nav position
            # Try to find it in filtered_action_space first
            found_idx = None
            try:
                found_idx, _ = find_action_range(
                    filtered_action_space, "oracle_nav_action"
                )
            except (KeyError, ValueError):
                pass

            if found_idx is not None:
                self._oracle_nav_ac_idx = found_idx
            else:
                # Fallback: for humanoid with arm, oracle_nav typically comes after:
                # arm_action (7) + grip (1) + base_vel (2) = position 10
                if self._full_ac_size == 12:  # humanoid with oracle_nav
                    self._oracle_nav_ac_idx = 10
                else:
                    # For other cases, oracle_nav is at the end
                    self._oracle_nav_ac_idx = self._full_ac_size - 1
        else:
            # For Dict action space, use find_action_range
            try:
                self._oracle_nav_ac_idx, _ = find_action_range(
                    action_space, "oracle_nav_action"
                )
            except (KeyError, ValueError):
                raise ValueError(
                    f"Could not find oracle_nav_action in action space {action_space}"
                )

    def set_pddl_problem(self, pddl_prob):
        super().set_pddl_problem(pddl_prob)
        self._all_entities = self._pddl_problem.get_ordered_entities_list()

    def on_enter(
        self,
        skill_arg,
        batch_idx,
        observations,
        rnn_hidden_states,
        prev_actions,
        skill_name,
    ):
        ret = super().on_enter(
            skill_arg,
            batch_idx,
            observations,
            rnn_hidden_states,
            prev_actions,
            skill_name,
        )
        self._was_running_on_prev_step = False
        return ret

    @classmethod
    def from_config(
        cls, config, observation_space, action_space, batch_size, full_config
    ):
        # If action_space is a Box (agent-specific for social nav), 
        # we need to get the full Dict action space from the environment config
        if isinstance(action_space, spaces.Box):
            # For multi-agent setups, we need the full action space from the env
            # This is typically stored in full_config if available
            try:
                # Try to get action space from full_config's habitat settings
                from habitat_baselines.rl.hrl.hl.high_level_policy import HighLevelPolicy
                # We'll use the orig_action_space if available in full_config
                if hasattr(full_config, 'habitat') and hasattr(full_config.habitat, 'task'):
                    # This is a navigation task with Box action space
                    # Create a simple filtered action space for oracle nav
                    filtered_action_space = ActionSpace(
                        {config.action_name: action_space}
                    )
                else:
                    filtered_action_space = ActionSpace(
                        {config.action_name: action_space}
                    )
            except Exception as e:
                baselines_logger.warning(f"Could not construct action space: {e}")
                filtered_action_space = ActionSpace(
                    {config.action_name: action_space}
                )
        else:
            # Original logic for Dict action spaces (manipulation tasks)
            try:
                filtered_action_space = ActionSpace(
                    {config.action_name: action_space[config.action_name]}
                )
            except KeyError as e:
                try:
                    filtered_action_space = ActionSpace(
                        {config.action_name: action_space["oracle_nav_action"][config.action_name]}
                    )
                except KeyError:
                    baselines_logger.error(f"Could not find action {config.action_name} in action space")
                    raise
           
        baselines_logger.debug(
            f"Loaded action space {filtered_action_space} for skill {config.skill_name}"
        )
        return cls(
            None,
            config,
            action_space,
            observation_space,
            filtered_action_space,
            batch_size,
            full_config.habitat.task.pddl_domain_def,
            osp.join(
                full_config.habitat.task.task_spec_base_path,
                full_config.habitat.task.task_spec + ".yaml",
            ),
            full_config.habitat.task,
        )

    def _is_skill_done(
        self,
        observations,
        rnn_hidden_states,
        prev_actions,
        masks,
        batch_idx,
    ) -> torch.BoolTensor:
        ret = torch.zeros(masks.shape[0], dtype=torch.bool)

        finish_oracle_nav = observations[
            HasFinishedOracleNavSensor.cls_uuid
        ].cpu()
        ret = finish_oracle_nav.to(torch.bool)[:, 0]

        return ret

    def _parse_skill_arg(self, skill_name: str, skill_arg):
        if len(skill_arg) == 2:
            search_target, _ = skill_arg
        elif len(skill_arg) == 3:
            _, search_target, _ = skill_arg
        else:
            raise ValueError(
                f"Unexpected number of skill arguments in {skill_arg}"
            )
            

        target = self._pddl_problem.get_entity(search_target)
        if target is None:
            raise ValueError(
                f"Cannot find matching entity for {search_target}"
            )
        match_i = self._all_entities.index(target)

        return OracleNavPolicy.OracleNavActionArgs(match_i)

    @property
    def required_obs_keys(self):
        ret = [HasFinishedOracleNavSensor.cls_uuid]
        if self._should_keep_hold_state:
            ret.append(IsHoldingSensor.cls_uuid)
        return ret

    def _internal_act(
        self,
        observations,
        rnn_hidden_states,
        prev_actions,
        masks,
        cur_batch_idx,
        deterministic=False,
    ):
        full_action = torch.zeros(
            (masks.shape[0], self._full_ac_size), device=masks.device
        )
        action_idxs = torch.FloatTensor(
            [self._cur_skill_args[i].action_idx + 1 for i in cur_batch_idx]
        )

        full_action[:, self._oracle_nav_ac_idx] = action_idxs

        print(f"[OracleNavPolicy._internal_act] _oracle_nav_ac_idx={self._oracle_nav_ac_idx}, action_idxs={action_idxs}, full_action={full_action}", flush=True)

        return PolicyActionData(
            actions=full_action, rnn_hidden_states=rnn_hidden_states
        )


class OracleNavCoordPolicy(OracleNavPolicy):
    """The function produces a sequence of navigation targets and the oracle nav navigates to those targets"""

    @dataclass
    class OracleNavActionArgs:
        """
        :property action_idx: The index of the oracle action we want to execute
        """

        action_idx: int

    def __init__(
        self,
        wrap_policy,
        config,
        action_space,
        filtered_obs_space,
        filtered_action_space,
        batch_size,
        pddl_domain_path,
        pddl_task_path,
        task_config,
    ):
        NnSkillPolicy.__init__(
            self,
            wrap_policy,
            config,
            action_space,
            filtered_obs_space,
            filtered_action_space,
            batch_size,
        )
        # Random coordinate means that the navigation target is chosen randomly
        action_name = "oracle_nav_randcoord_action"
        self._oracle_nav_ac_idx, _ = find_action_range(
            action_space, action_name
        )

    def _parse_skill_arg(self, skill_arg):
        return OracleNavCoordPolicy.OracleNavActionArgs(skill_arg)

    def _internal_act(
        self,
        observations,
        rnn_hidden_states,
        prev_actions,
        masks,
        cur_batch_idx,
        deterministic=False,
    ):
        full_action = torch.zeros(
            (masks.shape[0], self._full_ac_size), device=masks.device
        )
        action_idxs = torch.FloatTensor(
            [self._cur_skill_args[i].action_idx for i in cur_batch_idx]
        )

        full_action[:, self._oracle_nav_ac_idx] = action_idxs

        return PolicyActionData(
            actions=full_action, rnn_hidden_states=rnn_hidden_states
        )


class OracleNavHumanPolicy(OracleNavCoordPolicy):
    """
    Navigate to human's location using oracle nav
    """

    @dataclass
    class OracleNavActionArgs:
        """
        :property action_idx: The index of the oracle action we want to execute
        """

        action_idx: int

    def __init__(
        self,
        wrap_policy,
        config,
        action_space,
        filtered_obs_space,
        filtered_action_space,
        batch_size,
        pddl_domain_path,
        pddl_task_path,
        task_config,
    ):
        NnSkillPolicy.__init__(
            self,
            wrap_policy,
            config,
            action_space,
            filtered_obs_space,
            filtered_action_space,
            batch_size,
        )
        action_name = "oracle_nav_human_action"
        self._oracle_nav_ac_idx, _ = find_action_range(
            action_space, action_name
        )

    def _parse_skill_arg(self, skill_arg):
        return OracleNavHumanPolicy.OracleNavActionArgs(skill_arg)
