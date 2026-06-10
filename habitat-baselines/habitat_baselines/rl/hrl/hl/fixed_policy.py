# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Tuple

import torch

from habitat.tasks.rearrange.multi_task.rearrange_pddl import parse_func
from habitat_baselines.common.logging import baselines_logger
from habitat_baselines.rl.hrl.hl.high_level_policy import HighLevelPolicy
from habitat_baselines.rl.ppo.policy import PolicyActionData
from IPython import embed

class FixedHighLevelPolicy(HighLevelPolicy):
    """
    Executes a fixed sequence of high-level actions as specified by the
    `solution` field of the PDDL problem file.
    :property _solution_actions: List of tuples where the first tuple element
        is the action name and the second is the action arguments. Stores a plan
        for each environment.
    """

    _solution_actions: List[List[Tuple[str, List[str]]]]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._update_solution_actions(
            [self._parse_solution_actions() for _ in range(self._num_envs)]
        )
        self._next_sol_idxs = torch.zeros(self._num_envs, dtype=torch.int32)
        self._steps_since_start = 0  # Track steps for backoff triggering

    def _update_solution_actions(
        self, solution_actions: List[List[Tuple[str, List[str]]]]
    ) -> None:
        if len(solution_actions) == 0:
            raise ValueError(
                "Solution actions must be non-empty (if want to execute no actions, just include a no-op)"
            )
        self._solution_actions = solution_actions

    def _parse_solution_actions(self) -> List[Tuple[str, List[str]]]:
        """
        Returns the sequence of actions to execute as a list of:
        - The action name.
        - A list of the action arguments.
        If no PDDL problem is available, returns agent-specific actions for multi-agent scenarios.
        Filters the solution to only include actions for this agent (multi-agent support).
        """
        if self._pddl_prob is None:
            # For multi-agent setups without PDDL available, provide simple hardcoded actions
            # Default: just wait (no navigation)
            baselines_logger.warning(
                f"No PDDL problem available for agent {self._agent_name}, using default wait action"
            )
            return [parse_func("wait(30)")]

        # Derive robot entity name from agent name for filtering
        # E.g., "agent_1" -> "robot_1"
        if self._agent_name is not None:
            robot_id = "robot_" + self._agent_name.split("_")[1]
        else:
            robot_id = None

        solution = self._pddl_prob.solution

        solution_actions = []
        for i, hl_action in enumerate(solution):
            # Only include actions where this agent's robot appears in parameters
            param_names = [x.name for x in hl_action.param_values]
            if robot_id is not None and robot_id not in param_names:
                # Skip actions meant for other agents
                continue

            sol_action = (hl_action.name, param_names)
            solution_actions.append(sol_action)

            if self._config.add_arm_rest and i < (len(solution) - 1):
                solution_actions.append(parse_func("reset_arm(0)"))

        # Add a wait action at the end.
        solution_actions.append(parse_func("wait(30)"))

        return solution_actions

    def apply_mask(self, mask):
        """
        Apply the given mask to the next skill index.

        Args:
            mask: Binary mask of shape (num_envs, ) to be applied to the next
                skill index.
        """
        self._next_sol_idxs *= mask.cpu().view(-1)

    def _get_next_sol_idx(self, batch_idx, immediate_end):
        """
        Get the next index to be used from the list of solution actions.

        Args:
            batch_idx: The index of the current environment.

        Returns:
            The next index to be used from the list of solution actions.
        """
        if self._next_sol_idxs[batch_idx] >= len(
            self._solution_actions[batch_idx]
        ):
            baselines_logger.info(
                f"Calling for immediate end with {self._next_sol_idxs[batch_idx]}"
            )
            immediate_end[batch_idx] = True
            # Just repeat the last action.
            return len(self._solution_actions[batch_idx]) - 1
        else:
            return self._next_sol_idxs[batch_idx].item()

    def get_value(self, observations, rnn_hidden_states, prev_actions, masks):
        # We assign a value of 0. This is needed so that we can concatenate values in multiagent
        # policies
        return torch.zeros(rnn_hidden_states.shape[0], 1).to(
            rnn_hidden_states.device
        )

    def get_next_skill(
        self,
        observations,
        rnn_hidden_states,
        prev_actions,
        masks,
        plan_masks,
        deterministic,
        log_info,
    ):
        batch_size = masks.shape[0]
        next_skill = torch.zeros(batch_size)
        skill_args_data = [None for _ in range(batch_size)]
        immediate_end = torch.zeros(batch_size, dtype=torch.bool)

        # Force backoff at step 15 to test the skill
        if self._steps_since_start >= 15:
            # Always select skill 0 (backoff in HierarchicalPolicy, oracle_nav in FixedHighLevelPolicy)
            for i in range(batch_size):
                next_skill[i] = 0  # Force first skill (backoff)
                skill_args_data[i] = {}
            return next_skill, skill_args_data, immediate_end, PolicyActionData()

        # Rule-based override: force backoff if human is detected
        if isinstance(observations, dict) and "humanoid_detector_sensor" in observations:
            detector = observations["humanoid_detector_sensor"]
            if isinstance(detector, torch.Tensor):
                detector = detector.cpu()
            elif isinstance(detector, np.ndarray):
                detector = torch.from_numpy(detector).float()

            for i in range(batch_size):
                if detector.shape[0] > i and detector[i, 0].item() > 0.5:
                    # Human detected - force backoff skill (should exist in skill_name_to_idx)
                    if "backoff" in self._skill_name_to_idx:
                        next_skill[i] = self._skill_name_to_idx["backoff"]
                        skill_args_data[i] = {}
                        continue

        for batch_idx, should_plan in enumerate(plan_masks):
            # Skip if already set to backoff via rule-based override
            if next_skill[batch_idx] > 0:  # If not zero (backoff), skip the plan-based logic
                continue

            if should_plan == 1.0:
                use_idx = self._get_next_sol_idx(batch_idx, immediate_end)

                pddl_action_name, skill_args = self._solution_actions[batch_idx][
                    use_idx
                ]
                # Map PDDL action name to skill name using the mapping passed during initialization
                skill_name = self._pddl_action_name_to_skill_name.get(pddl_action_name, pddl_action_name)

                if skill_name not in self._skill_name_to_idx:
                    raise ValueError(
                        f"Could not find skill named {skill_name} (from PDDL action {pddl_action_name}) in {self._skill_name_to_idx}"
                    )
                next_skill[batch_idx] = self._skill_name_to_idx[skill_name]

                skill_args_data[batch_idx] = skill_args  # type: ignore[call-overload]

                self._next_sol_idxs[batch_idx] += 1

        return next_skill, skill_args_data, immediate_end, PolicyActionData()

    def filter_envs(self, curr_envs_to_keep_active):
        """
        Cleans up stateful variables of the policy so that
        they match with the active environments
        """
        self._next_sol_idxs = self._next_sol_idxs[curr_envs_to_keep_active]
        parse_solution_actions = [
            self._parse_solution_actions() for _ in range(self._num_envs)
        ]
        self._update_solution_actions(
            [
                parse_solution_actions[i]
                for i in range(curr_envs_to_keep_active.shape[0])
                if curr_envs_to_keep_active[i]
            ]
        )
