import logging
import numpy as np
import torch

from habitat_baselines.rl.hrl.hl.high_level_policy import HighLevelPolicy
from habitat_baselines.rl.ppo.policy import PolicyActionData

logger = logging.getLogger(__name__)


class CyclingHighLevelPolicy(HighLevelPolicy):
    """
    High-level policy that cycles through all defined skills in order.
    Holds each skill for a fixed number of steps before advancing.
    Useful for testing skill implementations without a learned policy.

    Skill durations can be configured via skill max_skill_steps or defaults:
      - go_to_goal: 20 steps
      - backoff: 10 steps
      - wait: 10 steps
    """

    def __init__(
        self,
        config,
        pddl_problem,
        num_envs,
        skill_name_to_idx,
        observation_space,
        action_space,
        **kwargs,
    ):
        super().__init__(
            config,
            pddl_problem,
            num_envs,
            skill_name_to_idx,
            observation_space,
            action_space,
            **kwargs,
        )
        # List of skill indices in order (sorted values of skill_name_to_idx)
        self._skill_list = sorted(skill_name_to_idx.values())
        self._cursors = torch.zeros(num_envs, dtype=torch.long)
        self._first_call = True  # Track if this is the initial skill selection
        self._local_step_counter = 0  # Local counter (deprecated, use global step from observations)
        # Read backoff_switch_step from config, default to 30
        self._backoff_switch_step = getattr(config, "backoff_switch_step", 30)

    def get_value(self, observations, rnn_hidden_states, prev_actions, masks):
        """Return zero values for all envs (no learning signal)."""
        return torch.zeros(masks.shape[0], 1)

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
        """
        Intelligently select skills:
        - go_to_goal by default
        - Switch to backoff after _backoff_switch_step environment steps if human is detected
        """
        batch_size = masks.shape[0]
        next_skill = torch.zeros(batch_size)
        skill_args_data = [None] * batch_size
        immediate_end = torch.zeros(batch_size, dtype=torch.bool)

        # Get actual environment step from observations
        env_step = 0
        if isinstance(observations, dict) and "num_steps" in observations:
            num_steps_obs = observations["num_steps"]
            if isinstance(num_steps_obs, torch.Tensor):
                env_step = int(num_steps_obs[0].item()) if num_steps_obs.dim() > 0 else int(num_steps_obs.item())
            else:
                env_step = int(num_steps_obs[0]) if hasattr(num_steps_obs, '__getitem__') else int(num_steps_obs)

        # Log every call for debugging

        go_to_goal_idx = self._skill_name_to_idx.get("go_to_goal", max(self._skill_name_to_idx.values(), default=0))
        backoff_idx = self._skill_name_to_idx.get("backoff", go_to_goal_idx)

        for i in range(batch_size):
            skill_idx = go_to_goal_idx  # Default: go_to_goal

            # Only switch to backoff after threshold steps AND human is detected
            if backoff_idx != go_to_goal_idx and env_step >= self._backoff_switch_step:
                if isinstance(observations, dict) and "humanoid_detector_sensor" in observations:
                    detector = observations["humanoid_detector_sensor"]
                    if isinstance(detector, torch.Tensor):
                        detector = detector.cpu().numpy()

                    # If shape allows access and human is detected
                    if detector.shape[0] > i and detector[i, 0] > 0.5:
                        skill_idx = backoff_idx  # Switch to backoff

            next_skill[i] = float(skill_idx)
            skill_args_data[i] = []



        return next_skill, skill_args_data, immediate_end, PolicyActionData()

    @classmethod
    def from_config(cls, config, observation_space, action_space, num_envs, full_config, **kwargs):
        """Create CyclingHighLevelPolicy from config."""
        return cls(
            config,
            kwargs.get("pddl_problem", None),
            num_envs,
            kwargs.get("skill_name_to_idx", {}),
            observation_space,
            action_space,
            **{k: v for k, v in kwargs.items()
               if k not in ["pddl_problem", "skill_name_to_idx"]}
        )

    def apply_mask(self, mask):
        """Reset cursor state on episode resets."""
        self._cursors *= mask.cpu().long().view(-1)
