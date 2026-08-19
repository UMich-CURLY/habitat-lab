import logging

import numpy as np
import torch

from habitat_baselines.rl.hrl.hl.high_level_policy import HighLevelPolicy
from habitat_baselines.rl.ppo.policy import PolicyActionData

logger = logging.getLogger(__name__)


def _dist_to_human(observations, i):
    """Distance (m) from the robot to the human from ``other_agent_gps``
    (= robot_xz - human_xz). Returns ``None`` if unavailable."""
    if not isinstance(observations, dict):
        return None
    oag = observations.get("other_agent_gps")
    if oag is None:
        return None
    if isinstance(oag, torch.Tensor):
        oag = oag.detach().cpu().numpy()
    else:
        oag = np.asarray(oag)
    if oag.ndim < 2 or oag.shape[0] <= i:
        return None
    return float(np.linalg.norm(oag[i][:2]))


class CyclingHighLevelPolicy(HighLevelPolicy):
    """
    Rule-based high-level switch policy for social navigation.

    Selects between the low-level skills based on how close the human is to the
    robot (read from the ``other_agent_gps`` observation), with hysteresis to
    avoid flip-flopping at the boundary:

      - ``go_to_goal`` by default (robot navigates to its goal with RVO).
      - switch to ``backoff`` once the human is closer than ``backoff_enter_dist``.
      - switch back to ``go_to_goal`` once the human is farther than
        ``backoff_exit_dist``.

    The two low-level skills hand control back to this policy at the matching
    crossings (see GoToGoalSkill / BackOffSkill ``should_terminate``), so this
    policy is consulted exactly when a switch may be needed.

    Thresholds are read from the high_level_policy config (``backoff_enter_dist``,
    ``backoff_exit_dist``); defaults 1.0 m / 1.6 m.
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
        self._enter_dist = float(getattr(config, "backoff_enter_dist", 1.0))
        self._exit_dist = float(getattr(config, "backoff_exit_dist", 1.6))
        # Per-env hysteresis state: are we currently backing off?
        self._in_backoff = [False] * num_envs
        # Resolve skill indices by name (fall back to the documented order
        # backoff=0, wait=1, go_to_goal=2 if a name is missing).
        self._go_idx = skill_name_to_idx.get("go_to_goal", 2)
        self._backoff_idx = skill_name_to_idx.get("backoff", 0)

    def get_value(self, observations, rnn_hidden_states, prev_actions, masks):
        """Return zero values for all envs (no learning signal)."""
        return torch.zeros(masks.shape[0], 1, device=masks.device)

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
        skill_args_data = [None] * batch_size
        immediate_end = torch.zeros(batch_size, dtype=torch.bool)

        for i in range(batch_size):
            d = _dist_to_human(observations, i)
            if self._in_backoff[i]:
                # Stay backing off until the human is clearly far again.
                if d is None or d > self._exit_dist:
                    self._in_backoff[i] = False
            else:
                # Start backing off once the human gets close.
                if d is not None and d < self._enter_dist:
                    self._in_backoff[i] = True

            skill_idx = self._backoff_idx if self._in_backoff[i] else self._go_idx
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
        """Reset hysteresis state on episode resets (mask == 0)."""
        m = mask.cpu().view(-1)
        for i in range(min(len(self._in_backoff), m.shape[0])):
            if float(m[i]) == 0.0:
                self._in_backoff[i] = False
