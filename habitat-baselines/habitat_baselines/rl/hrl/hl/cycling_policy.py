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
    """Throwaway scaffolding HL policy with no learnable parameters, used only to
    verify that the low-level skills execute. Two modes (``mode`` config):

    - ``"fixed"`` (default): cycle through skills on a FIXED schedule
      ``go_to_goal -> wait -> backoff -> ...``, each held for its own
      ``max_skill_steps`` (run the skills with ``skill_data.cycle_demo: True`` so
      they terminate purely on the step counter).
    - ``"distance"``: pick ``backoff`` when the human is within
      ``backoff_enter_dist``, resume ``go_to_goal`` once farther than
      ``backoff_exit_dist`` (hysteresis). Sensible, reactive behaviour; nicer
      demo videos. Run the skills with ``cycle_demo: False``.

    For training, replace this with a learned HL (e.g. SocialNavNeuralHighLevelPolicy).
    """

    CYCLE_NAMES = ("go_to_goal", "wait", "backoff")

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
        self._mode = str(getattr(config, "mode", "fixed"))
        self._idx_to_name = {v: k for k, v in skill_name_to_idx.items()}

        # fixed-cycle state
        self._cycle = [
            skill_name_to_idx[n] for n in self.CYCLE_NAMES if n in skill_name_to_idx
        ]
        if not self._cycle:
            self._cycle = sorted(skill_name_to_idx.values())
        self._cursor = [0] * num_envs

        # distance-mode state
        self._enter_dist = float(getattr(config, "backoff_enter_dist", 1.0))
        self._exit_dist = float(getattr(config, "backoff_exit_dist", 1.2))
        self._in_backoff = [False] * num_envs
        self._go_idx = skill_name_to_idx.get("go_to_goal", 2)
        self._backoff_idx = skill_name_to_idx.get("backoff", 0)

    def get_value(self, observations, rnn_hidden_states, prev_actions, masks):
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
        batch_size = masks.shape[0]
        next_skill = torch.zeros(batch_size)
        skill_args_data = [None] * batch_size
        immediate_end = torch.zeros(batch_size, dtype=torch.bool)

        for i, should_plan in enumerate(plan_masks):
            if should_plan != 1.0:
                continue
            if self._mode == "distance":
                d = _dist_to_human(observations, i)
                if self._in_backoff[i]:
                    if d is None or d > self._exit_dist:
                        self._in_backoff[i] = False
                else:
                    if d is not None and d < self._enter_dist:
                        self._in_backoff[i] = True
                skill_idx = self._backoff_idx if self._in_backoff[i] else self._go_idx
            else:  # fixed cycle
                skill_idx = self._cycle[self._cursor[i] % len(self._cycle)]
                self._cursor[i] += 1

            next_skill[i] = float(skill_idx)
            skill_args_data[i] = []
            print(
                f"[CyclingPolicy] env{i} ({self._mode}) -> skill="
                f"{self._idx_to_name.get(skill_idx, skill_idx)} (idx={skill_idx})",
                flush=True,
            )
        return next_skill, skill_args_data, immediate_end, PolicyActionData()

    @classmethod
    def from_config(cls, config, observation_space, action_space, num_envs, full_config, **kwargs):
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
        """Reset per-env state on episode resets (mask == 0)."""
        m = mask.cpu().view(-1)
        for i in range(min(self._num_envs, m.shape[0])):
            if float(m[i]) == 0.0:
                if i < len(self._cursor):
                    self._cursor[i] = 0
                if i < len(self._in_backoff):
                    self._in_backoff[i] = False
