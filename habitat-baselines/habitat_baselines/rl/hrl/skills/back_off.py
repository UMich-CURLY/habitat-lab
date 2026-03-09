import torch

from habitat_baselines.rl.hrl.skills.skill import SkillPolicy
from habitat_baselines.rl.hrl.utils import find_action_range
from habitat_baselines.rl.ppo.policy import PolicyActionData


class BackOffSkillPolicy(SkillPolicy):
    """Scripted skill that commands a small backward base velocity for a
    fixed number of steps. Intended to be mapped to a PDDL `back_off` action.
    """

    def __init__(self, config, action_space, batch_size):
        # Keep hold-state to avoid changing grip behaviour (but most social
        # tasks will ignore grip).
        super().__init__(config, action_space, batch_size, True)
        # default backward power (negative forward)
        self._backoff_power = config.skill_data.get("backoff_power", -0.3)
        # locate base_velocity action start index
        try:
            self._nav_ac_start, _ = find_action_range(
                action_space, "base_velocity"
            )
        except Exception:
            # fallback index 0 if action not found; skill will fail noisily
            self._nav_ac_start = 0

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
        # set backward (negative forward) velocity
        full_action[:, self._nav_ac_start] = self._backoff_power

        return PolicyActionData(actions=full_action, rnn_hidden_states=rnn_hidden_states)
