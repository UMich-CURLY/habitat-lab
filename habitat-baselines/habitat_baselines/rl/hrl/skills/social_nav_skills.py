"""
Low-level skills for social navigation with Spot robot.

Three discrete skills (all observation-driven, all output ``base_velocity``;
per-step pathfinding lives in task sensors, which have sim access):
  1. BackOffSkill:  Back up IN REVERSE to a validated point ~2 m behind the door
                    on the robot's start side (``backoff_waypoint_delta`` sensor).
  2. WaitSkill:     Stay in place while rotating to face the robot goal.
  3. GoToGoalSkill: Navigate to the robot goal along the default navmesh
                    geodesic (``goal_waypoint_delta`` sensor) with NO human
                    avoidance -- yielding is the high-level planner's job.

Notes on the multi-agent key convention: each agent's policy receives its own
observation/action dict with the ``agent_{i}_`` prefix **stripped**
(see ``rl/multi_agent/utils.update_dict_with_agent_prefix``). So inside agent_0's
skill the keys are bare: ``base_velocity``, ``localization_sensor``,
``other_agent_gps``, ``goal_world_delta``, ``humanoid_detector_sensor``. The
``habitat.gym.obs_keys`` whitelist, however, lists the prefixed env-level names
(``agent_0_localization_sensor`` etc.).
"""

import logging
import math
from typing import Any, List

import numpy as np
import gym.spaces as spaces  # noqa: F401  (kept for parity with sibling skills)
import torch
import torch.nn as nn

from habitat.tasks.utils import get_angle
from habitat_baselines.rl.hrl.skills import SkillPolicy
from habitat_baselines.rl.hrl.utils import find_action_range
from habitat_baselines.rl.ppo.policy import PolicyActionData
from habitat_baselines.utils.common import get_num_actions

logger = logging.getLogger(__name__)


def _to_tensor(x: Any) -> torch.Tensor:
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).float()
    return x.float()


def _to_np(x: Any):
    """Convert an observation entry (tensor / ndarray / None) to a 2D ndarray."""
    if x is None:
        return None
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


class SocialNavSkillBase(nn.Module, SkillPolicy):
    """
    Base for social nav skills that output base-velocity actions.
    Bypasses SkillPolicy.__init__ to avoid PDDL requirements.
    """

    def __init__(self, config, action_space, batch_size, **kwargs):
        nn.Module.__init__(self)

        self._config = config
        self._batch_size = batch_size

        self.should_ignore_grip = getattr(config, "ignore_grip", True)
        self._apply_postconds = getattr(config, "apply_postconds", False)
        self._force_end_on_timeout = getattr(config, "force_end_on_timeout", False)
        self._max_skill_steps = getattr(config, "max_skill_steps", -1)
        self._pddl_ac_start = None
        self._cur_skill_step = 0
        self._cur_skill_args = [None] * batch_size
        self._raw_skill_args = [None] * batch_size
        self._pddl_problem = None
        self._last_num_steps = -1  # Track last num_steps to avoid duplicate termination

        self._full_ac_size = get_num_actions(action_space)

        # Locate the (stripped) base_velocity slot robustly instead of hardcoding
        # an index. ``BaseVelNonCylinderAction`` with enable_lateral_move=False is
        # 2-dim [longitudinal, angular]; the slot start is what we write into.
        try:
            self._base_vel_start, self._base_vel_end = find_action_range(
                action_space, "base_velocity"
            )
        except (ValueError, KeyError):
            self._base_vel_start, self._base_vel_end = 0, 2

    @property
    def num_recurrent_layers(self) -> int:
        return 0

    @property
    def required_obs_keys(self) -> List[str]:
        return []

    def on_enter(self, skill_args, batch_indices, observations,
                 rnn_hidden_states, prev_actions, skill_name=None):
        return rnn_hidden_states, prev_actions

    def _internal_act(self, observations, rnn_hidden_states, prev_actions, masks,
            cur_batch_idx=None, deterministic=False):
        raise NotImplementedError

    def act(self, observations, rnn_hidden_states, prev_actions, masks,
            cur_batch_idx=None, deterministic=False, **kwargs):
        """Call _internal_act and return PolicyActionData."""
        return self._internal_act(observations, rnn_hidden_states, prev_actions, masks,
                                  cur_batch_idx, deterministic)

    def should_terminate(self, observations, rnn_hidden_states, prev_actions,
                         masks, hl_wants_skill_term, actions, **kwargs):
        batch_size = masks.shape[0]
        call_hl = hl_wants_skill_term.clone()

        self._cur_skill_step += 1

        # No reactive early-termination here: WaitSkill ends only when the
        # high-level policy re-decides (fixed control interval) or on the
        # max_skill_steps cap below.
        if self._max_skill_steps > 0 and self._cur_skill_step >= self._max_skill_steps:
            call_hl[:] = True
            self._cur_skill_step = 0
        elif call_hl.any():
            self._cur_skill_step = 0

        bad_should_terminate = torch.zeros(batch_size, dtype=torch.bool)
        return call_hl, bad_should_terminate, actions

    @classmethod
    def from_config(cls, config, observation_space, action_space, num_envs, full_config):
        """Create skill from config."""
        return cls(config, action_space, num_envs)

    # ---- shared turn-then-go helpers (world (x,z) -> base_velocity) ----
    @staticmethod
    def _robot_forward(yaw: float) -> np.ndarray:
        """World (x, z) forward direction of the robot from LocalizationSensor
        yaw (heading = -atan2(fz, fx) => forward = (cos yaw, -sin yaw))."""
        return np.array([math.cos(yaw), -math.sin(yaw)], dtype=np.float64)

    @staticmethod
    def _compute_turn(rel, turn_vel, robot_forward):
        """Angular-only command to rotate toward `rel` (mirrors
        OracleNavAction._compute_turn)."""
        is_left = np.cross(robot_forward, rel) > 0
        return [0.0, -turn_vel] if is_left else [0.0, turn_vel]


class BackOffSkill(SocialNavSkillBase):
    """Yield to the human by backing the robot to a point ~2 m behind the door on
    its start side, using the DEFAULT geodesic pathfinding (no RVO). The
    ``backoff_waypoint_delta`` sensor computes the validated target and the next
    path waypoint; this skill only realizes REVERSE turn-then-go: rotate so the
    robot's BACK faces the waypoint, then drive backward (negative linear). Once
    the target is reached it holds in place and hands control back to the HL
    switch policy (which resumes go_to_goal once the human is clear).

    Observation-only (no sim access): direction comes from the sensor's
    ``[wp_dx, wp_dz, remaining]``; ``remaining`` drives the stop and termination.
    """

    def __init__(self, config, action_space, batch_size, **kwargs):
        super().__init__(config, action_space, batch_size, **kwargs)
        sd = getattr(config, "skill_data", None) or {}

        def _p(key, default):
            try:
                return sd.get(key, default)
            except AttributeError:
                return getattr(sd, key, default)

        # `lin = -v / lin_speed` then BaseVelAction does clip(.,-1,1)*
        # longitudinal_lin_speed, so lin_speed must match that action's
        # longitudinal_lin_speed for `v` to be the reverse speed in m/s.
        self._lin_speed = float(_p("lin_speed", 10.0))
        self._max_back_speed = float(_p("max_back_speed", 1.0))
        self._turn_thresh = float(_p("turn_thresh", 0.1))
        self._turn_velocity = float(_p("turn_velocity", 1.0))
        # Stop forcing motion within this radius of the target.
        self._start_stop_radius = float(_p("start_stop_radius", 0.15))
        # Reached target (within this radius) -> hand control back to the HL.
        self._start_done_radius = float(_p("start_done_radius", 0.3))
        # When enabled, print a concise per-step trace for env 0 (debug/tuning).
        self._backoff_debug = bool(_p("backoff_debug", False))

    @property
    def required_obs_keys(self) -> List[str]:
        return ["localization_sensor", "backoff_waypoint_delta"]

    def _internal_act(self, observations, rnn_hidden_states, prev_actions, masks,
            cur_batch_idx=None, deterministic=False):
        batch_size = masks.shape[0]
        action = torch.zeros(batch_size, self._full_ac_size, device=masks.device)

        loc_all = _to_np(observations.get("localization_sensor")) if isinstance(observations, dict) else None
        wp_all = _to_np(observations.get("backoff_waypoint_delta")) if isinstance(observations, dict) else None

        if loc_all is None or wp_all is None:
            logger.warning(
                "[BackOffSkill] missing required sensors; keys=%s",
                sorted(observations.keys()) if isinstance(observations, dict) else type(observations),
            )
            return PolicyActionData(actions=action, rnn_hidden_states=rnn_hidden_states)

        lin_i = self._base_vel_start
        ang_i = self._base_vel_start + 1

        for j in range(batch_size):
            # World forward (x, z) of the robot from LocalizationSensor yaw.
            yaw = float(loc_all[j][3])
            robot_forward = np.array([math.cos(yaw), -math.sin(yaw)], dtype=np.float64)

            # Sensor: next geodesic waypoint delta (world x, z) + distance left.
            wp_delta = np.asarray(wp_all[j][:2], dtype=np.float64)
            remaining = float(wp_all[j][2])

            if remaining < self._start_stop_radius or np.linalg.norm(wp_delta) < 1e-6:
                # Arrived / nothing to do -> hold in place and wait.
                lin, ang = 0.0, 0.0
            else:
                # REVERSE turn-then-go: rotate so the robot's BACK faces the
                # waypoint (front faces -wp_delta), then drive BACKWARD.
                v = min(self._max_back_speed, remaining)  # ramp down near target
                back_target = -wp_delta
                if float(get_angle(robot_forward, back_target)) < self._turn_thresh:
                    lin, ang = -v / self._lin_speed, 0.0
                else:
                    vel = self._compute_turn(back_target, self._turn_velocity, robot_forward)
                    lin, ang = vel[0], vel[1]

            action[j, lin_i] = lin
            action[j, ang_i] = ang

            if self._backoff_debug and j == 0:
                print(
                    f"[BackOffSkill] env0 yaw={yaw:.2f} "
                    f"wp_delta=({wp_delta[0]:.2f},{wp_delta[1]:.2f}) "
                    f"remaining={remaining:.2f} lin={lin:.2f} ang={ang:.2f}",
                    flush=True,
                )

        return PolicyActionData(actions=action, rnn_hidden_states=rnn_hidden_states)

    def should_terminate(self, observations, rnn_hidden_states, prev_actions,
                         masks, hl_wants_skill_term, actions, **kwargs):
        batch_size = masks.shape[0]
        call_hl = hl_wants_skill_term.clone()
        self._cur_skill_step += 1

        wp_all = _to_np(observations.get("backoff_waypoint_delta")) if isinstance(observations, dict) else None
        for i in range(batch_size):
            if call_hl[i]:
                continue
            # Reached the backoff target -> hand back so the HL re-decides.
            if wp_all is not None and wp_all.shape[0] > i:
                if float(wp_all[i][2]) < self._start_done_radius:
                    call_hl[i] = True

        if self._max_skill_steps > 0 and self._cur_skill_step >= self._max_skill_steps:
            call_hl[:] = True
            self._cur_skill_step = 0
        elif call_hl.any():
            self._cur_skill_step = 0
        bad_should_terminate = torch.zeros(batch_size, dtype=torch.bool)
        return call_hl, bad_should_terminate, actions


class WaitSkill(SocialNavSkillBase):
    """
    Stay in place while rotating to face the robot goal.
    Terminates only when the high-level policy requests it.
    """

    @property
    def required_obs_keys(self) -> List[str]:
        return ["localization_sensor", "goal_world_delta"]

    def _internal_act(self, observations, rnn_hidden_states, prev_actions, masks,
            cur_batch_idx=None, deterministic=False):
        batch_size = masks.shape[0]
        action = torch.zeros(batch_size, self._full_ac_size, device=masks.device)

        loc = _to_np(observations.get("localization_sensor")) if isinstance(observations, dict) else None
        goal_delta = _to_np(observations.get("goal_world_delta")) if isinstance(observations, dict) else None
        if loc is not None and goal_delta is not None:
            for j in range(batch_size):
                yaw = float(loc[j][3])
                robot_forward = np.array([math.cos(yaw), -math.sin(yaw)])
                gd = np.asarray(goal_delta[j][:2], dtype=np.float64)
                if np.linalg.norm(gd) < 1e-6:
                    continue
                angle = float(get_angle(robot_forward, gd))
                is_left = np.cross(robot_forward, gd) > 0
                ang_vel = -angle / math.pi if is_left else angle / math.pi
                action[j, self._base_vel_start + 1] = float(np.clip(ang_vel, -1.0, 1.0))

        return PolicyActionData(actions=action, rnn_hidden_states=rnn_hidden_states)


class GoToGoalSkill(SocialNavSkillBase):
    """Navigate to the robot's goal along the DEFAULT navmesh geodesic (no RVO,
    no human avoidance). The ``goal_waypoint_delta`` sensor pathfinds each step
    and this skill realizes forward turn-then-go toward the next waypoint,
    outputting ``base_velocity`` ([linear, angular]) directly. Yielding to the
    human is the EXCLUSIVE job of the high-level planner (backoff / wait).

    Observation-only (no sim access), stateless: direction comes from the
    sensor's ``[wp_dx, wp_dz, remaining]``; ``remaining`` drives the stop and
    termination.
    """

    def __init__(self, config, action_space, batch_size, **kwargs):
        super().__init__(config, action_space, batch_size, **kwargs)
        sd = getattr(config, "skill_data", None) or {}

        def _p(key, default):
            try:
                return sd.get(key, default)
            except AttributeError:
                return getattr(sd, key, default)

        # `lin = v / lin_speed` then BaseVelAction does clip(.,-1,1)*
        # longitudinal_lin_speed, so lin_speed must match that action's
        # longitudinal_lin_speed for `v` to be the forward speed in m/s.
        self._lin_speed = float(_p("lin_speed", 10.0))
        self._max_speed = float(_p("max_speed", 1.0))
        self._turn_thresh = float(_p("turn_thresh", 0.1))
        self._turn_velocity = float(_p("turn_velocity", 1.0))
        # Stop forcing motion within this radius of the goal. Keep below the
        # social_nav_to_pos_succ success_distance (0.4) so success can fire.
        self._goal_stop_radius = float(_p("goal_stop_radius", 0.15))
        # Reached goal (within this radius) -> hand control back to the HL.
        self._goal_done_radius = float(_p("goal_done_radius", 0.3))
        # When enabled, print a concise per-step trace for env 0 (debug/tuning).
        self._goal_debug = bool(_p("goal_debug", False))

    @property
    def required_obs_keys(self) -> List[str]:
        return ["localization_sensor", "goal_waypoint_delta"]

    def _internal_act(self, observations, rnn_hidden_states, prev_actions, masks,
            cur_batch_idx=None, deterministic=False):
        batch_size = masks.shape[0]
        action = torch.zeros(batch_size, self._full_ac_size, device=masks.device)

        loc_all = _to_np(observations.get("localization_sensor")) if isinstance(observations, dict) else None
        wp_all = _to_np(observations.get("goal_waypoint_delta")) if isinstance(observations, dict) else None

        if loc_all is None or wp_all is None:
            logger.warning(
                "[GoToGoalSkill] missing required sensors; keys=%s",
                sorted(observations.keys()) if isinstance(observations, dict) else type(observations),
            )
            return PolicyActionData(actions=action, rnn_hidden_states=rnn_hidden_states)

        lin_i = self._base_vel_start
        ang_i = self._base_vel_start + 1

        for j in range(batch_size):
            # World forward (x, z) of the robot from LocalizationSensor yaw.
            yaw = float(loc_all[j][3])
            robot_forward = np.array([math.cos(yaw), -math.sin(yaw)], dtype=np.float64)

            # Sensor: next geodesic waypoint delta (world x, z) + distance left.
            wp_delta = np.asarray(wp_all[j][:2], dtype=np.float64)
            remaining = float(wp_all[j][2])

            if remaining < self._goal_stop_radius or np.linalg.norm(wp_delta) < 1e-6:
                # Arrived / nothing to do -> hold in place.
                lin, ang = 0.0, 0.0
            else:
                # Forward turn-then-go toward the waypoint.
                v = min(self._max_speed, remaining)  # ramp down near goal
                if float(get_angle(robot_forward, wp_delta)) < self._turn_thresh:
                    lin, ang = v / self._lin_speed, 0.0
                else:
                    vel = self._compute_turn(wp_delta, self._turn_velocity, robot_forward)
                    lin, ang = vel[0], vel[1]

            action[j, lin_i] = lin
            action[j, ang_i] = ang

            if self._goal_debug and j == 0:
                print(
                    f"[GoToGoalSkill] env0 yaw={yaw:.2f} "
                    f"wp_delta=({wp_delta[0]:.2f},{wp_delta[1]:.2f}) "
                    f"remaining={remaining:.2f} lin={lin:.2f} ang={ang:.2f}",
                    flush=True,
                )

        return PolicyActionData(actions=action, rnn_hidden_states=rnn_hidden_states)

    def should_terminate(self, observations, rnn_hidden_states, prev_actions,
                         masks, hl_wants_skill_term, actions, **kwargs):
        batch_size = masks.shape[0]
        call_hl = hl_wants_skill_term.clone()
        self._cur_skill_step += 1

        wp_all = _to_np(observations.get("goal_waypoint_delta")) if isinstance(observations, dict) else None
        for i in range(batch_size):
            if call_hl[i]:
                continue
            # Goal reached -> hand back so the HL re-decides. Reactive "human
            # got close" conditions are left out on purpose; switching is driven
            # by the fixed control interval.
            if wp_all is not None and wp_all.shape[0] > i:
                if float(wp_all[i][2]) < self._goal_done_radius:
                    call_hl[i] = True

        if self._max_skill_steps > 0 and self._cur_skill_step >= self._max_skill_steps:
            call_hl[:] = True
            self._cur_skill_step = 0
        elif call_hl.any():
            self._cur_skill_step = 0

        bad_should_terminate = torch.zeros(batch_size, dtype=torch.bool)
        return call_hl, bad_should_terminate, actions
