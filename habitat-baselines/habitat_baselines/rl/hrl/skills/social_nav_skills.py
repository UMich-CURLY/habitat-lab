"""
Low-level skills for social navigation with Spot robot.

Three discrete skills:
  1. BackOffSkill:  Retrace the recent trajectory backward to yield to the human.
  2. WaitSkill:     Stay in place while rotating to face the robot goal.
  3. GoToGoalSkill: Navigate toward the robot goal with **self-contained RVO/ORCA
                    collision avoidance**. The skill builds a small per-env ORCA
                    simulation (robot + human, no static obstacles -- walls are
                    handled downstream by the navmesh ``step_filter`` inside
                    ``BaseVelNonCylinderAction``) from observation sensors and
                    outputs ``base_velocity`` ([linear, angular]) directly.

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
    """Yield to the human by backing toward the robot's EPISODE START (spawn)
    using the SAME self-contained RVO/ORCA avoidance as ``GoToGoalSkill``, but
    driven in REVERSE: the ORCA preferred velocity points toward the spawn, and
    the world ORCA velocity is realized by rotating so the robot's BACK faces it
    and driving backward (negative linear). Net effect: the robot backs toward
    its spawn while its front keeps facing the human / original-goal side. ORCA
    still treats the human as an obstacle to avoid.

    Per-env state mirrors ``GoToGoalSkill``: one keyed ``RVOManager`` (agents
    ``"robot"``, ``"human"``), the last human position (finite-difference velocity
    estimate), and the last robot ORCA velocity (warm start). Hands control back
    to the HL switch policy once the human is far again (resume go_to_goal), the
    spawn is reached, or on timeout.
    """

    def __init__(self, config, action_space, batch_size, **kwargs):
        super().__init__(config, action_space, batch_size, **kwargs)
        # --- ORCA params (mirror GoToGoalSkill; the defaults already match this
        # env, so no `skill_data` block is required in the yaml) ---
        sd = getattr(config, "skill_data", None) or {}

        def _p(key, default):
            try:
                return sd.get(key, default)
            except AttributeError:
                return getattr(sd, key, default)

        self._rvo_radius = float(_p("rvo_radius", 0.4))
        self._rvo_human_radius = float(_p("rvo_human_radius", 0.4))
        self._rvo_max_speed = float(_p("rvo_max_speed", 1.0))
        self._rvo_neighbor_dist = float(_p("rvo_neighbor_dist", 2.0))
        self._rvo_max_neighbors = int(_p("rvo_max_neighbors", 10))
        self._rvo_time_horizon = float(_p("rvo_time_horizon", 2.0))
        self._rvo_time_horizon_obst = float(_p("rvo_time_horizon_obst", 4.0))
        # Stop forcing the ORCA preferred velocity within this radius of spawn.
        self._start_stop_radius = float(_p("start_stop_radius", 0.15))
        # Must match the env control step (ac_freq_ratio / ctrl_freq).
        self._rvo_time_step = float(_p("rvo_time_step", 1.0 / 30.0))
        # Must match the agent_0_base_velocity action's longitudinal_lin_speed so
        # the post-clip effective speed equals the ORCA-solved speed.
        self._rvo_lin_speed = float(_p("rvo_lin_speed", 10.0))
        self._turn_thresh = float(_p("turn_thresh", 0.1))
        self._turn_velocity = float(_p("turn_velocity", 1.0))
        # Reached spawn (within this radius) -> hand control back to the HL.
        self._start_done_radius = float(_p("start_done_radius", 0.3))
        # Resume go_to_goal once the human is farther than this (>= the HL's
        # backoff_exit_dist).
        self._human_clear_dist = float(_p("human_clear_dist", 1.2))
        # When enabled, print a concise per-step trace for env 0 (debug/tuning).
        self._rvo_debug = bool(_p("backoff_debug", False))

        # --- per-env state ---
        self._rvo = [None] * batch_size
        self._last_human_xz = [None] * batch_size
        self._last_robot_vel = [(0.0, 0.0)] * batch_size

    @property
    def required_obs_keys(self) -> List[str]:
        return ["localization_sensor", "other_agent_gps", "start_world_delta"]

    def on_enter(self, skill_args, batch_indices, observations,
                 rnn_hidden_states, prev_actions, skill_name=None):
        for e in batch_indices:
            if e < len(self._rvo):
                self._rvo[e] = None
                self._last_human_xz[e] = None
                self._last_robot_vel[e] = (0.0, 0.0)
        return rnn_hidden_states, prev_actions

    def _build_rvo(self, robot_xz, human_xz):
        # Lazy import so importing this module does not hard-require rvo2.
        from habitat.tasks.rearrange.social_nav.rvo_manager import RVOManager

        mgr = RVOManager(
            time_step=self._rvo_time_step,
            neighbor_dist=self._rvo_neighbor_dist,
            max_neighbors=self._rvo_max_neighbors,
            time_horizon=self._rvo_time_horizon,
            time_horizon_obst=self._rvo_time_horizon_obst,
            radius=self._rvo_radius,
            default_max_speed=self._rvo_max_speed,
            static_obstacles=None,
        )
        mgr.add_agent(
            "robot",
            (float(robot_xz[0]), float(robot_xz[1])),
            radius=self._rvo_radius,
            max_speed=self._rvo_max_speed,
        )
        mgr.add_agent(
            "human",
            (float(human_xz[0]), float(human_xz[1])),
            radius=self._rvo_human_radius,
            max_speed=self._rvo_max_speed,
        )
        return mgr

    def _internal_act(self, observations, rnn_hidden_states, prev_actions, masks,
            cur_batch_idx=None, deterministic=False):
        batch_size = masks.shape[0]
        action = torch.zeros(batch_size, self._full_ac_size, device=masks.device)
        if cur_batch_idx is None:
            cur_batch_idx = list(range(batch_size))

        loc_all = _to_np(observations.get("localization_sensor")) if isinstance(observations, dict) else None
        oag_all = _to_np(observations.get("other_agent_gps")) if isinstance(observations, dict) else None
        start_all = _to_np(observations.get("start_world_delta")) if isinstance(observations, dict) else None

        if loc_all is None or oag_all is None or start_all is None:
            logger.warning(
                "[BackOffSkill][RVO] missing required sensors; keys=%s",
                sorted(observations.keys()) if isinstance(observations, dict) else type(observations),
            )
            return PolicyActionData(actions=action, rnn_hidden_states=rnn_hidden_states)

        dt = self._rvo_time_step
        lin_i = self._base_vel_start
        ang_i = self._base_vel_start + 1

        for j in range(batch_size):
            e = cur_batch_idx[j]

            # Episode reset -> drop per-env ORCA state.
            if float(masks[j].reshape(-1)[0].item()) == 0.0:
                self._rvo[e] = None
                self._last_human_xz[e] = None
                self._last_robot_vel[e] = (0.0, 0.0)

            loc = loc_all[j]
            robot_xz = np.array([loc[0], loc[2]], dtype=np.float64)
            yaw = float(loc[3])
            # Derived from LocalizationSensor: heading = -atan2(fz, fx), so the
            # world forward (x, z) of the robot is (cos yaw, -sin yaw).
            robot_forward = np.array([math.cos(yaw), -math.sin(yaw)], dtype=np.float64)

            # other_agent_gps = (robot_xz - human_xz) -> human_xz = robot_xz - oag
            human_xz = robot_xz - np.asarray(oag_all[j][:2], dtype=np.float64)

            # Human velocity via finite difference (zero on first step / reset).
            if self._last_human_xz[e] is not None:
                human_vel = (human_xz - self._last_human_xz[e]) / dt
                sp = float(np.linalg.norm(human_vel))
                if sp > self._rvo_max_speed and sp > 1e-6:
                    human_vel = human_vel * (self._rvo_max_speed / sp)
            else:
                human_vel = np.zeros(2, dtype=np.float64)
            self._last_human_xz[e] = human_xz

            if self._rvo[e] is None:
                self._rvo[e] = self._build_rvo(robot_xz, human_xz)
            mgr = self._rvo[e]

            mgr.sync_agent_pose(
                "robot", (float(robot_xz[0]), float(robot_xz[1])), self._last_robot_vel[e]
            )
            mgr.sync_agent_pose(
                "human", (float(human_xz[0]), float(human_xz[1])),
                (float(human_vel[0]), float(human_vel[1])),
            )

            # Preferred velocity toward the SPAWN (world-frame delta from sensor).
            start_delta = np.asarray(start_all[j][:2], dtype=np.float64)
            rho = float(np.linalg.norm(start_delta))
            if rho >= self._start_stop_radius and rho > 1e-6:
                pref = start_delta * (self._rvo_max_speed / rho)
            else:
                pref = np.zeros(2, dtype=np.float64)
            mgr.set_pref_velocity("robot", (float(pref[0]), float(pref[1])))
            mgr.set_pref_velocity("human", (float(human_vel[0]), float(human_vel[1])))

            mgr.step()
            orca_vel = np.asarray(mgr.get_agent_velocity("robot"), dtype=np.float64)
            self._last_robot_vel[e] = (float(orca_vel[0]), float(orca_vel[1]))

            # Convert world ORCA velocity -> [linear, angular] via REVERSE
            # turn-then-go: rotate so the robot's BACK faces orca_vel (front faces
            # -orca_vel), then drive BACKWARD (negative linear) along orca_vel.
            dist = float(np.linalg.norm(orca_vel))
            if dist < 1e-3 or rho < self._start_stop_radius:
                lin, ang = 0.0, 0.0
            else:
                back_target = -orca_vel
                angle_to_target = float(get_angle(robot_forward, back_target))
                if angle_to_target < self._turn_thresh:
                    # Back aligned with orca_vel -> reverse at effective speed `dist`.
                    lin, ang = -dist / self._rvo_lin_speed, 0.0
                else:
                    # Rotate the front toward -orca_vel (== back toward orca_vel).
                    vel = self._compute_turn(back_target, self._turn_velocity, robot_forward)
                    lin, ang = vel[0], vel[1]

            action[j, lin_i] = lin
            action[j, ang_i] = ang

            if self._rvo_debug and e == 0:
                print(
                    f"[BackOffSkill][RVO] env0 robot_xz=({robot_xz[0]:.2f},{robot_xz[1]:.2f}) "
                    f"yaw={yaw:.2f} human_xz=({human_xz[0]:.2f},{human_xz[1]:.2f}) "
                    f"start_rho={rho:.2f} pref=({pref[0]:.2f},{pref[1]:.2f}) "
                    f"orca_vel=({orca_vel[0]:.2f},{orca_vel[1]:.2f}) lin={lin:.2f} ang={ang:.2f}",
                    flush=True,
                )

        return PolicyActionData(actions=action, rnn_hidden_states=rnn_hidden_states)

    def should_terminate(self, observations, rnn_hidden_states, prev_actions,
                         masks, hl_wants_skill_term, actions, **kwargs):
        batch_size = masks.shape[0]
        call_hl = hl_wants_skill_term.clone()
        self._cur_skill_step += 1

        start_all = _to_np(observations.get("start_world_delta")) if isinstance(observations, dict) else None
        for i in range(batch_size):
            if call_hl[i]:
                continue
            # Reached spawn (retreat complete / cannot continue) -> hand back so
            # the HL re-decides. The "human cleared" reactive condition is left
            # out on purpose; switching is driven by the fixed control interval.
            if start_all is not None and start_all.shape[0] > i:
                if float(np.linalg.norm(start_all[i][:2])) < self._start_done_radius:
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
    """
    Navigate toward the robot's goal with self-contained RVO/ORCA collision
    avoidance, outputting ``base_velocity`` ([linear, angular]) directly.

    Per-env state: one keyed ``RVOManager`` (agents ``"robot"``, ``"human"``),
    the last human position (for a finite-difference velocity estimate), and the
    last robot ORCA velocity (warm start). The ORCA sim is rebuilt lazily on
    episode reset / skill (re)entry.
    """

    def __init__(self, config, action_space, batch_size, **kwargs):
        super().__init__(config, action_space, batch_size, **kwargs)
        # --- ORCA params (from the skill config's `skill_data` escape hatch;
        # HrlDefinedSkillConfig is a closed schema, so custom keys live there) ---
        sd = getattr(config, "skill_data", None) or {}

        def _p(key, default):
            try:
                return sd.get(key, default)
            except AttributeError:
                return getattr(sd, key, default)

        self._rvo_radius = float(_p("rvo_radius", 0.4))
        self._rvo_human_radius = float(_p("rvo_human_radius", 0.4))
        self._rvo_max_speed = float(_p("rvo_max_speed", 1.0))
        self._rvo_neighbor_dist = float(_p("rvo_neighbor_dist", 2.0))
        self._rvo_max_neighbors = int(_p("rvo_max_neighbors", 10))
        self._rvo_time_horizon = float(_p("rvo_time_horizon", 2.0))
        self._rvo_time_horizon_obst = float(_p("rvo_time_horizon_obst", 4.0))
        self._rvo_goal_stop_radius = float(_p("rvo_goal_stop_radius", 0.3))
        # Must match the env control step (ac_freq_ratio / ctrl_freq).
        self._rvo_time_step = float(_p("rvo_time_step", 1.0 / 30.0))
        # Must match the agent_0_base_velocity action's longitudinal_lin_speed so
        # the post-clip effective forward speed equals the ORCA-solved speed.
        self._rvo_lin_speed = float(_p("rvo_lin_speed", 10.0))
        self._turn_thresh = float(_p("turn_thresh", 0.1))
        self._turn_velocity = float(_p("turn_velocity", 1.0))
        self._goal_done_radius = float(_p("goal_done_radius", 0.5))
        # Hand control back to the HL switch policy once the human is closer than
        # this (so it can switch to backoff). Keep >= the CyclingHighLevelPolicy
        # backoff_enter_dist.
        self._human_switch_dist = float(_p("human_switch_dist", 1.0))
        # When enabled, print a concise per-step trace for env 0 (debug/tuning).
        self._rvo_debug = bool(_p("rvo_debug", False))

        # --- per-env state ---
        self._rvo = [None] * batch_size
        self._last_human_xz = [None] * batch_size
        self._last_robot_vel = [(0.0, 0.0)] * batch_size

    @property
    def required_obs_keys(self) -> List[str]:
        return [
            "localization_sensor",
            "other_agent_gps",
            "goal_world_delta",
            "humanoid_detector_sensor",
        ]

    def on_enter(self, skill_args, batch_indices, observations,
                 rnn_hidden_states, prev_actions, skill_name=None):
        for e in batch_indices:
            if e < len(self._rvo):
                self._rvo[e] = None
                self._last_human_xz[e] = None
                self._last_robot_vel[e] = (0.0, 0.0)
        return rnn_hidden_states, prev_actions

    def _build_rvo(self, robot_xz, human_xz):
        # Lazy import so importing this module does not hard-require rvo2.
        from habitat.tasks.rearrange.social_nav.rvo_manager import RVOManager

        mgr = RVOManager(
            time_step=self._rvo_time_step,
            neighbor_dist=self._rvo_neighbor_dist,
            max_neighbors=self._rvo_max_neighbors,
            time_horizon=self._rvo_time_horizon,
            time_horizon_obst=self._rvo_time_horizon_obst,
            radius=self._rvo_radius,
            default_max_speed=self._rvo_max_speed,
            static_obstacles=None,
        )
        mgr.add_agent(
            "robot",
            (float(robot_xz[0]), float(robot_xz[1])),
            radius=self._rvo_radius,
            max_speed=self._rvo_max_speed,
        )
        mgr.add_agent(
            "human",
            (float(human_xz[0]), float(human_xz[1])),
            radius=self._rvo_human_radius,
            max_speed=self._rvo_max_speed,
        )
        return mgr

    @staticmethod
    def _compute_turn(rel, turn_vel, robot_forward):
        # Mirrors OracleNavAction._compute_turn.
        is_left = np.cross(robot_forward, rel) > 0
        return [0.0, -turn_vel] if is_left else [0.0, turn_vel]

    def _internal_act(self, observations, rnn_hidden_states, prev_actions, masks,
            cur_batch_idx=None, deterministic=False):
        batch_size = masks.shape[0]
        action = torch.zeros(batch_size, self._full_ac_size, device=masks.device)
        if cur_batch_idx is None:
            cur_batch_idx = list(range(batch_size))

        loc_all = _to_np(observations.get("localization_sensor")) if isinstance(observations, dict) else None
        oag_all = _to_np(observations.get("other_agent_gps")) if isinstance(observations, dict) else None
        goal_all = _to_np(observations.get("goal_world_delta")) if isinstance(observations, dict) else None

        if loc_all is None or oag_all is None or goal_all is None:
            logger.warning(
                "[GoToGoalSkill][RVO] missing required sensors; keys=%s",
                sorted(observations.keys()) if isinstance(observations, dict) else type(observations),
            )
            return PolicyActionData(actions=action, rnn_hidden_states=rnn_hidden_states)

        dt = self._rvo_time_step
        lin_i = self._base_vel_start
        ang_i = self._base_vel_start + 1

        for j in range(batch_size):
            e = cur_batch_idx[j]

            # Episode reset -> drop per-env ORCA state.
            if float(masks[j].reshape(-1)[0].item()) == 0.0:
                self._rvo[e] = None
                self._last_human_xz[e] = None
                self._last_robot_vel[e] = (0.0, 0.0)

            loc = loc_all[j]
            robot_xz = np.array([loc[0], loc[2]], dtype=np.float64)
            yaw = float(loc[3])
            # Derived from LocalizationSensor: heading = -atan2(fz, fx), so the
            # world forward (x, z) of the robot is (cos yaw, -sin yaw).
            robot_forward = np.array([math.cos(yaw), -math.sin(yaw)], dtype=np.float64)

            # other_agent_gps = (robot_xz - human_xz) -> human_xz = robot_xz - oag
            human_xz = robot_xz - np.asarray(oag_all[j][:2], dtype=np.float64)

            # Human velocity via finite difference (zero on first step / reset).
            if self._last_human_xz[e] is not None:
                human_vel = (human_xz - self._last_human_xz[e]) / dt
                sp = float(np.linalg.norm(human_vel))
                if sp > self._rvo_max_speed and sp > 1e-6:
                    human_vel = human_vel * (self._rvo_max_speed / sp)
            else:
                human_vel = np.zeros(2, dtype=np.float64)
            self._last_human_xz[e] = human_xz

            if self._rvo[e] is None:
                self._rvo[e] = self._build_rvo(robot_xz, human_xz)
            mgr = self._rvo[e]

            mgr.sync_agent_pose(
                "robot", (float(robot_xz[0]), float(robot_xz[1])), self._last_robot_vel[e]
            )
            mgr.sync_agent_pose(
                "human", (float(human_xz[0]), float(human_xz[1])),
                (float(human_vel[0]), float(human_vel[1])),
            )

            # Preferred velocity toward the goal (world-frame delta from sensor).
            goal_delta = np.asarray(goal_all[j][:2], dtype=np.float64)
            rho = float(np.linalg.norm(goal_delta))
            if rho >= self._rvo_goal_stop_radius and rho > 1e-6:
                pref = goal_delta * (self._rvo_max_speed / rho)
            else:
                pref = np.zeros(2, dtype=np.float64)
            mgr.set_pref_velocity("robot", (float(pref[0]), float(pref[1])))
            mgr.set_pref_velocity("human", (float(human_vel[0]), float(human_vel[1])))

            mgr.step()
            orca_vel = np.asarray(mgr.get_agent_velocity("robot"), dtype=np.float64)
            self._last_robot_vel[e] = (float(orca_vel[0]), float(orca_vel[1]))

            # Convert world ORCA velocity -> [linear, angular] via turn-then-go.
            dist = float(np.linalg.norm(orca_vel))
            if dist < 1e-3 or rho < self._rvo_goal_stop_radius:
                lin, ang = 0.0, 0.0
            else:
                angle_to_target = float(get_angle(robot_forward, orca_vel))
                if angle_to_target < self._turn_thresh:
                    # BaseVelAction clips lin to [-1,1] then *longitudinal_lin_speed,
                    # so dist/lin_speed yields an effective forward speed of `dist`.
                    lin, ang = dist / self._rvo_lin_speed, 0.0
                else:
                    vel = self._compute_turn(orca_vel, self._turn_velocity, robot_forward)
                    lin, ang = vel[0], vel[1]

            action[j, lin_i] = lin
            action[j, ang_i] = ang

            if self._rvo_debug and e == 0:
                print(
                    f"[GoToGoalSkill][RVO] env0 robot_xz=({robot_xz[0]:.2f},{robot_xz[1]:.2f}) "
                    f"yaw={yaw:.2f} human_xz=({human_xz[0]:.2f},{human_xz[1]:.2f}) "
                    f"goal_rho={rho:.2f} pref=({pref[0]:.2f},{pref[1]:.2f}) "
                    f"orca_vel=({orca_vel[0]:.2f},{orca_vel[1]:.2f}) lin={lin:.2f} ang={ang:.2f}",
                    flush=True,
                )

        return PolicyActionData(actions=action, rnn_hidden_states=rnn_hidden_states)

    def should_terminate(self, observations, rnn_hidden_states, prev_actions,
                         masks, hl_wants_skill_term, actions, **kwargs):
        batch_size = masks.shape[0]
        call_hl = hl_wants_skill_term.clone()
        self._cur_skill_step += 1

        goal_all = _to_np(observations.get("goal_world_delta")) if isinstance(observations, dict) else None

        for i in range(batch_size):
            if call_hl[i]:
                continue
            # Goal reached (task complete) -> hand back so the HL re-decides.
            # Reactive "human got close" / detector conditions are left out on
            # purpose; switching is driven by the fixed control interval, while
            # the in-skill RVO keeps avoiding the human every step.
            if goal_all is not None and goal_all.shape[0] > i:
                if float(np.linalg.norm(goal_all[i][:2])) < self._goal_done_radius:
                    call_hl[i] = True

        if self._max_skill_steps > 0 and self._cur_skill_step >= self._max_skill_steps:
            call_hl[:] = True
            self._cur_skill_step = 0
        elif call_hl.any():
            self._cur_skill_step = 0

        bad_should_terminate = torch.zeros(batch_size, dtype=torch.bool)
        return call_hl, bad_should_terminate, actions
