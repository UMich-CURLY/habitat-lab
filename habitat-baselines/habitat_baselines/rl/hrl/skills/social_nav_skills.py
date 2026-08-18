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
import os
import json
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

    # Proportional-turn gain: saturate the command (|ang|=1) once the heading
    # error exceeds ~pi/GAIN, so mid/large-angle turns run at full ang_speed
    # instead of a sluggish fraction; below that it scales down smoothly to
    # avoid overshoot/limit-cycling near the target. GAIN=2 -> saturate by 90 deg.
    _TURN_GAIN = 2.0

    @staticmethod
    def _proportional_turn(rel, robot_forward):
        """Proportional-with-gain angular command toward `rel`: full ang_speed
        for headings beyond ~pi/GAIN, smooth decel inside that band (no bang-bang
        limit cycle near the target)."""
        rel = np.asarray(rel, dtype=np.float64)
        if np.linalg.norm(rel) < 1e-6:
            return 0.0
        angle = float(get_angle(robot_forward, rel))
        is_left = np.cross(robot_forward, rel) > 0
        ang = SocialNavSkillBase._TURN_GAIN * angle / math.pi
        ang = -ang if is_left else ang
        return float(np.clip(ang, -1.0, 1.0))


class ClearCorridorYieldSkill(SocialNavSkillBase):
    """Yield to the human by clearing the human's path corridor -- moving to a
    validated start-side point that keeps clear of the human's route (NOT
    necessarily straight back: the motion may be lateral or oblique). The
    ``backoff_waypoint_delta`` sensor (``yield_mode=corridor``) computes the
    target and the next path waypoint; this skill realizes REVERSE turn-then-go
    toward it (drive so the robot's back faces the waypoint). With
    ``hold_at_target`` it holds at the target -- rotating to face the door if the
    5-D sensor provides the door direction -- and does NOT self-terminate on
    arrival; the HL re-decides on its regular control interval instead (restores
    the 1 Hz temporal abstraction, no 30 Hz replan storm).

    Observation-only (no sim access): direction comes from the sensor's
    ``[wp_dx, wp_dz, remaining(, door_dx, door_dz)]``.
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
        # Proportional turns (kills bang-bang limit cycles).
        self._proportional = bool(_p("proportional_turn", False))
        # Reverse-and-drive: back up WHILE turning (no dead in-place spin) with
        # a floored angular rate. Mirrors GoToGoal's turn-and-drive.
        self._drive_while_turning = bool(_p("drive_while_turning", False))
        self._min_turn_rate = float(_p("min_turn_rate", 0.0))
        # Hold at the target until the HL re-decides on its control interval
        # (do not self-terminate on arrival) -> restores 1 Hz decisions.
        self._hold_at_target = bool(_p("hold_at_target", False))
        # Reverse-safety floor: freeze instead of backing into a human closer
        # than this (m). The skill reverses blind (the waypoint is behind it);
        # with a non-reciprocal human whose route crosses the retreat line this
        # produced back-into-human collisions. 0.0 = off (legacy).
        self._human_stop_dist = float(_p("human_stop_dist", 0.0))
        # When enabled, print a concise per-step trace for env 0 (debug/tuning).
        self._backoff_debug = bool(_p("backoff_debug", False))
        # LL demo logging: when the env var LL_LOG is a path, append one jsonl
        # row per env step while this skill is active -- (local obs -> action)
        # pairs for distilling the scripted yield into a learned LL policy.
        self._ll_log = os.environ.get("LL_LOG", "")

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
                "[ClearCorridorYieldSkill] missing required sensors; keys=%s",
                sorted(observations.keys()) if isinstance(observations, dict) else type(observations),
            )
            return PolicyActionData(actions=action, rnn_hidden_states=rnn_hidden_states)

        lin_i = self._base_vel_start
        ang_i = self._base_vel_start + 1
        has_door = wp_all.shape[1] >= 5
        feat_all = (
            _to_np(observations.get("social_nav_policy_state"))
            if isinstance(observations, dict) and self._human_stop_dist > 0
            else None
        )

        for j in range(batch_size):
            # World forward (x, z) of the robot from LocalizationSensor yaw.
            yaw = float(loc_all[j][3])
            robot_forward = np.array([math.cos(yaw), -math.sin(yaw)], dtype=np.float64)

            # Sensor: next geodesic waypoint delta (world x, z) + distance left.
            wp_delta = np.asarray(wp_all[j][:2], dtype=np.float64)
            remaining = float(wp_all[j][2])

            if (
                feat_all is not None
                and float(feat_all[j][0]) < self._human_stop_dist
            ):
                # Human closer than the safety floor: freeze (never reverse
                # into a body). The HL keeps yielding until the human settles.
                lin, ang = 0.0, 0.0
            elif remaining < self._start_stop_radius or np.linalg.norm(wp_delta) < 1e-6:
                # Arrived. Optionally rotate to face the door while holding.
                lin, ang = 0.0, 0.0
                if self._hold_at_target and has_door:
                    door_dir = np.asarray(wp_all[j][3:5], dtype=np.float64)
                    ang = self._proportional_turn(door_dir, robot_forward)
            else:
                # REVERSE turn-then-go: rotate so the robot's BACK faces the
                # waypoint (front faces -wp_delta), then drive BACKWARD.
                v = min(self._max_back_speed, remaining)  # ramp down near target
                back_target = -wp_delta
                angle = float(get_angle(robot_forward, back_target))
                if self._proportional and self._drive_while_turning:
                    # Reverse-and-drive: rotate (floored) AND back up scaled by
                    # alignment — no dead in-place spin.
                    ang = self._proportional_turn(back_target, robot_forward)
                    if angle > self._turn_thresh and abs(ang) < self._min_turn_rate:
                        ang = math.copysign(self._min_turn_rate, ang)
                    lin = -(v / self._lin_speed) * max(0.0, math.cos(angle))
                elif angle < self._turn_thresh:
                    lin, ang = -v / self._lin_speed, 0.0
                    if self._proportional:
                        ang = self._proportional_turn(back_target, robot_forward)
                elif self._proportional:
                    lin, ang = 0.0, self._proportional_turn(back_target, robot_forward)
                else:
                    vel = self._compute_turn(back_target, self._turn_velocity, robot_forward)
                    lin, ang = vel[0], vel[1]

            action[j, lin_i] = lin
            action[j, ang_i] = ang

            if self._ll_log:
                lidar = observations.get("lidar_scan")
                feat = observations.get("social_nav_policy_state")
                if lidar is not None and feat is not None:
                    ns = observations.get("num_steps")
                    row = {
                        "lidar": [float(x) for x in _to_np(lidar)[j]],
                        "feat": [float(x) for x in _to_np(feat)[j][:4]],
                        "door": [float(wp_all[j][3]), float(wp_all[j][4])] if has_door else [0.0, 0.0],
                        "wp": [float(wp_all[j][0]), float(wp_all[j][1]), float(wp_all[j][2])],
                        "act": [float(lin), float(ang)],
                        "branch": int(wp_all[j][5]) if wp_all.shape[1] >= 6 else -1,
                        # Cumulative step counter: joins rows to episodes via
                        # the EVAL_STATS num_steps cumsum (same trick as the HL
                        # decision log), enabling failed-segment removal.
                        "t": int(_to_np(ns).flatten()[0]) if ns is not None else -1,
                        "env": j,
                    }
                    with open(self._ll_log, "a") as lf:
                        lf.write(json.dumps(row) + "\n")

            if self._backoff_debug and j == 0:
                print(
                    f"[ClearCorridorYieldSkill] env0 yaw={yaw:.2f} "
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

        if not self._hold_at_target:
            wp_all = _to_np(observations.get("backoff_waypoint_delta")) if isinstance(observations, dict) else None
            for i in range(batch_size):
                if call_hl[i]:
                    continue
                # Reached the yield target -> hand back so the HL re-decides.
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


class LearnedYieldSkill(SocialNavSkillBase):
    """LEARNED yield LL (L4): a small MLP distilled from the scripted
    ClearCorridorYieldSkill, consuming only deployment-legal observations --
    lidar_scan (16), human polar state (4) and the door direction -- and
    emitting base_vel [lin, ang] directly. No privileged pocket planner at
    execution time. Never self-terminates (hold semantics); the HL re-decides
    on its control interval.
    """

    def __init__(self, config, action_space, batch_size, **kwargs):
        super().__init__(config, action_space, batch_size, **kwargs)
        sd = getattr(config, "skill_data", None) or {}

        def _p(key, default):
            try:
                return sd.get(key, default)
            except AttributeError:
                return getattr(sd, key, default)

        # LL_MODEL env overrides the config path -- lets the ES driver evaluate
        # candidate checkpoints without touching yaml.
        self._model_path = os.environ.get("LL_MODEL", "") or str(
            _p("model_path", "/habitat-lab/hrl_pipeline/ll_yield_bc.pth"))
        self._max_range = float(_p("lidar_max_range", 3.0))
        # DAgger teacher-label freeze floor; must match the scripted skill's
        # human_stop_dist (v2 default 1.0).
        self._t_human_stop_dist = float(_p("human_stop_dist", 1.0))
        # DAgger: when LL_DAGGER is a path, log (obs -> SCRIPTED-teacher action)
        # for every state the LEARNED policy visits. The privileged yield sensor
        # keeps computing wp_delta regardless of who drives, so teacher labels
        # are free on the learner's own state distribution.
        self._dagger_log = os.environ.get("LL_DAGGER", "")
        d = torch.load(self._model_path, map_location="cpu")
        in_dim = int(d.get("in_dim", 22))
        import torch.nn as nn
        self._net = nn.Sequential(
            nn.Linear(in_dim, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, 2), nn.Tanh(),
        )
        sdict = {k.replace("net.", ""): v for k, v in d["model"].items()}
        self._net.load_state_dict(sdict)
        self._net.eval()
        # Joint HL-LL training hygiene: the LL is FROZEN during HL PPO. Skills
        # live in a plain dict (not registered modules) so params never reach
        # the optimizer anyway, but requires_grad=False makes it explicit and
        # future-proofs against a ModuleDict refactor.
        self._net.requires_grad_(False)

    @property
    def required_obs_keys(self) -> List[str]:
        return ["lidar_scan", "social_nav_policy_state", "backoff_waypoint_delta"]

    def _internal_act(self, observations, rnn_hidden_states, prev_actions, masks,
            cur_batch_idx=None, deterministic=False):
        batch_size = masks.shape[0]
        action = torch.zeros(batch_size, self._full_ac_size, device=masks.device)
        lidar = observations.get("lidar_scan") if isinstance(observations, dict) else None
        feat = observations.get("social_nav_policy_state") if isinstance(observations, dict) else None
        wp = observations.get("backoff_waypoint_delta") if isinstance(observations, dict) else None
        if lidar is None or feat is None:
            logger.warning("[LearnedYieldSkill] missing sensors; keys=%s",
                           sorted(observations.keys()) if isinstance(observations, dict) else type(observations))
            return PolicyActionData(actions=action, rnn_hidden_states=rnn_hidden_states)
        li = _to_np(lidar); fe = _to_np(feat)
        dw = _to_np(wp) if wp is not None else None
        xs = []
        for j in range(batch_size):
            l = np.asarray(li[j], dtype=np.float32) / self._max_range
            f = np.asarray(fe[j][:4], dtype=np.float32).copy()
            f[0] = min(f[0], 6.0) / 6.0
            f[1] /= np.pi
            f[2] /= np.pi
            f[3] = float(np.clip(f[3], -1.5, 1.5) / 1.5)
            if dw is not None and dw.shape[1] >= 5:
                door = np.asarray(dw[j][3:5], dtype=np.float32)
                n = float(np.linalg.norm(door))
                if n > 1e-6:
                    door = door / n
            else:
                door = np.zeros(2, dtype=np.float32)
            xs.append(np.concatenate([l, f, door]))
        dev = next(self._net.parameters()).device
        with torch.no_grad():
            out = self._net(torch.tensor(np.stack(xs), device=dev)).cpu()
        lin_i = self._base_vel_start
        for j in range(batch_size):
            action[j, lin_i] = float(out[j, 0])
            action[j, lin_i + 1] = float(out[j, 1])

        if self._dagger_log and dw is not None:
            loc = observations.get("localization_sensor")
            ln = _to_np(loc) if loc is not None else None
            if ln is not None:
                ns = observations.get("num_steps")
                with open(self._dagger_log, "a") as lf:
                    for j in range(batch_size):
                        t_lin, t_ang = self._teacher_action(
                            dw[j], float(ln[j][3]), float(fe[j][0])
                        )
                        row = {
                            "lidar": [float(x) for x in li[j]],
                            "feat": [float(x) for x in fe[j][:4]],
                            "door": [float(dw[j][3]), float(dw[j][4])] if dw.shape[1] >= 5 else [0.0, 0.0],
                            "act": [t_lin, t_ang],
                            "t": int(_to_np(ns).flatten()[0]) if ns is not None else -1,
                            "env": j,
                        }
                        lf.write(json.dumps(row) + "\n")
        return PolicyActionData(actions=action, rnn_hidden_states=rnn_hidden_states)

    def _teacher_action(self, wp_row, yaw, human_dist=np.inf):
        """The scripted ClearCorridorYieldSkill control law applied to the
        CURRENT state -- the DAgger label. Mirrors the scripted skill's
        branches; keep in sync (esp. the human_stop_dist freeze, without which
        DAgger teaches reversing into a too-close human)."""
        if human_dist < getattr(self, "_t_human_stop_dist", 1.0):
            return 0.0, 0.0
        robot_forward = np.array([math.cos(yaw), -math.sin(yaw)], dtype=np.float64)
        wp_delta = np.asarray(wp_row[:2], dtype=np.float64)
        remaining = float(wp_row[2])
        if remaining < 0.15 or np.linalg.norm(wp_delta) < 1e-6:
            lin, ang = 0.0, 0.0
            if len(wp_row) >= 5:
                door_dir = np.asarray(wp_row[3:5], dtype=np.float64)
                if np.linalg.norm(door_dir) > 1e-6:
                    ang = self._proportional_turn(door_dir, robot_forward)
            return float(lin), float(ang)
        v = min(1.0, remaining)
        back = -wp_delta
        angle = float(get_angle(robot_forward, back))
        ang = self._proportional_turn(back, robot_forward)
        if angle > 0.3 and abs(ang) < 0.35:
            ang = math.copysign(0.35, ang)
        lin = -(v / 10.0) * max(0.0, math.cos(angle))
        return float(lin), float(ang)

    def should_terminate(self, observations, rnn_hidden_states, prev_actions,
                         masks, hl_wants_skill_term, actions, **kwargs):
        batch_size = masks.shape[0]
        call_hl = hl_wants_skill_term.clone()
        self._cur_skill_step += 1
        if self._max_skill_steps > 0 and self._cur_skill_step >= self._max_skill_steps:
            call_hl[:] = True
            self._cur_skill_step = 0
        elif call_hl.any():
            self._cur_skill_step = 0
        bad = torch.zeros(batch_size, dtype=torch.bool)
        return call_hl, bad, actions


# Back-compat alias: the semantic role is "clear the human's corridor", but old
# configs (and hierarchical_policy's eval(skill_name)) still refer to BackOffSkill.
BackOffSkill = ClearCorridorYieldSkill


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
                # Same proportional-with-gain turn as go/yield (not sluggish).
                action[j, self._base_vel_start + 1] = self._proportional_turn(
                    gd, robot_forward
                )

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
        # Proportional turns (kills bang-bang limit cycles).
        self._proportional = bool(_p("proportional_turn", False))
        # Turn-and-drive: while proportional, drive with lin = v*cos(angle) so the
        # robot moves toward the goal WHILE turning (no dead in-place spin at the
        # start), and floor the angular rate so the alignment tail isn't sluggish.
        self._drive_while_turning = bool(_p("drive_while_turning", False))
        self._min_turn_rate = float(_p("min_turn_rate", 0.0))
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
                v = min(self._max_speed, remaining)  # ramp down near goal
                angle = float(get_angle(robot_forward, wp_delta))
                if self._proportional and self._drive_while_turning:
                    # Turn-and-drive: rotate (floored proportional) AND move
                    # forward scaled by alignment — no dead in-place spin.
                    ang = self._proportional_turn(wp_delta, robot_forward)
                    if angle > self._turn_thresh and abs(ang) < self._min_turn_rate:
                        ang = math.copysign(self._min_turn_rate, ang)
                    lin = (v / self._lin_speed) * max(0.0, math.cos(angle))
                elif angle < self._turn_thresh:
                    lin, ang = v / self._lin_speed, 0.0
                    if self._proportional:
                        ang = self._proportional_turn(wp_delta, robot_forward)
                elif self._proportional:
                    lin, ang = 0.0, self._proportional_turn(wp_delta, robot_forward)
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


class FlatNavSkill(LearnedYieldSkill):
    """Flat end-to-end BC baseline: one MLP drives base_vel for the WHOLE
    episode (no HL decisions, no skill decomposition). Input is the
    LearnedYieldSkill 22-D vector plus the goal in robot-frame polar (2), all
    deployment-legal. Evaluate with the harness in always_backoff mode so the
    HL is forced onto this skill every macro step.
    """

    @property
    def required_obs_keys(self) -> List[str]:
        return ["lidar_scan", "social_nav_policy_state",
                "backoff_waypoint_delta", "goal_world_delta",
                "localization_sensor"]

    def _internal_act(self, observations, rnn_hidden_states, prev_actions,
                      masks, cur_batch_idx=None, deterministic=False):
        batch_size = masks.shape[0]
        action = torch.zeros(batch_size, self._full_ac_size, device=masks.device)
        lidar = observations.get("lidar_scan")
        feat = observations.get("social_nav_policy_state")
        wp = observations.get("backoff_waypoint_delta")
        goal = observations.get("goal_world_delta")
        loc = observations.get("localization_sensor")
        if lidar is None or feat is None or goal is None or loc is None:
            logger.warning("[FlatNavSkill] missing sensors; keys=%s",
                           sorted(observations.keys()))
            return PolicyActionData(actions=action, rnn_hidden_states=rnn_hidden_states)
        li, fe, gl, ln = _to_np(lidar), _to_np(feat), _to_np(goal), _to_np(loc)
        dw = _to_np(wp) if wp is not None else None
        xs = []
        for j in range(batch_size):
            l = np.asarray(li[j], dtype=np.float32) / self._max_range
            f = np.asarray(fe[j][:4], dtype=np.float32).copy()
            f[0] = min(f[0], 6.0) / 6.0
            f[1] /= np.pi
            f[2] /= np.pi
            f[3] = float(np.clip(f[3], -1.5, 1.5) / 1.5)
            if dw is not None and dw.shape[1] >= 5:
                door = np.asarray(dw[j][3:5], dtype=np.float32)
                n = float(np.linalg.norm(door))
                if n > 1e-6:
                    door = door / n
            else:
                door = np.zeros(2, dtype=np.float32)
            yaw = float(ln[j][3])
            fwd = np.array([math.cos(yaw), -math.sin(yaw)], dtype=np.float32)
            lat = np.array([fwd[1], -fwd[0]], dtype=np.float32)
            g = np.asarray(gl[j][:2], dtype=np.float32)
            gf, gla = float(np.dot(g, fwd)), float(np.dot(g, lat))
            gd = min(float(np.hypot(gf, gla)), 6.0) / 6.0
            ga = float(np.arctan2(gla, gf)) / np.pi
            xs.append(np.concatenate([l, f, door, [gd, ga]]))
        dev = next(self._net.parameters()).device
        with torch.no_grad():
            out = self._net(torch.tensor(np.stack(xs), device=dev)).cpu()
        lin_i = self._base_vel_start
        for j in range(batch_size):
            action[j, lin_i] = float(out[j, 0])
            action[j, lin_i + 1] = float(out[j, 1])
        return PolicyActionData(actions=action, rnn_hidden_states=rnn_hidden_states)
