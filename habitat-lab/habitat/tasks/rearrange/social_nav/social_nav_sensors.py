#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import magnum as mn
import math
import numpy as np
from gym import spaces

import habitat_sim
from habitat.articulated_agents.humanoids.kinematic_humanoid import (
    KinematicHumanoid,
)
from habitat.core.embodied_task import Measure
from habitat.core.registry import registry
from habitat.core.simulator import Sensor, SensorTypes
from habitat.tasks.rearrange.multi_agent_sensors import DidAgentsCollide
from habitat.tasks.rearrange.rearrange_sensors import RearrangeReward
from habitat.tasks.rearrange.social_nav.utils import (
    robot_human_vec_dot_product,
)
from habitat.tasks.rearrange.sub_tasks.nav_to_obj_sensors import (
    DistToGoal,
    NavToPosSucc,
    RotDistToGoal,
)
from habitat.tasks.rearrange.utils import (
    UsesArticulatedAgentInterface,
    batch_transform_point,
)
from habitat.tasks.utils import cartesian_to_polar
from IPython import embed
BASE_ACTION_NAME = "base_velocity"


@registry.register_measure
class SocialNavReward(RearrangeReward):
    """
    Reward that gives a continuous reward for the social navigation task.
    """

    cls_uuid: str = "social_nav_reward"

    @staticmethod
    def _get_uuid(*args, **kwargs):
        return SocialNavReward.cls_uuid

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        config = kwargs["config"]
        # Get the config and setup the hyperparameters
        self._config = config
        self._sim = kwargs["sim"]
        self._safe_dis_min = config.safe_dis_min
        self._safe_dis_max = config.safe_dis_max
        self._safe_dis_reward = config.safe_dis_reward
        self._facing_human_dis = config.facing_human_dis
        self._facing_human_reward = config.facing_human_reward
        self._toward_human_reward = config.toward_human_reward
        self._near_human_bonus = config.near_human_bonus
        self._explore_reward = config.explore_reward
        self._use_geo_distance = config.use_geo_distance
        self._collide_penalty = config.collide_penalty
        # Dense shaping coefficients (see update_metric):
        #   goal_progress_reward: reward per metre of progress toward the goal
        #   backoff_reward: reward per metre of backing away from a too-close human
        self._goal_progress_reward = getattr(config, "goal_progress_reward", 1.0)
        self._backoff_reward = getattr(config, "backoff_reward", 1.0)
        # Awareness radius for the dense yield shaping (>= safe_dis_min).
        self._yield_dis = getattr(config, "yield_dis", self._safe_dis_min)
        # Whether a robot-human collision ends the episode (default) or only
        # incurs collide_penalty and lets the episode continue.
        self._end_on_collide = getattr(config, "end_on_collide", True)
        # Layer-2/3 shaping (see default_structured_configs.SocialNavReward).
        self._eff_success_reward = getattr(config, "eff_success_reward", 0.0)
        self._eff_step_cap = getattr(config, "eff_step_cap", 1200.0)
        self._corridor_coef = getattr(config, "corridor_potential_coef", 0.0)
        self._corridor_safe = getattr(config, "corridor_safe_clear", 1.0)
        self._release_bonus = getattr(config, "release_bonus", 0.0)
        self._release_mpd = getattr(config, "release_mpd_min", 0.75)
        self._release_hold = getattr(config, "release_hold_steps", 30)
        self._v_robot = getattr(config, "release_robot_speed", 0.008)
        self.interm_goal_bonus = 1.0   #Change to get from config
        # Record the previous distance to human
        self._prev_dist = -1.0
        self._prev_dist_to_goal = -1.0
        self._robot_idx = config.robot_idx
        self._human_idx = config.human_idx
        # Add exploration reward dictionary tracker
        self._visited_pos = set()

    def reset_metric(self, *args, episode, task, observations, **kwargs):
        self._prev_dist = -1.0
        self._prev_dist_to_goal = -1.0
        # Per-episode running sums of each reward component. The measure's own
        # _metric is the PER-STEP reward (the base class assigns, not
        # accumulates), and the eval stats only capture a measure's final-step
        # value -- so component totals have to be accumulated explicitly here.
        # Read out by SocialNavRewardBreakdown.
        self.comp_sums = {
            "goal_progress": 0.0,
            "backoff": 0.0,
            "proximity": 0.0,
            "collide": 0.0,
            "efficiency": 0.0,
            "corridor": 0.0,
            "release": 0.0,
        }
        # Layer-2/3 per-episode state.
        self._num_steps = 0
        self._eff_paid = False
        self._prev_phi = None
        self._hold_steps = 0
        self._release_paid = False
        self._prev_human_xz = None
        self._human_speed_ema = 0.0
        super().reset_metric(
            *args,
            episode=episode,
            task=task,
            observations=observations,
            **kwargs,
        )
        # Reset the location visit tracker for the agent
        self._visited_pos = set()

    @staticmethod
    def _dist_to_polyline(p, poly):
        """Min distance from XZ point p to polyline poly[N,2] (segment-exact)."""
        a, b = poly[:-1], poly[1:]
        ab = b - a
        t = np.clip(
            ((p - a) * ab).sum(1) / ((ab * ab).sum(1) + 1e-9), 0.0, 1.0
        )
        proj = a + t[:, None] * ab
        return float(np.min(np.linalg.norm(proj - p, axis=1)))

    @staticmethod
    def _walk_polyline(poly, dists):
        """Points reached after walking each of dists along poly from start."""
        seg = np.linalg.norm(np.diff(poly, axis=0), axis=1)
        cum = np.concatenate([[0.0], np.cumsum(seg)])
        d = np.clip(dists, 0.0, cum[-1])
        idx = np.clip(np.searchsorted(cum, d, side="right") - 1, 0, len(seg) - 1)
        frac = (d - cum[idx]) / (seg[idx] + 1e-9)
        return poly[idx] + frac[:, None] * (poly[idx + 1] - poly[idx])

    def _human_plan(self, human_pos, task):
        """Human's REMAINING planned path as an XZ polyline (its oracle nav
        plan to its goal -- privileged sim info, fine for a training reward)."""
        goal = task.my_nav_to_info.human_info.nav_goal_pos
        path = habitat_sim.ShortestPath()
        path.requested_start = np.array(human_pos)
        path.requested_end = np.array(goal)
        if self._sim.pathfinder.find_path(path) and len(path.points) >= 2:
            return np.array([[p[0], p[2]] for p in path.points])
        return np.array(
            [[human_pos[0], human_pos[2]], [float(goal[0]), float(goal[2])]]
        )

    def update_metric(self, *args, episode, task, observations, **kwargs):
        super().update_metric(
            *args,
            episode=episode,
            task=task,
            observations=observations,
            **kwargs,
        )

        # Get the pos
        use_k_human = f"agent_{self._human_idx}_localization_sensor"
        human_pos = observations[use_k_human][:3]
        use_k_robot = f"agent_{self._robot_idx}_localization_sensor"
        robot_pos = observations[use_k_robot][:3]

        # If we consider using geo distance
        if self._use_geo_distance:
            path = habitat_sim.ShortestPath()
            path.requested_start = np.array(robot_pos)
            path.requested_end = human_pos
            found_path = self._sim.pathfinder.find_path(path)

        # Compute the distance between the robot and the human
        if self._use_geo_distance and found_path:
            dis = self._sim.geodesic_distance(robot_pos, human_pos)
        else:
            dis = np.linalg.norm(human_pos - robot_pos)

        # Start social nav reward
        social_nav_reward = 0.0
        # Snapshot so the first-step zeroing below can roll the component sums
        # back too (otherwise the breakdown would not add up to `reward`).
        _comp_snapshot = dict(self.comp_sums)

        ### CADRL Reward structure ###

        
        # Componet 1: Social nav reward three stage design
        # if dis >= self._safe_dis_min and dis < self._safe_dis_max:
        #     # If the distance is within the safety interval
        #     social_nav_reward += self._safe_dis_reward
        # elif dis < self._safe_dis_min:
        #     # If the distance is too samll
        #     social_nav_reward += dis - self._prev_dist
        # else:
        #     # if the distance is too large
        #     social_nav_reward += self._prev_dist - dis
        # social_nav_reward = (
        #     self._config.toward_human_reward * social_nav_reward
        # )

        # # Componet 2: Social nav reward for facing human
        # if dis < self._facing_human_dis and self._facing_human_reward != -1:
        #     base_T = self._sim.get_agent_data(
        #         self.agent_id
        #     ).articulated_agent.base_transformation
        #     # Dot product
        #     social_nav_reward += (
        #         self._facing_human_reward
        #         * robot_human_vec_dot_product(robot_pos, human_pos, base_T)
        #     )

        # # Componet 3: Social nav reward bonus for getting closer to human
        # if (
        #     dis < self._facing_human_dis
        #     and self._facing_human_reward != -1
        #     and self._near_human_bonus != -1
        # ):
        #     social_nav_reward += self._near_human_bonus

        # # Componet 4: Social nav reward for exploration
        # # There is no exploration reward once the agent finds the human
        # # round off float to nearest 0.5 in python
        robot_pos_key = (
            round(robot_pos[0] * 2) / 2,
            round(robot_pos[2] * 2) / 2,
        )
        social_nav_stats = task.measurements.measures[
            SocialNavStats.cls_uuid
        ].get_metric()
        if (
            self._explore_reward != -1
            and robot_pos_key not in self._visited_pos
        ):
            self._visited_pos.add(robot_pos_key)
            # Give the reward if the agent visits the new location
            social_nav_reward += self._explore_reward

        # Distance from the robot to its own navigation goal (XZ plane).
        goal = task.my_nav_to_info.robot_info.nav_goal_pos
        dist_to_goal = np.linalg.norm(np.array(robot_pos)[[0, 2]] - goal[[0, 2]])

        # Componet 5: Collision detection for two agents
        did_collide = task.measurements.measures[
            DidAgentsCollide._get_uuid()
        ].get_metric()
        if did_collide:
            social_nav_reward -= self._collide_penalty
            self.comp_sums["collide"] -= self._collide_penalty
            if self._end_on_collide:
                task.should_end = True
        #### CADRL style reward ####
        # Shaping runs unless the episode is being terminated by this collision.
        # With end_on_collide=False the robot keeps a dense learning signal
        # right through a (non-terminal) collision, so it can recover and still
        # experience the human clearing — instead of every contact truncating
        # the episode with a large negative return.
        if not (did_collide and self._end_on_collide):
            # Dense goal-progress reward: positive when the robot reduces the
            # distance to its goal, negative when it moves away. This gives the
            # high-level policy a gradient toward picking `go_to_goal` when it is
            # safe to do so (otherwise the only goal signal is the sparse success
            # bonus, which the policy never reaches).
            if self._prev_dist_to_goal >= 0.0:
                _gp = self._goal_progress_reward * (
                    self._prev_dist_to_goal - dist_to_goal
                )
                social_nav_reward += _gp
                self.comp_sums["goal_progress"] += _gp

            # Proximity penalty: being closer than the safety radius is bad.
            if dis < self._safe_dis_min:
                social_nav_reward -= (self._safe_dis_min - dis)
                self.comp_sums["proximity"] -= (self._safe_dis_min - dis)

            # Dense yield shaping (potential-based): reward actively increasing
            # the robot-human distance whenever the human is within the wider
            # awareness radius `yield_dis`, not only inside safe_dis_min. This
            # gives an early, dense gradient to start yielding as the human
            # approaches — instead of the switch paying off only at the terminal
            # success. Being potential-based (Φ = backoff_reward * dis, so the
            # per-step reward telescopes), it densifies credit assignment
            # without rewarding fleeing over a full episode.
            if dis < self._yield_dis and self._prev_dist >= 0.0:
                _bo = self._backoff_reward * (dis - self._prev_dist)
                social_nav_reward += _bo
                self.comp_sums["backoff"] += _bo

        # --- Layer-2/3 shaping (2026-08 yield-geometry diagnosis) ---
        self._num_steps += 1
        robot_xz = np.array([robot_pos[0], robot_pos[2]], dtype=np.float64)
        human_xz = np.array([human_pos[0], human_pos[2]], dtype=np.float64)
        if self._prev_human_xz is not None:
            self._human_speed_ema = (
                0.9 * self._human_speed_ema
                + 0.1 * float(np.linalg.norm(human_xz - self._prev_human_xz))
            )
        self._prev_human_xz = human_xz

        plan = None
        if self._corridor_coef > 0 or self._release_bonus > 0:
            plan = self._human_plan(human_pos, task)
            plan_clear = self._dist_to_polyline(robot_xz, plan)

        # Corridor potential: credit for LEAVING the human's planned corridor,
        # nothing for retreating beyond corridor_safe, negative for re-entry.
        if self._corridor_coef > 0 and not (
            did_collide and self._end_on_collide
        ):
            phi = -max(0.0, self._corridor_safe - plan_clear)
            if self._prev_phi is not None:
                _cor = self._corridor_coef * (phi - self._prev_phi)
                social_nav_reward += _cor
                self.comp_sums["corridor"] += _cor
            self._prev_phi = phi

        # Timely-release bonus: once per episode, on resuming goal progress
        # after a hold, IF going now is kinematically clear of the human's
        # planned motion (mpd_go >= release_mpd_min).
        if self._release_bonus > 0:
            # Scale-aware: nominal speed is ~0.008 m/step, so an absolute
            # threshold (e.g. 1 cm/step) would never fire.
            progressing = (
                self._prev_dist_to_goal >= 0.0
                and self._prev_dist_to_goal - dist_to_goal
                > 0.5 * self._v_robot
            )
            if (
                progressing
                and not self._release_paid
                and self._hold_steps >= self._release_hold
            ):
                to_goal = np.array(goal)[[0, 2]] - robot_xz
                n = float(np.linalg.norm(to_goal))
                tau = np.arange(0.0, 300.0, 5.0)
                adv = np.minimum(self._v_robot * tau, n)
                rpos = robot_xz + (to_goal / (n + 1e-9)) * adv[:, None]
                hpos = self._walk_polyline(plan, self._human_speed_ema * tau)
                mpd_go = float(
                    np.min(np.linalg.norm(rpos - hpos, axis=1))
                )
                if mpd_go >= self._release_mpd:
                    social_nav_reward += self._release_bonus
                    self.comp_sums["release"] += self._release_bonus
                    self._release_paid = True
            self._hold_steps = 0 if progressing else self._hold_steps + 1

        # Success-conditioned efficiency: same-step success recomputation
        # (SocialNavToPosSucc = dist_to_goal < success_distance, but it
        # updates AFTER this measure and the episode ends on success, so the
        # measure's own flag would arrive one step too late).
        if self._eff_success_reward > 0 and not self._eff_paid:
            succ_cfg = task.measurements.measures[
                "social_nav_to_pos_success"
            ]._config
            if dist_to_goal < succ_cfg.success_distance:
                _eff = self._eff_success_reward * max(
                    0.0, 1.0 - self._num_steps / self._eff_step_cap
                )
                social_nav_reward += _eff
                self.comp_sums["efficiency"] += _eff
                self._eff_paid = True

        # No shaping on the first step (no valid previous distances yet).
        if self._prev_dist < 0 or self._prev_dist_to_goal < 0.0:
            social_nav_reward = 0.0
            self.comp_sums = _comp_snapshot

        self._metric += social_nav_reward
        # Update the distance
        self._prev_dist = dis  # type: ignore
        self._prev_dist_to_goal = dist_to_goal


@registry.register_measure
class SocialNavRewardBreakdown(Measure):
    """Per-episode running sums of each `SocialNavReward` component.

    `SocialNavReward._metric` is the PER-STEP reward and the evaluator only
    stores a measure's final-step value, so component totals are invisible
    downstream. This measure republishes the running sums that SocialNavReward
    accumulates, as a dict -- `extract_scalars_from_info` flattens it into
    `social_nav_reward_breakdown.goal_progress` etc., so per-episode eval stats
    carry the full decomposition.

    The two env-level terms (slack, success bonus) are NOT here: they are added
    outside the measure (habitat/core/environments.py) and are exactly
    recoverable as slack_reward * num_steps and success_reward * success.
    """

    cls_uuid: str = "social_nav_reward_breakdown"

    @staticmethod
    def _get_uuid(*args, **kwargs):
        return SocialNavRewardBreakdown.cls_uuid

    def reset_metric(self, *args, task, **kwargs):
        task.measurements.check_measure_dependencies(
            self.uuid, [SocialNavReward.cls_uuid]
        )
        self.update_metric(*args, task=task, **kwargs)

    def update_metric(self, *args, task, **kwargs):
        src = task.measurements.measures[SocialNavReward.cls_uuid]
        self._metric = dict(getattr(src, "comp_sums", {}) or {})


@registry.register_measure
class SocialNavStats(UsesArticulatedAgentInterface, Measure):
    """
    The measure for social navigation
    """

    cls_uuid: str = "social_nav_stats"

    def __init__(self, sim, config, *args, **kwargs):
        super().__init__(**kwargs)
        self._sim = sim
        self._config = config

        # Hyper-parameters
        self._check_human_in_frame = self._config.check_human_in_frame
        self._min_dis_human = self._config.min_dis_human
        self._max_dis_human = self._config.max_dis_human
        self._human_id = self._config.human_id
        self._human_detect_threshold = (
            self._config.human_detect_pixel_threshold
        )
        self._total_step = self._config.total_steps
        self._dis_threshold_for_backup_yield = (
            self._config.dis_threshold_for_backup_yield
        )
        self._min_abs_vel_for_yield = self._config.min_abs_vel_for_yield
        self._robot_face_human_threshold = (
            self._config.robot_face_human_threshold
        )
        self._enable_shortest_path_computation = (
            self._config.enable_shortest_path_computation
        )
        self._robot_idx = config.robot_idx
        self._human_idx = config.human_idx

        # For the variable tracking
        self._val_dict = {
            "min_start_end_episode_step": float("inf"),
            "has_found_human": False,
            "has_found_human_step": self._total_step,
            "found_human_times": 0,
            "after_found_human_times": 0,
            "step": 0,
            "step_after_found": 1,
            "dis": 0,
            "dis_after_found": 0,
            "backup_count": 0,
            "yield_count": 0,
        }

        # Robot's info
        self._prev_robot_base_T = None
        self._robot_init_pos = None
        self._robot_init_trans = None

        # Store pos of human and robot
        self.human_pos_list = []
        self.robot_pos_list = []

    @staticmethod
    def _get_uuid(*args, **kwargs):
        return SocialNavStats.cls_uuid

    def reset_metric(self, *args, task, **kwargs):
        if (self.agent_id is None):
            self.agent_id = 0
        robot_pos = np.array(
            self._sim.get_agent_data(self.agent_id).articulated_agent.base_pos
        )

        # For the variable track
        self._val_dict = {
            "min_start_end_episode_step": float("inf"),
            "has_found_human": False,
            "has_found_human_step": 1500,
            "found_human_times": 0,
            "after_found_human_times": 0,
            "step": 0,
            "step_after_found": 1,
            "dis": 0,
            "dis_after_found": 0,
            "backup_count": 0,
            "yield_count": 0,
        }

        # Robot's info
        self._robot_init_pos = robot_pos
        self._robot_init_trans = mn.Matrix4(
            self._sim.get_agent_data(
                0
            ).articulated_agent.sim_obj.transformation
        )

        self._prev_robot_base_T = mn.Matrix4(
            self._sim.get_agent_data(
                0
            ).articulated_agent.sim_obj.transformation
        )

        # Store pos of human and robot
        self.human_pos_list = []
        self.robot_pos_list = []

        # Update metrics
        self.update_metric(*args, task=task, **kwargs)

    def _check_human_dis(self, robot_pos, human_pos):
        # We use geo geodesic distance here
        dis = self._sim.geodesic_distance(robot_pos, human_pos)
        return dis >= self._min_dis_human and dis <= self._max_dis_human

    def _check_human_frame(self, obs):
        if not self._check_human_in_frame:
            return True
        use_k = f"agent_{self._robot_idx}_articulated_agent_arm_panoptic"
        panoptic = obs[use_k]
        return (
            np.sum(panoptic == self._human_id) > self._human_detect_threshold
        )

    def _check_robot_facing_human(self, human_pos, robot_pos):
        base_T = self._sim.get_agent_data(
            self._robot_idx
        ).articulated_agent.sim_obj.transformation
        facing = (
            robot_human_vec_dot_product(robot_pos, human_pos, base_T)
            > self._robot_face_human_threshold
        )
        return facing

    def update_metric(self, *args, episode, task, observations, **kwargs):
        # Get the agent locations
        robot_pos = np.array(
            self._sim.get_agent_data(
                self._robot_idx
            ).articulated_agent.base_pos
        )
        human_pos = np.array(
            self._sim.get_agent_data(
                self._human_idx
            ).articulated_agent.base_pos
        )

        # Store the human/robot position info
        self.human_pos_list.append(human_pos)
        self.robot_pos_list.append(robot_pos)

        # Compute the distance based on the L2 norm
        dis = np.linalg.norm(robot_pos - human_pos, ord=2, axis=-1)

        # Add the current distance to compute average distance
        self._val_dict["dis"] += dis

        # Compute the robot moving velocity for backup and yiled metrics
        robot_move_vec = np.array(
            self._prev_robot_base_T.inverted().transform_point(robot_pos)
        )[[0, 2]]
        robot_move_vel = (
            np.linalg.norm(robot_move_vec)
            / (1.0 / 120.0)
            * np.sign(robot_move_vec[0])
        )

        # Compute the metrics for backing up and yield
        if (
            dis <= self._dis_threshold_for_backup_yield
            and robot_move_vel < 0.0
        ):
            self._val_dict["backup_count"] += 1
        elif (
            dis <= self._dis_threshold_for_backup_yield
            and abs(robot_move_vel) < self._min_abs_vel_for_yield
        ):
            self._val_dict["yield_count"] += 1

        # Increase the step counter
        self._val_dict["step"] += 1

        # Check if human has been found
        found_human = False
        if self._check_human_dis(
            robot_pos, human_pos
        ) and self._check_robot_facing_human(human_pos, robot_pos):
            found_human = True
            self._val_dict["has_found_human"] = True
            self._val_dict["found_human_times"] += 1

        # Compute the metrics after finding the human
        if self._val_dict["has_found_human"]:
            self._val_dict["dis_after_found"] += dis
            self._val_dict["after_found_human_times"] += found_human

        # Record the step taken to find the human
        if (
            self._val_dict["has_found_human"]
            and self._val_dict["has_found_human_step"] == 1500
        ):
            self._val_dict["has_found_human_step"] = self._val_dict["step"]

        # Compute the minimum distance only when the minimum distance has not found yet
        if (
            self._val_dict["min_start_end_episode_step"] == float("inf")
            and self._enable_shortest_path_computation
        ):
            use_k_human = (
                f"agent_{self._human_idx}_oracle_nav_randcoord_action"
            )
            robot_to_human_min_step = task.actions[
                use_k_human
            ]._compute_robot_to_human_min_step(
                self._robot_init_trans, human_pos, self.human_pos_list
            )

            if robot_to_human_min_step <= self._val_dict["step"]:
                robot_to_human_min_step = self._val_dict["step"]
            else:
                robot_to_human_min_step = float("inf")

            # Update the minimum SPL
            self._val_dict["min_start_end_episode_step"] = min(
                self._val_dict["min_start_end_episode_step"],
                robot_to_human_min_step,
            )

        # Compute the SPL before finding the human
        first_encounter_spl = (
            self._val_dict["has_found_human"]
            * self._val_dict["min_start_end_episode_step"]
            / max(
                self._val_dict["min_start_end_episode_step"],
                self._val_dict["has_found_human_step"],
            )
        )

        # Make sure the first_encounter_spl is not NaN
        if np.isnan(first_encounter_spl):
            first_encounter_spl = 0.0

        self._prev_robot_base_T = mn.Matrix4(
            self._sim.get_agent_data(
                0
            ).articulated_agent.sim_obj.transformation
        )

        # Compute the metrics
        self._metric = {
            "has_found_human": self._val_dict["has_found_human"],
            "found_human_rate_over_epi": self._val_dict["found_human_times"]
            / self._val_dict["step"],
            "found_human_rate_after_encounter_over_epi": self._val_dict[
                "after_found_human_times"
            ]
            / self._val_dict["step_after_found"],
            "avg_robot_to_human_dis_over_epi": self._val_dict["dis"]
            / self._val_dict["step"],
            "avg_robot_to_human_after_encounter_dis_over_epi": self._val_dict[
                "dis_after_found"
            ]
            / self._val_dict["step_after_found"],
            "first_encounter_spl": first_encounter_spl,
            "frist_ecnounter_steps": self._val_dict["has_found_human_step"],
            "frist_ecnounter_steps_ratio": self._val_dict[
                "has_found_human_step"
            ]
            / self._val_dict["min_start_end_episode_step"],
            "follow_human_steps_after_frist_encounter": self._val_dict[
                "after_found_human_times"
            ],
            "follow_human_steps_ratio_after_frist_encounter": self._val_dict[
                "after_found_human_times"
            ]
            / (
                self._total_step - self._val_dict["min_start_end_episode_step"]
            ),
            "backup_ratio": self._val_dict["backup_count"]
            / self._val_dict["step"],
            "yield_ratio": self._val_dict["yield_count"]
            / self._val_dict["step"],
        }

        # Update the counter
        if self._val_dict["has_found_human"]:
            self._val_dict["step_after_found"] += 1


@registry.register_measure
class SocialNavSeekSuccess(Measure):
    """Social nav seek success meassurement"""

    cls_uuid: str = "nav_seek_success"  #KL: add social

    @staticmethod
    def _get_uuid(*args, **kwargs):
        return SocialNavSeekSuccess.cls_uuid

    def reset_metric(self, *args, task, **kwargs):
        """Reset the metrics"""
        task.measurements.check_measure_dependencies(
            self.uuid,
            [NavToPosSucc.cls_uuid, RotDistToGoal.cls_uuid],
        )
        self._following_step = 0
        self.update_metric(*args, task=task, **kwargs)

    def __init__(self, *args, config, sim, **kwargs):
        self._config = config
        self._sim = sim

        super().__init__(*args, config=config, **kwargs)
        # Setup the parameters
        self._following_step = 0
        self._following_step_succ_threshold = (
            config.following_step_succ_threshold
        )
        self._safe_dis_min = config.safe_dis_min
        self._safe_dis_max = config.safe_dis_max
        self._use_geo_distance = config.use_geo_distance
        self._need_to_face_human = config.need_to_face_human
        self._facing_threshold = config.facing_threshold
        self._robot_idx = config.robot_idx
        self._human_idx = config.human_idx

    def update_metric(self, *args, episode, task, observations, **kwargs):
        # Get the angle distance
        angle_dist = task.measurements.measures[
            RotDistToGoal.cls_uuid
        ].get_metric()

        # Get the positions of the human and the robot
        use_k_human = f"agent_{self._human_idx}_localization_sensor"
        human_pos = observations[use_k_human][:3]
        use_k_robot = f"agent_{self._robot_idx}_localization_sensor"
        robot_pos = observations[use_k_robot][:3]

        # If we want to use the geo distance
        if self._use_geo_distance:
            dist = self._sim.geodesic_distance(robot_pos, human_pos)
        else:
            dist = task.measurements.measures[DistToGoal.cls_uuid].get_metric()

        # Compute facing to human
        base_T = self._sim.get_agent_data(
            0
        ).articulated_agent.base_transformation
        if self._need_to_face_human:
            facing = (
                robot_human_vec_dot_product(robot_pos, human_pos, base_T)
                > self._facing_threshold
            )
        else:
            facing = True

        # Check if the agent follows the human within the safe distance
        if dist >= self._safe_dis_min and dist < self._safe_dis_max and facing:
            self._following_step += 1

        nav_pos_succ = False
        if self._following_step >= self._following_step_succ_threshold:
            nav_pos_succ = True

        # If the robot needs to look at the target
        if self._config.must_look_at_targ:
            self._metric = (
                nav_pos_succ and angle_dist < self._config.success_angle_dist
            )
        else:
            self._metric = nav_pos_succ


#############SocialNavToPosSucc######################

@registry.register_measure
class SocialDistToGoal(UsesArticulatedAgentInterface, Measure):
    cls_uuid: str = "social_dist_to_goal"

    def __init__(self, *args, sim, config, task, **kwargs):
        self._config = config
        self._sim = sim
        self._prev_dist = None
        super().__init__(*args, sim=sim, config=config, task=task, **kwargs)

    def reset_metric(self, *args, episode, task, observations, **kwargs):
        self._prev_dist = self._get_cur_geo_dist(task)
        self.update_metric(
            *args,
            episode=episode,
            task=task,
            observations=observations,
            **kwargs,
        )

    def _get_cur_geo_dist(self, task):
        if (self.agent_id is None):
            self.agent_id = 0
        if (self.agent_id == 0):
            goal = task.my_nav_to_info.robot_info.nav_goal_pos
            # print("For robot goal is ", goal, "pos is " ,self._sim.get_agent_data(self.agent_id).articulated_agent.base_pos)
        else:
            goal = task.my_nav_to_info.human_info.nav_goal_pos
            # print("For human goal is ", goal, "pos is " ,self._sim.get_agent_data(self.agent_id).articulated_agent.base_pos)
        return np.linalg.norm(
            np.array(
                self._sim.get_agent_data(
                    self.agent_id
                ).articulated_agent.base_pos
            )[[0, 2]]
            - goal[[0, 2]]
        )


    @staticmethod
    def _get_uuid(*args, **kwargs):
        return SocialDistToGoal.cls_uuid

    def update_metric(self, *args, episode, task, observations, **kwargs):
        self._metric = self._get_cur_geo_dist(task)

@registry.register_measure
class SocialNavToPosSucc(Measure):
    cls_uuid: str = "social_nav_to_pos_success"

    @staticmethod
    def _get_uuid(*args, **kwargs):
        return SocialNavToPosSucc.cls_uuid

    def __init__(self, *args, config, **kwargs):
        self._config = config
        super().__init__(*args, config=config, **kwargs)

    def reset_metric(self, *args, task, **kwargs):
        task.measurements.check_measure_dependencies(
            self.uuid,
            [SocialDistToGoal.cls_uuid],
        )
        self.update_metric(*args, task=task, **kwargs)

    def update_metric(self, *args, episode, task, observations, **kwargs):
        dist = task.measurements.measures[SocialDistToGoal.cls_uuid].get_metric()
        self._metric = dist < self._config.success_distance
        if self._metric:
            task.should_end = True
        # print("social navtopos distance is ", dist)


###########################################





@registry.register_sensor
class HumanoidDetectorSensor(UsesArticulatedAgentInterface, Sensor):
    def __init__(self, sim, config, *args, **kwargs):
        self._sim = sim
        self._human_id = config.human_id
        self._human_pixel_threshold = config.human_pixel_threshold
        self._return_image = config.return_image
        self._is_return_image_bbox = config.is_return_image_bbox

        # Check the observation size
        arm_panoptic_shape = None
        head_depth_shape = None
        for key in self._sim.sensor_suite.observation_spaces.spaces:
            if "articulated_agent_arm_panoptic" in key:
                arm_panoptic_shape = (
                    self._sim.sensor_suite.observation_spaces.spaces[key].shape
                )
            if "head_depth" in key:
                head_depth_shape = (
                    self._sim.sensor_suite.observation_spaces.spaces[key].shape
                )

        # Set the correct size
        if arm_panoptic_shape is not None:
            self._height = arm_panoptic_shape[0]
            self._width = arm_panoptic_shape[1]
        else:
            self._height = head_depth_shape[0]
            self._width = head_depth_shape[1]
        super().__init__(config=config)

    def _get_uuid(self, *args, **kwargs):
        return "humanoid_detector_sensor"

    def _get_sensor_type(self, *args, **kwargs):
        return SensorTypes.TENSOR

    def _get_observation_space(self, *args, config, **kwargs):
        if config.return_image:
            return spaces.Box(
                shape=(
                    self._height,
                    self._width,
                    1,
                ),
                low=np.finfo(np.float32).min,
                high=np.finfo(np.float32).max,
                dtype=np.float32,
            )
        else:
            return spaces.Box(
                shape=(1,),
                low=np.finfo(np.float32).min,
                high=np.finfo(np.float32).max,
                dtype=np.float32,
            )

    def _get_bbox(self, img):
        """Simple function to get the bounding box, assuming that only one object of interest in the image"""
        rows = np.any(img, axis=1)
        cols = np.any(img, axis=0)
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        return rmin, rmax, cmin, cmax

    def get_observation(self, observations, episode, *args, **kwargs):
        found_human = False
        use_k = f"agent_{self.agent_id}_articulated_agent_arm_panoptic"
        if use_k in observations:
            panoptic = observations[use_k]
        else:
            if self._return_image:
                return np.zeros(
                    (self._height, self._width, 1), dtype=np.float32
                )
            else:
                return np.zeros(1, dtype=np.float32)

        if self._return_image:
            tgt_mask = np.float32(panoptic == self._human_id)
            if self._is_return_image_bbox:
                # Get the bounding box
                bbox = np.zeros(tgt_mask.shape)
                if np.sum(tgt_mask) != 0:
                    rmin, rmax, cmin, cmax = self._get_bbox(tgt_mask)
                    bbox[rmin:rmax, cmin:cmax] = 1.0
                return np.float32(bbox)
            else:
                return tgt_mask
        else:
            if (
                np.sum(panoptic == self._human_id)
                > self._human_pixel_threshold
            ):
                found_human = True

            if found_human:
                return np.ones(1, dtype=np.float32)
            else:
                return np.zeros(1, dtype=np.float32)


@registry.register_sensor
class InitialGpsCompassSensor(UsesArticulatedAgentInterface, Sensor):
    """
    Get the relative distance to the initial starting location of the robot
    """

    def __init__(self, sim, config, *args, **kwargs):
        self._sim = sim
        super().__init__(config=config)

    def _get_uuid(self, *args, **kwargs):
        return "initial_gps_compass_sensor"

    def _get_sensor_type(self, *args, **kwargs):
        return SensorTypes.TENSOR

    def _get_observation_space(self, *args, config, **kwargs):
        return spaces.Box(
            shape=(2,),
            low=np.finfo(np.float32).min,
            high=np.finfo(np.float32).max,
            dtype=np.float32,
        )

    def get_observation(self, task, *args, **kwargs):
        if (self.agent_id is None):
            self.agent_id = 0
        agent_data = self._sim.get_agent_data(self.agent_id).articulated_agent
        agent_pos = np.array(agent_data.base_pos)
        init_articulated_agent_T = task.initial_robot_trans

        # Do not support human relative GPS
        if init_articulated_agent_T is None or isinstance(
            agent_data, KinematicHumanoid
        ):
            return np.zeros(2, dtype=np.float32)
        else:
            rel_pos = batch_transform_point(
                np.array([agent_pos]),
                init_articulated_agent_T.inverted(),
                np.float32,
            )
            rho, phi = cartesian_to_polar(rel_pos[0][0], rel_pos[0][1])
            init_rel_pos = np.array([rho, -phi], dtype=np.float32)

            return init_rel_pos


@registry.register_sensor
class NavGoalWorldDeltaSensor(UsesArticulatedAgentInterface, Sensor):
    """World-frame (x, z) vector from the agent to its navigation goal.

    Returns ``(nav_goal_pos - base_pos)[[0, 2]]`` in **world** coordinates,
    unlike ``NavGoalPointGoalSensor`` which returns robot-local polar
    ``[rho, -phi]``. The self-contained RVO ``GoToGoalSkill`` consumes this to
    set its ORCA preferred velocity (direction = normalized delta) and to detect
    goal arrival (distance = ``norm(delta)``) without having to invert the agent
    base transform in the policy. Per-agent duplication (``agent_0_*`` /
    ``agent_1_*``) is handled by ``RearrangeTask._duplicate_sensor_suite``.
    """

    cls_uuid: str = "goal_world_delta"

    def __init__(self, sim, config, *args, **kwargs):
        self._sim = sim
        super().__init__(config=config)

    def _get_uuid(self, *args, **kwargs):
        return NavGoalWorldDeltaSensor.cls_uuid

    def _get_sensor_type(self, *args, **kwargs):
        return SensorTypes.TENSOR

    def _get_observation_space(self, *args, config, **kwargs):
        return spaces.Box(
            shape=(2,),
            low=np.finfo(np.float32).min,
            high=np.finfo(np.float32).max,
            dtype=np.float32,
        )

    def get_observation(self, task, *args, **kwargs):
        agent_id = self.agent_id if self.agent_id is not None else 0
        base_pos = np.array(
            self._sim.get_agent_data(agent_id).articulated_agent.base_pos
        )
        info = getattr(task, "my_nav_to_info", None)
        if info is None:
            return np.zeros(2, dtype=np.float32)
        if agent_id == 1 and info.human_info is not None:
            goal = info.human_info.nav_goal_pos
        elif info.robot_info is not None:
            goal = info.robot_info.nav_goal_pos
        else:
            return np.zeros(2, dtype=np.float32)
        goal = np.array(goal, dtype=np.float32)
        return np.array(
            [goal[0] - base_pos[0], goal[2] - base_pos[2]], dtype=np.float32
        )


def _geodesic_next_waypoint(sim, base_pos, target):
    """``[wp_dx, wp_dz, remaining]``: world (x, z) delta to the next waypoint of
    the default geodesic path from ``base_pos`` to ``target``, plus the
    straight-line XZ distance left to ``target``."""
    path = habitat_sim.ShortestPath()
    path.requested_start = np.array(base_pos, dtype=np.float32)
    path.requested_end = np.array(target, dtype=np.float32)
    found = sim.pathfinder.find_path(path)
    if found and len(path.points) > 1:
        wp = np.array(path.points[1], dtype=np.float64)
    else:
        wp = np.array(target, dtype=np.float64)
    wp_delta = (wp - base_pos)[[0, 2]]
    remaining = float(np.linalg.norm((np.array(target) - base_pos)[[0, 2]]))
    return np.array([wp_delta[0], wp_delta[1], remaining], dtype=np.float32)


@registry.register_sensor
class NavGoalWaypointDeltaSensor(UsesArticulatedAgentInterface, Sensor):
    """Next-waypoint vector of the default geodesic path to the ROBOT GOAL
    (agent_0 only). Snaps the goal to the navmesh once per episode, pathfinds
    each step, and returns ``[wp_dx, wp_dz, remaining]``. ``GoToGoalSkill``
    consumes this for plain shortest-path driving with no human avoidance."""

    cls_uuid: str = "goal_waypoint_delta"

    def __init__(self, sim, config, *args, **kwargs):
        self._sim = sim
        self._target = None
        self._prev_ep_id = None
        super().__init__(config=config)

    def _get_uuid(self, *args, **kwargs):
        return NavGoalWaypointDeltaSensor.cls_uuid

    def _get_sensor_type(self, *args, **kwargs):
        return SensorTypes.TENSOR

    def _get_observation_space(self, *args, config, **kwargs):
        return spaces.Box(
            shape=(3,),
            low=np.finfo(np.float32).min,
            high=np.finfo(np.float32).max,
            dtype=np.float32,
        )

    def _compute_target(self, task):
        info = getattr(task, "my_nav_to_info", None)
        robot_info = getattr(info, "robot_info", None) if info else None
        if robot_info is None:
            return None
        goal = np.array(robot_info.nav_goal_pos, dtype=np.float64)
        snapped = np.array(self._sim.pathfinder.snap_point(goal), dtype=np.float64)
        return goal if np.any(np.isnan(snapped)) else snapped

    def get_observation(self, task, *args, **kwargs):
        agent_id = self.agent_id if self.agent_id is not None else 0
        if agent_id != 0:
            return np.zeros(3, dtype=np.float32)

        ep_id = getattr(self._sim.ep_info, "episode_id", None)
        if self._target is None or ep_id != self._prev_ep_id:
            self._target = self._compute_target(task)
            self._prev_ep_id = ep_id
        if self._target is None:
            return np.zeros(3, dtype=np.float32)

        base_pos = np.array(
            self._sim.get_agent_data(agent_id).articulated_agent.base_pos
        )
        return _geodesic_next_waypoint(self._sim, base_pos, self._target)


@registry.register_sensor
class BackoffWaypointDeltaSensor(UsesArticulatedAgentInterface, Sensor):
    """Next-waypoint vector for the door-relative reverse backoff (agent_0 only).

    Computes ONCE per episode a backoff target ~``backoff_dist`` m behind the
    door on the robot's start side -- default straight back along the door
    normal, otherwise the nearest navigable corridor direction on that side --
    validated to be navigable, on the start side, with >= ``obstacle_clearance``
    m clearance, and reachable from the robot. Each step returns
    ``[wp_dx, wp_dz, remaining]`` where ``(wp_dx, wp_dz)`` is the world (x, z)
    delta to the next waypoint of the **default geodesic path** to that target
    and ``remaining`` is the straight-line distance left. ``BackOffSkill`` drives
    the robot BACKWARD toward the waypoint and stops/terminates on ``remaining``.
    Per-agent duplication is handled by ``RearrangeTask._duplicate_sensor_suite``.
    """

    cls_uuid: str = "backoff_waypoint_delta"

    def __init__(self, sim, config, *args, **kwargs):
        self._sim = sim
        self._backoff_dist = float(getattr(config, "backoff_dist", 2.0))
        self._clearance = float(getattr(config, "obstacle_clearance", 0.3))
        # --- S2 clear-corridor yield knobs (default 'legacy' = old behavior) ---
        self._yield_mode = str(getattr(config, "yield_mode", "legacy"))
        self._corridor_clearance = float(
            getattr(config, "corridor_clearance", 1.1)
        )
        self._human_goal_clearance = float(
            getattr(config, "human_goal_clearance", 1.5)
        )
        self._include_door_dir = bool(
            getattr(config, "include_door_dir", False)
        )
        self._recompute_human_motion = float(
            getattr(config, "recompute_human_motion", 0.75)
        )
        self._include_branch = bool(getattr(config, "include_branch", False))
        self._retreat_corridor_clearance = float(
            getattr(config, "retreat_corridor_clearance", 0.0)
        )
        self._target = None
        self._last_branch = 0
        self._prev_ep_id = None
        # Human position at the last target computation (corridor recompute).
        self._last_human_xz = None
        super().__init__(config=config)

    def _get_uuid(self, *args, **kwargs):
        return BackoffWaypointDeltaSensor.cls_uuid

    def _get_sensor_type(self, *args, **kwargs):
        return SensorTypes.TENSOR

    def _get_observation_space(self, *args, config, **kwargs):
        dim = 5 if bool(getattr(config, "include_door_dir", False)) else 3
        if bool(getattr(config, "include_branch", False)):
            dim += 1
        return spaces.Box(
            shape=(dim,),
            low=np.finfo(np.float32).min,
            high=np.finfo(np.float32).max,
            dtype=np.float32,
        )

    @staticmethod
    def _side_sign(door_start, door_end, p):
        # Mirrors door_sampler.py: sign of the 2D cross product about the door
        # line in the XZ plane -> which side of the door a point lies on.
        r1, r2 = door_start[0], door_end[0]
        c1, c2 = door_start[2], door_end[2]
        det = (p[0] - r1) * (c2 - c1) - (p[2] - c1) * (r2 - r1)
        return 1.0 if det > 0 else -1.0

    @staticmethod
    def _point_seg_dist_xz(p, a, b):
        """XZ distance from point p to segment [a, b] (all 3-vectors)."""
        p = np.array([p[0], p[2]], dtype=np.float64)
        a = np.array([a[0], a[2]], dtype=np.float64)
        b = np.array([b[0], b[2]], dtype=np.float64)
        ab = b - a
        denom = float(np.dot(ab, ab))
        t = 0.0 if denom < 1e-9 else float(np.clip(np.dot(p - a, ab) / denom, 0.0, 1.0))
        return float(np.linalg.norm(p - (a + t * ab)))

    def _dist_to_corridor(self, cand, corridor):
        """Min XZ distance from cand to the human corridor polyline."""
        return min(
            self._point_seg_dist_xz(cand, corridor[i], corridor[i + 1])
            for i in range(len(corridor) - 1)
        )

    def _human_route(self, pf, human_pos, human_goal, door_mid):
        """The human's ACTUAL future route: its navmesh shortest path to the
        goal. The straight [pos, door, goal] polyline under-covers the swept
        area (the real path curves around furniture, and once the human is past
        the door it no longer routes via the door), so a pocket that "clears"
        the polyline can still sit ON the real route -> the firm human either
        freezes against the robot (blocked-gate deadlock) or pushes into it
        (pursuit collision). Falls back to the legacy polyline if pathing fails.
        """
        path = habitat_sim.ShortestPath()
        path.requested_start = np.array(human_pos, dtype=np.float32)
        path.requested_end = np.array(human_goal, dtype=np.float32)
        if pf.find_path(path) and len(path.points) >= 2:
            return [np.array(p, dtype=np.float64) for p in path.points]
        return [human_pos, door_mid, human_goal]

    def _compute_target(self, task, robot_pos):
        """Yield target on the robot's start side, validated against the navmesh.

        ``legacy``: straight-back point ~backoff_dist behind the door (old).
        ``corridor``: clear the human's path corridor -- pick the navigable
        start-side point nearest the robot that keeps >= corridor_clearance
        from the human polyline [human_pos, door_mid, human_goal] and
        >= human_goal_clearance from the human goal.

        Returns ``(target, branch)``: branch 0 = validated pocket, 1 = no valid
        pocket. On branch 1 corridor mode holds IN PLACE (target = robot
        position, an explicit defined behavior) instead of the old silent
        reverse-to-spawn, which was never validated against the human's route.
        Legacy mode keeps the spawn fallback byte-for-byte.
        """
        pf = self._sim.pathfinder
        info = getattr(self._sim.ep_info, "info", {}) or {}
        nav_info = getattr(task, "my_nav_to_info", None)
        robot_info = getattr(nav_info, "robot_info", None) if nav_info else None
        if robot_info is None or "door_start" not in info or "door_end" not in info:
            return None, 1

        door_start = np.array(info["door_start"], dtype=np.float64)
        door_end = np.array(info["door_end"], dtype=np.float64)
        start_pos = np.array(
            robot_info.articulated_agent_start_pos, dtype=np.float64
        )

        door_mid = (door_start + door_end) / 2.0
        d = door_end - door_start
        d_xz = np.array([d[0], d[2]], dtype=np.float64)
        nrm = float(np.linalg.norm(d_xz))
        d_xz = d_xz / nrm if nrm > 1e-6 else np.array([1.0, 0.0])
        n_xz = np.array([-d_xz[1], d_xz[0]])  # door normal (XZ)

        # Pick the normal pointing to the robot's start side.
        start_side = self._side_sign(door_start, door_end, start_pos)
        probe = door_mid + np.array([n_xz[0], 0.0, n_xz[1]]) * 0.5
        if self._side_sign(door_start, door_end, probe) != start_side:
            n_xz = -n_xz

        def _rotate(v, deg):
            t = np.radians(deg)
            c, s = np.cos(t), np.sin(t)
            return np.array([c * v[0] - s * v[1], s * v[0] + c * v[1]])

        corridor_mode = self._yield_mode == "corridor"
        if corridor_mode:
            # Fan wider (incl. lateral / oblique-back) and DROP the straight-back
            # priority; selection is by proximity to the robot + corridor clearance.
            angles = [0, 30, -30, 60, -60, 90, -90, 120, -120, 150, -150]
            radii = [
                self._backoff_dist,
                self._backoff_dist * 0.75,
                self._backoff_dist * 0.5,
            ]
        else:
            angles = [0, 30, -30, 60, -60, 90, -90]
            radii = [
                self._backoff_dist,
                self._backoff_dist * 0.75,
                self._backoff_dist * 0.5,
            ]
        candidates = []
        for r in radii:
            for a in angles:
                dir_xz = _rotate(n_xz, a)
                cand = door_mid + np.array([dir_xz[0], 0.0, dir_xz[1]]) * r
                cand[1] = door_mid[1]
                candidates.append(cand)

        if corridor_mode:
            # Door-anchored candidates only cover ~backoff_dist around the door.
            # When the door sits in a narrow single-file corridor there is NO
            # lateral clearance beside it -- the yield pocket is further down the
            # corridor (a side room the robot passed). Also fan a FULL circle
            # around the ROBOT out to a few metres so the nearest corridor-clearing
            # pocket (in any direction) becomes a candidate. Selection below still
            # picks the valid one nearest the robot (minimum detour).
            robot_xz = np.array([robot_pos[0], robot_pos[2]], dtype=np.float64)
            # Radii scale off backoff_dist (config-driven); at the default 2.0
            # this reproduces the validated 1.0..3.5 m sweep. Nearest valid wins,
            # so a generous sweep costs nothing when close pockets exist.
            for frac in (0.5, 0.75, 1.0, 1.25, 1.5, 1.75):
                r = self._backoff_dist * frac
                for a in range(0, 360, 30):
                    dir2 = _rotate(np.array([1.0, 0.0]), a)
                    cand = np.array(
                        [robot_xz[0] + dir2[0] * r, door_mid[1],
                         robot_xz[1] + dir2[1] * r],
                        dtype=np.float64,
                    )
                    candidates.append(cand)

        # Human corridor polyline (corridor mode only).
        corridor = None
        human_goal = None
        if corridor_mode and nav_info is not None and getattr(nav_info, "human_info", None) is not None:
            human_pos = np.array(
                self._sim.get_agent_data(1).articulated_agent.base_pos,
                dtype=np.float64,
            )
            human_goal = np.array(
                nav_info.human_info.nav_goal_pos, dtype=np.float64
            )
            corridor = self._human_route(pf, human_pos, human_goal, door_mid)

        def _validate(cand, clearance, corridor_clear, retreat_clear=0.0):
            snapped = np.array(pf.snap_point(cand), dtype=np.float64)
            if np.any(np.isnan(snapped)) or not pf.is_navigable(snapped):
                return None
            if self._side_sign(door_start, door_end, snapped) != start_side:
                return None  # snap flipped it across the door
            if (
                pf.distance_to_closest_obstacle(snapped, max_search_radius=2.0)
                < clearance
            ):
                return None
            if corridor is not None:
                if self._dist_to_corridor(snapped, corridor) < corridor_clear:
                    return None
                if (
                    human_goal is not None
                    and float(np.linalg.norm((snapped - human_goal)[[0, 2]]))
                    < self._human_goal_clearance
                ):
                    return None
            path = habitat_sim.ShortestPath()
            path.requested_start = np.array(robot_pos, dtype=np.float32)
            path.requested_end = snapped.astype(np.float32)
            if not pf.find_path(path):
                return None  # unreachable
            if retreat_clear > 0.0 and corridor is not None:
                # The RETREAT PATH must clear the human's route too: a pocket
                # that itself clears the corridor is useless if the reverse
                # drive to it crosses the corridor (the robot backs blind into
                # the oncoming human). The first ~1 m around the robot is
                # exempt -- the robot usually starts ON the corridor, escaping
                # it is the whole point.
                pts = [np.array(p, dtype=np.float64) for p in path.points]
                for a, b in zip(pts[:-1], pts[1:]):
                    seg = b - a
                    seg_len = float(np.linalg.norm(seg[[0, 2]]))
                    n = max(1, int(seg_len / 0.25))
                    for t in range(n + 1):
                        q = a + seg * (t / n)
                        if float(np.linalg.norm((q - robot_pos)[[0, 2]])) < 1.0:
                            continue
                        if self._dist_to_corridor(q, corridor) < retreat_clear:
                            return None
            return snapped

        if corridor_mode:
            # Pass 1-2: full then relaxed corridor clearance, both requiring
            # the retreat path to clear the corridor; pass 3 drops the retreat
            # check (a crossing pocket beats no pocket -- the yield skill's
            # human_stop_dist floor prevents actually backing into the human).
            # With retreat_corridor_clearance=0 (legacy) passes collapse to the
            # old two.
            passes = []
            if self._retreat_corridor_clearance > 0.0:
                passes += [
                    (self._corridor_clearance, self._retreat_corridor_clearance),
                    (
                        self._corridor_clearance * 0.75,
                        self._retreat_corridor_clearance,
                    ),
                ]
            passes += [
                (self._corridor_clearance, 0.0),
                (self._corridor_clearance * 0.75, 0.0),
            ]
            for corridor_clear, retreat_clear in passes:
                best, best_d = None, None
                for cand in candidates:
                    ok = _validate(
                        cand, self._clearance, corridor_clear, retreat_clear
                    )
                    if ok is None:
                        continue
                    dr = float(np.linalg.norm((ok - robot_pos)[[0, 2]]))
                    if best_d is None or dr < best_d:
                        best, best_d = ok, dr
                if best is not None:
                    self._last_corridor_dist = (
                        self._dist_to_corridor(best, corridor)
                        if corridor is not None
                        else -1.0
                    )
                    print(
                        f"[YieldSensor] branch=pocket "
                        f"dist_to_corridor={self._last_corridor_dist:.2f} "
                        f"robot_detour={best_d:.2f}",
                        flush=True,
                    )
                    return best, 0
            print(
                "[YieldSensor] corridor-mode fallback to legacy candidates",
                flush=True,
            )

        # Legacy ordering: first valid (straight-back preferred).
        for clearance in (self._clearance, self._clearance / 2.0):
            for cand in candidates:
                ok = _validate(cand, clearance, 0.0)
                if ok is not None:
                    return ok, 0

        if corridor_mode:
            # No valid pocket anywhere: hold in place. The stand-still is the
            # DEFINED fallback (remaining ~ 0 keeps the yield skill parked); the
            # HL teacher/student sees branch=1 in the observation.
            print("[YieldSensor] branch=hold (no valid pocket)", flush=True)
            snapped = np.array(pf.snap_point(robot_pos), dtype=np.float64)
            if np.any(np.isnan(snapped)):
                return np.array(robot_pos, dtype=np.float64), 1
            return snapped, 1

        # Legacy fallback: original spawn behavior.
        print(
            "[BackoffWaypointDeltaSensor] no valid door-relative point; "
            "falling back to spawn",
            flush=True,
        )
        snapped = np.array(pf.snap_point(start_pos), dtype=np.float64)
        if np.any(np.isnan(snapped)):
            return np.array(robot_pos, dtype=np.float64), 1
        return snapped, 1

    def _door_dir_xz(self):
        """World (x, z) delta from the robot to the door mid, for facing."""
        info = getattr(self._sim.ep_info, "info", {}) or {}
        if "door_start" not in info or "door_end" not in info:
            return np.zeros(2, dtype=np.float32)
        ds = np.array(info["door_start"], dtype=np.float64)
        de = np.array(info["door_end"], dtype=np.float64)
        door_mid = (ds + de) / 2.0
        base_pos = np.array(
            self._sim.get_agent_data(0).articulated_agent.base_pos
        )
        delta = (door_mid - base_pos)[[0, 2]]
        return delta.astype(np.float32)

    def _target_still_valid(self, task):
        """Hysteresis: is the CURRENT yield target still clearing the human's
        corridor? If so we keep it (do not hop to a new candidate) so the
        reverse-drive controller doesn't oscillate as the human moves."""
        if self._target is None:
            return False
        if self._last_branch != 0:
            # Hold-in-place has no pocket to invalidate; re-picking every
            # human step would churn. A pocket may open later, so the episode
            # boundary (not human motion) is the only recompute trigger.
            return True
        info = getattr(self._sim.ep_info, "info", {}) or {}
        nav_info = getattr(task, "my_nav_to_info", None)
        human_info = getattr(nav_info, "human_info", None) if nav_info else None
        if "door_start" not in info or human_info is None:
            return True  # no door/human info -> don't churn the target
        ds = np.array(info["door_start"], dtype=np.float64)
        de = np.array(info["door_end"], dtype=np.float64)
        door_mid = (ds + de) / 2.0
        human_pos = np.array(
            self._sim.get_agent_data(1).articulated_agent.base_pos,
            dtype=np.float64,
        )
        human_goal = np.array(human_info.nav_goal_pos, dtype=np.float64)
        corridor = self._human_route(
            self._sim.pathfinder, human_pos, human_goal, door_mid
        )
        d_corr = self._dist_to_corridor(self._target, corridor)
        d_hgoal = float(np.linalg.norm((self._target - human_goal)[[0, 2]]))
        return (
            d_corr >= self._corridor_clearance
            and d_hgoal >= self._human_goal_clearance
        )

    def get_observation(self, task, *args, **kwargs):
        out_dim = (5 if self._include_door_dir else 3) + (
            1 if self._include_branch else 0
        )
        agent_id = self.agent_id if self.agent_id is not None else 0
        if agent_id != 0:
            return np.zeros(out_dim, dtype=np.float32)  # yield is robot-only

        base_pos = np.array(
            self._sim.get_agent_data(agent_id).articulated_agent.base_pos
        )

        ep_id = getattr(self._sim.ep_info, "episode_id", None)
        human_xz = np.array(
            self._sim.get_agent_data(1).articulated_agent.base_pos
        )[[0, 2]]
        need_recompute = self._target is None or ep_id != self._prev_ep_id
        # Corridor mode: when the human has moved enough, only RE-pick the yield
        # target if the current one is no longer clearing the corridor
        # (hysteresis) -- otherwise keep it to avoid fore/aft oscillation.
        if (
            self._yield_mode == "corridor"
            and self._last_human_xz is not None
            and float(np.linalg.norm(human_xz - self._last_human_xz))
            >= self._recompute_human_motion
        ):
            self._last_human_xz = human_xz  # reset the motion reference
            if not self._target_still_valid(task):
                need_recompute = True
        # Hold-in-place must track the robot: once it has moved off the stored
        # hold point, re-plan (also the upgrade path if a pocket opened up).
        if (
            not need_recompute
            and self._last_branch != 0
            and self._target is not None
            and float(np.linalg.norm((base_pos - self._target)[[0, 2]])) > 0.5
        ):
            need_recompute = True
        if need_recompute:
            self._target, self._last_branch = self._compute_target(
                task, base_pos
            )
            self._prev_ep_id = ep_id
            self._last_human_xz = human_xz

        if self._target is None:
            out = np.zeros(out_dim, dtype=np.float32)
            if self._include_branch:
                out[-1] = float(self._last_branch)
            return out

        wp = _geodesic_next_waypoint(self._sim, base_pos, self._target)
        vals = [wp[0], wp[1], wp[2]]
        if self._include_door_dir:
            door_dir = self._door_dir_xz()
            vals += [door_dir[0], door_dir[1]]
        if self._include_branch:
            vals.append(float(self._last_branch))
        return np.array(vals, dtype=np.float32)


@registry.register_sensor
class TopDownConflictMapSensor(UsesArticulatedAgentInterface, Sensor):
    """S7-lite egocentric top-down conflict map, uint8 (S, S, 4), robot-centred
    and heading-aligned (forward = up). Channels: (0) static occupancy,
    (1) human disc + fading trail, (2) robot goal disc, (3) door line.
    Static occupancy is cached once per scene; each step is one cv2 affine.
    NOTE: verify orientation visually once per install (dump snippet in
    hrl_pipeline/FRAMEWORKS_HOWTO.md) before large runs.
    """

    cls_uuid: str = "topdown_conflict_map"

    def __init__(self, sim, config, *args, **kwargs):
        self._sim = sim
        self._size = int(getattr(config, "map_size", 64))
        self._mpp = float(getattr(config, "meters_per_pixel", 0.1))
        self._trail_len = int(getattr(config, "trail_len", 30))
        super().__init__(config=config)
        self._cache = {}
        self._trail: list = []
        self._prev_ep = None

    def _get_uuid(self, *args, **kwargs):
        return TopDownConflictMapSensor.cls_uuid

    def _get_sensor_type(self, *args, **kwargs):
        return SensorTypes.TENSOR

    def _get_observation_space(self, *args, config, **kwargs):
        return spaces.Box(
            shape=(self._size, self._size, 4), low=0, high=255, dtype=np.uint8
        )

    def _static_grid(self):
        from habitat.utils.visualizations import maps

        key = getattr(self._sim.ep_info, "scene_id", "?")
        if key not in self._cache:
            base_y = float(
                self._sim.get_agent_data(0).articulated_agent.base_pos[1]
            )
            td = maps.get_topdown_map(
                self._sim.pathfinder, height=base_y,
                meters_per_pixel=self._mpp, draw_border=False,
            )
            occ = ((td == 0) * 255).astype(np.uint8)  # 255 = obstacle
            lower, _ = self._sim.pathfinder.get_bounds()
            self._cache[key] = (occ, np.array(lower, dtype=np.float64))
        return self._cache[key]

    def _world_to_grid(self, p, lower):
        # maps.get_topdown_map convention: row ~ z, col ~ x.
        return (
            (p[0] - lower[0]) / self._mpp,   # col
            (p[2] - lower[2]) / self._mpp,   # row
        )

    def get_observation(self, task, *args, **kwargs):
        import cv2

        agent_id = self.agent_id if self.agent_id is not None else 0
        out = np.zeros((self._size, self._size, 4), dtype=np.uint8)
        if agent_id != 0:
            return out
        grid, lower = self._static_grid()
        agent = self._sim.get_agent_data(0).articulated_agent
        pos = np.array(agent.base_pos, dtype=np.float64)
        yaw = float(agent.base_rot)
        col, row = self._world_to_grid(pos, lower)

        # Rotate so the robot's forward axis points UP in the crop, then
        # translate the robot to the crop centre.
        fwd = np.array([math.cos(yaw), -math.sin(yaw)])  # (x, z)
        ang_deg = math.degrees(math.atan2(fwd[1], fwd[0]))  # angle in (col,row)
        M = cv2.getRotationMatrix2D((col, row), ang_deg + 90.0, 1.0)
        M[0, 2] += self._size / 2.0 - col
        M[1, 2] += self._size / 2.0 - row
        out[:, :, 0] = cv2.warpAffine(
            grid, M, (self._size, self._size), flags=cv2.INTER_NEAREST,
            borderValue=255,
        )

        def to_crop(p_world):
            c, r = self._world_to_grid(np.asarray(p_world, dtype=np.float64), lower)
            v = M @ np.array([c, r, 1.0])
            return int(round(v[0])), int(round(v[1]))

        # Human disc + fading trail.
        ep_id = getattr(self._sim.ep_info, "episode_id", None)
        if ep_id != self._prev_ep:
            self._trail = []
            self._prev_ep = ep_id
        human = np.array(
            self._sim.get_agent_data(1).articulated_agent.base_pos,
            dtype=np.float64,
        )
        self._trail.append(human.copy())
        if len(self._trail) > self._trail_len:
            self._trail = self._trail[-self._trail_len:]
        ch1 = out[:, :, 1].copy()
        n = len(self._trail)
        for i, hp in enumerate(self._trail[:-1]):
            cv2.circle(ch1, to_crop(hp), 1, int(60 + 140 * (i + 1) / n), -1)
        cv2.circle(ch1, to_crop(human), max(1, int(0.3 / self._mpp)), 255, -1)
        out[:, :, 1] = ch1

        # Robot goal disc.
        nav_info = getattr(task, "my_nav_to_info", None)
        if nav_info is not None and getattr(nav_info, "robot_info", None) is not None:
            goal = np.asarray(nav_info.robot_info.nav_goal_pos, dtype=np.float64)
            ch2 = out[:, :, 2].copy()
            cv2.circle(ch2, to_crop(goal), max(1, int(0.3 / self._mpp)), 255, -1)
            out[:, :, 2] = ch2

        # Door line.
        info = getattr(self._sim.ep_info, "info", {}) or {}
        if "door_start" in info and "door_end" in info:
            ch3 = out[:, :, 3].copy()
            cv2.line(ch3, to_crop(info["door_start"]), to_crop(info["door_end"]), 255, 2)
            out[:, :, 3] = ch3
        return out


@registry.register_sensor
class LidarScanSensor(UsesArticulatedAgentInterface, Sensor):
    """Robot-frame K-ray range scan sampled on the RUNTIME navmesh (furniture
    included) -- a simulated 2D lidar. Ray 0 points along the robot's forward
    axis; rays proceed counter-clockwise in the same frame convention as
    SocialNavPolicyStateSensor (forward=(cos yaw, -sin yaw)). This is the
    deployment-legal local-geometry observation for the LEARNED yield skill
    (replaces the privileged pocket planner's map access).
    """

    cls_uuid: str = "lidar_scan"

    def __init__(self, sim, config, *args, **kwargs):
        # Set fields BEFORE super().__init__ -- the Sensor base constructor
        # calls _get_observation_space, which needs them.
        self._sim = sim
        self._num_rays = int(getattr(config, "num_rays", 16))
        self._max_range = float(getattr(config, "max_range", 3.0))
        self._step = float(getattr(config, "ray_step", 0.1))
        super().__init__(config=config)

    def _get_uuid(self, *args, **kwargs):
        return LidarScanSensor.cls_uuid

    def _get_sensor_type(self, *args, **kwargs):
        return SensorTypes.TENSOR

    def _get_observation_space(self, *args, config, **kwargs):
        return spaces.Box(
            shape=(self._num_rays,), low=0.0, high=self._max_range,
            dtype=np.float32,
        )

    def get_observation(self, task, *args, **kwargs):
        agent_id = self.agent_id if self.agent_id is not None else 0
        agent = self._sim.get_agent_data(agent_id).articulated_agent
        pos = np.array(agent.base_pos, dtype=np.float64)
        yaw = float(agent.base_rot)
        fwd = np.array([math.cos(yaw), -math.sin(yaw)])
        lat = np.array([fwd[1], -fwd[0]])
        pf = self._sim.pathfinder
        out = np.full(self._num_rays, self._max_range, dtype=np.float32)
        n_steps = int(self._max_range / self._step)
        for k in range(self._num_rays):
            phi = 2.0 * math.pi * k / self._num_rays
            d = math.cos(phi) * fwd + math.sin(phi) * lat
            for i in range(1, n_steps + 1):
                t = i * self._step
                q = np.array([pos[0] + d[0] * t, pos[1], pos[2] + d[1] * t],
                             dtype=np.float32)
                if not pf.is_navigable(q):
                    out[k] = t
                    break
        return out


@registry.register_sensor
class SocialNavPolicyStateSensor(UsesArticulatedAgentInterface, Sensor):
    """
    Robot-frame state features for the social-nav high-level policy. Every
    quantity is expressed in the robot's own local frame, so the observation is
    invariant to the robot's absolute world pose. Returns a 6-D vector:

        [human_dist, human_bearing, human_rel_heading, human_rel_speed,
         goal_dist, goal_bearing]

    - human_dist / human_bearing: polar position of the human in the robot frame
      (bearing in [-pi, pi], 0 = straight ahead).
    - human_rel_heading: human heading minus robot heading, wrapped to [-pi, pi].
    - human_rel_speed: magnitude of the human's velocity relative to the robot
      (m/s), from a finite difference of base positions. NOTE: unsigned and
      polluted by ego motion (a parked human "moves" at the robot's own speed).
    - goal_dist / goal_bearing: polar position of the robot's nav goal in the
      robot frame.

    With ``include_approach_speed`` a 7th dim is appended:

    - human_approach_speed: the human's ABSOLUTE velocity projected onto the
      human->robot direction (m/s, EMA-smoothed). Positive = human closing in,
      ~0 = parked, negative = receding — regardless of what the robot does.

    Only supports 2-agent setups (robot = this agent, human = the other agent).
    Per-agent duplication (``agent_0_*`` / ``agent_1_*``) is handled by
    ``RearrangeTask._duplicate_sensor_suite``.
    """

    cls_uuid: str = "social_nav_policy_state"

    def __init__(self, sim, config, *args, **kwargs):
        self._sim = sim
        self._prev_robot_xz = None
        self._prev_human_xz = None
        self._prev_episode_id = None
        self._include_approach = bool(
            getattr(config, "include_approach_speed", False)
        )
        self._approach_alpha = float(getattr(config, "approach_ema_alpha", 0.2))
        self._approach_ema = 0.0
        # Control timestep used for the relative-speed finite difference. Falls
        # back to 1.0 (i.e. per-step displacement) if the sim config is missing.
        self._dt = 1.0
        try:
            ctrl_freq = float(self._sim.habitat_config.ctrl_freq)
            ac_freq_ratio = float(self._sim.habitat_config.ac_freq_ratio)
            if ctrl_freq > 0:
                self._dt = ac_freq_ratio / ctrl_freq
        except Exception:
            self._dt = 1.0
        super().__init__(config=config)

    def _get_uuid(self, *args, **kwargs):
        return SocialNavPolicyStateSensor.cls_uuid

    def _get_sensor_type(self, *args, **kwargs):
        return SensorTypes.TENSOR

    def _get_observation_space(self, *args, config, **kwargs):
        dim = 7 if getattr(config, "include_approach_speed", False) else 6
        return spaces.Box(
            shape=(dim,),
            low=np.finfo(np.float32).min,
            high=np.finfo(np.float32).max,
            dtype=np.float32,
        )

    def get_observation(self, *args, task=None, episode=None, **kwargs):
        agent_id = self.agent_id if self.agent_id is not None else 0
        other_id = (agent_id + 1) % 2

        robot = self._sim.get_agent_data(agent_id).articulated_agent
        human = self._sim.get_agent_data(other_id).articulated_agent

        # Robot orthonormal frame in the world XZ plane (rotation only, so this
        # is robust to where the base transform's translation actually sits).
        # forward = local +x; lateral = forward rotated +90 deg in XZ.
        fwd = robot.base_transformation.transform_vector(mn.Vector3(1.0, 0.0, 0.0))
        forward = np.array([fwd[0], fwd[2]], dtype=np.float64)
        n = np.linalg.norm(forward)
        forward = forward / n if n > 1e-8 else np.array([1.0, 0.0])
        lateral = np.array([forward[1], -forward[0]], dtype=np.float64)

        robot_xz = np.array(
            [robot.base_pos[0], robot.base_pos[2]], dtype=np.float64
        )

        def _to_robot_frame(world_xz):
            d = world_xz - robot_xz
            fwd_c = float(d @ forward)
            lat_c = float(d @ lateral)
            return np.hypot(fwd_c, lat_c), np.arctan2(lat_c, fwd_c)

        # Human polar position in the robot frame.
        human_xz = np.array(
            [human.base_pos[0], human.base_pos[2]], dtype=np.float64
        )
        human_dist, human_bearing = _to_robot_frame(human_xz)

        # Human heading relative to the robot heading.
        hfwd = human.base_transformation.transform_vector(
            mn.Vector3(1.0, 0.0, 0.0)
        )
        hfwd_xz = np.array([hfwd[0], hfwd[2]], dtype=np.float64)
        human_rel_heading = float(
            np.arctan2(hfwd_xz @ lateral, hfwd_xz @ forward)
        )

        # Goal polar position in the robot frame.
        goal_dist = 0.0
        goal_bearing = 0.0
        info = getattr(task, "my_nav_to_info", None)
        if info is not None and info.robot_info is not None:
            g = info.robot_info.nav_goal_pos
            goal_dist, goal_bearing = _to_robot_frame(
                np.array([float(g[0]), float(g[2])], dtype=np.float64)
            )

        # Relative speed via finite difference of world base positions
        # (robot_xz / human_xz were computed above).
        ep_id = getattr(episode, "episode_id", None)
        if ep_id != self._prev_episode_id:
            # New episode: reset the finite-difference state.
            self._prev_robot_xz = None
            self._prev_human_xz = None
            self._prev_episode_id = ep_id
            self._approach_ema = 0.0
        human_rel_speed = 0.0
        approach_raw = 0.0
        if self._prev_robot_xz is not None and self._dt > 0:
            rel_vel = (
                (human_xz - self._prev_human_xz)
                - (robot_xz - self._prev_robot_xz)
            ) / self._dt
            human_rel_speed = float(np.linalg.norm(rel_vel))
            # Human ABSOLUTE velocity projected onto human->robot: positive
            # only when the human itself is closing in (ego-motion invariant).
            human_vel = (human_xz - self._prev_human_xz) / self._dt
            to_robot = robot_xz - human_xz
            n_tr = np.linalg.norm(to_robot)
            if n_tr > 1e-8:
                approach_raw = float(human_vel @ (to_robot / n_tr))
        self._prev_robot_xz = robot_xz
        self._prev_human_xz = human_xz

        feats = [
            human_dist,
            human_bearing,
            human_rel_heading,
            human_rel_speed,
            goal_dist,
            goal_bearing,
        ]
        if self._include_approach:
            self._approach_ema = (
                self._approach_alpha * approach_raw
                + (1.0 - self._approach_alpha) * self._approach_ema
            )
            feats.append(self._approach_ema)
        return np.array(feats, dtype=np.float32)


@registry.register_sensor
class RobotTrajectoryBufferSensor(UsesArticulatedAgentInterface, Sensor):
    """Fixed-length ring buffer of the agent's recent world-frame (x, z) base
    positions: newest last, front-padded with the oldest real point.

    Consumed by ``BackOffSkill`` to retrace the robot's own traversed path
    backward (a path it already walked is guaranteed navigable, so the retreat
    is collision-free) when yielding to the human.

    State is keyed by ``agent_id`` because ``RearrangeTask._duplicate_sensor_suite``
    shallow-copies (``copy.copy``) this sensor for each agent -- the copies share
    these dicts, but distinct ``agent_id`` keys keep the per-agent buffers
    isolated. The buffer resets whenever the episode id changes.
    """

    cls_uuid: str = "trajectory_buffer"

    def __init__(self, sim, config, *args, **kwargs):
        self._sim = sim
        self._buffer_size = int(getattr(config, "buffer_size", 100))
        self._min_step_dist = float(getattr(config, "min_step_dist", 0.1))
        self._bufs = {}     # agent_id -> list[np.ndarray(2,)]
        self._prev_ep = {}  # agent_id -> episode_id
        super().__init__(config=config)

    def _get_uuid(self, *args, **kwargs):
        return RobotTrajectoryBufferSensor.cls_uuid

    def _get_sensor_type(self, *args, **kwargs):
        return SensorTypes.TENSOR

    def _get_observation_space(self, *args, config, **kwargs):
        # Flattened to 1-D (buffer_size*2,) so the video renderer
        # (observations_to_image) skips it -- it treats any >1-D obs as an image.
        # BackOffSkill reshapes it back to (buffer_size, 2).
        n = int(getattr(config, "buffer_size", 100))
        return spaces.Box(
            shape=(n * 2,),
            low=np.finfo(np.float32).min,
            high=np.finfo(np.float32).max,
            dtype=np.float32,
        )

    def get_observation(self, observations, episode, *args, **kwargs):
        agent_id = self.agent_id if self.agent_id is not None else 0
        base_pos = np.asarray(
            self._sim.get_agent_data(agent_id).articulated_agent.base_pos,
            dtype=np.float32,
        )
        xz = np.array([base_pos[0], base_pos[2]], dtype=np.float32)

        ep_id = getattr(episode, "episode_id", None) if episode is not None else None
        if self._prev_ep.get(agent_id, "__unset__") != ep_id:
            self._bufs[agent_id] = []
            self._prev_ep[agent_id] = ep_id
        buf = self._bufs.setdefault(agent_id, [])

        # Append only when the agent has moved a meaningful step (keeps the
        # breadcrumbs spaced ~min_step_dist apart).
        if len(buf) == 0 or float(np.linalg.norm(xz - buf[-1])) >= self._min_step_dist:
            buf.append(xz)
            if len(buf) > self._buffer_size:
                buf.pop(0)

        out = np.zeros((self._buffer_size, 2), dtype=np.float32)
        n = len(buf)
        if n > 0:
            arr = np.asarray(buf, dtype=np.float32)
            out[self._buffer_size - n :] = arr
            out[: self._buffer_size - n] = arr[0]  # front-pad with the oldest point
        return out.reshape(-1)  # (buffer_size*2,)


def _door_side_sign(door_start, door_end, p):
    """Sign of the 2D cross product about the door line in the XZ plane.
    Mirrors ``BackoffWaypointDeltaSensor._side_sign`` so measures and the
    yield sensor agree on which side of the door a point lies on."""
    r1, r2 = door_start[0], door_end[0]
    c1, c2 = door_start[2], door_end[2]
    det = (p[0] - r1) * (c2 - c1) - (p[2] - c1) * (r2 - r1)
    return 1.0 if det > 0 else -1.0


def _door_perp_dist(door_start, door_end, p):
    """Perpendicular XZ distance from point ``p`` to the (infinite) door line."""
    a = np.array([door_start[0], door_start[2]], dtype=np.float64)
    b = np.array([door_end[0], door_end[2]], dtype=np.float64)
    q = np.array([p[0], p[2]], dtype=np.float64)
    ab = b - a
    n = float(np.linalg.norm(ab))
    if n < 1e-9:
        return float(np.linalg.norm(q - a))
    return float(abs(np.cross(ab, q - a)) / n)


@registry.register_measure
class HumanPassedDoor(Measure):
    """1.0 (latched) once the human has crossed to the robot's start side of
    the door AND is > 0.3 m past the door line. Robot-only episodes leave it 0."""

    cls_uuid: str = "human_passed_door"

    def __init__(self, *args, sim, config, **kwargs):
        self._sim = sim
        self._config = config
        self._passed = False
        super().__init__(*args, sim=sim, config=config, **kwargs)

    @staticmethod
    def _get_uuid(*args, **kwargs):
        return HumanPassedDoor.cls_uuid

    def reset_metric(self, *args, task, **kwargs):
        self._passed = False
        self.update_metric(*args, task=task, **kwargs)

    def update_metric(self, *args, task, **kwargs):
        if self._passed:
            self._metric = 1.0
            return
        info = getattr(self._sim.ep_info, "info", {}) or {}
        if "door_start" not in info or "door_end" not in info:
            self._metric = 0.0
            return
        ds = np.array(info["door_start"], dtype=np.float64)
        de = np.array(info["door_end"], dtype=np.float64)
        robot_start = np.array(task.my_nav_to_info.robot_info.articulated_agent_start_pos, dtype=np.float64)
        robot_side = _door_side_sign(ds, de, robot_start)
        human_pos = np.array(self._sim.get_agent_data(1).articulated_agent.base_pos, dtype=np.float64)
        if (
            _door_side_sign(ds, de, human_pos) == robot_side
            and _door_perp_dist(ds, de, human_pos) > 0.3
        ):
            self._passed = True
        self._metric = 1.0 if self._passed else 0.0


@registry.register_measure
class HumanDelay(Measure):
    """Extra seconds the human spent reaching its goal vs. an unobstructed
    nominal (geodesic / speed). Freezes once the human reaches its goal."""

    cls_uuid: str = "human_delay"

    def __init__(self, *args, sim, config, **kwargs):
        self._sim = sim
        self._config = config
        self._steps = 0
        self._nominal = 0.0
        self._frozen_val = None
        self._dt = 1.0
        try:
            cf = float(self._sim.habitat_config.ctrl_freq)
            afr = float(self._sim.habitat_config.ac_freq_ratio)
            if cf > 0:
                self._dt = afr / cf
        except Exception:
            self._dt = 1.0
        super().__init__(*args, sim=sim, config=config, **kwargs)

    @staticmethod
    def _get_uuid(*args, **kwargs):
        return HumanDelay.cls_uuid

    def reset_metric(self, *args, task, **kwargs):
        self._steps = 0
        self._frozen_val = None
        speed = float(getattr(self._config, "human_speed", 0.8)) or 0.8
        try:
            info = getattr(self._sim.ep_info, "info", {}) or {}
            hs = np.array(info["human_start"], dtype=np.float32)
            hg = np.array(task.my_nav_to_info.human_info.nav_goal_pos, dtype=np.float32)
            geo = self._sim.geodesic_distance(hs, hg)
            if not np.isfinite(geo):
                geo = float(np.linalg.norm((hs - hg)[[0, 2]]))
            self._nominal = geo / speed
        except Exception:
            self._nominal = 0.0
        self.update_metric(*args, task=task, **kwargs)

    def update_metric(self, *args, task, **kwargs):
        self._steps += 1
        if self._frozen_val is not None:
            self._metric = self._frozen_val
            return
        hg = np.array(task.my_nav_to_info.human_info.nav_goal_pos, dtype=np.float64)
        hp = np.array(self._sim.get_agent_data(1).articulated_agent.base_pos, dtype=np.float64)
        elapsed = self._steps * self._dt
        val = max(0.0, elapsed - self._nominal)
        if float(np.linalg.norm((hp - hg)[[0, 2]])) < 0.4:
            self._frozen_val = val
        self._metric = val


@registry.register_measure
class MinAgentClearance(Measure):
    """Running minimum XZ distance between the robot and human base positions."""

    cls_uuid: str = "min_agent_clearance"

    def __init__(self, *args, sim, config, **kwargs):
        self._sim = sim
        self._config = config
        self._min = float("inf")
        super().__init__(*args, sim=sim, config=config, **kwargs)

    @staticmethod
    def _get_uuid(*args, **kwargs):
        return MinAgentClearance.cls_uuid

    def reset_metric(self, *args, task, **kwargs):
        self._min = float("inf")
        self.update_metric(*args, task=task, **kwargs)

    def update_metric(self, *args, task, **kwargs):
        r = np.array(self._sim.get_agent_data(0).articulated_agent.base_pos, dtype=np.float64)
        h = np.array(self._sim.get_agent_data(1).articulated_agent.base_pos, dtype=np.float64)
        d = float(np.linalg.norm((r - h)[[0, 2]]))
        self._min = min(self._min, d)
        self._metric = self._min


@registry.register_measure
class HumanFrozenWhileClear(Measure):
    """Max consecutive steps where the human is nearly still (< 0.05 m/s) while
    the robot is > 1.0 m off the human's straight line to its goal and the
    human has not reached its goal. Detects a firm human that fails to resume."""

    cls_uuid: str = "human_frozen_while_clear"

    def __init__(self, *args, sim, config, **kwargs):
        self._sim = sim
        self._config = config
        self._prev_hp = None
        self._run = 0
        self._max_run = 0
        self._dt = 1.0
        try:
            cf = float(self._sim.habitat_config.ctrl_freq)
            afr = float(self._sim.habitat_config.ac_freq_ratio)
            if cf > 0:
                self._dt = afr / cf
        except Exception:
            self._dt = 1.0
        super().__init__(*args, sim=sim, config=config, **kwargs)

    @staticmethod
    def _get_uuid(*args, **kwargs):
        return HumanFrozenWhileClear.cls_uuid

    def reset_metric(self, *args, task, **kwargs):
        self._prev_hp = None
        self._run = 0
        self._max_run = 0
        self.update_metric(*args, task=task, **kwargs)

    @staticmethod
    def _point_seg_dist(p, a, b):
        p = np.asarray(p, dtype=np.float64)
        a = np.asarray(a, dtype=np.float64)
        b = np.asarray(b, dtype=np.float64)
        ab = b - a
        denom = float(np.dot(ab, ab))
        t = 0.0 if denom < 1e-9 else float(np.clip(np.dot(p - a, ab) / denom, 0.0, 1.0))
        return float(np.linalg.norm(p - (a + t * ab)))

    def update_metric(self, *args, task, **kwargs):
        hp = np.array(self._sim.get_agent_data(1).articulated_agent.base_pos, dtype=np.float64)
        hg = np.array(task.my_nav_to_info.human_info.nav_goal_pos, dtype=np.float64)
        rp = np.array(self._sim.get_agent_data(0).articulated_agent.base_pos, dtype=np.float64)
        speed = 0.0
        if self._prev_hp is not None and self._dt > 0:
            speed = float(np.linalg.norm((hp - self._prev_hp)[[0, 2]]) / self._dt)
        self._prev_hp = hp
        at_goal = float(np.linalg.norm((hp - hg)[[0, 2]])) < 0.4
        robot_off = (
            self._point_seg_dist(rp[[0, 2]], hp[[0, 2]], hg[[0, 2]]) > 1.0
        )
        if speed < 0.05 and robot_off and not at_goal:
            self._run += 1
            self._max_run = max(self._max_run, self._run)
        else:
            self._run = 0
        self._metric = float(self._max_run)
