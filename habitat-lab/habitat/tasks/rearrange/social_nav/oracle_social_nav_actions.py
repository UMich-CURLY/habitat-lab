# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import magnum as mn
import numpy as np
from gym import spaces

import habitat_sim
from habitat.core.registry import registry
from habitat.tasks.rearrange.actions.actions import (
    BaseVelAction,
    HumanoidJointAction,
)
from habitat.tasks.rearrange.actions.oracle_nav_action import (
    OracleNavAction,
    SimpleVelocityControlEnv,
)
from habitat.tasks.rearrange.social_nav.utils import (
    robot_human_vec_dot_product,
)
from habitat.tasks.rearrange.utils import place_agent_at_dist_from_pos
from habitat.tasks.utils import get_angle

from IPython import embed
@registry.register_task_action
class OracleNavCoordAction(OracleNavAction):  # type: ignore
    """
    An action that comments the agent to navigate to a sequence of random navigation targets (or we call these targets (x,y) coordinates)
    """

    def __init__(self, *args, task, **kwargs):
        super().__init__(*args, task=task, **kwargs)
        self.nav_mode = None
        self.simple_backward = False

    @property
    def action_space(self):
        return spaces.Dict(
            {
                self._action_arg_prefix
                + "oracle_nav_coord_action": spaces.Box(
                    shape=(3,),
                    low=np.finfo(np.float32).min,
                    high=np.finfo(np.float32).max,
                    dtype=np.float32,
                )
            }
        )

    def _get_target_for_coord(self, obj_pos):
        """Get the targets by recording them in the dict"""
        precision = 0.25
        pos_key = np.around(obj_pos / precision, decimals=0) * precision
        pos_key = tuple(pos_key)
        if pos_key not in self._targets:
            start_pos, _, _ = place_agent_at_dist_from_pos(
                np.array(obj_pos),
                0.0,
                self._config.spawn_max_dist_to_obj,
                self._sim,
                self._config.num_spawn_attempts,
                True,
                self.cur_articulated_agent,
            )
            self._targets[pos_key] = start_pos
        else:
            start_pos = self._targets[pos_key]
        if self.motion_type == "human_joints":
            self.humanoid_controller.reset(
                self.cur_articulated_agent.base_transformation
            )
        return (start_pos, np.array(obj_pos))

    def step(self, *args, **kwargs):
        self.skill_done = False
        nav_to_target_coord = kwargs.get(
            self._action_arg_prefix + "oracle_nav_coord_action",
            self._action_arg_prefix + "oracle_nav_human_action",
        )
        # print("kwargs  here are ", kwargs)
        if np.linalg.norm(nav_to_target_coord) == 0:
            return {}
        final_nav_targ, obj_targ_pos = self._get_target_for_coord( #debug
            nav_to_target_coord
        )
        door_pos = (np.array(self._sim.ep_info.info["door_start"]) + np.array(self._sim.ep_info.info["door_end"]))/2
        # if self.motion_type == "human_joints":
        #     final_nav_targ = self._task.my_nav_to_info.human_info.nav_goal_pos #kl
        #     obj_targ_pos  = self._task.my_nav_to_info.human_info.nav_goal_pos
        # else:
            # final_nav_targ = nav_to_target_coord
            # print("final nav target here is", final_nav_targ)
        #KL: generate humanoidpath from here

        base_T = self.cur_articulated_agent.base_transformation
        curr_path_points = self._path_to_point(final_nav_targ)
        robot_pos = np.array(self.cur_articulated_agent.base_pos)
        if curr_path_points is None:
            raise Exception
        else:
            # Compute distance and angle to target
            if len(curr_path_points) == 1:
                curr_path_points += curr_path_points
            cur_nav_targ = curr_path_points[1]
            
            forward = np.array([1.0, 0, 0])
            robot_forward = np.array(base_T.transform_vector(forward))

            # Compute relative target.
            rel_targ = cur_nav_targ - robot_pos

            # Compute heading angle (2D calculation)
            robot_forward = robot_forward[[0, 2]]
            rel_targ = rel_targ[[0, 2]]
            rel_pos = (obj_targ_pos - robot_pos)[[0, 2]]
            rel_door_pos = (door_pos - robot_pos)[[0, 2]]
            angle_to_target = get_angle(robot_forward, rel_targ)
            angle_to_obj = get_angle(robot_forward, rel_pos)
            angle_to_door = get_angle(robot_forward, rel_door_pos)
            dist_to_final_nav_targ = np.linalg.norm(
                (final_nav_targ - robot_pos)[[0, 2]]
            )
            self._config.dist_thresh = 0.1
            at_goal = (
                dist_to_final_nav_targ < self._config.dist_thresh
                and angle_to_door < self._config.turn_thresh # Used to be abgle to obj, TK
            ) or dist_to_final_nav_targ < self._config.dist_thresh / 10.0
            if self.motion_type == "base_velocity":
                # print("Current nav path is ", curr_path_points)
                # print("Robot pose and current nav target is ", robot_pos, final_nav_targ)
                # print("Dist to goal is ", dist_to_final_nav_targ, self._config.dist_thresh)
                if not at_goal:
                    backward = np.array([-1.0, 0, 0])
                    robot_backward = np.array(
                        base_T.transform_vector(backward)
                    )
                    robot_backward = robot_backward[[0, 2]]
                    angle_to_target_back = get_angle(robot_backward, rel_targ)
                    if angle_to_target_back < 0.5:
                        self.nav_mode = "avoid"
                    else:
                        self.nav_mode = "dont_avoid"
                    # print(self.nav_mode)
                    if self.nav_mode == "avoid":
                        backward = np.array([-1.0, 0, 0])
                        robot_backward = np.array(
                            base_T.transform_vector(backward)
                        )
                        robot_backward = robot_backward[[0, 2]]
                        angle_to_target = get_angle(robot_backward, rel_targ)
                        # self.simple_backward = True
                        # print("Angle to target is ", angle_to_target)
                        if (
                            self.simple_backward
                            or angle_to_target < self._config.turn_thresh
                        ):
                            # Move backwards the target
                            vel = [-self._config.forward_velocity, 0]
                            # print ("Backward thresh is ", self._config.turn_thresh)
                            # Robot's rear looks at the target waypoint.
                        else:
                            print("Got here? ")
                            vel = OracleNavAction._compute_turn(
                                rel_targ,
                                self._config.turn_velocity,
                                robot_backward,
                            )
                    else:
                        
                        if dist_to_final_nav_targ < self._config.dist_thresh:
                            # Look at the object
                            print("Looking to the door now")
                            vel = OracleNavAction._compute_turn(
                                rel_door_pos,  # Used to be rel_pos, TK
                                self._config.turn_velocity,
                                robot_forward,
                            )
                        elif angle_to_target < self._config.turn_thresh:
                            # Move towards the target
                            vel = [self._config.forward_velocity, 0]
                        else:
                            HumanoidJointAction# Look at the target waypoint.
                            vel = OracleNavAction._compute_turn(
                                rel_targ,
                                self._config.turn_velocity,
                                robot_forward,
                            )
                else:
                    vel = [0, 0]
                    self.skill_done = True
                kwargs[f"{self._action_arg_prefix}base_vel"] = np.array(vel)
                # print("Got base_velocity here" , vel)
                return BaseVelAction.step(self, *args, **kwargs)

            elif self.motion_type == "human_joints":
                # Update the humanoid base
                self.humanoid_controller.obj_transform_base = base_T
                if not at_goal:
                    if dist_to_final_nav_targ < self._config.dist_thresh:
                        # Look at the object
                        self.humanoid_controller.calculate_turn_pose(
                            mn.Vector3([rel_pos[0], 0.0, rel_pos[1]])
                        )
                    else:
                        # Move towards the target
                        if self._config["lin_speed"] == 0:
                            distance_multiplier = 0.0
                        else:
                            distance_multiplier = 1.0
                        self.humanoid_controller.calculate_walk_pose(
                            mn.Vector3([rel_targ[0], 0.0, rel_targ[1]]),
                            distance_multiplier,
                        )
                else:
                    self.humanoid_controller.calculate_stop_pose()
                    self.skill_done = True
                # This line is important to reset the controller
                self._update_controller_to_navmesh()
                base_action = self.humanoid_controller.get_pose()
                kwargs[
                    f"{self._action_arg_prefix}human_joints_trans"
                ] = base_action

                return HumanoidJointAction.step(self, *args, **kwargs)
            else:
                raise ValueError(
                    "Unrecognized motion type for oracle nav action"
                )
            


@registry.register_task_action
class OracleNavRandCoordAction(OracleNavCoordAction):  # type: ignore
    """
    Oracle Nav RandCoord Action. Selects a random position in the scene and navigates
    there until reaching. When the arg is 1, does replanning.
    """

    def __init__(self, *args, task, **kwargs):
        super().__init__(*args, task=task, **kwargs)
        self._config = kwargs["config"]

    @property
    def action_space(self):
        return spaces.Dict(
            {
                self._action_arg_prefix
                + "oracle_nav_randcoord_action": spaces.Box(
                    shape=(1,),
                    low=np.finfo(np.float32).min,
                    high=np.finfo(np.float32).max,
                    dtype=np.float32,
                )
            }
        )

    def reset(self, *args, **kwargs):
        super().reset(*args, **kwargs)
        if self._task._episode_id != self._prev_ep_id:
            self._targets = {}
            self._prev_ep_id = self._task._episode_id
        self.skill_done = False
        self.coord_nav = self._task.my_nav_to_info.human_info.nav_goal_pos
        # self.coord_nav = None

    def _find_path_given_start_end(self, start, end):
        """Helper function to find the path given starting and end locations"""
        path = habitat_sim.ShortestPath()
        path.requested_start = start
        path.requested_end = end
        found_path = self._sim.pathfinder.find_path(path)
        if not found_path:
            return [start, end]
        return path.points

    def _reach_human(self, robot_pos, human_pos, base_T):
        """Check if the agent reaches the human or not"""
        facing = (
            robot_human_vec_dot_product(robot_pos, human_pos, base_T) > 0.5
        )
        # Use geodesic distance here
        dis = self._sim.geodesic_distance(robot_pos, human_pos)
        
        return dis <= 2.0 and facing

    def _compute_robot_to_human_min_step(
        self, robot_trans, human_pos, human_pos_list
    ):
        """The function to compute the minimum step to reach the goal"""
        _vel_scale = self._config.lin_speed

        # Copy the robot transformation
        base_T = mn.Matrix4(robot_trans)

        vc = SimpleVelocityControlEnv()

        # Compute the step taken to reach the human
        robot_pos = np.array(base_T.translation)
        robot_pos[1] = human_pos[1]
        self._reach_human(robot_pos, human_pos, base_T)
        step_taken = 0
        while (
            not self._reach_human(robot_pos, human_pos, base_T) 
            and step_taken <= 1500
        ):
            path_points = self._find_path_given_start_end(robot_pos, human_pos)
            cur_nav_targ = path_points[1]
            obj_targ_pos = path_points[1]
            forward = np.array([1.0, 0, 0])
            robot_forward = np.array(base_T.transform_vector(forward))

            # Compute relative target.
            rel_targ = cur_nav_targ - robot_pos
            rel_pos = (obj_targ_pos - robot_pos)[[0, 2]]

            # Compute heading angle (2D calculation)
            robot_forward = robot_forward[[0, 2]]
            rel_targ = rel_targ[[0, 2]]
            angle_to_target = get_angle(robot_forward, rel_targ)
            dist_to_final_nav_targ = np.linalg.norm(
                (human_pos - robot_pos)[[0, 2]]
            )

            if dist_to_final_nav_targ < self._config.dist_thresh:
                # Look at the object
                vel = OracleNavAction._compute_turn(
                    rel_pos,
                    self._config.turn_velocity * _vel_scale,
                    robot_forward,
                )
            elif angle_to_target < self._config.turn_thresh:
                # Move towards the target
                vel = [self._config.forward_velocity * _vel_scale, 0]
            else:
                # Look at the target waypoint.
                vel = OracleNavAction._compute_turn(
                    rel_targ,
                    self._config.turn_velocity * _vel_scale,
                    robot_forward,
                )

            # Update the robot's info
            base_T = vc.act(base_T, vel)
            robot_pos = np.array(base_T.translation)
            step_taken += 1

            robot_pos[1] = human_pos[1]
        return step_taken

    def _get_target_for_coord(self, obj_pos):
        start_pos = obj_pos
        if self.motion_type == "human_joints":
            self.humanoid_controller.reset(
                self.cur_articulated_agent.base_transformation
            )
        return (start_pos, np.array(obj_pos))

    def step(self, *args, **kwargs):
        max_tries = 10
        self.skill_done = False
        
        # #KL: test human agent pos
        # robot_pos = self._sim.get_agent_data(0).articulated_agent.base_pos
        # human_pos = self._sim.get_agent_data(1).articulated_agent.base_pos
        # print("TEST in step: ", human_pos)
        # print("-------TEST in step for COORD_NAV:", self.coord_nav, "is called--------")


        # if self.coord_nav is None:
        #     self.coord_nav = self._sim.pathfinder.get_random_navigable_point(
        #         max_tries,
        #         island_index=self._sim.largest_island_idx,
        #     )

        kwargs[
            self._action_arg_prefix + "oracle_nav_coord_action"
        ] = self.coord_nav
        #KL: debug
        kwargs[
            self._action_arg_prefix + "oracle_nav_randcoord_action"
        ] = self.coord_nav

        ret_val = super().step(*args, **kwargs)
        # if self.skill_done:
        #     self.coord_nav = None

        # # If the robot is nearby, the human starts to walk, otherwise, the human
        # # just stops there and waits for robot to find it
        # if self._config.human_stop_and_walk_to_robot_distance_threshold != -1:
        #     assert (
        #         len(self._sim.agents_mgr) == 2
        #     ), "Does not support more than two agents when you want human to stop and walk based on the distance to the robot"
        #     robot_id = int(1 - self._agent_index)
        #     robot_pos = self._sim.get_agent_data(
        #         robot_id
        #     ).articulated_agent.base_pos
        #     human_pos = self.cur_articulated_agent.base_pos
        #     dis = self._sim.geodesic_distance(robot_pos, human_pos)
        #     # The human needs to stop and wait for robot to come if the distance is too larget
        #     if (
        #         dis
        #         > self._config.human_stop_and_walk_to_robot_distance_threshold
        #     ):
        #         self.humanoid_controller.set_framerate_for_linspeed(
        #             0.0, 0.0, self._sim.ctrl_freq
        #         )
        #     # The human needs to walk otherwise
        #     else:
        #         speed = np.random.uniform(
        #             self._config.lin_speed / 5.0, self._config.lin_speed
        #         )
        #         lin_speed = speed
        #         ang_speed = speed
        #         self.humanoid_controller.set_framerate_for_linspeed(
        #             lin_speed, ang_speed, self._sim.ctrl_freq
        #         )
        speed = np.random.uniform(
            self._config.lin_speed / 5.0, self._config.lin_speed
        )
        if self.motion_type == "human_joints":
            lin_speed = speed
            ang_speed = speed
            self.humanoid_controller.set_framerate_for_linspeed(
                lin_speed, ang_speed, self._sim.ctrl_freq
            )

            try:
                kwargs["task"].measurements.measures[
                    "social_nav_stats"
                ].update_human_pos = self.coord_nav

            except Exception:
                pass
        return ret_val


@registry.register_task_action
class OracleNavWaypointAction(OracleNavAction):  # type: ignore
    """Steer the agent toward a fed (x, y, z) waypoint with turn-then-go
    kinematics, moving at the speed encoded in the waypoint.

    ``TwoAgentSocialNavTask`` solves ORCA and injects
    ``waypoint = base_pos + orca_vel * 1.0`` into the ``oracle_nav_action`` arg,
    so ``|waypoint - base_pos|`` equals the desired speed in m/s. We deliberately
    keep the arg name ``oracle_nav_action`` (not a new ``*_coord_action``) so the
    habitat-baselines ``OracleNavPolicy`` / ``find_action_range`` still resolve it;
    the policy's scalar write into that slot is overwritten by the task.

    No door logic, no backward "avoid" mode, no ``_get_target_for_coord`` re-snap,
    no pathfinder: ORCA handles avoidance and ``step_filter`` (inside
    ``BaseVelAction`` / ``_update_controller_to_navmesh``) keeps us on the navmesh.

    Firm-human blocked-gate (human_joints only, ``firm_mode=True``): under
    ``kinematic_mode`` the humanoid's ``step_filter`` respects the navmesh but
    NOT the robot body, so a firm (non-reciprocal) human would clip through a
    stationary robot. The gate stops the human only when it is physically
    blocked — the other agent within ``block_clearance`` and roughly ahead —
    with hysteresis both ways, and resumes automatically once clear.
    """

    def __init__(self, *args, task, **kwargs):
        super().__init__(*args, task=task, **kwargs)
        self._firm_mode = bool(getattr(self._config, "firm_mode", False))
        self._block_clearance = float(
            getattr(self._config, "block_clearance", 0.8)
        )
        self._block_hysteresis_steps = int(
            getattr(self._config, "block_hysteresis_steps", 5)
        )
        self._block_other_agent_idx = int(
            getattr(self._config, "block_other_agent_idx", 0)
        )
        self._blocked_state = False
        self._blocked_raw_count = 0
        self._clear_raw_count = 0
        self._blocked_prev_ep = None

    @property
    def action_space(self):
        return spaces.Dict(
            {
                self._action_arg_prefix
                + "oracle_nav_action": spaces.Box(
                    shape=(3,),
                    low=np.finfo(np.float32).min,
                    high=np.finfo(np.float32).max,
                    dtype=np.float32,
                )
            }
        )

    def _update_blocked_gate(self, robot_pos, rel_targ) -> bool:
        """Return True when the human is physically blocked by the other agent.

        Raw block = other agent within ``block_clearance`` AND roughly ahead
        (angle between commanded motion and the vector to it < 75 deg). The
        latched state flips only after ``block_hysteresis_steps`` consecutive
        raw observations, both entering and leaving.
        """
        ep_id = getattr(self._sim.ep_info, "episode_id", None)
        if ep_id != self._blocked_prev_ep:
            self._blocked_prev_ep = ep_id
            self._blocked_state = False
            self._blocked_raw_count = 0
            self._clear_raw_count = 0

        try:
            other = self._sim.get_agent_data(
                self._block_other_agent_idx
            ).articulated_agent
            other_xz = np.array(other.base_pos)[[0, 2]]
        except Exception:
            return self._blocked_state

        self_xz = np.asarray(robot_pos)[[0, 2]]
        to_other = other_xz - self_xz
        d = float(np.linalg.norm(to_other))
        raw = False
        if d < self._block_clearance and d > 1e-6:
            motion = np.asarray(rel_targ, dtype=float)
            if float(np.linalg.norm(motion)) > 1e-6:
                cos_ang = float(
                    np.dot(motion, to_other)
                    / (np.linalg.norm(motion) * d)
                )
                # cos(75 deg) ~= 0.2588
                raw = cos_ang > 0.2588

        if raw:
            self._blocked_raw_count += 1
            self._clear_raw_count = 0
        else:
            self._clear_raw_count += 1
            self._blocked_raw_count = 0

        if (
            not self._blocked_state
            and self._blocked_raw_count >= self._block_hysteresis_steps
        ):
            self._blocked_state = True
        elif (
            self._blocked_state
            and self._clear_raw_count >= self._block_hysteresis_steps
        ):
            self._blocked_state = False
        return self._blocked_state

    def step(self, *args, **kwargs):
        self.skill_done = False
        targ = np.asarray(
            kwargs[self._action_arg_prefix + "oracle_nav_action"], dtype=float
        )
        # Idle until the task injects a real 3-D waypoint (the policy may write a
        # bare scalar entity index into this slot before the task overwrites it).
        if targ.size < 3 or not np.any(targ):
            return

        base_T = self.cur_articulated_agent.base_transformation
        robot_pos = np.array(self.cur_articulated_agent.base_pos)
        robot_forward = np.array(
            base_T.transform_vector(mn.Vector3(1.0, 0.0, 0.0))
        )[[0, 2]]
        rel_targ = (targ - robot_pos)[[0, 2]]
        # 1 s lookahead => distance to the waypoint IS the desired speed (m/s).
        dist = float(np.linalg.norm(rel_targ))
        angle_to_target = get_angle(robot_forward, rel_targ)

        if self.motion_type == "base_velocity":
            if dist < self._config.dist_thresh:
                vel = [0.0, 0.0]
                self.skill_done = True
            elif angle_to_target < self._config.turn_thresh:
                # BaseVelAction does clip(lin,-1,1)*lin_speed, so commanding
                # dist/lin_speed yields an effective forward speed of `dist`.
                vel = [dist / self._lin_speed, 0.0]
            else:
                vel = OracleNavAction._compute_turn(
                    rel_targ, self._config.turn_velocity, robot_forward
                )
            kwargs[f"{self._action_arg_prefix}base_vel"] = np.array(vel)
            return BaseVelAction.step(self, *args, **kwargs)

        elif self.motion_type == "human_joints":
            self.humanoid_controller.obj_transform_base = base_T
            physically_blocked = False
            if self._firm_mode:
                physically_blocked = self._update_blocked_gate(
                    robot_pos, rel_targ
                )
            if dist < self._config.dist_thresh or physically_blocked:
                self.humanoid_controller.calculate_stop_pose()
                # Only a true goal arrival ends the skill; a physical block is
                # a transient wait, so leave skill_done False to keep replanning.
                if dist < self._config.dist_thresh:
                    self.skill_done = True
            else:
                # Match the ORCA speed; the controller turns gradually itself.
                self.humanoid_controller.set_framerate_for_linspeed(
                    max(dist, 1e-3),
                    self._config.ang_speed,
                    self._sim.ctrl_freq,
                )
                self.humanoid_controller.calculate_walk_pose(
                    mn.Vector3([rel_targ[0], 0.0, rel_targ[1]])
                )
            self._update_controller_to_navmesh()
            kwargs[
                f"{self._action_arg_prefix}human_joints_trans"
            ] = self.humanoid_controller.get_pose()
            return HumanoidJointAction.step(self, *args, **kwargs)
        else:
            raise ValueError(
                "Unrecognized motion type for oracle nav waypoint action"
            )
