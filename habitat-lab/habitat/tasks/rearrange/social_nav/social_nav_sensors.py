#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import magnum as mn
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
        super().reset_metric(
            *args,
            episode=episode,
            task=task,
            observations=observations,
            **kwargs,
        )
        # Reset the location visit tracker for the agent
        self._visited_pos = set()

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
            task.should_end = True
            social_nav_reward -= self._collide_penalty
        #### CADRL style reward ####
        else:
            # Dense goal-progress reward: positive when the robot reduces the
            # distance to its goal, negative when it moves away. This gives the
            # high-level policy a gradient toward picking `go_to_goal` when it is
            # safe to do so (otherwise the only goal signal is the sparse success
            # bonus, which the policy never reaches).
            if self._prev_dist_to_goal >= 0.0:
                social_nav_reward += self._goal_progress_reward * (
                    self._prev_dist_to_goal - dist_to_goal
                )

            # Social back-off shaping: when the human is inside the safety
            # radius, penalize being too close AND reward actively increasing the
            # robot-human distance (backing off). This rewards the desired
            # behavior of backing off / yielding when the human gets close.
            if dis < self._safe_dis_min:
                social_nav_reward -= (self._safe_dis_min - dis)
                if self._prev_dist >= 0.0:
                    social_nav_reward += self._backoff_reward * (
                        dis - self._prev_dist
                    )

        # No shaping on the first step (no valid previous distances yet).
        if self._prev_dist < 0 or self._prev_dist_to_goal < 0.0:
            social_nav_reward = 0.0

        self._metric += social_nav_reward
        # Update the distance
        self._prev_dist = dis  # type: ignore
        self._prev_dist_to_goal = dist_to_goal


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
      (m/s), from a finite difference of base positions.
    - goal_dist / goal_bearing: polar position of the robot's nav goal in the
      robot frame.

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
        return spaces.Box(
            shape=(6,),
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
        human_rel_speed = 0.0
        if self._prev_robot_xz is not None and self._dt > 0:
            rel_vel = (
                (human_xz - self._prev_human_xz)
                - (robot_xz - self._prev_robot_xz)
            ) / self._dt
            human_rel_speed = float(np.linalg.norm(rel_vel))
        self._prev_robot_xz = robot_xz
        self._prev_human_xz = human_xz

        return np.array(
            [
                human_dist,
                human_bearing,
                human_rel_heading,
                human_rel_speed,
                goal_dist,
                goal_bearing,
            ],
            dtype=np.float32,
        )


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
