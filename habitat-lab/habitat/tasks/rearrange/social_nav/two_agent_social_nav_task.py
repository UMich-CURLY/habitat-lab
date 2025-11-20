#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import copy
import numpy as np
from collections import OrderedDict
from typing import Optional, Any

from habitat.core.dataset import Episode
from habitat.core.registry import registry
from habitat.core.simulator import Sensor, SensorSuite
from habitat.tasks.nav.nav import NavigationTask
from habitat.tasks.rearrange.sub_tasks.nav_to_obj_task import (
    NavToInfo,
    MyNavToInfo,
)
from habitat.tasks.rearrange.utils import UsesArticulatedAgentInterface


@registry.register_task(name="TwoAgentSocialNavTask-v0")
class TwoAgentSocialNavTask(NavigationTask):
    """
    Minimal two-agent social navigation task.

    This task places two articulated agents (agent_0 robot, agent_1 human)
    according to the episode's start positions and exposes a navigation
    goal for each agent from episode.info (robot_goal / human_goal). It
    purposefully avoids PDDL/problem binding and object manipulation.
    """

    _nav_to_info: Optional[NavToInfo]
    my_nav_to_info: Optional[MyNavToInfo]

    def _duplicate_sensor_suite(self, sensor_suite: SensorSuite) -> None:
        """
        Duplicate articulated-agent sensors between agents so that each
        articulated agent has its own sensor instance in the task sensor
        suite. Adapted from RearrangeTask._duplicate_sensor_suite.
        """

        task_new_sensors: dict = {}
        task_obs_spaces = OrderedDict()
        for agent_idx, agent_id in enumerate(self._sim.agents_mgr.agent_names):
            for sensor_name, sensor in sensor_suite.sensors.items():
                if isinstance(sensor, UsesArticulatedAgentInterface):
                    new_sensor = copy.copy(sensor)
                    new_sensor.agent_id = agent_idx
                    full_name = f"{agent_id}_{sensor_name}"
                    task_new_sensors[full_name] = new_sensor
                    task_obs_spaces[full_name] = new_sensor.observation_space
                else:
                    task_new_sensors[sensor_name] = sensor
                    task_obs_spaces[sensor_name] = sensor.observation_space

        sensor_suite.sensors = task_new_sensors
        sensor_suite.observation_spaces = SensorSuite(
            list(task_new_sensors.values())
        ).observation_spaces

    def __init__(self, config: Any, sim, dataset=None):
        # Call NavigationTask initializer (this creates sensor_suite, actions, etc.)
        super().__init__(config=config, sim=sim, dataset=dataset)

        # If there are multiple articulated agents, duplicate per-agent sensors
        try:
            if len(self._sim.agents_mgr) > 1:
                self._duplicate_sensor_suite(self.sensor_suite)
        except Exception:
            # Be conservative: if agents_mgr is not present or duplication fails,
            # continue without duplication.
            pass

        self._min_start_distance = getattr(config, "min_start_distance", 0.0)
        self._nav_to_info = None

    def _generate_nav_start_goal(self, episode, agent_idx, force_idx=None) -> NavToInfo:
        """
        Generate start pose and navigation goal for a given agent index.
        Expects episode.info to contain 'human_start', 'human_goal',
        and top-level episode.start_position / start_rotation for the robot.
        """
        start_hold_obj_idx: Optional[int] = None

        # Use episode info already prepared for social-nav episodes.
        if agent_idx == 0:  # robot
            articulated_agent_pos = np.array(episode.start_position)
            try:
                articulated_agent_angle = episode.start_rotation[2]
            except Exception:
                # If rotation is stored differently, fall back to 0.
                articulated_agent_angle = 0.0
            nav_to_pos = np.array(episode.info.get("robot_goal", episode.start_position))
        elif agent_idx == 1:  # human
            articulated_agent_pos = np.array(episode.info["human_start"])
            try:
                articulated_agent_angle = episode.info.get("human_rot", [0.0, 0.0, 0.0])[2]
            except Exception:
                articulated_agent_angle = 0.0
            nav_to_pos = np.array(episode.info.get("human_goal", episode.info.get("human_start")))
        else:
            raise IndexError("TwoAgentSocialNavTask supports only two articulated agents (0 and 1)")

        return NavToInfo(
            nav_goal_pos=nav_to_pos,
            articulated_agent_start_pos=articulated_agent_pos,
            articulated_agent_start_angle=articulated_agent_angle,
            start_hold_obj_idx=start_hold_obj_idx,
        )

    def reset(self, episode: Episode):
        # Reset the sim without fetching observations (we'll return them later)
        super().reset(episode, fetch_observations=False)

        # Generate and set start/goal for each articulated agent
        for agent_id in range(self._sim.num_articulated_agents):
            self._nav_to_info = self._generate_nav_start_goal(
                episode, agent_id, force_idx=None
            )
            # Set articulated agent pose on the simulator agent data
            self._sim.get_agent_data(agent_id).articulated_agent.base_pos = (
                self._nav_to_info.articulated_agent_start_pos
            )
            self._sim.get_agent_data(agent_id).articulated_agent.base_rot = (
                self._nav_to_info.articulated_agent_start_angle
            )

            if agent_id == 0:
                robot_info = self._nav_to_info
            else:
                human_info = self._nav_to_info

        # Store combined info for convenience
        self.my_nav_to_info = MyNavToInfo(robot_info, human_info)

        # Update simulator and return observations
        if self._sim.habitat_config.debug_render:
            # Visualize robot goal position
            self._sim.viz_ids["nav_targ_pos"] = self._sim.visualize_position(
                self.my_nav_to_info.robot_info.nav_goal_pos,
                self._sim.viz_ids.get("nav_targ_pos"),
                r=0.2,
            )

        self._sim.maybe_update_articulated_agent()

        return self._get_observations(episode)
