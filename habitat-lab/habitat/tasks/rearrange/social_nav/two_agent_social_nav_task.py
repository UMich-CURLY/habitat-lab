#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
from typing import Optional, Any

from habitat.core.dataset import Episode
from habitat.core.registry import registry
from habitat.tasks.rearrange.multi_task.pddl_task import PddlTask

from habitat.tasks.rearrange.sub_tasks.nav_to_obj_task import (
    NavToInfo,
    MyNavToInfo,
)


@registry.register_task(name="TwoAgentSocialNavTask-v0")
class TwoAgentSocialNavTask(PddlTask):
    """
    Minimal two-agent social navigation task.

    This task places two articulated agents (agent_0 robot, agent_1 human)
    according to the episode's start positions and exposes a navigation
    goal for each agent from episode.info (robot_goal / human_goal).
    """

    _nav_to_info: Optional[NavToInfo]
    my_nav_to_info: Optional[MyNavToInfo]

    def __init__(self, config: Any, sim, dataset=None):
        # Call PddlTask initializer 
        super().__init__(config=config, sim=sim, dataset=dataset)
        
        self._min_start_distance = getattr(config, "min_start_distance", 0.0)
        self._nav_to_info = None
        print("task.actions keys:", list(self.actions.keys()))
        # Inspect PDDL problem and actions
        pddl_prob = self.pddl_problem  # or task._pddl_problem depending on your Task impl
        print("Available PDDL actions:", list(pddl_prob.actions.keys()))
        # Examine the post-conditions for a named action (e.g., 'nav' or 'nav_to_receptacle_by_name')
        # for a_name, a_obj in pddl_prob.actions.items():
        #     print("Action:", a_name)
        #     print("  preconds:", [p.compact_str for p in a_obj.pre_cond])
        #     print("  postconds:", [p.compact_str for p in a_obj.post_cond])
        #     print("  n_args:", a_obj.n_args)
        

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
        # Generate and set start/goal for each articulated agent before the
        # task-level reset so the simulator and any downstream binders see the
        # intended articulated-agent start poses.
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

        # Now call the parent reset which will perform simulator reset steps,
        # pddl binding (in PddlTask), and any other initialization. We don't
        # pass fetch_observations here — the parent manages that internally.
        super().reset(episode)

        # Debug visualization of the goal position (if enabled)
        if self._sim.habitat_config.debug_render:
            self._sim.viz_ids["nav_targ_pos"] = self._sim.visualize_position(
                self.my_nav_to_info.robot_info.nav_goal_pos,
                self._sim.viz_ids.get("nav_targ_pos"),
                r=0.2,
            )
       

        self._sim.maybe_update_articulated_agent()

        return self._get_observations(episode)