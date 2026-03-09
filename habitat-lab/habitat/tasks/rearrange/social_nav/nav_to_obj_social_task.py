#!/usr/bin/env python3
"""Social navigation task that runs a local RVO/ORCA manager each episode.

This task subclasses the existing DynNavRLEnv (NavToObjTask-v0) and
initializes an RVOManager on reset. On each step it computes preferred
velocities toward each agent's nav goal, steps the RVO manager, and
applies the ORCA-computed velocity to the robot base using velocity
integration similar to `BaseVelAction.update_base`.
"""
from typing import Optional
import numpy as np

import magnum as mn
import habitat_sim

from habitat.core.registry import registry
from habitat.tasks.rearrange.sub_tasks.nav_to_obj_task import DynNavRLEnv
from habitat.tasks.rearrange.social_nav.rvo_manager import RVOManager


@registry.register_task(name="NavToObjSocialTask-v0")
class NavToObjSocialTask(DynNavRLEnv):
    def __init__(self, *args, config, dataset=None, **kwargs):
        super().__init__(*args, config=config, dataset=dataset, **kwargs)
        # RVO manager will be created per-episode in reset
        self._rvo_manager: Optional[RVOManager] = None
        self._agent_keys = []

    def reset(self, episode):
        # call parent reset (which sets up nav start/goal and agent poses)
        obs = super().reset(episode)

        # RVO initialized on-demand via `enable_rvo()`; keep manager None
        # until the social-nav skill requests it.
        self._rvo_manager = None
        self._agent_keys = []
        self._rvo_enabled = False

        return obs

    def step(self, action, episode):
        """Before performing the usual step, compute ORCA velocities and
        apply the robot base motion accordingly."""
        # Only run ORCA when explicitly enabled by the controller/skill.
        if not getattr(self, "_rvo_enabled", False) or self._rvo_manager is None:
            return super().step(action, episode)

        # compute preferred velocities for each agent toward their nav_goal
        # assume there are two agents in many social scenarios; fall back
        # to using per-agent nav goals stored in task if available
        for idx, key in enumerate(self._agent_keys):
            try:
                agent_data = self._sim.get_agent_data(idx)
                pos = np.array(agent_data.articulated_agent.base_pos)
                # determine nav goal for this agent
                if hasattr(self, "my_nav_to_info") and self.my_nav_to_info is not None:
                    if idx == 0:
                        goal = self.my_nav_to_info.robot_info.nav_goal_pos
                    else:
                        goal = self.my_nav_to_info.human_info.nav_goal_pos
                else:
                    # best-effort: use the task nav_goal_pos
                    goal = getattr(self, "nav_goal_pos", pos)

                # world-space 2D vector (x,z)
                rel = np.array([float(goal[0]) - float(pos[0]), float(goal[2]) - float(pos[2])])
                dist = np.linalg.norm(rel)
                if dist < 1e-6:
                    pref = (0.0, 0.0)
                else:
                    # normalize and scale by default max speed
                    pref_dir = rel / dist
                    max_speed = 1.0
                    pref = (pref_dir[0] * max_speed, pref_dir[1] * max_speed)
                self._rvo_manager.set_pref_velocity(key, pref)
            except Exception:
                # ignore agent-specific errors
                continue

        # step RVO
        self._rvo_manager.step()

        # get robot velocity (agent_0)
        robot_vel = self._rvo_manager.get_agent_velocity("agent_0")
        # Convert ORCA 2D velocity (vx, vy) in world frame to a local
        # longitudinal and angular velocity and integrate transform for the robot.
        try:
            # fetch agent 0 data
            agent_data = self._sim.get_agent_data(0)
            articulated = agent_data.articulated_agent
            sim_obj = articulated.sim_obj

            # current transform and rigid state
            trans = sim_obj.transformation
            rigid_state = habitat_sim.RigidState(mn.Quaternion.from_matrix(trans.rotation()), trans.translation)

            # compute forward vector in world frame
            forward = np.array(trans.transform_vector(np.array([1.0, 0.0, 0.0])))
            forward2 = np.array([forward[0], forward[2]])
            if np.linalg.norm(forward2) < 1e-6:
                forward2 = np.array([1.0, 0.0])
            forward2 = forward2 / np.linalg.norm(forward2)

            vel_world = np.array([robot_vel[0], robot_vel[1]])
            # forward speed is projection of vel_world onto forward2
            lin_speed = float(np.dot(vel_world, forward2))

            # desired heading angle of vel_world
            ang_speed = 0.0
            if np.linalg.norm(vel_world) > 1e-6:
                vel_dir = vel_world / np.linalg.norm(vel_world)
                # angle between forward2 and vel_dir
                ang = np.arctan2(vel_dir[1], vel_dir[0]) - np.arctan2(forward2[1], forward2[0])
                # normalize angle to [-pi, pi]
                ang = (ang + np.pi) % (2 * np.pi) - np.pi
                # simple proportional mapping to angular velocity
                ang_speed = float(ang)

            # Use VelocityControl to integrate transform similar to BaseVelAction
            vc = habitat_sim.physics.VelocityControl()
            vc.controlling_lin_vel = True
            vc.lin_vel_is_local = True
            vc.controlling_ang_vel = True
            vc.ang_vel_is_local = True

            # map lin_speed/ang_speed to the controller velocities
            # here we keep magnitudes conservative; users can tune via config
            lin = np.clip(lin_speed, -1.0, 1.0) * getattr(self._config, "lin_speed", 1.0)
            ang = np.clip(ang_speed, -1.0, 1.0) * getattr(self._config, "ang_speed", 1.0)

            vc.linear_velocity = mn.Vector3(lin, 0.0, 0.0)
            vc.angular_velocity = mn.Vector3(0.0, ang, 0.0)

            ctrl_freq = getattr(self._sim, "ctrl_freq", 60)
            target_rigid_state = vc.integrate_transform(1.0 / ctrl_freq, rigid_state)
            end_pos = self._sim.step_filter(rigid_state.translation, target_rigid_state.translation)

            # if step_filter failed (no movement), skip
            did_try_step_fail = np.allclose(end_pos, rigid_state.translation)
            if not did_try_step_fail:
                # apply base offset if present
                try:
                    end_pos = end_pos - articulated.params.base_offset
                except Exception:
                    pass

            target_trans = mn.Matrix4.from_(target_rigid_state.rotation.to_matrix(), end_pos)
            sim_obj.transformation = target_trans

            # update grasped objects (kinematic) if necessary
            if agent_data.grasp_mgr is not None and agent_data.grasp_mgr.snap_idx is not None:
                agent_data.grasp_mgr.update_object_to_grasp()

        except Exception:
            # If anything goes wrong, perform the normal task step as a fallback
            return super().step(action, episode)

        # Now continue with the normal task step (observations, metrics, etc.)
        return super().step(action, episode)

    def enable_rvo(self):
        """Enable and lazily initialize per-episode RVO manager.

        Safe to call multiple times.
        """
        if getattr(self, "_rvo_enabled", False):
            return
        try:
            from habitat.tasks.rearrange.social_nav.rvo_manager import (
                RVOManager,
            )
            ctrl_freq = getattr(self._sim, "ctrl_freq", 60)
            time_step = 1.0 / float(ctrl_freq)
            self._rvo_manager = RVOManager(time_step=time_step)
            # register articulated agents (robot=0, humans=1..N)
            self._agent_keys = []
            n_agents = getattr(self._sim, "num_articulated_agents", 1)
            for aid in range(n_agents):
                agent_data = self._sim.get_agent_data(aid)
                pos = np.array(agent_data.articulated_agent.base_pos)
                key = f"agent_{aid}"
                self._agent_keys.append(key)
                self._rvo_manager.add_agent(key, (float(pos[0]), float(pos[2])))
            self._rvo_enabled = True
            print("RVO enabled for episode (task)")
        except Exception:
            self._rvo_manager = None
            self._agent_keys = []
            self._rvo_enabled = False

    def disable_rvo(self):
        self._rvo_enabled = False
        self._rvo_manager = None
        self._agent_keys = []
