#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
from typing import Any, Dict, List, Optional, Sequence, Tuple

import magnum as mn
import habitat_sim

from habitat.core.dataset import Episode
from habitat.core.registry import registry
from habitat.tasks.rearrange.multi_task.pddl_task import PddlTask

from habitat.tasks.rearrange.social_nav.rvo_manager import (
    RVOManager,
    save_rvo_navmesh_obstacle_debug_figure,
    static_obstacles_from_sim_topdown,
)
from habitat.tasks.rearrange.sub_tasks.nav_to_obj_task import (
    NavToInfo,
    MyNavToInfo,
)
from IPython import embed

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

        # --- RVO / ORCA state (see enable_rvo / step) ---
        # These names mirror the ``rvo_*`` fields on ``TaskConfig``; Hydra
        # structured configs will reject any key not defined there.
        self._rvo_manager: Optional[RVOManager] = None
        self._rvo_enabled: bool = False
        self._rvo_agent_keys: List[str] = []
        self._rvo_controlled_agents: List[int] = []
        # ``rvo_use_orca_nav`` is the master switch; when True we enable RVO at
        # the end of every ``reset``.
        self._rvo_auto_enable: bool = bool(
            getattr(config, "rvo_use_orca_nav", False)
        )
        self._rvo_static_map_enabled: bool = bool(
            getattr(config, "rvo_static_map_enabled", True)
        )
        self._rvo_default_controlled: List[int] = list(
            getattr(config, "rvo_controlled_agents", [0])
        )
        self._rvo_radius: float = float(
            getattr(config, "rvo_agent_radius", 0.4)
        )
        self._rvo_max_speed: float = float(
            getattr(config, "rvo_default_max_speed", 1.0)
        )
        self._rvo_neighbor_dist: float = float(
            getattr(config, "rvo_neighbor_dist", 2.0)
        )
        self._rvo_max_neighbors: int = int(
            getattr(config, "rvo_max_neighbors", 10)
        )
        self._rvo_time_horizon: float = float(
            getattr(config, "rvo_time_horizon", 2.0)
        )
        self._rvo_time_horizon_obst: float = float(
            getattr(config, "rvo_time_horizon_obst", 4.0)
        )
        # Not a TaskConfig field; ORCA's preferred velocity is forced to 0 once
        # an agent is within this many metres of its nav goal (same 0.3 m
        # threshold that trajectory mode uses).
        self._rvo_goal_stop_radius: float = 0.3
        self._rvo_map_resolution: int = int(
            getattr(config, "rvo_map_resolution", 512)
        )
        self._rvo_meters_per_pixel: Optional[float] = getattr(
            config, "rvo_meters_per_pixel", None
        )
        self._rvo_lin_scale: float = float(
            getattr(config, "rvo_lin_speed_scale", 1.0)
        )
        self._rvo_ang_scale: float = float(
            getattr(config, "rvo_ang_speed_scale", 1.0)
        )
        # Per-agent overrides (both default to None -> fall back to the
        # shared ``rvo_agent_radius`` / ``rvo_default_max_speed`` values).
        self._rvo_agent_radius: Dict[int, Optional[float]] = {
            0: getattr(config, "rvo_agent_0_radius", None),
            1: getattr(config, "rvo_agent_1_radius", None),
        }
        self._rvo_agent_max_speed: Dict[int, Optional[float]] = {
            0: getattr(config, "rvo_agent_0_max_speed", None),
            1: getattr(config, "rvo_agent_1_max_speed", None),
        }
        self._rvo_debug_save_obstacle_figure: bool = bool(
            getattr(config, "rvo_debug_save_obstacle_figure", False)
        )
        self._rvo_debug_obstacle_figure_dir: str = str(
            getattr(
                config,
                "rvo_debug_obstacle_figure_dir",
                "video_dir/rvo_static_obstacles",
            )
        )
        self._rvo_debug_episode_id: Optional[str] = None
        self._rvo_debug_step_trail: Dict[str, List[Tuple[float, float]]] = {}
        self._rvo_debug_agent_paths: Dict[str, Dict[str, Any]] = {}
        self._rvo_debug_static_obs: List[Any] = []
        # Cache last applied (vx, vz) per agent so sync_agent_pose keeps ORCA
        # warm-started even though Habitat doesn't expose a base velocity.
        self._rvo_last_vel: Dict[str, Tuple[float, float]] = {}
        self._rvo_last_pref: Dict[str, Tuple[float, float]] = {}
        self._rvo_last_goal_dist: Dict[int, float] = {}
        print(
            f"[RVO][init] auto_enable={self._rvo_auto_enable} "
            f"static_map={self._rvo_static_map_enabled} "
            f"controlled={self._rvo_default_controlled} "
            f"radius={self._rvo_radius} max_speed={self._rvo_max_speed}"
        )
        
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

        if self._rvo_debug_save_obstacle_figure and self._rvo_debug_episode_id is not None:
            try:
                self._save_rvo_waypoint_debug_figure()
            except Exception as e:  # pragma: no cover
                print(f"[RVO] waypoint debug PNG flush failed: {e}")

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

        # Rebuild the RVO manager every episode: navmesh / agent poses change.
        self.disable_rvo()
        self._rvo_step_counter = 0
        if self._rvo_auto_enable:
            self.enable_rvo(self._rvo_default_controlled, episode=episode)

        # embed()
        return self._get_observations(episode)

    # ------------------------------------------------------------------ RVO

    def enable_rvo(
        self,
        controlled_agents: Optional[Sequence[int]] = None,
        *,
        episode: Optional[Episode] = None,
    ) -> None:
        """Lazily build a keyed RVOManager with the current scene's static obstacles.

        Safe to call multiple times per episode; the manager is rebuilt only
        when RVO is currently disabled. Agents are registered in simulator order
        (agent_0 == robot, agent_1 == human) with their current base pose.

        Parameters
        ----------
        controlled_agents
            Indices of articulated agents whose base transform should be
            overwritten by the ORCA velocity each :meth:`step`. If ``None``,
            uses ``rvo_controlled_agents`` from config (defaults to ``[0]``).
        episode
            Current episode; when ``rvo_debug_save_obstacle_figure`` is enabled,
            used to name the dumped obstacle PNG.
        """
        if self._rvo_enabled and self._rvo_manager is not None:
            return

        import time as _time

        ctrl_freq = getattr(self._sim, "ctrl_freq", 60)
        time_step = 1.0 / float(ctrl_freq)

        static_obs: List = []
        if self._rvo_static_map_enabled:
            print(
                f"[RVO] rasterizing navmesh obstacles "
                f"(map_resolution={self._rvo_map_resolution})..."
            )
            t0 = _time.time()
            try:
                static_obs = static_obstacles_from_sim_topdown(
                    self._sim,
                    map_resolution=self._rvo_map_resolution,
                    meters_per_pixel=self._rvo_meters_per_pixel,
                )
            except Exception as e:  # pragma: no cover - scene / navmesh issues
                print(f"[RVO] failed to rasterize navmesh obstacles: {e}")
                static_obs = []
            print(
                f"[RVO] obstacle rasterization done: "
                f"{len(static_obs)} polys in {_time.time() - t0:.2f}s"
            )

        print("[RVO] building PyRVOSimulator + KD tree...")
        t0 = _time.time()
        try:
            self._rvo_manager = RVOManager(
                time_step=time_step,
                neighbor_dist=self._rvo_neighbor_dist,
                max_neighbors=self._rvo_max_neighbors,
                time_horizon=self._rvo_time_horizon,
                time_horizon_obst=self._rvo_time_horizon_obst,
                radius=self._rvo_radius,
                default_max_speed=self._rvo_max_speed,
                static_obstacles=static_obs if static_obs else None,
            )
        except Exception as e:  # pragma: no cover
            print(f"[RVO] manager construction failed: {e}")
            self._rvo_manager = None
            self._rvo_enabled = False
            return
        print(
            f"[RVO] PyRVOSimulator ready in {_time.time() - t0:.2f}s"
        )

        self._rvo_agent_keys = []
        self._rvo_last_vel = {}
        n_agents = getattr(self._sim, "num_articulated_agents", 1)
        for aid in range(n_agents):
            agent_data = self._sim.get_agent_data(aid)
            pos = np.asarray(agent_data.articulated_agent.base_pos)
            # Precedence for radius/max_speed:
            #   per-agent config override > shared rvo_agent_radius config
            override_r = self._rvo_agent_radius.get(aid)
            radius = (
                float(override_r)
                if override_r is not None
                else self._rvo_radius
            )
            override_ms = self._rvo_agent_max_speed.get(aid)
            max_speed = (
                float(override_ms)
                if override_ms is not None
                else self._rvo_max_speed
            )
            key = f"agent_{aid}"
            self._rvo_manager.add_agent(
                key,
                (float(pos[0]), float(pos[2])),
                radius=radius,
                max_speed=max_speed,
            )
            self._rvo_agent_keys.append(key)
            self._rvo_last_vel[key] = (0.0, 0.0)

        if controlled_agents is None:
            controlled_agents = self._rvo_default_controlled
        self._rvo_controlled_agents = [
            int(a) for a in controlled_agents if 0 <= int(a) < n_agents
        ]
        self._rvo_enabled = True
        print(
            f"[RVO] enabled: {n_agents} agents, "
            f"{len(static_obs)} static obstacle polygons, "
            f"controlled={self._rvo_controlled_agents}, dt={time_step:.4f}s"
        )
        # Debug (no behavior change): confirm which ORCA params are in effect.
        print(
            "[RVO][debug:init] "
            f"neighbor_dist={self._rvo_neighbor_dist} "
            f"time_horizon={self._rvo_time_horizon} "
            f"time_horizon_obst={self._rvo_time_horizon_obst} "
            f"default_radius={self._rvo_radius} "
            f"default_max_speed={self._rvo_max_speed} "
            f"max_neighbors={self._rvo_max_neighbors}"
        )

        if self._rvo_debug_save_obstacle_figure and episode is not None:
            self._rvo_debug_static_obs = list(static_obs) if static_obs else []
            self._rvo_debug_episode_id = str(episode.episode_id)
            self._rvo_debug_step_trail = {f"agent_{aid}": [] for aid in range(n_agents)}
            self._rvo_debug_agent_paths = {}
            for aid in range(n_agents):
                agent_data = self._sim.get_agent_data(aid)
                pos3 = np.asarray(agent_data.articulated_agent.base_pos, dtype=float)
                start_xz = (float(pos3[0]), float(pos3[2]))
                goal_xz = self._rvo_goal_xz_for_agent(aid)
                key = f"agent_{aid}"
                path_xz: Optional[List[Tuple[float, float]]] = None
                status = "no_goal"
                if goal_xz is not None:
                    goal3 = np.array(
                        [goal_xz[0], float(pos3[1]), goal_xz[1]], dtype=float
                    )
                    path_xz, status = self._navmesh_path_xz(pos3, goal3)
                    self._log_navmesh_waypoints_for_agent(
                        aid,
                        str(episode.episode_id),
                        start_xz,
                        goal_xz,
                        path_xz,
                        status,
                    )
                else:
                    self._log_navmesh_waypoints_for_agent(
                        aid,
                        str(episode.episode_id),
                        start_xz,
                        None,
                        None,
                        "no_goal",
                    )
                self._rvo_debug_agent_paths[key] = {
                    "start": start_xz,
                    "goal": goal_xz,
                    "path": path_xz,
                    "step_trail": [],
                    "status": status if goal_xz is not None else "no_goal",
                }

    def disable_rvo(self) -> None:
        # Debug PNG is flushed at the start of the next reset(), not here, so a
        # mid-episode disable (e.g. back_off) does not write an empty step_trail.

        self._rvo_enabled = False
        self._rvo_manager = None
        self._rvo_agent_keys = []
        self._rvo_controlled_agents = []
        self._rvo_last_vel = {}
        self._rvo_last_pref = {}
        self._rvo_last_goal_dist = {}
        self._rvo_debug_tick = 0
        self._rvo_warned_snapshot = False
        self._rvo_debug_episode_id = None
        self._rvo_debug_step_trail = {}
        self._rvo_debug_agent_paths = {}
        self._rvo_debug_static_obs = []

    def _save_rvo_waypoint_debug_figure(self) -> None:
        """Write one PNG for the current debug episode (navmesh path + per-step trail)."""
        if self._rvo_debug_episode_id is None:
            return
        from pathlib import Path as _Path

        out_dir = _Path(self._rvo_debug_obstacle_figure_dir)
        stem = "".join(
            c if c.isalnum() or c in "-_." else "_"
            for c in str(self._rvo_debug_episode_id)
        )
        waypoint_png = out_dir / f"rvo_waypoints_ep_{stem}.png"
        agent_paths: Dict[str, Dict[str, Any]] = {}
        for key, base in self._rvo_debug_agent_paths.items():
            entry = dict(base)
            entry["step_trail"] = list(self._rvo_debug_step_trail.get(key, []))
            agent_paths[key] = entry

        trail_lens = {
            k: len(self._rvo_debug_step_trail.get(k, []))
            for k in agent_paths
        }
        print(
            f"[RVO] flushing waypoint PNG episode={self._rvo_debug_episode_id} "
            f"step_trail_lens={trail_lens}"
        )
        save_rvo_navmesh_obstacle_debug_figure(
            self._sim,
            waypoint_png,
            map_resolution=self._rvo_map_resolution,
            meters_per_pixel=self._rvo_meters_per_pixel,
            static_obstacles=self._rvo_debug_static_obs,
            agent_paths=agent_paths,
            title=(
                f"episode={self._rvo_debug_episode_id} | obstacles={len(self._rvo_debug_static_obs)} "
                f"| step_trail = per-tick RVO pref target (final goal)"
            ),
        )
        print(f"[RVO] waypoint debug PNG -> {waypoint_png.resolve()}")
        self._rvo_debug_episode_id = None

    def _rvo_goal_xz_for_agent(self, aid: int) -> Optional[Tuple[float, float]]:
        """Return the (x, z) navigation goal for agent ``aid`` if one is known."""
        info = getattr(self, "my_nav_to_info", None)
        if info is None:
            return None
        if aid == 0 and info.robot_info is not None:
            g = info.robot_info.nav_goal_pos
        elif aid == 1 and info.human_info is not None:
            g = info.human_info.nav_goal_pos
        else:
            return None
        g = np.asarray(g)
        if g.shape[-1] < 3:
            return None
        return (float(g[0]), float(g[2]))

    def _navmesh_path_xz(
        self,
        start_xyz: np.ndarray,
        goal_xyz: np.ndarray,
    ) -> Tuple[Optional[List[Tuple[float, float]]], str]:
        """Return full navmesh shortest-path as ``[(x, z), ...]`` and a status string."""
        pathfinder = getattr(self._sim, "pathfinder", None)
        if pathfinder is None or not getattr(pathfinder, "is_loaded", True):
            return None, "no_pathfinder"
        try:
            sp = habitat_sim.ShortestPath()
            sp.requested_start = np.asarray(start_xyz, dtype=np.float32)
            sp.requested_end = np.asarray(goal_xyz, dtype=np.float32)
            if not pathfinder.find_path(sp):
                return None, "find_path_failed"
            pts = list(sp.points)
            if len(pts) < 2:
                return None, "too_few_points"
            path_xz = [(float(p[0]), float(p[2])) for p in pts]
            return path_xz, "ok"
        except Exception as e:
            return None, f"exception:{e}"

    def _log_navmesh_waypoints_for_agent(
        self,
        aid: int,
        episode_id: str,
        start_xz: Tuple[float, float],
        goal_xz: Optional[Tuple[float, float]],
        path_xz: Optional[List[Tuple[float, float]]],
        status: str,
    ) -> None:
        """Print navmesh shortest path at reset (debug)."""
        ep = episode_id
        if goal_xz is None:
            print(
                f"[RVO][waypoints] agent_{aid} episode={ep} status=no_goal "
                "(no waypoints)"
            )
            return
        if path_xz is None or status != "ok":
            print(
                f"[RVO][waypoints] agent_{aid} episode={ep} status={status} "
                "(no waypoints)"
            )
            return
        print(
            f"[RVO][waypoints] agent_{aid} episode={ep} status=ok "
            f"n_sparse={len(path_xz)}"
        )
        for i, (px, pz) in enumerate(path_xz):
            label = "start" if i == 0 else ("goal" if i == len(path_xz) - 1 else "")
            extra = f" ({label})" if label else ""
            print(f"  sparse[{i}] (x={px:.4f}, z={pz:.4f}){extra}")

    def _snapshot_pre_step_state(self) -> Dict[int, np.ndarray]:
        """Record every controlled agent's pre-step (x, y, z) base position.

        Needed because we can only compute the correct ORCA-applied pose by
        starting from where the agent *was* this tick, not where Oracle-nav
        (run inside ``super().step(...)``) has already moved it to.
        """
        snapshot: Dict[int, np.ndarray] = {}
        if not self._rvo_enabled or self._rvo_manager is None:
            return snapshot
        for aid in self._rvo_controlled_agents:
            if aid >= len(self._rvo_agent_keys):
                continue
            try:
                pos = np.asarray(
                    self._sim.get_agent_data(aid).articulated_agent.base_pos,
                    dtype=float,
                )
                snapshot[aid] = pos.copy()
            except Exception:
                continue
        return snapshot

    def _rvo_compute_and_apply(
        self, pre_step_pos: Dict[int, np.ndarray]
    ) -> None:
        """Compute ORCA velocities and overwrite controlled agents' base poses.

        Controlled agents are *synced* into ORCA at their pre-step positions and
        *written back* to ``pre_step_pos + orca_vel * dt``. Uncontrolled agents
        are synced at their current (post-Oracle) positions so they still act
        as dynamic obstacles.
        """
        if not self._rvo_enabled or self._rvo_manager is None:
            self._rvo_step_counter = getattr(self, "_rvo_step_counter", 0) + 1
            ctrl_freq = int(getattr(self._sim, "ctrl_freq", 30))
            if ctrl_freq > 0 and self._rvo_step_counter % max(1, ctrl_freq) == 0:
                print(
                    f"[RVO][skip step {self._rvo_step_counter}] "
                    f"enabled={self._rvo_enabled} manager="
                    f"{self._rvo_manager is not None}"
                )
            return

        dt = 1.0 / float(getattr(self._sim, "ctrl_freq", 30))
        controlled_set = set(self._rvo_controlled_agents)

        # 1) Sync ORCA with the correct per-agent reference position.
        for aid, key in enumerate(self._rvo_agent_keys):
            if aid in controlled_set and aid in pre_step_pos:
                pos3 = pre_step_pos[aid]
            else:
                pos3 = np.asarray(
                    self._sim.get_agent_data(aid).articulated_agent.base_pos,
                    dtype=float,
                )
            self._rvo_manager.sync_agent_pose(
                key,
                (float(pos3[0]), float(pos3[2])),
                self._rvo_last_vel.get(key, (0.0, 0.0)),
            )

        # 2) Preferred velocities toward each agent's final nav goal.
        for aid, key in enumerate(self._rvo_agent_keys):
            if aid in controlled_set and aid in pre_step_pos:
                pos3 = pre_step_pos[aid]
            else:
                pos3 = np.asarray(
                    self._sim.get_agent_data(aid).articulated_agent.base_pos,
                    dtype=float,
                )
            goal_xz = self._rvo_goal_xz_for_agent(aid)
            if goal_xz is None:
                pref = (0.0, 0.0)
                self._rvo_manager.set_pref_velocity(key, pref)
                self._rvo_last_pref[key] = pref
                continue
            goal_dist = float(
                np.hypot(goal_xz[0] - pos3[0], goal_xz[1] - pos3[2])
            )
            if aid in controlled_set:
                self._rvo_last_goal_dist[aid] = goal_dist
            if goal_dist < self._rvo_goal_stop_radius:
                pref = (0.0, 0.0)
                self._rvo_manager.set_pref_velocity(key, pref)
                self._rvo_last_pref[key] = pref
                continue
            tx, tz = float(goal_xz[0]), float(goal_xz[1])
            dx = tx - float(pos3[0])
            dz = tz - float(pos3[2])
            dist = float(np.hypot(dx, dz))
            if dist < 1e-6:
                pref = (0.0, 0.0)
                self._rvo_manager.set_pref_velocity(key, pref)
                self._rvo_last_pref[key] = pref
                continue
            override_ms = self._rvo_agent_max_speed.get(aid)
            max_speed = (
                float(override_ms)
                if override_ms is not None
                else self._rvo_max_speed
            )
            inv = max_speed / dist
            pref = (dx * inv, dz * inv)
            self._rvo_manager.set_pref_velocity(key, pref)
            self._rvo_last_pref[key] = pref

            if (
                self._rvo_debug_save_obstacle_figure
                and self._rvo_debug_episode_id is not None
            ):
                trail = self._rvo_debug_step_trail.setdefault(key, [])
                trail.append((float(tx), float(tz)))

        # 3) Advance ORCA by dt and cache the solved velocities.
        self._rvo_manager.step()
        for key in self._rvo_agent_keys:
            v = self._rvo_manager.get_agent_velocity(key)
            self._rvo_last_vel[key] = (float(v[0]), float(v[1]))

        # Debug (no behavior change): when both agents are close, log ORCA vels vs
        # line-of-sight between them. Throttle to avoid flooding the console.
        # - along_0: signed component of agent_0 velocity along 0→1 (large + ⇒
        #   moving toward the other along the connecting segment; large lateral
        #   split means |along| is small while |v| is not).
        self._rvo_debug_tick = getattr(self, "_rvo_debug_tick", 0) + 1
        if (
            len(self._rvo_agent_keys) >= 2
            and 0 in pre_step_pos
            and 1 in pre_step_pos
        ):
            p0 = pre_step_pos[0]
            p1 = pre_step_pos[1]
            sep = float(
                np.hypot(
                    float(p0[0]) - float(p1[0]), float(p0[2]) - float(p1[2])
                )
            )
            if sep < 3.0 and self._rvo_debug_tick % 6 == 0:
                udx = float(p1[0]) - float(p0[0])
                udz = float(p1[2]) - float(p0[2])
                ulen = float(np.hypot(udx, udz)) + 1e-9
                udx /= ulen
                udz /= ulen
                v0 = self._rvo_last_vel.get("agent_0", (0.0, 0.0))
                v1 = self._rvo_last_vel.get("agent_1", (0.0, 0.0))
                v0n = float(np.hypot(v0[0], v0[1])) + 1e-9
                v1n = float(np.hypot(v1[0], v1[1])) + 1e-9
                along_0 = (v0[0] * udx + v0[1] * udz) / v0n
                along_1 = (v1[0] * (-udx) + v1[1] * (-udz)) / v1n
                lat0 = float(
                    np.hypot(v0[0] * udz - v0[1] * udx, 0.0) / v0n
                )
                lat1 = float(
                    np.hypot(v1[0] * udz - v1[1] * udx, 0.0) / v1n
                )
                # Remaining straight-line dist to each agent's nav goal (not
                # path length). If still far but pref gets zeroed, check
                # goal_stop; door choke is often narrow-portal + dual ORCA.
                dg0: Optional[float] = None
                dg1: Optional[float] = None
                g0 = self._rvo_goal_xz_for_agent(0)
                g1 = self._rvo_goal_xz_for_agent(1)
                if g0 is not None:
                    dg0 = float(
                        np.hypot(
                            float(p0[0]) - g0[0], float(p0[2]) - g0[1]
                        )
                    )
                if g1 is not None:
                    dg1 = float(
                        np.hypot(
                            float(p1[0]) - g1[0], float(p1[2]) - g1[1]
                        )
                    )
                gs = self._rvo_goal_stop_radius
                print(
                    f"[RVO][debug:close] tick={self._rvo_debug_tick} "
                    f"sep_m={sep:.2f} "
                    f"dist2goal0={dg0} dist2goal1={dg1} goal_stop_r={gs} "
                    f"(0_stops={dg0 is not None and dg0 < gs}) "
                    f"(1_stops={dg1 is not None and dg1 < gs}) "
                    f"v0=({v0[0]:.3f},{v0[1]:.3f})|v|={v0n:.3f} "
                    f"v1=({v1[0]:.3f},{v1[1]:.3f})|v|={v1n:.3f} "
                    f"along0_toward1={along_0:+.2f} lat0={lat0:.2f} "
                    f"along1_toward0={along_1:+.2f} lat1={lat1:.2f}"
                )

        # 4) Teleport controlled agents to (pre_pos + orca_vel * dt), keeping
        #    the ambient y (height) and snapping back onto the navmesh so we
        #    don't go through walls missed by the 2D ORCA obstacle list.
        pathfinder = getattr(self._sim, "pathfinder", None)
        for aid in self._rvo_controlled_agents:
            if aid not in pre_step_pos or aid >= len(self._rvo_agent_keys):
                continue
            key = self._rvo_agent_keys[aid]
            vx, vz = self._rvo_last_vel[key]
            pre3 = pre_step_pos[aid]
            new_x = float(pre3[0]) + vx * dt * self._rvo_lin_scale
            new_z = float(pre3[2]) + vz * dt * self._rvo_lin_scale
            new_y = float(pre3[1])
            start = mn.Vector3(float(pre3[0]), new_y, float(pre3[2]))
            target = mn.Vector3(new_x, new_y, new_z)
            try:
                filtered = self._sim.step_filter(start, target)
            except Exception:
                filtered = target
            final_pos = np.array(
                [float(filtered[0]), float(filtered[1]), float(filtered[2])],
                dtype=float,
            )
            # Snap back onto the navmesh surface to preserve height across slopes.
            if pathfinder is not None and getattr(pathfinder, "is_loaded", True):
                try:
                    snapped = pathfinder.snap_point(final_pos)
                    snap_arr = np.array(
                        [float(snapped[0]), float(snapped[1]), float(snapped[2])],
                        dtype=float,
                    )
                    if np.all(np.isfinite(snap_arr)):
                        final_pos = snap_arr
                except Exception:
                    pass

            try:
                agent_data = self._sim.get_agent_data(aid)
                agent_data.articulated_agent.base_pos = mn.Vector3(
                    float(final_pos[0]),
                    float(final_pos[1]),
                    float(final_pos[2]),
                )
                # Orient the base toward the ORCA velocity when we actually moved.
                if np.hypot(vx, vz) > 1e-3:
                    # Habitat base_rot is yaw around +Y; forward is -Z in local
                    # frame, so a world-space (vx, vz) direction of motion
                    # corresponds to yaw = atan2(vx, -vz).
                    try:
                        agent_data.articulated_agent.base_rot = float(
                            np.arctan2(vx, -vz)
                        )
                    except Exception:
                        pass
                # Keep grasped objects anchored while we teleport the base.
                if (
                    agent_data.grasp_mgr is not None
                    and agent_data.grasp_mgr.snap_idx is not None
                ):
                    agent_data.grasp_mgr.update_object_to_grasp()
            except Exception as e:
                print(f"[RVO] apply pose for agent {aid} failed: {e}")

    def step(self, action: Dict[str, Any], episode: Episode):
        """Snapshot pre-step poses, let the base task process the action,
        then overwrite controlled agents' poses with ``pre_pos + ORCA_vel * dt``."""
        pre_pos = self._snapshot_pre_step_state()
        # Debug: snapshot should cover every controlled id when RVO is on.
        if self._rvo_enabled and self._rvo_manager is not None:
            expected = set(self._rvo_controlled_agents)
            have = set(pre_pos.keys())
            if expected and not expected.issubset(have):
                if not getattr(self, "_rvo_warned_snapshot", False):
                    print(
                        "[RVO][debug] snapshot missing some controlled agents: "
                        f"expected={sorted(expected)} got={sorted(have)}"
                    )
                    self._rvo_warned_snapshot = True
        obs = super().step(action=action, episode=episode)
        if self._rvo_enabled and self._rvo_manager is not None:
            try:
                self._rvo_compute_and_apply(pre_pos)
                # Print velocities once per second so you can verify RVO is live.
                self._rvo_step_counter = getattr(self, "_rvo_step_counter", 0) + 1
                ctrl_freq = int(getattr(self._sim, "ctrl_freq", 30))
                if ctrl_freq > 0 and self._rvo_step_counter % max(1, ctrl_freq) == 0:
                    preview = {
                        k: (round(v[0], 3), round(v[1], 3))
                        for k, v in self._rvo_last_vel.items()
                    }
                    pref0 = self._rvo_last_pref.get("agent_0")
                    dg0 = self._rvo_last_goal_dist.get(0)
                    extra = ""
                    if pref0 is not None:
                        extra = (
                            f" agent_0 pref={tuple(round(x, 3) for x in pref0)}"
                            f" dist2goal={round(dg0, 3) if dg0 is not None else None}"
                            f" goal_stop={self._rvo_goal_stop_radius}"
                        )
                    print(
                        f"[RVO][step {self._rvo_step_counter}] "
                        f"controlled={self._rvo_controlled_agents} vel={preview}{extra}"
                    )
            except Exception as e:  # pragma: no cover - defensive
                print(f"[RVO] step failed, skipping overlay: {e}")
        return obs