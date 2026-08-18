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

# Seconds of ORCA velocity to project the per-step nav waypoint. A 1 s horizon
# makes the waypoint distance numerically equal to the ORCA-solved speed (m/s),
# which OracleNavWaypointAction reads back as its forward speed (speed-matching).
RVO_WAYPOINT_LOOKAHEAD = 1.0


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
        # --- Firm-human knobs (defaults reproduce legacy behavior) ---
        # 1.0: ORCA models every agent as cooperatively goal-seeking (legacy).
        # 0.0: uncontrolled agents (the robot) enter ORCA with their OBSERVED
        # velocity only, so the human stops counting on them to give way.
        self._rvo_reciprocity: float = float(
            getattr(config, "rvo_reciprocity", 1.0)
        )
        # Controlled agents' preferred velocity follows the navmesh shortest
        # path instead of the straight line to the goal (lets ORCA detour).
        self._rvo_navmesh_pref_vel: bool = bool(
            getattr(config, "rvo_navmesh_pref_vel", False)
        )
        # Lower bound on a controlled agent's solved speed (fraction of its
        # max speed) while its preferred velocity is non-zero — guarantees the
        # ORCA solution can recover from a near-zero standoff.
        self._rvo_resume_speed_floor: float = float(
            getattr(config, "rvo_resume_speed_floor", 0.0)
        )
        # Previous base positions for finite-difference observed velocities.
        self._rvo_prev_pos: Dict[int, np.ndarray] = {}
        self._rvo_step_dt: Optional[float] = None
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
        print(
            f"[RVO][init] auto_enable={self._rvo_auto_enable} "
            f"static_map={self._rvo_static_map_enabled} "
            f"controlled={self._rvo_default_controlled} "
            f"radius={self._rvo_radius} max_speed={self._rvo_max_speed}"
        )

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
        if self._rvo_auto_enable:
            self.enable_rvo(self._rvo_default_controlled, episode=episode)

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
        # Reset the finite-difference state for observed velocities and pin the
        # per-env-step dt (ORCA drive runs once per env step, not per sim tick).
        self._rvo_prev_pos = {}
        try:
            _cf = float(self._sim.habitat_config.ctrl_freq)
            _afr = float(self._sim.habitat_config.ac_freq_ratio)
            self._rvo_step_dt = _afr / _cf if _cf > 0 else 1.0
        except Exception:
            self._rvo_step_dt = 1.0
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

    def _rvo_drive_action(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """Solve ORCA at the agents' current poses and rewrite ``action`` so each
        controlled agent's ``oracle_nav_action`` carries a 1 s-ahead waypoint
        (``base_pos + orca_vel * RVO_WAYPOINT_LOOKAHEAD``). The waypoint distance
        therefore equals the desired speed, which ``OracleNavWaypointAction``
        reads back as its forward speed. Other action components (e.g.
        ``pddl_apply_action``) are preserved; uncontrolled agents stay in ORCA as
        dynamic obstacles at their current positions.
        """
        mgr = self._rvo_manager
        n = len(self._rvo_agent_keys)

        # Current (pre-step) base positions for every registered agent.
        cur_pos: Dict[int, np.ndarray] = {
            aid: np.asarray(
                self._sim.get_agent_data(aid).articulated_agent.base_pos,
                dtype=float,
            )
            for aid in range(n)
        }

        # Observed (finite-difference) XZ velocity per agent, used to feed
        # uncontrolled agents into ORCA non-reciprocally (firm human).
        dt = self._rvo_step_dt or 1.0
        observed_vel: Dict[int, Tuple[float, float]] = {}
        for aid in range(n):
            prev = self._rvo_prev_pos.get(aid)
            if prev is None:
                observed_vel[aid] = (0.0, 0.0)
            else:
                observed_vel[aid] = (
                    float((cur_pos[aid][0] - prev[0]) / dt),
                    float((cur_pos[aid][2] - prev[2]) / dt),
                )
            self._rvo_prev_pos[aid] = cur_pos[aid].copy()

        # 1) Sync ORCA with each agent's current position. Uncontrolled agents
        #    are warm-started with their OBSERVED velocity when reciprocity is
        #    reduced, so ORCA does not assume they will cooperatively avoid.
        for aid, key in enumerate(self._rvo_agent_keys):
            pos3 = cur_pos[aid]
            if aid in self._rvo_controlled_agents:
                warm = self._rvo_last_vel.get(key, (0.0, 0.0))
            else:
                warm = (
                    tuple(
                        self._rvo_reciprocity * c
                        + (1.0 - self._rvo_reciprocity) * o
                        for c, o in zip(
                            self._rvo_last_vel.get(key, (0.0, 0.0)),
                            observed_vel[aid],
                        )
                    )
                )
            mgr.sync_agent_pose(
                key,
                (float(pos3[0]), float(pos3[2])),
                warm,
            )

        # 2) Preferred velocity toward each agent's nav goal (0 within stop
        #    radius). Controlled agents optionally follow the navmesh next
        #    waypoint (detour capability); uncontrolled agents blend
        #    goal-seeking with observed motion by the reciprocity factor.
        pref_dir: Dict[int, Tuple[float, float]] = {}
        for aid, key in enumerate(self._rvo_agent_keys):
            pos3 = cur_pos[aid]
            goal_xz = self._rvo_goal_xz_for_agent(aid)
            override_ms = self._rvo_agent_max_speed.get(aid)
            max_speed = (
                float(override_ms)
                if override_ms is not None
                else self._rvo_max_speed
            )
            pref = (0.0, 0.0)
            if goal_xz is not None:
                target_xz = goal_xz
                # Controlled agents route via the navmesh: aim at the next
                # path waypoint rather than straight at the goal.
                if (
                    self._rvo_navmesh_pref_vel
                    and aid in self._rvo_controlled_agents
                ):
                    path_xz, status = self._navmesh_path_xz(
                        cur_pos[aid],
                        np.array(
                            [goal_xz[0], float(pos3[1]), goal_xz[1]],
                            dtype=np.float32,
                        ),
                    )
                    if status == "ok" and path_xz is not None:
                        nxt = path_xz[1]
                        if (
                            np.hypot(nxt[0] - pos3[0], nxt[1] - pos3[2])
                            < self._rvo_goal_stop_radius
                            and len(path_xz) > 2
                        ):
                            nxt = path_xz[2]
                        target_xz = nxt
                dx = target_xz[0] - float(pos3[0])
                dz = target_xz[1] - float(pos3[2])
                goal_dist = float(np.hypot(dx, dz))
                # Stop only on the TRUE goal proximity, not an intermediate wp.
                true_dist = float(
                    np.hypot(
                        goal_xz[0] - float(pos3[0]),
                        goal_xz[1] - float(pos3[2]),
                    )
                )
                if true_dist >= self._rvo_goal_stop_radius and goal_dist > 1e-6:
                    inv = max_speed / goal_dist
                    pref = (dx * inv, dz * inv)
            # Uncontrolled agents: blend the goal-seeking pref with observed
            # motion so a non-reciprocal robot is modeled by what it does.
            if aid not in self._rvo_controlled_agents:
                pref = (
                    self._rvo_reciprocity * pref[0]
                    + (1.0 - self._rvo_reciprocity) * observed_vel[aid][0],
                    self._rvo_reciprocity * pref[1]
                    + (1.0 - self._rvo_reciprocity) * observed_vel[aid][1],
                )
            pref_dir[aid] = pref
            mgr.set_pref_velocity(key, pref)

        # 3) Advance ORCA and cache the solved velocities. Apply the resume
        #    speed floor so a controlled agent whose preferred velocity is
        #    non-zero cannot be pinned to a near-zero solution (deadlock).
        mgr.step()
        for aid, key in enumerate(self._rvo_agent_keys):
            v = mgr.get_agent_velocity(key)
            vx, vz = float(v[0]), float(v[1])
            if (
                self._rvo_resume_speed_floor > 0.0
                and aid in self._rvo_controlled_agents
            ):
                pdx, pdz = pref_dir.get(aid, (0.0, 0.0))
                pmag = float(np.hypot(pdx, pdz))
                if pmag > 1e-6:
                    override_ms = self._rvo_agent_max_speed.get(aid)
                    max_speed = (
                        float(override_ms)
                        if override_ms is not None
                        else self._rvo_max_speed
                    )
                    floor = self._rvo_resume_speed_floor * max_speed
                    if float(np.hypot(vx, vz)) < floor:
                        vx = floor * pdx / pmag
                        vz = floor * pdz / pmag
            self._rvo_last_vel[key] = (vx, vz)

        # 4) Project a 1 s-ahead waypoint into each controlled agent's
        #    oracle_nav_action. RVO is the sole driver of controlled agents, so
        #    drop their *_base_velocity: when oracle_nav stops (agent at/near its
        #    waypoint) it sets skill_done, the HL policy churns and may emit a
        #    non-zero base_velocity that would otherwise push the agent past ORCA.
        raw_names = (
            list(action["action"])
            if isinstance(action.get("action"), (tuple, list))
            else ([action["action"]] if action.get("action") else [])
        )
        drop = {f"agent_{aid}_base_velocity" for aid in self._rvo_controlled_agents}
        names = [nm for nm in raw_names if nm not in drop]
        args = dict(action.get("action_args") or {})
        for aid in self._rvo_controlled_agents:
            if aid >= n:
                continue
            vx, vz = self._rvo_last_vel[self._rvo_agent_keys[aid]]
            p = cur_pos[aid]
            waypoint = np.array(
                [
                    float(p[0]) + vx * RVO_WAYPOINT_LOOKAHEAD,
                    float(p[1]),
                    float(p[2]) + vz * RVO_WAYPOINT_LOOKAHEAD,
                ],
                dtype=np.float32,
            )
            nav = f"agent_{aid}_oracle_nav_action"
            args[nav] = waypoint
            if nav not in names:
                names.append(nav)
            if (
                self._rvo_debug_save_obstacle_figure
                and self._rvo_debug_episode_id is not None
            ):
                self._rvo_debug_step_trail.setdefault(
                    self._rvo_agent_keys[aid], []
                ).append((float(waypoint[0]), float(waypoint[2])))

        return {"action": tuple(names), "action_args": args}

    def step(self, action: Dict[str, Any], episode: Episode):
        """Solve ORCA at the current poses and rewrite ``action`` so each
        controlled agent's oracle-nav action receives a 1 s-ahead waypoint; the
        action then walks the agent there (turn-then-go / walk animation). No
        teleport: locomotion and rotation are handled by the action + step_filter.
        """
        if self._rvo_enabled and self._rvo_manager is not None:
            try:
                action = self._rvo_drive_action(action)
            except Exception as e:  # pragma: no cover - defensive
                import traceback

                traceback.print_exc()
                print(f"[RVO] waypoint injection failed: {e}")
        return super().step(action=action, episode=episode)