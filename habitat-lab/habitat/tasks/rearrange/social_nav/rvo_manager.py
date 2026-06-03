#!/usr/bin/env python3
"""ORCA (RVO2) manager for social-nav.

**Construction (two modes)**

1. **Trajectory mode** (same as ``examples/get_trajectory_rvo.py``): pass ``my_env`` with
   ``initial_state`` rows ``[x, y, vx, vy, gx, gy]``. Use ``get_velocity``,
   ``get_future_position``, ``load_obs_from_map``, ``load_obstacles``, ``plot_obstacles``.
2. **Keyed mode** (for :class:`NavToObjSocialTask` / :class:`TwoAgentSocialNavTask`): omit ``my_env``; call
   ``add_agent("agent_0", (x, z))``, ``sync_agent_pose``, ``set_pref_velocity``, ``step``, ``get_agent_velocity``.
   For **walls in ORCA velocity space**, build polygons with :func:`static_obstacles_from_sim_topdown` and pass
   ``static_obstacles=...`` into ``RVOManager`` (before ``add_agent``). Habitat ``step_filter`` remains a safety net.

Requires: ``rvo2`` (Python-RVO2). Optional: ``toml``, ``PIL``, ``matplotlib``, ``tf``,
``habitat.utils.visualizations.maps``.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import rvo2

try:
    import toml
except ImportError:
    toml = None  # type: ignore

ObstaclePolygon = Sequence[Tuple[float, float]]


def static_obstacles_from_sim_topdown(
    sim: Any,
    map_resolution: int = 512,
    meters_per_pixel: Optional[float] = None,
    agent_id: int = 0,
    max_polys: Optional[int] = 20000,
    verbose: bool = True,
) -> List[ObstaclePolygon]:
    """
    Build axis-aligned obstacle quads from the navmesh top-down occupancy map.

    Non-walkable cells are merged per row via run-length encoding so we emit
    one quad per contiguous horizontal bar instead of one per cell. On HSSD at
    ``map_resolution=512`` this typically cuts obstacle count from ~200k to
    ~5–20k, which is the difference between RVO stalling for minutes and
    building in under a second.

    World (x, z) matches Habitat; RVO uses (x, y) = (x, z).

    Parameters
    ----------
    sim
        Simulator with a valid ``pathfinder`` (e.g. RearrangeSim).
    map_resolution
        Longest side of the rasterized map (see :py:`maps.get_topdown_map`).
    meters_per_pixel
        Optional override; if None, derived from ``map_resolution``.
    agent_id
        Which agent supplies floor height for the top-down slice.
    max_polys
        If the RLE output still exceeds this count, the resolution is
        progressively halved until it fits (or 64 is reached). Set to
        ``None`` to disable the safeguard. Default 20000.
    verbose
        Print a one-line summary with the final polygon count.
    """
    from habitat.utils.visualizations import maps

    if sim is None or getattr(sim, "pathfinder", None) is None:
        return []
    pathfinder = sim.pathfinder
    if not getattr(pathfinder, "is_loaded", True):
        return []
    try:
        lower, upper = pathfinder.get_bounds()
    except Exception:
        return []

    # world_x spans upper[0]-lower[0], world_z spans upper[2]-lower[2].
    world_x_min, world_x_max = float(lower[0]), float(upper[0])
    world_z_min, world_z_max = float(lower[2]), float(upper[2])

    def _rasterize(res: int) -> Tuple[Optional[np.ndarray], int, int]:
        try:
            td = maps.get_topdown_map_from_sim(
                sim,
                map_resolution=res,
                draw_border=False,
                meters_per_pixel=meters_per_pixel,
                agent_id=agent_id,
            )
        except Exception:
            return None, 0, 0
        return td, td.shape[0], td.shape[1]

    res = int(map_resolution)
    obstacles: List[ObstaclePolygon] = []

    while res >= 64:
        td, height, width = _rasterize(res)
        if td is None or height < 2 or width < 2:
            return []

        # grid size is (height rows, width cols). `from_grid` maps row i ->
        # world_z and col j -> world_x. We do the same via a direct linear
        # interpolation from pathfinder bounds so we avoid ~HxW Python calls.
        row_to_z = np.linspace(world_z_min, world_z_max, height, dtype=np.float64)
        col_to_x = np.linspace(world_x_min, world_x_max, width, dtype=np.float64)

        non_walk = td != maps.MAP_VALID_POINT
        obstacles = []
        # Row-wise run-length encoding: emit one polygon per contiguous
        # non-walkable run [j0, j1] in each row [i, i+1].
        for i in range(height - 1):
            row = non_walk[i]
            if not row.any():
                continue
            diff = np.diff(row.astype(np.int8))
            starts = np.where(diff == 1)[0] + 1
            ends = np.where(diff == -1)[0] + 1
            if row[0]:
                starts = np.concatenate(([0], starts))
            if row[-1]:
                ends = np.concatenate((ends, [len(row)]))
            z0 = row_to_z[i]
            z1 = row_to_z[i + 1]
            for s, e in zip(starts, ends):
                if e <= s:
                    continue
                # e is exclusive; clamp to valid index for the right x edge.
                j_right = min(e, width - 1)
                x0 = col_to_x[s]
                x1 = col_to_x[j_right]
                obstacles.append(
                    (
                        (x0, z0),
                        (x1, z0),
                        (x1, z1),
                        (x0, z1),
                    )
                )

        if max_polys is None or len(obstacles) <= max_polys:
            break
        if verbose:
            print(
                f"[RVO][obstacles] {len(obstacles)} polys at res={res} "
                f"exceeds max_polys={max_polys}; halving resolution"
            )
        res //= 2

    if verbose:
        print(
            f"[RVO][obstacles] rasterized {len(obstacles)} obstacle polys "
            f"at res={res} (bounds x=[{world_x_min:.2f},{world_x_max:.2f}] "
            f"z=[{world_z_min:.2f},{world_z_max:.2f}])"
        )
    return obstacles


def my_ceil(a, precision=2):
    return np.true_divide(np.ceil(a * 10**precision), 10**precision)


def my_floor(a, precision=2):
    return np.true_divide(np.floor(a * 10**precision), 10**precision)


def _point_in_polygon(x: float, y: float, poly: np.ndarray) -> bool:
    """Ray-casting point-in-polygon test. ``poly`` shape is (N, 2)."""
    n = len(poly)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        if ((yi > y) != (yj > y)) and (
            x < (xj - xi) * (y - yi) / ((yj - yi) + 1e-12) + xi
        ):
            inside = not inside
        j = i
    return inside


def plan_path_astar(
    start: Tuple[float, float],
    goal: Tuple[float, float],
    obstacles: Sequence[ObstaclePolygon],
    agent_radius: float,
    cell: float = 0.1,
    bounds_pad: float = 2.0,
) -> Optional[List[Tuple[float, float]]]:
    """Grid A* path around inflated static obstacle polygons.

    ORCA does no global planning; when the straight line to the goal is
    blocked by a symmetric obstacle, the preferred velocity is balanced and
    ORCA slows to zero instead of detouring. This planner produces waypoints
    that break that symmetry.

    Returns a line-of-sight smoothed list of ``(x, y)`` waypoints ending at
    ``goal`` (the first waypoint is the exact ``start``), or ``None`` if no
    feasible path was found.
    """
    import heapq
    from collections import deque

    start = (float(start[0]), float(start[1]))
    goal = (float(goal[0]), float(goal[1]))

    xs: List[float] = [start[0], goal[0]]
    ys: List[float] = [start[1], goal[1]]
    for poly in obstacles:
        for px, py in poly:
            xs.append(float(px))
            ys.append(float(py))
    xmin = min(xs) - bounds_pad
    xmax = max(xs) + bounds_pad
    ymin = min(ys) - bounds_pad
    ymax = max(ys) + bounds_pad

    nx = int(np.ceil((xmax - xmin) / cell)) + 1
    ny = int(np.ceil((ymax - ymin) / cell)) + 1
    occ = np.zeros((nx, ny), dtype=bool)

    for poly in obstacles:
        arr = np.asarray(poly, dtype=float)
        pxmin, pymin = arr.min(axis=0)
        pxmax, pymax = arr.max(axis=0)
        ilo = max(0, int(np.floor((pxmin - xmin) / cell)) - 1)
        ihi = min(nx, int(np.ceil((pxmax - xmin) / cell)) + 2)
        jlo = max(0, int(np.floor((pymin - ymin) / cell)) - 1)
        jhi = min(ny, int(np.ceil((pymax - ymin) / cell)) + 2)
        for ii in range(ilo, ihi):
            x_w = xmin + ii * cell
            for jj in range(jlo, jhi):
                if occ[ii, jj]:
                    continue
                if _point_in_polygon(x_w, ymin + jj * cell, arr):
                    occ[ii, jj] = True

    inflate = int(np.ceil(agent_radius / cell))
    for _ in range(inflate):
        nxt = occ.copy()
        nxt[1:] |= occ[:-1]
        nxt[:-1] |= occ[1:]
        nxt[:, 1:] |= occ[:, :-1]
        nxt[:, :-1] |= occ[:, 1:]
        occ = nxt

    def to_cell(p: Tuple[float, float]) -> Tuple[int, int]:
        return (
            min(max(int(round((p[0] - xmin) / cell)), 0), nx - 1),
            min(max(int(round((p[1] - ymin) / cell)), 0), ny - 1),
        )

    def to_world(c: Tuple[int, int]) -> Tuple[float, float]:
        return (xmin + c[0] * cell, ymin + c[1] * cell)

    def nearest_free(c: Tuple[int, int]) -> Optional[Tuple[int, int]]:
        if not occ[c]:
            return c
        seen = {c}
        q = deque([c])
        while q:
            cur = q.popleft()
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    if dx == 0 and dy == 0:
                        continue
                    nc = (cur[0] + dx, cur[1] + dy)
                    if nc in seen:
                        continue
                    if not (0 <= nc[0] < nx and 0 <= nc[1] < ny):
                        continue
                    seen.add(nc)
                    if not occ[nc]:
                        return nc
                    q.append(nc)
        return None

    s = nearest_free(to_cell(start))
    g = nearest_free(to_cell(goal))
    if s is None or g is None:
        return None

    def h(c: Tuple[int, int]) -> float:
        return float(np.hypot(c[0] - g[0], c[1] - g[1]))

    openh: List[Tuple[float, float, Tuple[int, int]]] = [(h(s), 0.0, s)]
    gscore: Dict[Tuple[int, int], float] = {s: 0.0}
    came: Dict[Tuple[int, int], Optional[Tuple[int, int]]] = {s: None}
    sqrt2 = float(np.sqrt(2.0))
    while openh:
        _f, gg, c = heapq.heappop(openh)
        if c == g:
            break
        if gg > gscore[c]:
            continue
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                nc = (c[0] + dx, c[1] + dy)
                if not (0 <= nc[0] < nx and 0 <= nc[1] < ny):
                    continue
                if occ[nc]:
                    continue
                if dx != 0 and dy != 0:
                    if occ[c[0] + dx, c[1]] or occ[c[0], c[1] + dy]:
                        continue
                step = sqrt2 if (dx != 0 and dy != 0) else 1.0
                ng = gg + step
                if ng < gscore.get(nc, float("inf")):
                    gscore[nc] = ng
                    came[nc] = c
                    heapq.heappush(openh, (ng + h(nc), ng, nc))

    if g not in came:
        return None

    raw: List[Tuple[float, float]] = []
    cur: Optional[Tuple[int, int]] = g
    while cur is not None:
        raw.append(to_world(cur))
        cur = came[cur]
    raw.reverse()
    raw[0] = start
    raw[-1] = goal

    def line_of_sight(a: Tuple[float, float], b: Tuple[float, float]) -> bool:
        d = float(np.hypot(b[0] - a[0], b[1] - a[1]))
        n_samples = max(2, int(d / (cell * 0.5)) + 1)
        for t in np.linspace(0.0, 1.0, n_samples):
            px = a[0] + (b[0] - a[0]) * t
            py = a[1] + (b[1] - a[1]) * t
            ci = int(round((px - xmin) / cell))
            cj = int(round((py - ymin) / cell))
            if not (0 <= ci < nx and 0 <= cj < ny):
                return False
            if occ[ci, cj]:
                return False
        return True

    smoothed: List[Tuple[float, float]] = [raw[0]]
    i = 0
    while i < len(raw) - 1:
        j = len(raw) - 1
        while j > i + 1 and not line_of_sight(raw[i], raw[j]):
            j -= 1
        smoothed.append(raw[j])
        i = j
    return smoothed


class RVOManager:
    """ORCA simulator wrapper: trajectory-style (``get_trajectory_rvo``) or keyed agents."""

    def __init__(
        self,
        time_step: float = 1.0 / 60.0,
        neighbor_dist: float = 1.5,
        max_neighbors: int = 10,
        time_horizon: float = 1.5,
        time_horizon_obst: float = 6.0,
        radius: float = 0.4,
        default_max_speed: float = 1.0,
        static_obstacles: Optional[List[ObstaclePolygon]] = None,
        *,
        my_env: Any = None,
        map_path: str = "",
        config_file: Optional[str] = None,
        resolution: float = 0.025,
        enable_plotting: bool = True,
    ):
        """
        Parameters
        ----------
        my_env
            If set, **trajectory mode**: must provide ``grid_dimensions``,
            ``control_frequency``, ``initial_state`` (list of ``[x,y,vx,vy,gx,gy]``).
        static_obstacles
            Optional list of polygons ``[(x,y), ...]`` (closed implicitly by RVO2).
            Used in **both** modes; in trajectory mode they are inserted **before**
            agents are added (same as keyed mode).
        enable_plotting
            If True (default), trajectory mode creates matplotlib ``fig/ax`` for
            ``get_velocity`` / ``plot_obstacles``. Set False in headless training.
        """
        self._mode = "trajectory" if my_env is not None else "keys"
        self.obs: List[Any] = []
        self.enable_plotting = enable_plotting
        self.fig = None
        self.ax = None
        self.orca_ped: List[int] = []
        self._agents: Dict[str, Dict[str, Any]] = {}
        self.dt = time_step
        self.config: Dict[str, Any] = {}
        self.update_number = 0
        self.max_counter = 0
        self.agent_backed = False
        self.backup_pos = None
        self.orca_max_speed = default_max_speed
        self.orca_radius = radius
        # trajectory-mode A* cache; populated per-agent in get_velocity
        self.static_obstacles_polys: List[ObstaclePolygon] = list(
            static_obstacles or []
        )
        self._planned_paths: Dict[int, List[Tuple[float, float]]] = {}
        self._planned_goals: Dict[int, Tuple[float, float]] = {}
        self.use_path_planner: bool = True
        self.plan_cell: float = 0.1
        self.plan_inflate_margin: float = 0.05
        self.waypoint_advance_dist: float = 0.6

        if self._mode == "trajectory":
            self._init_trajectory_mode(
                my_env,
                map_path,
                config_file,
                resolution,
                time_step,
                static_obstacles,
            )
            return

        # --- keyed mode (NavToObjSocialTask) ---
        self.time_step = time_step
        self.neighbor_dist = neighbor_dist
        self.max_neighbors = max_neighbors
        self.time_horizon = time_horizon
        self.time_horizon_obst = time_horizon_obst
        self.default_radius = radius
        self.default_max_speed = default_max_speed

        self.orca_sim = rvo2.PyRVOSimulator(
            self.time_step,
            self.neighbor_dist,
            self.max_neighbors,
            self.time_horizon,
            self.time_horizon_obst,
            self.default_radius,
            self.default_max_speed,
        )
        if static_obstacles:
            for poly in static_obstacles:
                self.orca_sim.addObstacle(
                    [tuple(float(p[i]) for i in range(2)) for p in poly]
                )
            self.orca_sim.processObstacles()

    def _init_trajectory_mode(
        self,
        my_env: Any,
        map_path: str,
        config_file: Optional[str],
        resolution: float,
        time_step: float,
        static_obstacles: Optional[List[ObstaclePolygon]] = None,
    ) -> None:
        num_sqrt_meter = np.sqrt(
            my_env.grid_dimensions[0] * my_env.grid_dimensions[1] * 0.025 * 0.025
        )
        if config_file is not None and Path(config_file).is_file() and toml is not None:
            self.config.update(toml.load(config_file))
        self.num_sqrt_meter_per_ped = self.config.get("num_sqrt_meter_per_ped", 8)
        self.num_pedestrians = max(1, int(num_sqrt_meter / self.num_sqrt_meter_per_ped))

        self.num_steps_stop = [0] * self.num_pedestrians
        self.neighbor_stop_radius = self.config.get("neighbor_stop_radius", 1.0)
        self.num_steps_stop_thresh = self.config.get("num_steps_stop_thresh", 20)
        self.backoff_radian_thresh = self.config.get(
            "backoff_radian_thresh", np.deg2rad(135.0)
        )

        self.neighbor_dist = self.config.get("orca_neighbor_dist", 2)
        self.max_neighbors = self.num_pedestrians
        self.time_horizon = self.config.get("orca_time_horizon", 4.0)
        self.time_horizon_obst = self.config.get("orca_time_horizon_obst", 4.0)
        self.orca_radius = self.config.get("orca_radius", 0.40)
        self.orca_max_speed = self.config.get("orca_max_speed", 1.3)
        self.dt = 1.0 / (my_env.control_frequency)
        self.dt = 1.0 / 12.0
        self.backup_pos = None

        self.orca_sim = rvo2.PyRVOSimulator(
            self.dt,
            self.neighbor_dist,
            self.max_neighbors,
            self.time_horizon,
            self.time_horizon_obst,
            self.orca_radius,
            self.orca_max_speed,
        )

        if static_obstacles:
            for poly in static_obstacles:
                self.orca_sim.addObstacle(
                    [tuple(float(p[i]) for i in range(2)) for p in poly]
                )
            self.orca_sim.processObstacles()

        if self.enable_plotting:
            try:
                import matplotlib

                matplotlib.use("Agg")
                import matplotlib.pyplot as plt

                self.fig, self.ax = plt.subplots()
            except Exception:
                self.fig, self.ax = None, None
        else:
            self.fig, self.ax = None, None

        self.max_counter = int(3 / self.dt)
        self.update_number = 0
        initial_state = my_env.initial_state
        for i in range(len(initial_state)):
            if i == 0:
                self.orca_ped.append(
                    self.orca_sim.addAgent(
                        (initial_state[i][0], initial_state[i][1]),
                        velocity=(initial_state[i][2], initial_state[i][3]),
                        radius=0.5,
                    )
                )
            else:
                self.orca_ped.append(
                    self.orca_sim.addAgent(
                        (initial_state[i][0], initial_state[i][1]),
                        velocity=(initial_state[i][2], initial_state[i][3]),
                    )
                )
            desired_vel = np.array(
                [
                    initial_state[i][4] - initial_state[i][0],
                    initial_state[i][5] - initial_state[i][1],
                ]
            )
            dn = np.linalg.norm(desired_vel)
            if dn > 1e-9:
                desired_vel = desired_vel / dn * self.orca_max_speed
            else:
                desired_vel = np.array([0.0, 0.0])
            self.orca_sim.setAgentPrefVelocity(self.orca_ped[i], tuple(desired_vel))
        self.orca_sim.setAgentRadius(self.orca_ped[0], 0.5)
        self.agent_backed = False

    # --- keyed API (skip if trajectory-only) ---

    def add_agent(
        self,
        key: str,
        pos: Tuple[float, float],
        radius: Optional[float] = None,
        max_speed: Optional[float] = None,
        velocity: Tuple[float, float] = (0.0, 0.0),
    ) -> None:
        if self._mode != "keys":
            raise RuntimeError("add_agent is only for keyed RVOManager (omit my_env).")
        radius = radius if radius is not None else self.default_radius
        max_speed = max_speed if max_speed is not None else self.default_max_speed
        # Python-RVO2 binding: addAgent accepts either (pos,) alone or all 8
        # params (pos, neighborDist, maxNeighbors, timeHorizon, timeHorizonObst,
        # radius, maxSpeed, velocity). Passing 7 raises
        # ``ValueError: Either pass only 'pos', or pass all parameters.``
        agent_id = self.orca_sim.addAgent(
            (float(pos[0]), float(pos[1])),
            self.neighbor_dist,
            self.max_neighbors,
            self.time_horizon,
            self.time_horizon_obst,
            radius,
            max_speed,
            (float(velocity[0]), float(velocity[1])),
        )
        self._agents[key] = {
            "rvo_id": agent_id,
            "radius": radius,
            "max_speed": max_speed,
        }

    def set_pref_velocity(self, key: str, vel_world: Tuple[float, float]) -> None:
        if self._mode != "keys":
            return
        if key not in self._agents:
            return
        aid = self._agents[key]["rvo_id"]
        self.orca_sim.setAgentPrefVelocity(
            aid, (float(vel_world[0]), float(vel_world[1]))
        )

    def step(self) -> None:
        if self._mode != "keys":
            return
        self.orca_sim.doStep()

    def get_agent_velocity(self, key: str) -> np.ndarray:
        if self._mode != "keys":
            return np.array([0.0, 0.0], dtype=np.float32)
        if key not in self._agents:
            return np.array([0.0, 0.0], dtype=np.float32)
        aid = self._agents[key]["rvo_id"]
        vel = self.orca_sim.getAgentVelocity(aid)
        return np.array([vel[0], vel[1]], dtype=np.float32)

    def sync_agent_pose(
        self,
        key: str,
        pos_xz: Tuple[float, float],
        vel_xz: Tuple[float, float] = (0.0, 0.0),
    ) -> None:
        """Align keyed ORCA state with Habitat before ``set_pref_velocity`` / ``step``."""
        if self._mode != "keys":
            return
        if key not in self._agents:
            return
        aid = self._agents[key]["rvo_id"]
        self.orca_sim.setAgentPosition(
            aid, (float(pos_xz[0]), float(pos_xz[1]))
        )
        self.orca_sim.setAgentVelocity(
            aid, (float(vel_xz[0]), float(vel_xz[1]))
        )

    # --- trajectory API (get_trajectory_rvo parity) ---

    def load_obstacles(self, env, grid_dimensions: Tuple[int, int] = (40, 74)):
        if self._mode != "trajectory":
            raise RuntimeError("load_obstacles is only for trajectory RVOManager (pass my_env).")
        try:
            import tf as tf_ros
            from habitat.utils.visualizations import maps
        except ImportError as e:
            raise ImportError(
                "load_obstacles requires ROS ``tf`` and habitat ``maps`` (see get_trajectory_rvo.py)."
            ) from e

        sem_scene = env._sim.semantic_annotations()
        for obj in sem_scene.objects:
            if obj.category.name() in ["floor", "ceiling"]:
                continue

            center = obj.aabb.center
            x_len, _, z_len = obj.aabb.sizes / 2.0
            corners = [
                center + np.array([x, 0, z])
                for x, z in [
                    (-x_len, -z_len),
                    (-x_len, z_len),
                    (x_len, z_len),
                    (x_len, -z_len),
                    (-x_len, -z_len),
                ]
            ]
            quat = tf_ros.transformations.quaternion_inverse(obj.obb.rotation)
            trans = tf_ros.transformations.quaternion_matrix(quat)
            map_corners = [
                np.dot(trans, np.array([p[0], p[1], p[2], 1.0])) for p in corners
            ]
            map_corners = [
                maps.to_grid(p[2], p[0], grid_dimensions, sim=env._sim)
                for p in map_corners
            ]
            self.obs.append(
                [map_corners[0][0], map_corners[1][0], map_corners[0][1], map_corners[1][1]]
            )
            self.obs.append(
                [map_corners[1][0], map_corners[2][0], map_corners[1][1], map_corners[2][1]]
            )
            self.obs.append(
                [map_corners[2][0], map_corners[3][0], map_corners[2][1], map_corners[3][1]]
            )
            self.obs.append(
                [map_corners[3][0], map_corners[4][0], map_corners[3][1], map_corners[4][1]]
            )

    def load_obs_from_map(self, map_path: str, resolution: float) -> None:
        if self._mode != "trajectory":
            raise RuntimeError(
                "load_obs_from_map is only for trajectory RVOManager (pass my_env)."
            )
        from PIL import Image

        img = Image.open(map_path).convert("L")
        img_np = np.array(img)
        for i in np.arange(img_np.shape[0]):
            for j in np.arange(img_np.shape[1]):
                if img_np[i][j] == 0:
                    self.orca_sim.addObstacle(
                        [
                            tuple([my_floor(j * resolution), my_floor(i * resolution)]),
                            tuple([my_ceil(j * resolution), my_floor(i * resolution)]),
                            tuple([my_ceil(j * resolution), my_ceil(i * resolution)]),
                            tuple([my_floor(j * resolution), my_ceil(i * resolution)]),
                        ]
                    )
        print("building the obstacle tree")
        self.orca_sim.processObstacles()

    def _pick_waypoint(
        self,
        agent_idx: int,
        cur_pos: Tuple[float, float],
        goal_pos: Tuple[float, float],
    ) -> Tuple[float, float]:
        """Return the next A* waypoint on the route from ``cur_pos`` to ``goal_pos``.

        Falls back to ``goal_pos`` when path planning is disabled, there are no
        static obstacles, or the planner cannot find a route.
        """
        if not self.use_path_planner or not self.static_obstacles_polys:
            return goal_pos

        cached_goal = self._planned_goals.get(agent_idx)
        path = self._planned_paths.get(agent_idx)
        needs_plan = (
            path is None
            or cached_goal is None
            or np.hypot(cached_goal[0] - goal_pos[0], cached_goal[1] - goal_pos[1])
            > 1e-3
            or len(path) == 0
        )
        if needs_plan:
            try:
                radius = self.orca_sim.getAgentRadius(self.orca_ped[agent_idx])
            except Exception:
                radius = self.orca_radius
            planned = plan_path_astar(
                cur_pos,
                goal_pos,
                self.static_obstacles_polys,
                agent_radius=radius + self.plan_inflate_margin,
                cell=self.plan_cell,
            )
            if planned is None or len(planned) < 2:
                # No feasible path; let ORCA fall back to raw goal direction.
                self._planned_paths[agent_idx] = [goal_pos]
            else:
                # Drop the start waypoint; keep intermediate waypoints + goal.
                self._planned_paths[agent_idx] = list(planned[1:])
            self._planned_goals[agent_idx] = goal_pos
            path = self._planned_paths[agent_idx]

        # Advance the cursor: pop waypoints we are already close to.
        while len(path) > 1:
            wp = path[0]
            if np.hypot(wp[0] - cur_pos[0], wp[1] - cur_pos[1]) < self.waypoint_advance_dist:
                path.pop(0)
            else:
                break
        return path[0] if path else goal_pos

    def get_velocity(
        self,
        initial_state,
        current_heading=None,
        groups=None,
        filename=None,
        save_anim=False,
    ):
        if self._mode != "trajectory":
            raise RuntimeError("get_velocity is only for trajectory RVOManager (pass my_env).")

        for i in range(len(initial_state)):
            self.orca_sim.setAgentPosition(
                self.orca_ped[i], tuple(np.array([initial_state[i][0], initial_state[i][1]]))
            )
            self.orca_sim.setAgentVelocity(
                self.orca_ped[i], tuple(np.array([initial_state[i][2], initial_state[i][3]]))
            )
            cur_pos = (float(initial_state[i][0]), float(initial_state[i][1]))
            goal_pos = (float(initial_state[i][4]), float(initial_state[i][5]))
            goal_dist = float(np.hypot(goal_pos[0] - cur_pos[0], goal_pos[1] - cur_pos[1]))
            if goal_dist < 0.3:
                desired_vel = np.array([0.0, 0.0])
                print("Agent ", i, " has reached its goal")
            else:
                target = self._pick_waypoint(i, cur_pos, goal_pos)
                desired_vel = np.array(
                    [target[0] - cur_pos[0], target[1] - cur_pos[1]],
                    dtype=float,
                )
                gn = float(np.linalg.norm(desired_vel))
                if gn > 1e-9:
                    desired_vel = desired_vel / gn * self.orca_max_speed
                else:
                    desired_vel = np.array([0.0, 0.0])
            self.orca_sim.setAgentPrefVelocity(self.orca_ped[i], tuple(desired_vel))
        self.orca_sim.doStep()

        import matplotlib.pyplot as plt

        plotting = self.ax is not None and self.update_number < self.max_counter
        if plotting:
            colors = plt.cm.rainbow(np.linspace(0, 1, len(initial_state)))
            # Only compute alpha when we actually draw; max_counter can be huge.
            denom = max(1, self.max_counter)
            alpha_val = 0.5 + 0.5 * min(1.0, self.update_number / denom)
        computed_velocity = []
        actual_velocity = []
        for j in range(len(initial_state)):
            x, y = self.orca_sim.getAgentPosition(self.orca_ped[j])
            velx = (x - initial_state[j][0]) / self.dt
            vely = (y - initial_state[j][1]) / self.dt
            computed_velocity.append([velx, vely])
            velx, vely = self.orca_sim.getAgentVelocity(self.orca_ped[j])
            actual_velocity.append([velx, vely])
            if plotting:
                self.ax.plot(
                    x,
                    y,
                    "-o",
                    label=f"ped {j}",
                    markersize=2.5,
                    color=colors[j],
                    alpha=alpha_val,
                )
                self.ax.plot(
                    initial_state[j][4],
                    initial_state[j][5],
                    "-x",
                    label=f"ped {j}",
                    markersize=2.5,
                    color=colors[j],
                    alpha=alpha_val,
                )
            print("Initial state is ", initial_state[j])
            print("Point reaches in this step is ", [x, y])
        if self.fig is not None and self.update_number == self.max_counter:
            print("saving the offline plot!!")
            self.fig.savefig("save_stepwise_rvo2.png", dpi=300)
            plt.close(self.fig)
        self.update_number += 1

        print("Computed velocity by rvo2 is ", computed_velocity)
        print("Given velocity by rvo2 is ", actual_velocity)

        if save_anim and self.ax is not None:
            self.plot_obstacles()
            num_steps = 1000
            for _i in range(num_steps):
                self.orca_sim.doStep()
                colors = plt.cm.rainbow(np.linspace(0, 1, len(initial_state)))
                vel = []
                for j in range(len(initial_state)):
                    x, y = self.orca_sim.getAgentPosition(self.orca_ped[j])
                    velx = (x - initial_state[j][0]) / self.dt
                    vely = (y - initial_state[j][1]) / self.dt
                    initial_state[j][0] = x
                    initial_state[j][1] = y
                    vel.append([velx, vely])
                    print(velx, vely)
                    self.ax.plot(x, y, "-o", label=f"ped {j}", markersize=0.5, color=colors[j])
                if np.all(vel[0] == [0.0, 0.0] and vel[1] == [0.0, 0.0]):
                    break
            if filename:
                self.fig.savefig(filename + ".png", dpi=300)
            plt.close(self.fig)
        return np.array(computed_velocity)

    def reset_peds(self, initial_state) -> None:
        if self._mode != "trajectory":
            return
        for i in range(len(initial_state)):
            self.orca_sim.setAgentPosition(
                self.orca_ped[i], (initial_state[i][0], initial_state[i][1])
            )
            self.orca_sim.setAgentVelocity(
                self.orca_ped[i], (initial_state[i][2], initial_state[i][3])
            )
            desired_vel = np.array(
                [
                    initial_state[i][4] - initial_state[i][0],
                    initial_state[i][5] - initial_state[i][1],
                ]
            )
            if np.linalg.norm(desired_vel) > 0.2:
                desired_vel = desired_vel / np.linalg.norm(desired_vel) * self.orca_max_speed
                self.orca_sim.setAgentPrefVelocity(self.orca_ped[i], tuple(desired_vel))
            else:
                self.orca_sim.setAgentPrefVelocity(self.orca_ped[i], (0, 0))
            self.orca_sim.setAgentVelocity(self.orca_ped[i], tuple(desired_vel))
            print("Desired velocity for agent ", i, " is ", desired_vel)

    def get_future_position(self, initial_state, sampled_goals=None, num_steps=1000):
        if self._mode != "trajectory":
            raise RuntimeError(
                "get_future_position is only for trajectory RVOManager (pass my_env)."
            )

        import matplotlib.pyplot as plt

        print("Initial state is ", initial_state)
        colors = plt.cm.rainbow(np.linspace(0, 1, len(initial_state)))
        if self.ax is not None:
            self.ax.plot(
                initial_state[0][0],
                initial_state[0][1],
                "-o",
                label="ped 0",
                markersize=2.5,
                color=colors[0],
            )
            self.ax.plot(
                initial_state[0][4],
                initial_state[0][5],
                "-x",
                label="ped 0",
                markersize=2.5,
                color=colors[0],
            )
            self.ax.plot(
                initial_state[1][0],
                initial_state[1][1],
                "-o",
                label="ped 1",
                markersize=2.5,
                color=colors[1],
            )
            self.ax.plot(
                initial_state[1][4],
                initial_state[1][5],
                "-x",
                label="ped 1",
                markersize=2.5,
                color=colors[1],
            )
        position = np.zeros((2, 2))
        position[0] = self.orca_sim.getAgentPosition(self.orca_ped[0])
        position[1] = self.orca_sim.getAgentPosition(self.orca_ped[1])
        vels = []
        for i in range(num_steps):
            self.orca_sim.doStep()
            vel = []
            for j in range(len(initial_state)):
                x, y = self.orca_sim.getAgentPosition(self.orca_ped[j])
                velx = (x - initial_state[j][0]) / ((i + 1) * self.dt)
                vely = (y - initial_state[j][1]) / ((i + 1) * self.dt)
                vel.append([velx, vely])
                position[j] = [x, y]
                if self.ax is not None:
                    self.ax.plot(x, y, "-o", label=f"ped {j}", markersize=0.5, color=colors[j])
            vels.append(vel)
        self.update_number += 1
        print(
            "Velocities are ",
            self.orca_sim.getAgentVelocity(self.orca_ped[0]),
            self.orca_sim.getAgentVelocity(self.orca_ped[1]),
        )
        print(
            "Agent radius is ",
            self.orca_sim.getAgentRadius(self.orca_ped[0]),
            self.orca_sim.getAgentRadius(self.orca_ped[1]),
        )
        if np.linalg.norm(initial_state[0][4:6] - initial_state[0][0:2]) > 0.1:
            position[0] = self.orca_sim.getAgentPosition(self.orca_ped[0])
        if np.linalg.norm(initial_state[1][4:6] - initial_state[1][0:2]) > 0.1:
            position[1] = self.orca_sim.getAgentPosition(self.orca_ped[1])
        return np.array(position)

    def plot_obstacles(self) -> None:
        if self.ax is None:
            return
        import matplotlib.pyplot as plt

        self.fig.set_tight_layout(True)
        self.ax.grid(linestyle="dotted")
        self.ax.set_aspect("equal")
        self.ax.margins(2.0)
        self.ax.set_axisbelow(True)
        self.ax.set_xlabel("x [m]")
        self.ax.set_ylabel("y [m]")
        plt.rcParams["animation.html"] = "jshtml"

        obstacles = []
        pt = []
        if self.orca_sim.getNumObstacleVertices() < 100:
            for i in range(self.orca_sim.getNumObstacleVertices()):
                pt_0 = self.orca_sim.getObstacleVertex(i)
                pt_1 = self.orca_sim.getObstacleVertex(
                    self.orca_sim.getNextObstacleVertexNo(i)
                )
                pt.append(pt_0)
                pt.append(pt_1)
                pt = np.array(pt)
                self.ax.plot(pt[:, 0], pt[:, 1], "-o", color="black", markersize=0.1)
                pt = []
                obstacles.append(self.orca_sim.getObstacleVertex(i))
        else:
            for i in range(self.orca_sim.getNumObstacleVertices()):
                obstacles.append(self.orca_sim.getObstacleVertex(i))
            xy_limits = np.array(obstacles)
            self.ax.plot(xy_limits[:, 0], xy_limits[:, 1], "o", color="black", markersize=0.1)
        xy_limits = np.array(obstacles)
        xmin, ymin, xmax, ymax = 10000, 10000, -10000, -10000
        for obs in xy_limits:
            xmin = min(xmin, obs[0])
            xmax = max(xmax, obs[0])
            ymin = min(ymin, obs[1])
            ymax = max(ymax, obs[1])
        self.ax.set(xlim=(xmin - 2, xmax + 3), ylim=(ymin - 2, ymax + 3))

    def save_figure(self, path: Union[str, Path]) -> None:
        """Save current trajectory-mode figure (if ``enable_plotting`` and ``fig`` exist)."""
        if self.fig is None:
            return
        import matplotlib.pyplot as plt

        p = Path(path)
        self.fig.savefig(str(p), dpi=150)
        plt.close(self.fig)
        self.fig = None
        self.ax = None


def ped_rvo(
    my_env: Any,
    map_path: str = "",
    config_file: Optional[str] = None,
    resolution: float = 0.025,
    enable_plotting: bool = True,
    static_obstacles: Optional[List[ObstaclePolygon]] = None,
) -> RVOManager:
    """Same call pattern as ``examples/get_trajectory_rvo.ped_rvo``."""
    return RVOManager(
        my_env=my_env,
        map_path=map_path,
        config_file=config_file,
        resolution=resolution,
        enable_plotting=enable_plotting,
        static_obstacles=static_obstacles,
    )


def _dist_to_goal(row) -> float:
    """Distance from current position to goal (indices 4,5), same as ``get_trajectory_rvo``."""
    return float(np.hypot(row[4] - row[0], row[5] - row[1]))


def _all_within_goal(st, goal_radius: float) -> bool:
    return all(_dist_to_goal(row) < goal_radius for row in st)


def save_trajectory_initial_figure(
    initial_state: List[List[float]],
    static_obstacles: Optional[List[ObstaclePolygon]],
    out_path: Union[str, Path],
    *,
    agent_radii: Optional[List[float]] = None,
    title: str = "Initial: agents + goals + obstacles",
) -> None:
    """Save a one-off PNG: obstacle polygons, start positions, goals (``[x,y,vx,vy,gx,gy]`` per row)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    colors = ["C0", "C1", "C2", "C3"]

    if static_obstacles:
        for k, poly in enumerate(static_obstacles):
            arr = np.array(list(poly) + [poly[0]])
            ax.fill(
                arr[:, 0],
                arr[:, 1],
                color="0.85",
                edgecolor="black",
                linewidth=1.5,
                label="obstacle" if k == 0 else None,
            )

    n = len(initial_state)
    if agent_radii is None:
        agent_radii = [0.5] + [0.4] * max(0, n - 1)
    while len(agent_radii) < n:
        agent_radii.append(0.4)

    for i, row in enumerate(initial_state):
        x0, y0 = float(row[0]), float(row[1])
        gx, gy = float(row[4]), float(row[5])
        r = float(agent_radii[i])
        c = colors[i % len(colors)]
        circ = plt.Circle((x0, y0), r, fill=False, color=c, linewidth=2.0, label=f"agent {i} start")
        ax.add_patch(circ)
        ax.plot(x0, y0, "o", color=c, markersize=6)
        ax.plot(gx, gy, "x", color=c, markersize=10, markeredgewidth=2)
        ax.annotate(f"g{i}", (gx, gy), textcoords="offset points", xytext=(4, 4), fontsize=9, color=c)

    ax.set_aspect("equal")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y / z [m]")
    ax.set_title(title)
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc="upper right")
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=150)
    plt.close(fig)


def save_rvo_navmesh_obstacle_debug_figure(
    sim: Any,
    out_path: Union[str, Path],
    *,
    map_resolution: int = 512,
    meters_per_pixel: Optional[float] = None,
    agent_id: int = 0,
    static_obstacles: Optional[List[ObstaclePolygon]] = None,
    max_polygons_to_draw: int = 8000,
    title: str = "",
    agent_paths: Optional[Dict[str, Dict[str, Any]]] = None,
) -> None:
    """Save PNG: habitat navmesh top-down slice used by ``static_obstacles_from_sim_topdown``.

    Overlays faint red outlines for merged axis-aligned quads handed to ORCA ``addObstacle``.
    Optionally overlays navmesh shortest-path waypoints per agent (see ``agent_paths``).

    Polygon vertices are Habitat ``(x, z)`` with ``z`` plotted on the vertical axis.

    ``agent_paths`` maps keys like ``agent_0`` to dicts with:
    ``start`` ``goal`` ``path`` ``step_trail``.
    """
    from habitat.utils.visualizations import maps

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    if sim is None or getattr(sim, "pathfinder", None) is None:
        return
    pathfinder = sim.pathfinder
    if not getattr(pathfinder, "is_loaded", True):
        return
    try:
        lower, upper = pathfinder.get_bounds()
    except Exception:
        return

    x0, x1 = float(lower[0]), float(upper[0])
    z0, z1 = float(lower[2]), float(upper[2])

    try:
        td = maps.get_topdown_map_from_sim(
            sim,
            map_resolution=int(map_resolution),
            draw_border=False,
            meters_per_pixel=meters_per_pixel,
            agent_id=agent_id,
        )
        rgb = maps.colorize_topdown_map(td)
    except Exception:
        return

    fig, ax = plt.subplots(figsize=(11, 11))
    ax.imshow(
        rgb,
        extent=[x0, x1, z0, z1],
        origin="lower",
        aspect="equal",
        interpolation="nearest",
    )

    polys = static_obstacles or []
    drawn = 0
    for poly in polys:
        if drawn >= max_polygons_to_draw:
            break
        plist = list(poly)
        if len(plist) < 2:
            continue
        xs = [float(p[0]) for p in plist] + [float(plist[0][0])]
        zs = [float(p[1]) for p in plist] + [float(plist[0][1])]
        ax.plot(xs, zs, color="red", linewidth=0.2, alpha=0.35)
        drawn += 1

    path_colors = {"agent_0": "C0", "agent_1": "C1"}
    if agent_paths:
        for key, info in agent_paths.items():
            color = path_colors.get(key, "C2")
            start = info.get("start")
            goal = info.get("goal")
            path = info.get("path")
            la_idx = info.get("lookahead_idx")
            lookahead_point = info.get("lookahead_point")
            step_trail = info.get("step_trail") or []
            if start is not None:
                ax.plot(
                    start[0],
                    start[1],
                    "o",
                    color=color,
                    markersize=8,
                    markeredgecolor="black",
                    markeredgewidth=0.8,
                    label=f"{key} start",
                )
            if goal is not None:
                ax.plot(
                    goal[0],
                    goal[1],
                    "x",
                    color=color,
                    markersize=10,
                    markeredgewidth=2.0,
                    label=f"{key} goal",
                )
            if path and len(path) >= 2:
                pxs = [p[0] for p in path]
                pzs = [p[1] for p in path]
                ax.plot(
                    pxs,
                    pzs,
                    "-",
                    color=color,
                    linewidth=1.5,
                    alpha=0.85,
                    label=f"{key} path",
                )
                for i, (px, pz) in enumerate(path):
                    ax.plot(px, pz, ".", color=color, markersize=4)
                    ax.annotate(
                        str(i),
                        (px, pz),
                        textcoords="offset points",
                        xytext=(3, 3),
                        fontsize=7,
                        color=color,
                    )
                if step_trail and len(step_trail) >= 1:
                    txs = [float(p[0]) for p in step_trail]
                    tzs = [float(p[1]) for p in step_trail]
                    ax.plot(
                        txs,
                        tzs,
                        "-",
                        color=color,
                        linewidth=0.9,
                        alpha=0.55,
                        label=f"{key} step_trail",
                    )
                    ax.plot(
                        txs,
                        tzs,
                        ".",
                        color=color,
                        markersize=3,
                        alpha=0.7,
                    )
                if lookahead_point is not None:
                    lx, lz = float(lookahead_point[0]), float(lookahead_point[1])
                    ax.plot(
                        lx,
                        lz,
                        "*",
                        color=color,
                        markersize=14,
                        markeredgecolor="black",
                        markeredgewidth=0.6,
                        label=f"{key} lookahead",
                    )
                elif la_idx is not None and path and 0 <= la_idx < len(path):
                    lx, lz = path[la_idx]
                    ax.plot(
                        lx,
                        lz,
                        "*",
                        color=color,
                        markersize=14,
                        markeredgecolor="black",
                        markeredgewidth=0.6,
                        label=f"{key} lookahead",
                    )

    n_poly = len(polys)
    extra = (
        f" (red: {drawn}/{n_poly} polys)"
        if n_poly > max_polygons_to_draw
        else f" ({n_poly} polys)"
    )
    ax.set_xlabel("world x [m]")
    ax.set_ylabel("world z [m]")
    ax.set_title(
        title
        if title
        else (
            "RVO static obstacles: navmesh top-down + ORCA quads"
            + extra
        )
    )
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(str(out), dpi=160)
    plt.close(fig)


def _main() -> None:
    from types import SimpleNamespace

    import matplotlib

    matplotlib.use("Agg")

    # Same threshold as ``get_trajectory_rvo.get_velocity``: goal_dist < 0.3 -> pref vel 0
    goal_radius = 0.3
    max_steps = 2000

    initial_state = [
        [0.0, 0.0, 0.0, 0.0, 4.5, 0.0],
        [6.0, 0.0, 0.0, 0.0, 6.1, 3.0],
    ]
    # Thin vertical wall in the middle so agents must go around (world x–y plane = Habitat x–z).
    demo_wall: List[ObstaclePolygon] = [
        [
            (3.0, -1.0),
            (3.2, -1.0),
            (3.2, 1.0),
            (3.0, 1.0),
        ]
    ]
    my_env = SimpleNamespace(
        grid_dimensions=[100, 100],
        control_frequency=12.0,
        initial_state=initial_state,
    )
    mgr = RVOManager(
        my_env=my_env,
        map_path="",
        config_file=None,
        enable_plotting=True,
        static_obstacles=demo_wall,
    )
    # Demo: ORCA obstacle avoidance only — no grid A* waypoints (see _pick_waypoint).
    mgr.use_path_planner = False
    init_png = Path(__file__).resolve().parent / "rvo_manager_demo_initial.png"
    save_trajectory_initial_figure(initial_state, demo_wall, init_png)
    print("Wrote", init_png)

    # Long runs would hit ``max_counter`` (~3 s) and ``get_velocity`` would close ``fig``;
    # disable that for this demo so the final PNG still has the full trace.
    mgr.max_counter = 10**9
    st = [list(row) for row in initial_state]
    step = 0
    while step < max_steps:
        mgr.get_velocity(st)
        for j in range(len(st)):
            p = mgr.orca_sim.getAgentPosition(mgr.orca_ped[j])
            v = mgr.orca_sim.getAgentVelocity(mgr.orca_ped[j])
            st[j][0], st[j][1] = p[0], p[1]
            st[j][2], st[j][3] = v[0], v[1]
        step += 1
        if _all_within_goal(st, goal_radius):
            print(
                f"Stopped at step {step}: all agents within goal_radius={goal_radius} "
                f"(distances: {[round(_dist_to_goal(r), 4) for r in st]})"
            )
            break
    else:
        print(
            f"Reached max_steps={max_steps} without all agents within {goal_radius}; "
            f"distances: {[round(_dist_to_goal(r), 4) for r in st]}"
        )
    out = Path(__file__).resolve().parent / "rvo_manager_demo.png"
    if mgr.fig is not None and mgr.ax is not None:
        wall_xy = np.array(list(demo_wall[0]) + [demo_wall[0][0]])
        mgr.ax.plot(wall_xy[:, 0], wall_xy[:, 1], "k-", linewidth=2.0, label="obstacle")
        mgr.ax.set_aspect("equal")
        mgr.ax.set_xlim(-0.5, 6.5)
        mgr.ax.set_ylim(-2.5, 2.5)
        mgr.fig.savefig(str(out), dpi=150)
        print("Wrote", out)


if __name__ == "__main__":
    _main()
