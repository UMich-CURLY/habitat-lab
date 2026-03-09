#!/usr/bin/env python3
"""Simple RVO/ORCA manager wrapper used by social-nav tasks.

This provides a minimal API around Python-RVO2 if available and
falls back to a stub that returns preferred velocities unmodified when
`rvo2` is not installed.

API:
  RVOManager(sim=None, time_step=1/60., neighbor_dist=1.5, max_neighbors=10,
             time_horizon=1.5, time_horizon_obst=2, radius=0.4)
  add_agent(agent_key, pos, radius=None, max_speed=None)
  set_pref_velocity(agent_key, vel_world)
  step()
  get_agent_velocity(agent_key) -> np.array([vx, vy])

Note: This manager runs in the calling thread. Do not call from multiple
threads without synchronization; Python-RVO2 bindings are not guaranteed
thread-safe.
"""
from typing import Dict, Optional, Tuple
import numpy as np

try:
    import rvo2
    _HAS_RVO2 = True
except Exception:
    rvo2 = None  # type: ignore
    _HAS_RVO2 = False


class RVOManager:
    def __init__(
        self,
        time_step: float = 1.0 / 60.0,
        neighbor_dist: float = 1.5,
        max_neighbors: int = 10,
        time_horizon: float = 1.5,
        time_horizon_obst: float = 2.0,
        radius: float = 0.4,
        default_max_speed: float = 1.0,
    ):
        self.time_step = time_step
        self.neighbor_dist = neighbor_dist
        self.max_neighbors = max_neighbors
        self.time_horizon = time_horizon
        self.time_horizon_obst = time_horizon_obst
        self.default_radius = radius
        self.default_max_speed = default_max_speed

        # Mapping from user keys to rvo agent ids (or to simple data in stub)
        self._agents: Dict[str, Dict] = {}

        if _HAS_RVO2:
            # create the PyRVOSimulator
            self._sim = rvo2.PyRVOSimulator(
                self.time_step,
                self.neighbor_dist,
                self.max_neighbors,
                self.time_horizon,
                self.time_horizon_obst,
                self.default_radius,
                self.default_max_speed,
            )
        else:
            self._sim = None
        import pdb; pdb.set_trace()
        print(f"RVOManager initialized with RVO2: {_HAS_RVO2}")

    def add_agent(self, key: str, pos: Tuple[float, float], radius: Optional[float] = None, max_speed: Optional[float] = None):
        radius = radius if radius is not None else self.default_radius
        max_speed = max_speed if max_speed is not None else self.default_max_speed
        if _HAS_RVO2 and self._sim is not None:
            # rvo2 expects 2D position tuple
            agent_id = self._sim.addAgent(
                (pos[0], pos[1]),
                self.neighbor_dist,
                self.max_neighbors,
                self.time_horizon,
                self.time_horizon_obst,
                radius,
                max_speed,
            )
            self._agents[key] = {
                "rvo_id": agent_id,
                "radius": radius,
                "max_speed": max_speed,
            }
        else:
            # stub: store the state and echo back pref vel
            self._agents[key] = {
                "pos": np.array(pos, dtype=np.float32),
                "radius": radius,
                "max_speed": max_speed,
                "vel": np.array([0.0, 0.0], dtype=np.float32),
                "pref_vel": np.array([0.0, 0.0], dtype=np.float32),
            }

    def set_pref_velocity(self, key: str, vel_world: Tuple[float, float]):
        if key not in self._agents:
            return
        if _HAS_RVO2 and self._sim is not None:
            aid = self._agents[key]["rvo_id"]
            self._sim.setAgentPrefVelocity(aid, (vel_world[0], vel_world[1]))
        else:
            self._agents[key]["pref_vel"] = np.array(vel_world, dtype=np.float32)

    def step(self):
        if _HAS_RVO2 and self._sim is not None:
            self._sim.doStep()
        else:
            # simple stub: assign pref_vel directly (no collision avoidance)
            for k, v in self._agents.items():
                v["vel"] = v.get("pref_vel", np.array([0.0, 0.0], dtype=np.float32))

    def get_agent_velocity(self, key: str) -> np.ndarray:
        if key not in self._agents:
            return np.array([0.0, 0.0], dtype=np.float32)
        if _HAS_RVO2 and self._sim is not None:
            aid = self._agents[key]["rvo_id"]
            vel = self._sim.getAgentVelocity(aid)
            return np.array([vel[0], vel[1]], dtype=np.float32)
        else:
            return self._agents[key].get("vel", np.array([0.0, 0.0], dtype=np.float32))

    def set_static_obstacles(self, obstacles):
        """Optional: accept a list of polygon obstacles. Not implemented in stub.

        obstacles: list of list of (x,y)
        """
        if not _HAS_RVO2 or self._sim is None:
            return
        # Clear current obstacles and add new ones
        # Python-RVO2 can add obstacles via addObstacle, finishObstacles
        # We'll add polygons provided as lists of (x,y)
        # Note: this will be destructive to previously added obstacles
        try:
            # No API to clear obstacles in PyRVOSimulator; this is a best-effort add-only.
            for poly in obstacles:
                self._sim.addObstacle([tuple(p) for p in poly])
            self._sim.processObstacles()
        except Exception:
            # best-effort: ignore obstacle setup errors
            pass
