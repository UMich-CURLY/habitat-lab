#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import json
import os
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence

from habitat.core.registry import registry
from habitat.core.simulator import AgentState, ShortestPathPoint
from habitat.core.utils import DatasetFloatJSONEncoder
from habitat.datasets.pointnav.pointnav_dataset import (
    CONTENT_SCENES_PATH_FIELD,
    DEFAULT_SCENE_PATH_PREFIX,
    PointNavDatasetV1,
)
from habitat.tasks.nav.object_nav_task import (
    ObjectGoal,
    ObjectGoalNavEpisode,
    ObjectViewLocation,
)
from IPython import embed
if TYPE_CHECKING:
    from omegaconf import DictConfig
import numpy as np
from math import atan2, cos, sin
@registry.register_dataset(name="ObjectNav-v1")
class ObjectNavDatasetV1(PointNavDatasetV1):
    r"""Class inherited from PointNavDataset that loads Object Navigation dataset."""
    category_to_task_category_id: Dict[str, int]
    category_to_scene_annotation_category_id: Dict[str, int]
    episodes: List[ObjectGoalNavEpisode] = []  # type: ignore
    content_scenes_path: str = "{data_path}/content/{scene}.json.gz"
    goals_by_category: Dict[str, Sequence[ObjectGoal]]

    @staticmethod
    def dedup_goals(dataset: Dict[str, Any]) -> Dict[str, Any]:
        if len(dataset["episodes"]) == 0:
            return dataset

        goals_by_category = {}
        for i, ep in enumerate(dataset["episodes"]):
            dataset["episodes"][i]["object_category"] = ep["goals"][0][
                "object_category"
            ]
            ep = ObjectGoalNavEpisode(**ep)

            goals_key = ep.goals_key
            if goals_key not in goals_by_category:
                goals_by_category[goals_key] = ep.goals

            dataset["episodes"][i]["goals"] = []

        dataset["goals_by_category"] = goals_by_category

        return dataset

    def to_json(self) -> str:
        for i in range(len(self.episodes)):
            self.episodes[i].goals = []

        result = DatasetFloatJSONEncoder().encode(self)

        for i in range(len(self.episodes)):
            goals = self.goals_by_category[self.episodes[i].goals_key]
            if not isinstance(goals, list):
                goals = list(goals)
            self.episodes[i].goals = goals

        return result

    def __init__(self, config: Optional["DictConfig"] = None) -> None:
        self.goals_by_category = {}
        super().__init__(config)
        self.episodes = list(self.episodes)

    @staticmethod
    def __deserialize_goal(serialized_goal: Dict[str, Any]) -> ObjectGoal:
        g = ObjectGoal(**serialized_goal)

        for vidx, view in enumerate(g.view_points):
            view_location = ObjectViewLocation(**view)  # type: ignore
            view_location.agent_state = AgentState(**view_location.agent_state)  # type: ignore
            g.view_points[vidx] = view_location

        return g

    def from_json(
        self, json_str: str, scenes_dir: Optional[str] = None
    ) -> None:
        deserialized = json.loads(json_str)
        if CONTENT_SCENES_PATH_FIELD in deserialized:
            self.content_scenes_path = deserialized[CONTENT_SCENES_PATH_FIELD]

        if "category_to_task_category_id" in deserialized:
            self.category_to_task_category_id = deserialized[
                "category_to_task_category_id"
            ]

        if "category_to_scene_annotation_category_id" in deserialized:
            self.category_to_scene_annotation_category_id = deserialized[
                "category_to_scene_annotation_category_id"
            ]

        if "category_to_mp3d_category_id" in deserialized:
            self.category_to_scene_annotation_category_id = deserialized[
                "category_to_mp3d_category_id"
            ]

        assert len(self.category_to_task_category_id) == len(
            self.category_to_scene_annotation_category_id
        )

        assert set(self.category_to_task_category_id.keys()) == set(
            self.category_to_scene_annotation_category_id.keys()
        ), "category_to_task and category_to_mp3d must have the same keys"
        
        if len(deserialized["episodes"]) == 0:
            return
        
        if "goals_by_category" not in deserialized:
            deserialized = self.dedup_goals(deserialized)

        for k, v in deserialized["goals_by_category"].items():
            self.goals_by_category[k] = [self.__deserialize_goal(g) for g in v]
        
        for i, episode in enumerate(deserialized["episodes"]):
            episode = ObjectGoalNavEpisode(**episode)
            episode.episode_id = str(i)
            
            if scenes_dir is not None:
                if episode.scene_id.startswith(DEFAULT_SCENE_PATH_PREFIX):
                    episode.scene_id = episode.scene_id[
                        len(DEFAULT_SCENE_PATH_PREFIX) :
                    ]

                episode.scene_id = os.path.join(scenes_dir, episode.scene_id)
            try:
                episode.goals = self.goals_by_category[episode.goals_key]
                print("Valid goals key:", episode.goals_key)
                
                # if (episode.goals_key != "sT4fr6TAbpF.glb_tv_monitor"):
                #     continue
            except:
                print("Episode goals key:", episode.goals_key)
                continue
            # --- inputs ---
            ref_pos   = np.array(episode.reference_replay[0]['agent_state']['position'], dtype=np.float32)   # world
            start_pos = np.array(episode.start_position, dtype=np.float32)                                   # world

            # direction from start to reference, projected onto XZ (ignore vertical)
            d = ref_pos - start_pos
            d[1] = 0.0
            norm = np.linalg.norm(d)
            if norm < 1e-8:
                # Positions coincide (or nearly) -> keep existing rotation
                desired_xyzw = episode.start_rotation
            else:
                d /= norm  # unit heading in world: [dx, 0, dz]

                # We want R_yaw to send local forward (0,0,-1) to d = [dx, 0, dz]
                # Under yaw ψ about +Y, forward becomes (sinψ, 0, -cosψ),
                # so: dx = sinψ, dz = -cosψ  ->  ψ = atan2(dx, -dz)
                psi = atan2(d[0], -d[2])

                # Quaternion for rotation about +Y by ψ:
                # [w,x,y,z] = [cos(ψ/2), 0, sin(ψ/2), 0]
                w = cos(psi * 0.5)
                y = sin(psi * 0.5)

                # Habitat commonly stores quats as [x,y,z,w]
                desired_xyzw = [0.0, y, 0.0, w]

            # Apply to episode/agent as needed:
            episode.start_rotation = desired_xyzw
            # or if setting a live agent state: sim.agents[0].state.rotation = desired_xyzw
            if episode.shortest_paths is not None:
                for path in episode.shortest_paths:
                    for p_index, point in enumerate(path):
                        if point is None or isinstance(point, (int, str)):
                            point = {
                                "action": point,
                                "rotation": None,
                                "position": None,
                            }
                        try:
                            path[p_index] = ShortestPathPoint(**point)
                        except:
                            print("Shortest path issue, dont think it matters")
            
            self.episodes.append(episode)  # type: ignore [attr-defined]
            # break
