import os
import os.path as osp
import random
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, List

import numpy as np
from omegaconf import OmegaConf

from habitat.core.logging import logger
from habitat.datasets.rearrange.rearrange_dataset import RearrangeDatasetV0
from habitat.datasets.rearrange.rearrange_generator import (
    RearrangeEpisodeGenerator,
)
from habitat.datasets.rearrange.samplers.receptacle import (
    get_all_scenedataset_receptacles,
)

if TYPE_CHECKING:
    from habitat.config import DictConfig
from habitat.datasets.rearrange.run_episode_generator import (
    SceneSamplerParamsConfig,
    SceneSamplerConfig,
    RearrangeEpisodeGeneratorConfig
)
import habitat_sim
from habitat.datasets.rearrange.navmesh_utils import (
    get_largest_island_index,
    path_is_navigable_given_robot,
)
from IPython import embed

import glob
import gzip
import json


if __name__ == "__main__":
    dataset = RearrangeDatasetV0()
    json_file_path = "/habitat-lab/data/scene_wise/test_dataset15_final.json"
    output_path = "/habitat-lab/data/scene_wise/test_dataset15_single_7.json.gz"
    with open(json_file_path, 'r') as j:
        contents = json.loads(j.read())
    remove_list = [18,17,15,12,14]
    for i in range(len(contents['episodes'])):
        if contents['episodes'][i]['episode_id'] in remove_list:
            continue
        if contents['episodes'][i]['episode_id'] ==7:
            dataset.episodes.append(contents['episodes'][i])
    
    with gzip.open(output_path, "wt") as f:
        f.write(dataset.to_json())