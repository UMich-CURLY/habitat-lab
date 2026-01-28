#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import inspect
import os.path as osp
from typing import Optional

from omegaconf import DictConfig, OmegaConf
from omegaconf import open_dict

from habitat.config.default import get_config as _habitat_get_config
from habitat.config.default_structured_configs import register_hydra_plugin
from habitat_baselines.config.default_structured_configs import (
    HabitatBaselinesConfigPlugin,
)

_BASELINES_CFG_DIR = osp.dirname(inspect.getabsfile(inspect.currentframe()))
# Habitat baselines config directory inside the installed package.
# Used to access default predefined configs.
# This is equivalent to doing osp.dirname(osp.abspath(__file__))
DEFAULT_CONFIG_DIR = "habitat-lab/habitat/config/"
CONFIG_FILE_SEPARATOR = ","


def get_config(
    config_path: str,
    overrides: Optional[list] = None,
    configs_dir: str = _BASELINES_CFG_DIR,
) -> DictConfig:
    """
    Returns habitat_baselines config object composed of configs from yaml file (config_path) and overrides.

    :param config_path: path to the yaml config file.
    :param overrides: list of config overrides. For example, :py:`overrides=["habitat_baselines.trainer_name=ddppo"]`.
    :param configs_dir: path to the config files root directory (defaults to :ref:`_BASELINES_CFG_DIR`).
    :return: composed config object.
    """
    register_hydra_plugin(HabitatBaselinesConfigPlugin)
    cfg = _habitat_get_config(config_path, overrides, configs_dir)
    # Some config files live in a group (e.g. "social_nav/social_nav_twoagent")
    # and their YAML may place keys under a top-level mapping named after
    # the group (e.g. `social_nav:`). That results in Hydra composing the
    # file under `cfg.social_nav.*` which can surprise code that expects
    # `cfg.habitat_baselines` at the job top-level. To be permissive and
    # match the behavior of other configs in this repo, if the composed
    # config contains a single group node matching the group name and that
    # node contains `habitat_baselines`, promote it to the top-level so
    # callers can access `cfg.habitat_baselines` directly.
    try:
        # Extract group name if config_path is like 'group/name'
        group_name = config_path.split(
            "," if "," in config_path else "/"
        )[0]
    except Exception:
        group_name = None

    if group_name and isinstance(cfg, DictConfig):
        # Only promote if habitat_baselines does not already exist top-level
        if "habitat_baselines" not in cfg and group_name in cfg:
            try:
                inner = cfg[group_name]
                if inner is not None and "habitat_baselines" in inner:
                    # Temporarily allow writing new keys on the structured cfg
                    with open_dict(cfg):
                        cfg["habitat_baselines"] = inner["habitat_baselines"]
            except Exception:
                # Be conservative: if anything goes wrong, leave cfg as-is
                pass

    return cfg
