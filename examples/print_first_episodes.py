#!/usr/bin/env python3
"""打印带 collision 的 dataset 前 N 个 episode。默认路径与 generate_episodes_config 中 OUTPUT_DATASET_WITH_COLLISION_PATH 一致。"""
import json
import gzip
import os

# 默认路径（相对本脚本所在目录的上一级，即 habitat-lab 根）
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
HABITAT_LAB_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
DATASET_PATH = os.path.join(HABITAT_LAB_ROOT, "data/data_xinyuan/dataset_with_collision_0220.json.gz")
NUM_EPISODES = 11


def load_dataset(path):
    if path.endswith(".gz"):
        with gzip.open(path, "rt", encoding="utf-8") as f:
            return json.load(f)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    path = DATASET_PATH
    if not os.path.exists(path):
        print(f"文件不存在: {path}")
        return
    data = load_dataset(path)
    episodes = data.get("episodes", [])
    n = min(NUM_EPISODES, len(episodes))
    print(f"共 {len(episodes)} 个 episodes，打印前 {n} 个:\n")
    for i in range(n):
        ep = episodes[i]
        print("=" * 60)
        print(f"Episode {i} (episode_id={ep.get('episode_id')})")
        print("=" * 60)
        print(json.dumps(ep, indent=2, ensure_ascii=False))
        print()


if __name__ == "__main__":
    main()
