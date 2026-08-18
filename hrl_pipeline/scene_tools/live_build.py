"""Build ONE runnable episode from a live click request (4 world-xz points on a
doorcand door). Reuses build_manual_episodes.build_one for snap+validate. Writes
a JSON status to status_json (stdout is unusable -- pybullet prints a banner
there on import); writes the 1-episode dataset to out_gz on success.

Usage: python hrl_pipeline/scene_tools/live_build.py <request.json> <out.json.gz> <status.json>
  request.json = {src_idx, scene_id, scene_dataset_config, door_start, door_end,
                  pts: [[x,z]_robot_start, _robot_goal, _human_start, _human_goal]}
"""
import gzip
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import build_manual_episodes as bme
from render_scene_rgb_topdown import make_sim

PTS = ["robot_start", "robot_goal", "human_start", "human_goal"]


def main():
    req = json.load(open(sys.argv[1]))
    out_gz = sys.argv[2]
    src = json.load(gzip.open(req.get("source", "data/social_nav_episode_doorcand_v3.json.gz"), "rt"))
    tmpl = src["episodes"][req["src_idx"]]

    sim, _ = make_sim(req["scene_id"], req["scene_dataset_config"])
    entry = {"id": "live", "src_idx": req["src_idx"],
             "scene_id": req["scene_id"], "scene_dataset_config": req["scene_dataset_config"],
             "door_start": req["door_start"], "door_end": req["door_end"]}
    for k, p in zip(PTS, req["pts"]):
        entry[k] = p
    ep, errors, warnings, _ = bme.build_one(entry, tmpl, sim.pathfinder)
    sim.close()

    status_json = sys.argv[3]
    if ep is None:
        json.dump({"ok": False, "errors": errors}, open(status_json, "w"))
        return
    ep["episode_id"] = "0"
    out = dict(src)
    out["episodes"] = [ep]
    json.dump(out, gzip.open(out_gz, "wt"))
    json.dump({"ok": True, "warnings": warnings}, open(status_json, "w"))


if __name__ == "__main__":
    main()
