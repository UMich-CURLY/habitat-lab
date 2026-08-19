"""Extract HL weights (incl. the map branch) from a multi-agent ckpt into the
pretrained_hl_weights format. Superset of _extract_hl.py: adds map_encoder /
map_fc and keeps the FULL policy head (map-augmented width)."""
import sys

import torch

src, dst = sys.argv[1], sys.argv[2]
d = torch.load(src, map_location="cpu", weights_only=False)
sd = d[0]["state_dict"]
out = {"state_encoder": {}, "policy": {}, "map_encoder": {}, "map_fc": {}}
for k, v in sd.items():
    for part, tag in (("._state_encoder.", "state_encoder"),
                      ("._map_encoder.", "map_encoder"),
                      ("._map_fc.", "map_fc")):
        if part in k:
            out[tag][k.split(part)[1]] = v
            break
    else:
        if "._policy." in k and ".linear." in k:
            out["policy"][k.split("._policy.")[1]] = v
for tag in ("map_encoder", "map_fc"):
    if not out[tag]:
        del out[tag]
assert out["state_encoder"] and out["policy"], f"empty split: {list(sd)[:6]}"
torch.save(out, dst)
print(f"saved {dst}: " + " ".join(f"{k}={len(v)}" for k, v in out.items()),
      "head_shape=", tuple(out["policy"]["linear.weight"].shape))
