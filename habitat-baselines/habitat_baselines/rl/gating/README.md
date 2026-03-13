Gating BC trainer
==================

Purpose
-------
This folder contains a small Behavior Cloning (BC) trainer for a visual gating
network that outputs NAV vs YIELD decisions from RGB images.

Files
-----
- `gating_model.py` - model factory (ResNet-18 backbone if torchvision is
  available, else a tiny convnet fallback).
- `gating_trainer.py` - training script that consumes .npz datasets and trains
  a binary classifier.

Dataset format
--------------
Provide either a single `.npz` file or a directory containing `.npz` files. Each
`.npz` must provide two arrays:

- `images`: numpy array of shape (N, H, W, 3), dtype=uint8 or float32.
- `labels`: numpy array of shape (N,), dtype=int, values in {0,1} where
  0 -> NAV, 1 -> YIELD.

Usage
-----

Train example:

```bash
python -m habitat_baselines.rl.gating.gating_trainer \
  --data /path/to/dataset_dir_or_file.npz \
  --out-dir /tmp/gate_ckpts \
  --epochs 20 \
  --batch-size 32
```

Notes & next steps
------------------
- The trainer is intentionally minimal: add augmentations, class weighting,
  or more sophisticated backbones as needed.
- To produce labels from episodes: write a small exporter that saved image
  frames and a label per frame (heuristic or oracle-based). The trainer
  assumes these npz files have already been produced.
- For reduced twitching, consider temporal smoothing (stack frames or add
  a small RNN on top of the backbone).
