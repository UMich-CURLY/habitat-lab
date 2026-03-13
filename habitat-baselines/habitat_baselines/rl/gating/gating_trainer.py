"""
Simple BC trainer for a binary visual gating network (NAV vs YIELD).

Dataset format (one of):
- A directory containing one or more .npz files. Each .npz must contain
  `images` (N,H,W,3) and `labels` (N,) where label 0=NAV, 1=YIELD.
- A single .npz file with the above arrays.

Usage examples:

python -m habitat_baselines.rl.gating.gating_trainer \
    --data /path/to/data_dir_or_file.npz \
    --out-dir /path/to/checkpoints \
    --epochs 20

"""
import argparse
import os
import time
from glob import glob

import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from .gating_model import make_resnet_backbone


class NPZImageDataset(Dataset):
    def __init__(self, path, transform=None):
        # path can be a dir or a single file
        files = []
        if os.path.isdir(path):
            files = glob(os.path.join(path, "*.npz"))
        elif os.path.isfile(path):
            files = [path]
        else:
            raise ValueError(f"Path not found: {path}")

        if len(files) == 0:
            raise ValueError(f"No .npz files found in {path}")

        self.images = []
        self.labels = []
        for f in files:
            data = np.load(f)
            imgs = data["images"]
            lbls = data["labels"]
            self.images.append(imgs)
            self.labels.append(lbls)
        self.images = np.concatenate(self.images, axis=0)
        self.labels = np.concatenate(self.labels, axis=0)
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = self.images[idx]
        lbl = int(self.labels[idx])
        # convert HWC uint8 to CHW float tensor
        if self.transform is not None:
            img = self.transform(img)
        else:
            img = torch.from_numpy(img).float().permute(2, 0, 1) / 255.0
        return img, lbl


def train_one_epoch(model, loader, opt, loss_fn, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for imgs, labels in loader:
        imgs = imgs.to(device)
        labels = labels.to(device)
        logits = model(imgs)
        loss = loss_fn(logits, labels)
        opt.zero_grad()
        loss.backward()
        opt.step()
        running_loss += loss.item() * imgs.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += imgs.size(0)
    return running_loss / total, correct / total


def eval_model(model, loader, loss_fn, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            labels = labels.to(device)
            logits = model(imgs)
            loss = loss_fn(logits, labels)
            running_loss += loss.item() * imgs.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += imgs.size(0)
    return running_loss / total, correct / total


def make_transforms(image_size=224):
    t = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((image_size, image_size)),
        # ToTensor already scales to [0,1]
        # optional normalization can be added
    ])
    return t


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to .npz or directory with .npz files")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--pretrained", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    transform = make_transforms(args.image_size)
    dataset = NPZImageDataset(args.data, transform=transform)

    # simple split
    n = len(dataset)
    n_val = max(1, int(0.1 * n))
    n_train = n - n_val
    train_set, val_set = torch.utils.data.random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = make_resnet_backbone(num_classes=2, pretrained=args.pretrained, in_channels=3)
    device = torch.device(args.device)
    model = model.to(device)

    loss_fn = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val = 1e9
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_one_epoch(model, train_loader, opt, loss_fn, device)
        val_loss, val_acc = eval_model(model, val_loader, loss_fn, device)
        t1 = time.time()
        print(f"Epoch {epoch:03d}: train loss {tr_loss:.4f}, acc {tr_acc:.3f}; val loss {val_loss:.4f}, acc {val_acc:.3f}; time {t1-t0:.1f}s")
        ckpt = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "opt_state": opt.state_dict(),
        }
        torch.save(ckpt, os.path.join(args.out_dir, f"gate_ckpt_epoch_{epoch:03d}.pth"))
        if val_loss < best_val:
            best_val = val_loss
            torch.save(ckpt, os.path.join(args.out_dir, f"gate_ckpt_best.pth"))


if __name__ == "__main__":
    main()
