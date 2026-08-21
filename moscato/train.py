"""Train the MOSP network on pseudo-labels (weak supervision).

Example:
    python -m moscato.train --data_root ./moscato_dataset --dataset EgoPER --task coffee \
        --epochs 20 --use_sai --out runs/egoper_coffee.pt
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import torch
import torch.nn as nn

from .config import DataPaths
from .data import MoscatoSplit, Vocab


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train MOSP on MOSCATO pseudo-labels")
    p.add_argument("--data_root", required=True)
    p.add_argument("--dataset", required=True, choices=["CMU", "EGTEA", "EgoPER"])
    p.add_argument("--task", default=None, help="EgoPER task: coffee/oatmeal/pinwheels/quesadilla/tea")
    p.add_argument("--feature_root", default=None,
                   help="Override: load features from <feature_root>/<video_id>.npy")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--nhead", type=int, default=4)
    p.add_argument("--use_sai", action="store_true", help="Enable State-Action Interaction module")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out", default="runs/mosp.pt")
    return p


def train(args) -> None:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    paths = DataPaths(args.data_root, args.dataset, args.task, args.feature_root)
    vocab = Vocab.load(paths)
    train_split = MoscatoSplit(paths, vocab, "train")
    feature_dim = train_split.feature_dim()
    print(f"[data] {len(train_split)} train videos | D={feature_dim} "
          f"O={vocab.num_objects} S={vocab.num_states} A={vocab.num_actions}")

    from .model import MOSP
    device = torch.device(args.device)
    model = MOSP(feature_dim, vocab.num_objects, vocab.num_states, vocab.num_actions,
                 hidden=args.hidden, num_layers=args.layers, nhead=args.nhead,
                 use_sai=args.use_sai).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.BCEWithLogitsLoss()

    order = np.arange(len(train_split))
    for epoch in range(args.epochs):
        model.train()
        np.random.shuffle(order)
        running = 0.0
        for i in order:
            s = train_split[int(i)]
            feats = torch.from_numpy(s.features).unsqueeze(0).to(device)      # [1,T,D]
            labels = torch.from_numpy(s.labels).unsqueeze(0).to(device)       # [1,T,O,S]
            actions = torch.from_numpy(s.action_ids).unsqueeze(0).to(device)  # [1,T]
            logits = model(feats, actions)
            loss = loss_fn(logits, labels)
            opt.zero_grad()
            loss.backward()
            opt.step()
            running += loss.item()
        print(f"[epoch {epoch + 1:3d}/{args.epochs}] loss={running / max(1, len(train_split)):.4f}")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    torch.save({
        "state_dict": model.state_dict(),
        "config": {
            "feature_dim": feature_dim,
            "num_objects": vocab.num_objects,
            "num_states": vocab.num_states,
            "num_actions": vocab.num_actions,
            "hidden": args.hidden, "layers": args.layers, "nhead": args.nhead,
            "use_sai": args.use_sai,
        },
        "args": vars(args),
    }, args.out)
    print(f"[saved] {args.out}")


if __name__ == "__main__":
    train(build_argparser().parse_args())
