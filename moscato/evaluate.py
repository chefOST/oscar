"""Evaluate a trained MOSP checkpoint on the ground-truth test split.

Example:
    python -m moscato.evaluate --data_root ./moscato_dataset --dataset EgoPER --task coffee \
        --ckpt runs/egoper_coffee.pt
"""

from __future__ import annotations

import argparse

import numpy as np
import torch

from .config import DataPaths
from .data import MoscatoSplit, Vocab
from .metrics import f1_max, mean_ap, segment_f1
from .model import MOSP


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Evaluate MOSP on MOSCATO ground truth")
    p.add_argument("--data_root", required=True)
    p.add_argument("--dataset", required=True, choices=["CMU", "EGTEA", "EgoPER"])
    p.add_argument("--task", default=None)
    p.add_argument("--feature_root", default=None)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--seg_thr", type=float, default=0.5,
                   help="Probability threshold for segment extraction")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p


@torch.no_grad()
def evaluate(args) -> dict:
    paths = DataPaths(args.data_root, args.dataset, args.task, args.feature_root)
    vocab = Vocab.load(paths)
    test_split = MoscatoSplit(paths, vocab, "test")

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    device = torch.device(args.device)
    model = MOSP(cfg["feature_dim"], cfg["num_objects"], cfg["num_states"],
                 cfg["num_actions"], hidden=cfg["hidden"], num_layers=cfg["layers"],
                 nhead=cfg["nhead"], use_sai=cfg["use_sai"]).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    OS = cfg["num_objects"] * cfg["num_states"]
    all_probs, all_targets = [], []
    pred_seqs, gt_seqs = [], []  # per (video, class) binary sequences for segment F1

    for i in range(len(test_split)):
        s = test_split[i]
        feats = torch.from_numpy(s.features).unsqueeze(0).to(device)
        actions = torch.from_numpy(s.action_ids).unsqueeze(0).to(device)
        logits = model(feats, actions)                     # [1,T,O,S]
        probs = torch.sigmoid(logits)[0].reshape(-1, OS).cpu().numpy()  # [T, OS]
        targets = s.labels.reshape(-1, OS)                 # [T, OS]
        all_probs.append(probs)
        all_targets.append(targets)

        pred_bin = probs >= args.seg_thr
        for c in range(OS):
            if targets[:, c].sum() == 0 and pred_bin[:, c].sum() == 0:
                continue  # skip empty classes for this video to keep it cheap
            pred_seqs.append(pred_bin[:, c].astype(np.int8))
            gt_seqs.append(targets[:, c].astype(np.int8))

    probs = np.concatenate(all_probs, axis=0)
    targets = np.concatenate(all_targets, axis=0)

    f1m, best_thr = f1_max(probs, targets)
    mapv = mean_ap(probs, targets)
    seg = segment_f1(pred_seqs, gt_seqs)

    results = {
        "F1-Max": f1m, "best_thr": best_thr, "mAP": mapv,
        "F1@0.1": seg[0.1], "F1@0.25": seg[0.25], "F1@0.5": seg[0.5],
        "num_test_videos": len(test_split),
    }
    _print_results(args, results)
    return results


def _print_results(args, r: dict) -> None:
    name = args.dataset + (f"/{args.task}" if args.task else "")
    print(f"\n=== MOSCATO evaluation: {name} ({r['num_test_videos']} test videos) ===")
    print(f"{'Metric':<10}{'Value':>10}")
    print("-" * 20)
    print(f"{'F1-Max':<10}{r['F1-Max']:>10.2f}   (thr={r['best_thr']:.2f})")
    print(f"{'mAP':<10}{r['mAP']:>10.2f}")
    print(f"{'F1@0.1':<10}{r['F1@0.1']:>10.2f}")
    print(f"{'F1@0.25':<10}{r['F1@0.25']:>10.2f}")
    print(f"{'F1@0.5':<10}{r['F1@0.5']:>10.2f}")


if __name__ == "__main__":
    evaluate(build_argparser().parse_args())
