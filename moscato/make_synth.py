"""Generate a tiny synthetic dataset in the exact MOSCATO on-disk schema.

This lets you smoke-test the full train -> evaluate pipeline before downloading
the real ~10 GB dataset.  The synthetic labels are *correlated* with the
features (plus noise), so training loss should drop and metrics should land
well above chance -- confirming the wiring end-to-end.

    python -m moscato.make_synth --out ./synth_dataset
    python -m moscato.train    --data_root ./synth_dataset --dataset CMU --use_sai --epochs 15 --out runs/synth.pt
    python -m moscato.evaluate --data_root ./synth_dataset --dataset CMU --ckpt runs/synth.pt
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np


def _write_json(path: str, obj) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f)


def generate(out: str, dataset: str = "CMU", n_train: int = 12, n_test: int = 5,
             num_actions: int = 4, num_objects: int = 3, num_states: int = 5,
             feat_dim: int = 64, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)

    # --- vocabulary -------------------------------------------------------
    a2i = {f"action_{i}": i for i in range(num_actions)}
    o2i = {f"object_{i}": i for i in range(num_objects)}
    s2i = {f"state_{i}": i for i in range(num_states)}
    voc = os.path.join(out, "vocabulary", dataset)
    _write_json(os.path.join(voc, "actions_dict.json"),
                {"a2i": {k: str(v) for k, v in a2i.items()}, "i2a": {str(v): k for k, v in a2i.items()}})
    _write_json(os.path.join(voc, "objects_dict.json"),
                {"o2i": {k: str(v) for k, v in o2i.items()}, "i2o": {str(v): k for k, v in o2i.items()}})
    _write_json(os.path.join(voc, "states_dict.json"),
                {"s2i": {k: str(v) for k, v in s2i.items()}, "i2s": {str(v): k for k, v in s2i.items()}})

    # Fixed "signature" vectors so labels are decodable from features.
    signatures = rng.normal(size=(num_objects, num_states, feat_dim)).astype(np.float32)

    def make_video(vid: str):
        L = int(rng.integers(30, 61))
        # Partition timeline into action segments.
        n_seg = int(rng.integers(2, 5))
        bounds = sorted(rng.choice(range(1, L), size=min(n_seg - 1, L - 1), replace=False).tolist())
        bounds = [0] + bounds + [L]
        segments, frame_action = [], np.zeros(L, dtype=int)
        for k in range(len(bounds) - 1):
            s, e = bounds[k], bounds[k + 1]
            aid = int(rng.integers(0, num_actions))
            segments.append([s, e, f"action_{aid}", str(aid)])
            frame_action[s:e] = aid

        # State rule: object o has state (o+aid) % S active when its action==aid.
        state_lists = {f"object_{o}": [[] for _ in range(L)] for o in range(num_objects)}
        active = np.zeros((L, num_objects, num_states), dtype=bool)
        for t in range(L):
            aid = frame_action[t]
            for o in range(num_objects):
                if (o + aid) % 2 == 0:  # only some object/action combos trigger a state
                    s = (o + aid) % num_states
                    state_lists[f"object_{o}"][t].append(f"state_{s}")
                    active[t, o, s] = True

        # Features = noise + signature of every active (object,state).
        feats = rng.normal(scale=0.5, size=(L, feat_dim)).astype(np.float32)
        for t in range(L):
            for o in range(num_objects):
                for s in range(num_states):
                    if active[t, o, s]:
                        feats[t] += signatures[o, s]
        return L, feats, segments, state_lists

    # --- per-split generation --------------------------------------------
    feat_dir = os.path.join(out, "features", "synthetic")
    os.makedirs(feat_dir, exist_ok=True)

    def build_split(n: int, prefix: str):
        pseudo, files = {}, {}
        for i in range(n):
            vid = f"{prefix}_{i:03d}"
            L, feats, segments, state_lists = make_video(vid)
            rel = os.path.join("features", "synthetic", f"{vid}.npy")
            np.save(os.path.join(out, rel), feats)
            pseudo[vid] = {"action": segments, "state": state_lists}
            files[vid] = {
                "video_path": f"/videos/{vid}.mp4",
                "feature_path": "/" + rel.replace("\\", "/"),
                "feature_len": L,
                "action_annts": segments,
            }
        return pseudo, files

    train_ann, train_files = build_split(n_train, "train")
    test_ann, test_files = build_split(n_test, "test")

    # ground truth also carries a per-frame "object" visibility list.
    for vid, ann in test_ann.items():
        L = len(next(iter(ann["state"].values())))
        vis = [[o for o in ann["state"] if ann["state"][o][t]] for t in range(L)]
        ann["object"] = vis

    _write_json(os.path.join(out, "annotations", "pseudo_labels", dataset, "pseudo_annotations.json"), train_ann)
    _write_json(os.path.join(out, "annotations", "ground_truth", dataset, "ground_truth_annotations.json"), test_ann)

    all_files = {**train_files, **test_files}
    _write_json(os.path.join(out, "metadata", dataset, "files_dict.json"), all_files)

    print(f"[synth] wrote dataset '{dataset}' to {out}")
    print(f"        {n_train} train + {n_test} test videos | "
          f"A={num_actions} O={num_objects} S={num_states} D={feat_dim}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="./synth_dataset")
    p.add_argument("--dataset", default="CMU")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    generate(args.out, args.dataset, seed=args.seed)
