"""Dataset loading, label-tensor construction and feature/label alignment."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass

import numpy as np

from .config import DataPaths


def _load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


@dataclass
class Vocab:
    a2i: dict[str, int]
    i2a: dict[int, str]
    o2i: dict[str, int]
    i2o: dict[int, str]
    s2i: dict[str, int]
    i2s: dict[int, str]

    @property
    def num_actions(self) -> int:
        return len(self.a2i)

    @property
    def num_objects(self) -> int:
        return len(self.o2i)

    @property
    def num_states(self) -> int:
        return len(self.s2i)

    @classmethod
    def load(cls, paths: DataPaths) -> "Vocab":
        a = _load_json(paths.actions_dict)
        o = _load_json(paths.objects_dict)
        s = _load_json(paths.states_dict)
        a2i = {k: int(v) for k, v in a["a2i"].items()}
        o2i = {k: int(v) for k, v in o["o2i"].items()}
        s2i = {k: int(v) for k, v in s["s2i"].items()}
        return cls(
            a2i=a2i, i2a={v: k for k, v in a2i.items()},
            o2i=o2i, i2o={v: k for k, v in o2i.items()},
            s2i=s2i, i2s={v: k for k, v in s2i.items()},
        )


@dataclass
class Sample:
    video_id: str
    features: np.ndarray      # [T, D] float32
    labels: np.ndarray        # [T, O, S] float32 (0/1)
    action_ids: np.ndarray    # [T] int64, action id per frame (num_actions == "no action")


def _resample_index(src_len: int, dst_len: int) -> np.ndarray:
    """Nearest-neighbour index map of length ``dst_len`` into ``[0, src_len)``."""
    if src_len == dst_len:
        return np.arange(dst_len)
    if src_len == 0:
        return np.zeros(dst_len, dtype=np.int64)
    pos = (np.arange(dst_len) + 0.5) * src_len / dst_len
    return np.clip(pos.astype(np.int64), 0, src_len - 1)


def _load_features(paths: DataPaths, video_id: str, feature_path: str | None) -> np.ndarray:
    fpath = paths.resolve_feature(video_id, feature_path)
    feat = np.load(fpath)
    feat = np.asarray(feat, dtype=np.float32)
    if feat.ndim == 1:
        feat = feat[None, :]
    # Features are expected as [T, D]. If it looks like [D, T] (D matches a
    # common backbone dim and the other axis is much longer) leave as-is unless
    # clearly transposed; we keep it simple and assume the longer axis is time
    # only when the array is 2D and the first axis is the smaller one.
    if feat.ndim == 2 and feat.shape[0] < feat.shape[1] and feat.shape[0] in (256, 512, 768, 1024, 2048):
        feat = feat.T
    return feat


def _build_labels(state_ann: dict, vocab: Vocab, ann_len: int) -> np.ndarray:
    """[ann_len, O, S] binary tensor from the per-object state lists."""
    y = np.zeros((ann_len, vocab.num_objects, vocab.num_states), dtype=np.float32)
    for obj_name, per_frame in state_ann.items():
        oi = vocab.o2i.get(obj_name)
        if oi is None:
            continue
        for t, states in enumerate(per_frame):
            if t >= ann_len:
                break
            for st in states:
                si = vocab.s2i.get(st)
                if si is not None:
                    y[t, oi, si] = 1.0
    return y


def _build_action_ids(action_segments: list, vocab: Vocab, length: int) -> np.ndarray:
    """[length] per-frame action id; ``num_actions`` marks 'no action'."""
    bg = vocab.num_actions
    a = np.full(length, bg, dtype=np.int64)
    for seg in action_segments or []:
        if len(seg) < 3:
            continue
        start, end, name = seg[0], seg[1], seg[2]
        ai = vocab.a2i.get(name)
        if ai is None and len(seg) >= 4:
            try:
                ai = int(seg[3])
            except (TypeError, ValueError):
                ai = None
        if ai is None:
            continue
        s = max(0, int(start))
        e = min(length, int(end))
        if e > s:
            a[s:e] = ai
    return a


def _ann_length(state_ann: dict, action_segments: list) -> int:
    n = 0
    for per_frame in state_ann.values():
        n = max(n, len(per_frame))
    for seg in action_segments or []:
        if len(seg) >= 2:
            n = max(n, int(seg[1]))
    return n


class MoscatoSplit:
    """A train (pseudo) or test (ground-truth) split for one dataset/task."""

    def __init__(self, paths: DataPaths, vocab: Vocab, split: str):
        assert split in ("train", "test")
        self.paths = paths
        self.vocab = vocab
        self.split = split
        ann_path = paths.pseudo_annotations if split == "train" else paths.ground_truth_annotations
        self.annotations = _load_json(ann_path)
        self.files = _load_json(paths.files_dict) if os.path.exists(paths.files_dict) else {}
        self.video_ids = list(self.annotations.keys())

    def __len__(self) -> int:
        return len(self.video_ids)

    def __getitem__(self, idx: int) -> Sample:
        vid = self.video_ids[idx]
        ann = self.annotations[vid]
        state_ann = ann.get("state", {})
        file_info = self.files.get(vid, {})
        action_segments = file_info.get("action_annts") or ann.get("action") or []

        feat = _load_features(self.paths, vid, file_info.get("feature_path"))
        T = feat.shape[0]

        ann_len = _ann_length(state_ann, action_segments)
        if ann_len == 0:
            ann_len = T
        y_full = _build_labels(state_ann, self.vocab, ann_len)
        a_full = _build_action_ids(action_segments, self.vocab, ann_len)

        # Align annotation timeline to the feature timeline.
        idx_map = _resample_index(ann_len, T)
        labels = y_full[idx_map]
        action_ids = a_full[idx_map]
        return Sample(video_id=vid, features=feat, labels=labels, action_ids=action_ids)

    def feature_dim(self) -> int:
        return self[0].features.shape[1]
