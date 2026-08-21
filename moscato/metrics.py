"""Frame-level and segment-level evaluation metrics.

All predictions/targets are per-(object,state) class.  A "class" is one flat
index into the O*S object-state grid.  Frame-level metrics pool every frame of
every video; segment-level metrics are computed per class per video and then
pooled following standard temporal-action-segmentation F1.
"""

from __future__ import annotations

import numpy as np


# --------------------------------------------------------------------------- #
# Frame-level
# --------------------------------------------------------------------------- #
def f1_max(probs: np.ndarray, targets: np.ndarray,
           thresholds=np.arange(0.1, 0.95, 0.1)) -> tuple[float, float]:
    """Micro-F1 maximised over a single global threshold.

    Returns (best_f1 in %, best_threshold).
    """
    best_f1, best_thr = 0.0, thresholds[0]
    pos = targets > 0.5
    n_pos = int(pos.sum())
    for thr in thresholds:
        pred = probs >= thr
        tp = int((pred & pos).sum())
        fp = int((pred & ~pos).sum())
        fn = n_pos - tp
        denom = 2 * tp + fp + fn
        f1 = (2 * tp / denom) if denom > 0 else 0.0
        if f1 > best_f1:
            best_f1, best_thr = f1, float(thr)
    return best_f1 * 100.0, best_thr


def mean_ap(probs: np.ndarray, targets: np.ndarray) -> float:
    """mAP over classes that contain at least one positive (returns %)."""
    try:
        from sklearn.metrics import average_precision_score
    except ImportError:
        return _mean_ap_numpy(probs, targets)
    aps = []
    for c in range(targets.shape[1]):
        col = targets[:, c]
        if col.sum() == 0:
            continue
        aps.append(average_precision_score(col, probs[:, c]))
    return float(np.mean(aps) * 100.0) if aps else 0.0


def _mean_ap_numpy(probs: np.ndarray, targets: np.ndarray) -> float:
    aps = []
    for c in range(targets.shape[1]):
        y = targets[:, c].astype(bool)
        if y.sum() == 0:
            continue
        order = np.argsort(-probs[:, c])
        y_sorted = y[order]
        tp = np.cumsum(y_sorted)
        precision = tp / (np.arange(len(y_sorted)) + 1)
        recall = tp / y.sum()
        # AP as area under precision-recall (step) curve.
        ap = np.sum((recall[1:] - recall[:-1]) * precision[1:]) if len(recall) > 1 else precision[0] * recall[0]
        aps.append(ap)
    return float(np.mean(aps) * 100.0) if aps else 0.0


# --------------------------------------------------------------------------- #
# Segment-level
# --------------------------------------------------------------------------- #
def _segments(binary_seq: np.ndarray) -> list[tuple[int, int]]:
    """Contiguous runs of 1s as (start, end_exclusive)."""
    segs = []
    start = None
    for t, v in enumerate(binary_seq):
        if v and start is None:
            start = t
        elif not v and start is not None:
            segs.append((start, t))
            start = None
    if start is not None:
        segs.append((start, len(binary_seq)))
    return segs


def _iou(a: tuple[int, int], b: tuple[int, int]) -> float:
    inter = max(0, min(a[1], b[1]) - max(a[0], b[0]))
    union = max(a[1], b[1]) - min(a[0], b[0])
    return inter / union if union > 0 else 0.0


def segment_f1(pred_seqs: list[np.ndarray], gt_seqs: list[np.ndarray],
               overlaps=(0.1, 0.25, 0.5)) -> dict[float, float]:
    """Segment F1 at each IoU overlap threshold, pooled over all (video,class).

    ``pred_seqs`` / ``gt_seqs`` are parallel lists of 1-D binary arrays, one per
    (video, class) pair.  Returns {overlap: f1 in %}.
    """
    results = {}
    for ov in overlaps:
        tp = fp = fn = 0
        for pred_bin, gt_bin in zip(pred_seqs, gt_seqs):
            p_segs = _segments(pred_bin)
            g_segs = _segments(gt_bin)
            matched = [False] * len(g_segs)
            for ps in p_segs:
                best, best_j = 0.0, -1
                for j, gs in enumerate(g_segs):
                    if matched[j]:
                        continue
                    iou = _iou(ps, gs)
                    if iou > best:
                        best, best_j = iou, j
                if best >= ov and best_j >= 0:
                    tp += 1
                    matched[best_j] = True
                else:
                    fp += 1
            fn += matched.count(False)
        denom = 2 * tp + fp + fn
        results[ov] = (2 * tp / denom * 100.0) if denom > 0 else 0.0
    return results
