"""Path resolution for the MOSCATO dataset layout.

Expected directory tree (see the dataset README):

    <data_root>/
      annotations/pseudo_labels/<DATASET>[/<TASK>]/pseudo_annotations.json
      annotations/ground_truth/<DATASET>[/<TASK>]/ground_truth_annotations.json
      vocabulary/<DATASET>[/<TASK>]/{actions_dict,objects_dict,states_dict}.json
      metadata/<DATASET>[/<TASK>]/files_dict.json
      features/{i3d_features,clip_features}/...

CMU and EGTEA have no <TASK> sub-level; EgoPER does
(coffee/oatmeal/pinwheels/quesadilla/tea).
"""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass
class DataPaths:
    data_root: str
    dataset: str            # "CMU" | "EGTEA" | "EgoPER"
    task: str | None = None  # only for EgoPER
    feature_root: str | None = None  # override: load <feature_root>/<video_id>.npy

    def _sub(self, top: str) -> str:
        parts = [self.data_root, top, self.dataset]
        if self.task:
            parts.append(self.task)
        return os.path.join(*parts)

    # --- vocabulary -------------------------------------------------------
    @property
    def actions_dict(self) -> str:
        return os.path.join(self._sub("vocabulary"), "actions_dict.json")

    @property
    def objects_dict(self) -> str:
        return os.path.join(self._sub("vocabulary"), "objects_dict.json")

    @property
    def states_dict(self) -> str:
        return os.path.join(self._sub("vocabulary"), "states_dict.json")

    # --- annotations ------------------------------------------------------
    @property
    def pseudo_annotations(self) -> str:
        return os.path.join(
            self.data_root, "annotations", "pseudo_labels",
            *( [self.dataset, self.task] if self.task else [self.dataset] ),
            "pseudo_annotations.json",
        )

    @property
    def ground_truth_annotations(self) -> str:
        return os.path.join(
            self.data_root, "annotations", "ground_truth",
            *( [self.dataset, self.task] if self.task else [self.dataset] ),
            "ground_truth_annotations.json",
        )

    # --- metadata ---------------------------------------------------------
    @property
    def files_dict(self) -> str:
        return os.path.join(self._sub("metadata"), "files_dict.json")

    def resolve_feature(self, video_id: str, feature_path: str | None) -> str:
        """Locate a video's feature .npy.

        If ``feature_root`` is set we use ``<feature_root>/<video_id>.npy``;
        otherwise we join the (relative) ``feature_path`` from files_dict onto
        ``data_root``.
        """
        if self.feature_root:
            return os.path.join(self.feature_root, f"{video_id}.npy")
        if feature_path is None:
            raise FileNotFoundError(f"No feature_path for {video_id} and no --feature_root given")
        rel = feature_path.lstrip("/\\")
        return os.path.join(self.data_root, rel)
