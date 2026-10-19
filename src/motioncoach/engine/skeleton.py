"""Skeleton definitions: joint names, left/right pairs and drawing edges.

Every pose backend maps its output onto a named skeleton. Features refer to joints by
name, never by raw index, so a backend that lacks a joint yields a missing feature
instead of silently comparing the wrong point.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Skeleton:
    name: str
    joints: tuple[str, ...]
    edges: tuple[tuple[str, str], ...]
    index: dict[str, int] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "index", {j: i for i, j in enumerate(self.joints)})

    def has(self, joint: str) -> bool:
        return joint in self.index

    def lr_permutation(self) -> list[int]:
        """Index permutation that swaps every left_* joint with its right_* counterpart."""
        perm = []
        for j in self.joints:
            if j.startswith("left_"):
                other = "right_" + j[5:]
            elif j.startswith("right_"):
                other = "left_" + j[6:]
            else:
                other = j
            perm.append(self.index.get(other, self.index[j]))
        return perm

    def edge_indices(self) -> list[tuple[int, int]]:
        return [(self.index[a], self.index[b]) for a, b in self.edges if a in self.index and b in self.index]


_MP33_JOINTS = (
    "nose", "left_eye_inner", "left_eye", "left_eye_outer", "right_eye_inner", "right_eye",
    "right_eye_outer", "left_ear", "right_ear", "mouth_left", "mouth_right",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist",
    "left_pinky", "right_pinky", "left_index", "right_index", "left_thumb", "right_thumb",
    "left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle",
    "left_heel", "right_heel", "left_foot_index", "right_foot_index",
)

_BODY_EDGES = (
    ("left_shoulder", "right_shoulder"), ("left_hip", "right_hip"),
    ("left_shoulder", "left_hip"), ("right_shoulder", "right_hip"),
    ("left_shoulder", "left_elbow"), ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"), ("right_elbow", "right_wrist"),
    ("left_hip", "left_knee"), ("left_knee", "left_ankle"),
    ("right_hip", "right_knee"), ("right_knee", "right_ankle"),
    ("left_ankle", "left_heel"), ("left_heel", "left_foot_index"), ("left_ankle", "left_foot_index"),
    ("right_ankle", "right_heel"), ("right_heel", "right_foot_index"), ("right_ankle", "right_foot_index"),
    ("left_wrist", "left_index"), ("right_wrist", "right_index"),
    ("left_ear", "nose"), ("right_ear", "nose"),
)

MEDIAPIPE33 = Skeleton("mediapipe33", _MP33_JOINTS, _BODY_EDGES)

# COCO-17 (YOLO / RTMPose body). Defined so future adapters share the same contract.
COCO17 = Skeleton(
    "coco17",
    (
        "nose", "left_eye", "right_eye", "left_ear", "right_ear",
        "left_shoulder", "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist",
        "left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle",
    ),
    _BODY_EDGES,
)

# COCO-WholeBody 133 (RTMW): body17 + feet6 + face68 + hands 2x21. Big toe is mapped to the
# MediaPipe name "foot_index" so foot features share one vocabulary. Face points are not
# left/right permuted by lr_permutation (they are not used by any feature).
COCO_WHOLEBODY133 = Skeleton(
    "coco_wholebody133",
    COCO17.joints
    + ("left_foot_index", "left_small_toe", "left_heel", "right_foot_index", "right_small_toe", "right_heel")
    + tuple(f"face_{i}" for i in range(68))
    + tuple(f"left_hand_{i}" for i in range(21))
    + tuple(f"right_hand_{i}" for i in range(21)),
    _BODY_EDGES,
)

_REGISTRY = {s.name: s for s in (MEDIAPIPE33, COCO17, COCO_WHOLEBODY133)}


def get_skeleton(name: str) -> Skeleton:
    try:
        return _REGISTRY[name]
    except KeyError as exc:
        raise ValueError(f"Unknown skeleton '{name}'. Known: {sorted(_REGISTRY)}") from exc
