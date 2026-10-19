"""Loading application profiles (rules) and reference assets (standard videos + annotations)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from ..engine.compare import Segment
from ..engine.features import FEATURES

PROFILES_DIR = Path(__file__).parent


@dataclass
class Profile:
    raw: dict

    @property
    def id(self) -> str:
        return self.raw["id"]

    @property
    def features(self) -> list[str]:
        return list(self.raw["features"])

    def tolerance(self, feature: str) -> dict:
        if feature in self.raw.get("overrides", {}):
            return self.raw["overrides"][feature]
        return self.raw["tolerances"][FEATURES[feature].unit]

    def segment_rule(self, seg_id: str) -> dict:
        return self.raw.get("segments", {}).get(seg_id, {"name": seg_id, "focus": []})

    @property
    def quality(self) -> dict:
        return self.raw.get("quality", {})


@dataclass
class Reference:
    raw: dict
    root: Path

    @property
    def id(self) -> str:
        return self.raw["id"]

    @property
    def video(self) -> Path:
        p = Path(self.raw["video"])
        return p if p.is_absolute() else self.root / p

    def segments(self, profile: Profile) -> list[Segment]:
        return [Segment(s["id"], profile.segment_rule(s["id"]).get("name", s["id"]), float(s["start"]), float(s["end"]))
                for s in self.raw.get("segments", [])]


def load_profile(profile_id: str) -> Profile:
    path = PROFILES_DIR / profile_id / "profile.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Profile not found: {profile_id}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    unknown = [f for f in raw["features"] if f not in FEATURES]
    if unknown:
        raise ValueError(f"Profile {profile_id} references unknown features: {unknown}")
    return Profile(raw)


def list_references(references_dir: Path, root: Path) -> list[Reference]:
    refs = []
    for p in sorted(Path(references_dir).glob("*.yaml")):
        refs.append(Reference(yaml.safe_load(p.read_text(encoding="utf-8")), root))
    return refs


def get_reference(ref_id: str, references_dir: Path, root: Path) -> Reference:
    for r in list_references(references_dir, root):
        if r.id == ref_id:
            return r
    raise KeyError(f"Reference not found: {ref_id}")
