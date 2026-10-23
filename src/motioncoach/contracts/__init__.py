"""Data contracts shared by the three logical roles (docs/MOBILE_DESIGN.md):

A high-precision analysis service, B algorithm testing service, C mobile capture/analysis.
Version strings live here; JSON Schemas live in ``schemas/`` and are the reference for any other
runtime (TypeScript / Kotlin / Swift ports) that produces or consumes these documents.
"""

from __future__ import annotations

import json
from pathlib import Path

REPORT_SCHEMA = "motioncoach.report/0.5"            # analysis report (apps/analysis.py)
POSE_PAYLOAD_SCHEMA = "motioncoach.poses/1"         # overlay / exchange form of a PoseSequence
REFPACK_SCHEMA = "motioncoach.refpack/1"            # reference package A -> C (apps/refpack.py)
GROUP_SCHEMA = "motioncoach.group/0"                # multi-person analysis (apps/group.py)
COACH_EVIDENCE_SCHEMA = "motioncoach.coach.evidence/0"   # LLM coaching: evidence in (interface only)
COACH_ADVICE_SCHEMA = "motioncoach.coach.advice/0"       # LLM coaching: advice out (interface only)

SCHEMA_DIR = Path(__file__).resolve().parent / "schemas"


def json_schema(name: str) -> dict:
    """Load ``schemas/<name>.schema.json`` (e.g. "refpack_manifest", "coach_advice")."""
    return json.loads((SCHEMA_DIR / f"{name}.schema.json").read_text(encoding="utf-8"))
