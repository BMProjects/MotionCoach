"""LLM coaching interfaces - definitions only, no implementation (docs/LLM_COACH_DESIGN.md).

Pipeline (role A): report --EvidenceBuilder--> evidence --KnowledgeIndex--> + knowledge
--CoachProvider--> advice --AdviceValidator--> advice shown in the report (or the rule texts on failure).
Documents follow ``schemas/coach_evidence.schema.json`` and ``schemas/coach_advice.schema.json``.
"""

from __future__ import annotations

from typing import Protocol


class EvidenceBuilder(Protocol):
    def build(self, report: dict, report_id: str) -> dict:
        """Deterministic: pick the items worth advising on (with JSON-pointer refs into the report)."""


class KnowledgeIndex(Protocol):
    def retrieve(self, evidence: dict, k: int = 8) -> list[dict]:
        """Coach-approved knowledge snippets for the segments/features in the evidence ({id, text, source})."""


class CoachProvider(Protocol):
    name: str          # e.g. "claude-opus-5-5", "qwen3-local"

    def advise(self, evidence: dict, knowledge: list[dict], *, locale: str = "zh-CN") -> dict:
        """Return a ``coach_advice`` document. Must not see video or faces unless separately authorised."""


class AdviceValidator(Protocol):
    def validate(self, advice: dict, evidence: dict) -> list[str]:
        """Problems found (empty = valid): unknown numbers, missing refs, directions for cannot-judge items, medical claims."""
