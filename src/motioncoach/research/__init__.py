"""Research harness: compare motion-correspondence methods side by side.

Nothing here is used by the analysis service. Every method reads the same cached poses and the
same section bounds and is scored by the same metrics (``metrics.py``) against a hand annotation
(``references/annotations/``), so a candidate can be judged before it replaces the production path.

    uv run python -m motioncoach.research.align_compare --out data/research/align_compare
"""
