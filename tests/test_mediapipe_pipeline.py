"""
Optional smoke test for the MediaPipe backend.

Default is skipped in pytest; run with RUN_MEDIAPIPE_TEST=1 or execute this
file directly in your IDE. Requires mediapipe and sample videos in assets/.
"""

import os
from pathlib import Path

import pytest

pytest.importorskip("librosa")
pytest.importorskip("moviepy")

from dancebattle import run_pipeline  # noqa: E402

RUN_MEDIAPIPE_TEST = os.getenv("RUN_MEDIAPIPE_TEST") == "1"

pytestmark = pytest.mark.skipif(
    not RUN_MEDIAPIPE_TEST,
    reason="Set RUN_MEDIAPIPE_TEST=1 to enable MediaPipe smoke test.",
)


def test_mediapipe_pipeline_smoke(tmp_path: Path):
    pytest.importorskip("mediapipe")

    video_a = Path("assets/chuu.mov")
    video_b = Path("assets/cyves.mov")
    if not video_a.exists() or not video_b.exists():
        pytest.skip("Sample videos not available")

    results = run_pipeline(
        video_a=video_a,
        video_b=video_b,
        output_dir=tmp_path,
        align_method="onset",  # faster for smoke test
        target_height=256,
        pose_backend="mediapipe",
        model_complexity=int(os.getenv("MP_MODEL_COMPLEXITY", "1")),
        min_detection_confidence=float(os.getenv("MP_MIN_DET_CONF", "0.5")),
        static_mode=bool(int(os.getenv("MP_STATIC_MODE", "0"))),
    )

    assert results["output_video"].exists()
    assert results["average_score"] >= 0


def _run_as_script():
    """
    Convenience entry point for IDE runs.
    """
    try:
        import mediapipe  # noqa: F401
    except ImportError:
        print("MediaPipe not installed. Install with: pip install -r requirements.txt")
        return

    video_a = Path("assets/chuu.mov")
    video_b = Path("assets/cyves.mov")
    if not video_a.exists() or not video_b.exists():
        print("Sample videos not found at assets/chuu.mov and assets/cyves.mov")
        return

    output_dir = Path(__file__).resolve().parent / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Running MediaPipe smoke test...")
    results = run_pipeline(
        video_a=video_a,
        video_b=video_b,
        output_dir=output_dir,
        align_method="onset",
        target_height=256,
        pose_backend="mediapipe",
        model_complexity=int(os.getenv("MP_MODEL_COMPLEXITY", "1")),
        min_detection_confidence=float(os.getenv("MP_MIN_DET_CONF", "0.5")),
        static_mode=bool(int(os.getenv("MP_STATIC_MODE", "0"))),
    )
    print(f"Done. Offset: {results['time_offset']:.2f}s, avg score: {results['average_score']:.2f}%")
    print(f"Output video: {results['output_video']}")


if __name__ == "__main__":
    _run_as_script()
