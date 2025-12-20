import argparse
from pathlib import Path

from .core import run_pipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Dance battle alignment demo")
    parser.add_argument("--video-a", default="assets/chuu.mov", help="Reference video path")
    parser.add_argument("--video-b", default="assets/cyves.mov", help="Comparison video path")
    parser.add_argument("--output-dir", default="output", help="Directory to store intermediate and output files")
    parser.add_argument(
        "--align-method",
        choices=["chroma", "onset"],
        default="chroma",
        help="Audio alignment strategy",
    )
    parser.add_argument("--target-height", type=int, default=720, help="Height of each video in the side-by-side output")
    parser.add_argument(
        "--static-mode",
        action="store_true",
        help="Use static image mode for MediaPipe (slower but more stable on single frames)",
    )
    parser.add_argument(
        "--model-complexity",
        type=int,
        choices=[0, 1, 2],
        default=1,
        help="MediaPipe model complexity (higher is more accurate and slower)",
    )
    parser.add_argument(
        "--min-detection-confidence",
        type=float,
        default=0.5,
        help="Minimum detection confidence for MediaPipe pose estimation",
    )
    parser.add_argument(
        "--pose-backend",
        choices=["mediapipe", "yolo"],
        default="mediapipe",
        help="Pose estimator backend. 'yolo' uses YOLO11m-pose (GPU recommended).",
    )
    parser.add_argument(
        "--yolo-model",
        default="yolo11m-pose.pt",
        help="YOLO pose model path/name (used when pose-backend=yolo).",
    )
    parser.add_argument(
        "--yolo-device",
        default="cuda",
        help="Device for YOLO pose inference, e.g., 'cuda' or 'cpu'.",
    )
    parser.add_argument(
        "--yolo-conf",
        type=float,
        default=0.25,
        help="Confidence threshold for YOLO pose detection.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    results = run_pipeline(
        video_a=Path(args.video_a),
        video_b=Path(args.video_b),
        output_dir=Path(args.output_dir),
        align_method=args.align_method,
        target_height=args.target_height,
        pose_backend=args.pose_backend,
        static_mode=bool(args.static_mode),
        model_complexity=int(args.model_complexity),
        min_detection_confidence=float(args.min_detection_confidence),
        yolo_model=args.yolo_model,
        yolo_device=args.yolo_device,
        yolo_conf=args.yolo_conf,
    )

    print(
        f"Finished. Offset: {results['time_offset']:.2f}s "
        f"({results['frame_offset']} frames), "
        f"Average similarity: {results['average_score']:.2f}%"
    )
    print(f"Output video: {results['output_video']}")


if __name__ == "__main__":
    # Support running directly from IDE without PYTHONPATH setup
    import sys
    from pathlib import Path as _Path

    # Add src/ to path for local development
    _project_root = _Path(__file__).resolve().parent.parent.parent
    _src_path = _project_root / "src"
    if str(_src_path) not in sys.path:
        sys.path.insert(0, str(_src_path))

    main()
