from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np
from moviepy.video.io.VideoFileClip import VideoFileClip
from tqdm import tqdm

from .audio_aligner import find_time_offset
from typing import Literal

from .pose_estimation.mediapipe_estimator import MediaPipePoseEstimator
from .pose_estimation.yolo_estimator import YOLOPoseEstimator
from .similarity_analysis.procrustes_analyzer import ProcrustesSimilarityAnalyzer
from .visualizer import create_comparison_video


def extract_audio(video_path: Path, output_path: Path, codec: str = "pcm_s16le") -> Path:
    """
    Extracts audio from a video file and saves it as WAV.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with VideoFileClip(str(video_path)) as clip:
        clip.audio.write_audiofile(str(output_path), codec=codec)
    return output_path


def process_video(video_path: Path, estimator: MediaPipePoseEstimator) -> List[np.ndarray | None]:
    """
    Runs pose estimation on every frame of a video.
    """
    cap = cv2.VideoCapture(str(video_path))
    poses: List[np.ndarray | None] = []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    for _ in tqdm(range(total_frames), desc=f"Processing {video_path.name}"):
        ret, frame = cap.read()
        if not ret:
            break
        poses.append(estimator.estimate(frame))

    cap.release()
    return poses


def run_pipeline(
    video_a: Path,
    video_b: Path,
    output_dir: Path = Path("output"),
    align_method: str = "chroma",
    target_height: int = 720,
    pose_backend: Literal["mediapipe", "yolo"] = "mediapipe",
    static_mode: bool = False,
    model_complexity: int = 1,
    min_detection_confidence: float = 0.5,
    yolo_model: str = "yolo11m-pose.pt",
    yolo_device: str = "cuda",
    yolo_conf: float = 0.25,
) -> Dict[str, object]:
    """
    Full pipeline: audio alignment -> pose extraction -> similarity scoring -> video export.
    Supports MediaPipe (lightweight) and YOLO pose (higher accuracy, GPU recommended).
    """
    video_a = Path(video_a)
    video_b = Path(video_b)
    for video in (video_a, video_b):
        if not video.exists():
            raise FileNotFoundError(f"Video not found: {video}")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    audio_a_path = output_dir / f"{video_a.stem}_audio.wav"
    audio_b_path = output_dir / f"{video_b.stem}_audio.wav"
    extract_audio(video_a, audio_a_path)
    extract_audio(video_b, audio_b_path)

    time_offset = find_time_offset(str(audio_a_path), str(audio_b_path), method=align_method)
    with VideoFileClip(str(video_a)) as clip_a:
        fps = clip_a.fps

    frame_offset = int(time_offset * fps)

    if pose_backend == "mediapipe":
        estimator = MediaPipePoseEstimator(
            static_mode=static_mode,
            model_complexity=model_complexity,
            min_detection_confidence=min_detection_confidence,
        )
    elif pose_backend == "yolo":
        estimator = YOLOPoseEstimator(
            model_path=yolo_model,
            device=yolo_device,
            conf=yolo_conf,
        )
    else:
        raise ValueError(f"Unsupported pose_backend: {pose_backend}")
    poses_a = process_video(video_a, estimator)
    poses_b = process_video(video_b, estimator)

    similarity_analyzer = ProcrustesSimilarityAnalyzer()
    start_a, start_b = (frame_offset, 0) if frame_offset > 0 else (0, -frame_offset)
    num_frames_to_compare = max(0, min(len(poses_a) - start_a, len(poses_b) - start_b))

    scores: List[float] = []
    for i in tqdm(range(num_frames_to_compare), desc="Comparing poses"):
        pose_a = poses_a[start_a + i]
        pose_b = poses_b[start_b + i]

        if pose_a is not None and pose_b is not None:
            score = similarity_analyzer.compare(pose_a, pose_b)
        else:
            score = 0.0
        scores.append(score)

    average_score = float(np.mean(scores)) if scores else 0.0

    output_video_path = output_dir / "comparison_video.mp4"
    create_comparison_video(
        video_path_a=str(video_a),
        video_path_b=str(video_b),
        poses_a=poses_a,
        poses_b=poses_b,
        scores=scores,
        frame_offset=frame_offset,
        output_path=str(output_video_path),
        fps=fps,
        target_height=target_height,
    )

    return {
        "time_offset": time_offset,
        "frame_offset": frame_offset,
        "average_score": average_score,
        "output_video": output_video_path,
        "scores": scores,
    }
