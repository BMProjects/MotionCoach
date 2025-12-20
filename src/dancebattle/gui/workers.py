"""
Background worker threads for the DanceBattle GUI.

CameraWorker: Captures webcam frames, runs pose estimation, computes live scores.
"""

import subprocess
import time
from pathlib import Path
from typing import List

import cv2
import numpy as np
from PySide6.QtCore import QThread, Signal

from ..core import MediaPipePoseEstimator, ProcrustesSimilarityAnalyzer


class CameraWorker(QThread):
    """
    Worker thread for capturing camera frames and computing real-time similarity scores.
    Uses ffmpeg subprocess for recording video with audio from microphone.

    Signals:
        frame_ready: Emitted with (BGR frame, user_pose, score) on each processed frame.
        error: Emitted when an error occurs.
        finished_recording: Emitted when recording stops, with the path to the temp video.
    """

    frame_ready = Signal(np.ndarray, object, float)  # frame, pose, score
    error = Signal(str)
    finished_recording = Signal(str)  # path to temp video

    def __init__(
        self,
        camera_id: int = 0,
        reference_poses: List[np.ndarray | None] | None = None,
        reference_fps: float = 30.0,
        output_path: Path | None = None,
        model_complexity: int = 1,
        min_detection_confidence: float = 0.5,
    ):
        super().__init__()
        self.camera_id = camera_id
        self.reference_poses = reference_poses or []
        self.reference_fps = reference_fps
        self.output_path = output_path

        self._running = False
        self._recording = False
        self._start_time = 0.0

        self.estimator = MediaPipePoseEstimator(
            static_mode=False,
            model_complexity=model_complexity,
            min_detection_confidence=min_detection_confidence,
        )
        self.analyzer = ProcrustesSimilarityAnalyzer()

        self._ffmpeg_process: subprocess.Popen | None = None
        self._frame_width = 640
        self._frame_height = 480

    def run(self) -> None:
        """Main thread loop: capture -> estimate -> compare -> emit."""
        cap = cv2.VideoCapture(self.camera_id)
        if not cap.isOpened():
            self.error.emit(f"Cannot open camera {self.camera_id}")
            return

        self._frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self._frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._camera_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

        self._running = True
        self._start_time = time.time()

        while self._running:
            ret, frame = cap.read()
            if not ret:
                continue

            elapsed = time.time() - self._start_time

            # Run pose estimation on user frame
            user_pose = self.estimator.estimate(frame)

            # Get corresponding reference pose
            ref_frame_idx = int(elapsed * self.reference_fps)
            ref_pose = None
            if 0 <= ref_frame_idx < len(self.reference_poses):
                ref_pose = self.reference_poses[ref_frame_idx]

            # Compute similarity
            score = 0.0
            if user_pose is not None and ref_pose is not None:
                score = self.analyzer.compare(user_pose, ref_pose)

            # Write frame to ffmpeg if recording
            if self._recording and self._ffmpeg_process is not None:
                try:
                    self._ffmpeg_process.stdin.write(frame.tobytes())
                except (BrokenPipeError, OSError):
                    pass

            # Emit signal
            self.frame_ready.emit(frame, user_pose, score)

        # Cleanup
        cap.release()
        self._stop_ffmpeg()

    def _start_ffmpeg(self, fps: float) -> None:
        """Start ffmpeg subprocess for recording video with audio."""
        if self.output_path is None:
            return

        # Determine audio input device based on platform
        import platform
        if platform.system() == "Linux":
            audio_input = ["-f", "pulse", "-i", "default"]
        elif platform.system() == "Darwin":  # macOS
            audio_input = ["-f", "avfoundation", "-i", ":0"]
        else:  # Windows
            audio_input = ["-f", "dshow", "-i", "audio=Microphone"]

        cmd = [
            "ffmpeg", "-y",
            # Video input from pipe
            "-f", "rawvideo",
            "-vcodec", "rawvideo",
            "-pix_fmt", "bgr24",
            "-s", f"{self._frame_width}x{self._frame_height}",
            "-r", str(fps),
            "-i", "-",
            # Audio input from microphone
            *audio_input,
            # Output settings
            "-c:v", "libx264",
            "-preset", "ultrafast",
            "-c:a", "aac",
            "-shortest",
            str(self.output_path),
        ]

        try:
            self._ffmpeg_process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except FileNotFoundError:
            self.error.emit("ffmpeg not found. Please install ffmpeg.")
            self._ffmpeg_process = None

    def _stop_ffmpeg(self) -> None:
        """Stop ffmpeg subprocess."""
        if self._ffmpeg_process is not None:
            try:
                self._ffmpeg_process.stdin.close()
                self._ffmpeg_process.wait(timeout=5)
            except Exception:
                self._ffmpeg_process.terminate()
            self.finished_recording.emit(str(self.output_path))
            self._ffmpeg_process = None

    def start_recording(self, output_path: Path) -> None:
        """Start recording frames to a video file with audio."""
        self.output_path = output_path
        self._recording = True
        self._start_time = time.time()
        self._start_ffmpeg(30.0)

    def stop_recording(self) -> None:
        """Stop recording."""
        self._recording = False
        self._stop_ffmpeg()

    def stop(self) -> None:
        """Stop the worker thread."""
        self._running = False
        self._recording = False


class VideoPlayerWorker(QThread):
    """
    Worker thread for playing a reference video in sync with the recording.

    Signals:
        frame_ready: Emitted with (BGR frame, pose) for each frame.
        finished: Emitted when the video ends.
    """

    frame_ready = Signal(np.ndarray, object)  # frame, pose
    finished = Signal()

    def __init__(
        self,
        video_path: Path,
        poses: List[np.ndarray | None] | None = None,
        fps: float = 30.0,
    ):
        super().__init__()
        self.video_path = video_path
        self.poses = poses or []
        self.fps = fps
        self._running = False
        self._paused = True
        self._start_time = 0.0

    def run(self) -> None:
        """Main thread loop: read frames and emit at correct timing."""
        cap = cv2.VideoCapture(str(self.video_path))
        if not cap.isOpened():
            return

        self._running = True
        frame_duration = 1.0 / self.fps
        frame_idx = 0

        while self._running:
            if self._paused:
                time.sleep(0.01)
                continue

            ret, frame = cap.read()
            if not ret:
                self.finished.emit()
                break

            # Get corresponding pose
            pose = self.poses[frame_idx] if frame_idx < len(self.poses) else None

            # Emit frame
            self.frame_ready.emit(frame, pose)

            frame_idx += 1

            # Synchronize to real time
            elapsed = time.time() - self._start_time
            expected_time = frame_idx * frame_duration
            sleep_time = expected_time - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

        cap.release()

    def play(self) -> None:
        """Start/resume playback."""
        self._paused = False
        self._start_time = time.time()

    def pause(self) -> None:
        """Pause playback."""
        self._paused = True

    def stop(self) -> None:
        """Stop the worker thread."""
        self._running = False
        self._paused = False


class PoseLoadWorker(QThread):
    """
    Worker thread for loading and caching poses from a video.
    Prevents UI from freezing during heavy processing.
    """

    finished = Signal(object)  # PoseCacher instance
    progress = Signal(str)  # status message
    error = Signal(str)

    def __init__(self, video_path: Path, model_complexity: int = 1):
        super().__init__()
        self.video_path = video_path
        self.model_complexity = model_complexity

    def run(self) -> None:
        try:
            from ..core import PoseCacher

            self.progress.emit("Loading video...")
            cacher = PoseCacher(model_complexity=self.model_complexity)
            cacher.load_video(self.video_path)
            self.finished.emit(cacher)
        except Exception as e:
            self.error.emit(str(e))


class PipelineWorker(QThread):
    """
    Worker thread for running the comparison pipeline.
    Prevents UI from freezing during heavy processing.
    """

    finished = Signal(dict)  # pipeline results
    progress = Signal(str)  # status message
    error = Signal(str)

    def __init__(
        self,
        video_a: Path,
        video_b: Path,
        output_dir: Path,
        align_method: str = "chroma",
        target_height: int = 720,
    ):
        super().__init__()
        self.video_a = video_a
        self.video_b = video_b
        self.output_dir = output_dir
        self.align_method = align_method
        self.target_height = target_height

    def run(self) -> None:
        try:
            from ..core import run_pipeline

            self.progress.emit("Extracting audio...")
            results = run_pipeline(
                video_a=self.video_a,
                video_b=self.video_b,
                output_dir=self.output_dir,
                align_method=self.align_method,
                target_height=self.target_height,
            )
            self.finished.emit(results)
        except Exception as e:
            self.error.emit(str(e))


class ResultVideoPlayer(QThread):
    """
    Worker thread for playing the comparison video with score overlay.
    """

    frame_ready = Signal(np.ndarray, float, float)  # frame, score, elapsed_time
    finished = Signal(float)  # average score
    progress = Signal(int, int)  # current_frame, total_frames

    def __init__(
        self,
        video_path: Path,
        scores: List[float],
        fps: float = 30.0,
    ):
        super().__init__()
        self.video_path = video_path
        self.scores = scores
        self.fps = fps
        self._running = False
        self._paused = False

    def run(self) -> None:
        cap = cv2.VideoCapture(str(self.video_path))
        if not cap.isOpened():
            return

        self._running = True
        frame_duration = 1.0 / self.fps
        frame_idx = 0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        start_time = time.time()

        while self._running:
            if self._paused:
                time.sleep(0.01)
                start_time = time.time() - frame_idx * frame_duration
                continue

            ret, frame = cap.read()
            if not ret:
                break

            # Get score for this frame
            score = self.scores[frame_idx] if frame_idx < len(self.scores) else 0.0
            elapsed = frame_idx * frame_duration

            # Emit frame with score
            self.frame_ready.emit(frame, score, elapsed)
            self.progress.emit(frame_idx + 1, total_frames)

            frame_idx += 1

            # Synchronize to real time
            actual_elapsed = time.time() - start_time
            expected_time = frame_idx * frame_duration
            sleep_time = expected_time - actual_elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

        cap.release()

        # Calculate and emit average score
        avg_score = sum(self.scores) / len(self.scores) if self.scores else 0.0
        self.finished.emit(avg_score)

    def play(self) -> None:
        self._paused = False

    def pause(self) -> None:
        self._paused = True

    def stop(self) -> None:
        self._running = False
