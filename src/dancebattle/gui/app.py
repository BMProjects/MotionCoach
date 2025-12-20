"""
DanceBattle GUI Application.

Provides an interactive dance comparison experience with:
- Video conference style layout (user full screen, reference PiP)
- 3-2-1 countdown before recording
- Real-time similarity scoring
- Final comparison video generation
"""

import sys
import time
from pathlib import Path
from enum import Enum, auto

import cv2
import numpy as np
from PySide6.QtCore import Qt, QTimer, Slot, Signal
from PySide6.QtGui import QImage, QPixmap, QFont, QPainter, QColor
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLabel,
    QFileDialog,
    QStackedWidget,
    QProgressBar,
    QMessageBox,
)
from PySide6.QtMultimedia import QMediaPlayer, QAudioOutput
from PySide6.QtCore import QUrl

from ..core import PoseCacher, run_pipeline, draw_pose
from .workers import (
    CameraWorker,
    VideoPlayerWorker,
    PoseLoadWorker,
    PipelineWorker,
    ResultVideoPlayer,
)


class AppState(Enum):
    """Application state machine states."""
    SETUP = auto()
    COUNTDOWN = auto()
    RECORDING = auto()
    PROCESSING = auto()
    RESULTS = auto()


def cv2_to_qimage(frame: np.ndarray) -> QImage:
    """Convert OpenCV BGR frame to QImage."""
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb.shape
    bytes_per_line = ch * w
    return QImage(rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)


class VideoWidget(QLabel):
    """Widget for displaying video frames."""

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(320, 240)
        self.setStyleSheet("background-color: #1a1a2e; border-radius: 8px;")

    def set_frame(self, frame: np.ndarray, pose: np.ndarray | None = None) -> None:
        """Display a frame, optionally with pose overlay."""
        display_frame = frame.copy()
        if pose is not None:
            draw_pose(display_frame, pose)

        qimg = cv2_to_qimage(display_frame)
        pixmap = QPixmap.fromImage(qimg)
        scaled = pixmap.scaled(
            self.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.setPixmap(scaled)


class OverlayWidget(QWidget):
    """Transparent overlay for countdown and score display."""

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)

        self._countdown_text = ""
        self._score = 0.0
        self._elapsed_time = 0.0
        self._show_score = False

    def set_countdown(self, text: str) -> None:
        self._countdown_text = text
        self._show_score = False
        self.update()

    def set_score(self, score: float, elapsed: float) -> None:
        self._score = score
        self._elapsed_time = elapsed
        self._show_score = True
        self._countdown_text = ""
        self.update()

    def clear(self) -> None:
        self._countdown_text = ""
        self._show_score = False
        self.update()

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        if self._countdown_text:
            # Draw large countdown number
            font = QFont("Arial", 200, QFont.Weight.Bold)
            painter.setFont(font)
            painter.setPen(QColor(255, 255, 255, 230))

            rect = self.rect()
            painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, self._countdown_text)

        if self._show_score:
            # Draw score box at bottom
            font = QFont("Arial", 24, QFont.Weight.Bold)
            painter.setFont(font)

            # Background box
            box_h = 60
            box_y = self.height() - box_h - 20
            painter.fillRect(20, box_y, self.width() - 40, box_h, QColor(0, 0, 0, 180))

            # Score text
            painter.setPen(QColor(255, 255, 255))
            score_text = f"Similarity: {self._score:.1f}%"
            time_text = f"Time: {self._elapsed_time:.1f}s"

            painter.drawText(40, box_y + 40, score_text)
            painter.drawText(self.width() - 200, box_y + 40, time_text)

        painter.end()


class SetupPage(QWidget):
    """Setup page for selecting reference video."""

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._init_ui()
        self.reference_path: Path | None = None
        self.user_video_path: Path | None = None  # For direct comparison mode

    def _init_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.setSpacing(20)

        # Title
        title = QLabel("🎵 DanceBattle")
        title.setFont(QFont("Arial", 48, QFont.Weight.Bold))
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet("color: #e94560;")
        layout.addWidget(title)

        # Subtitle
        subtitle = QLabel("Dance along with your favorite videos!")
        subtitle.setFont(QFont("Arial", 16))
        subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        subtitle.setStyleSheet("color: #eeeeee;")
        layout.addWidget(subtitle)

        layout.addSpacing(40)

        # File selection
        self.file_label = QLabel("No reference video selected")
        self.file_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.file_label.setStyleSheet("color: #888; font-size: 14px;")
        layout.addWidget(self.file_label)

        select_btn = QPushButton("📁 Select Reference Video")
        select_btn.setFont(QFont("Arial", 14))
        select_btn.setMinimumSize(300, 50)
        select_btn.setStyleSheet("""
            QPushButton {
                background-color: #16213e;
                color: white;
                border: 2px solid #e94560;
                border-radius: 8px;
            }
            QPushButton:hover {
                background-color: #0f3460;
            }
        """)
        select_btn.clicked.connect(self._select_file)
        layout.addWidget(select_btn, alignment=Qt.AlignmentFlag.AlignCenter)

        layout.addSpacing(20)

        # Start button
        self.start_btn = QPushButton("🚀 Start Dancing!")
        self.start_btn.setFont(QFont("Arial", 18, QFont.Weight.Bold))
        self.start_btn.setMinimumSize(300, 60)
        self.start_btn.setEnabled(False)
        self.start_btn.setStyleSheet("""
            QPushButton {
                background-color: #e94560;
                color: white;
                border: none;
                border-radius: 10px;
            }
            QPushButton:hover {
                background-color: #ff6b6b;
            }
            QPushButton:disabled {
                background-color: #444;
                color: #888;
            }
        """)
        layout.addWidget(self.start_btn, alignment=Qt.AlignmentFlag.AlignCenter)

        layout.addSpacing(10)

        # Compare button (skip recording)
        self.compare_btn = QPushButton("📹 Compare Existing Videos")
        self.compare_btn.setFont(QFont("Arial", 14))
        self.compare_btn.setMinimumSize(300, 50)
        self.compare_btn.setEnabled(False)
        self.compare_btn.setStyleSheet("""
            QPushButton {
                background-color: #16213e;
                color: white;
                border: 2px solid #4ecca3;
                border-radius: 8px;
            }
            QPushButton:hover {
                background-color: #0f3460;
            }
            QPushButton:disabled {
                background-color: #222;
                color: #666;
                border-color: #444;
            }
        """)
        self.compare_btn.clicked.connect(self._select_user_video)
        layout.addWidget(self.compare_btn, alignment=Qt.AlignmentFlag.AlignCenter)

    def _select_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Reference Video",
            str(Path("assets").absolute()),
            "Video Files (*.mp4 *.mov *.avi *.mkv)",
        )
        if path:
            self.reference_path = Path(path)
            self.file_label.setText(f"✓ {self.reference_path.name}")
            self.file_label.setStyleSheet("color: #4ecca3; font-size: 14px;")
            self.start_btn.setEnabled(True)
            self.compare_btn.setEnabled(True)

    def _select_user_video(self) -> None:
        """Select user's video for direct comparison (skip recording)."""
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Your Dance Video",
            str(Path.home()),
            "Video Files (*.mp4 *.mov *.avi *.mkv)",
        )
        if path:
            self.user_video_path = Path(path)


class RecordingPage(QWidget):
    """Recording page with camera view and reference PiP."""

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._init_ui()

    def _init_ui(self) -> None:
        # Main layout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Video container (for overlay positioning)
        self.video_container = QWidget()
        self.video_container.setStyleSheet("background-color: #0f0f23;")
        container_layout = QVBoxLayout(self.video_container)
        container_layout.setContentsMargins(0, 0, 0, 0)

        # Main camera view
        self.camera_view = VideoWidget()
        self.camera_view.setMinimumSize(800, 600)
        container_layout.addWidget(self.camera_view)

        layout.addWidget(self.video_container, stretch=1)

        # Overlay
        self.overlay = OverlayWidget(self.video_container)

        # PiP reference view (positioned in top-right)
        self.pip_view = VideoWidget(self.video_container)
        self.pip_view.setFixedSize(240, 180)
        self.pip_view.setStyleSheet(
            "background-color: #1a1a2e; border: 2px solid #e94560; border-radius: 8px;"
        )

        # Control bar
        control_bar = QHBoxLayout()
        control_bar.setContentsMargins(20, 10, 20, 20)

        self.stop_btn = QPushButton("⏹ Stop Recording")
        self.stop_btn.setFont(QFont("Arial", 14, QFont.Weight.Bold))
        self.stop_btn.setMinimumSize(200, 50)
        self.stop_btn.setStyleSheet("""
            QPushButton {
                background-color: #e94560;
                color: white;
                border: none;
                border-radius: 8px;
            }
            QPushButton:hover {
                background-color: #ff6b6b;
            }
        """)
        control_bar.addStretch()
        control_bar.addWidget(self.stop_btn)
        control_bar.addStretch()

        layout.addLayout(control_bar)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        # Reposition overlay
        self.overlay.setGeometry(self.video_container.geometry())
        # Reposition PiP
        margin = 20
        self.pip_view.move(
            self.video_container.width() - self.pip_view.width() - margin,
            margin,
        )


class ResultsPage(QWidget):
    """Results page showing comparison video playback with scores."""

    playback_finished = Signal()  # Emitted when video playback finishes

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._init_ui()
        self.scores: list[float] = []
        self.output_path: Path | None = None
        self._showing_final = False

    def _init_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Video display area
        self.video_container = QWidget()
        self.video_container.setStyleSheet("background-color: #0f0f23;")
        video_layout = QVBoxLayout(self.video_container)
        video_layout.setContentsMargins(0, 0, 0, 0)

        self.video_view = VideoWidget()
        self.video_view.setMinimumSize(800, 500)
        video_layout.addWidget(self.video_view)

        layout.addWidget(self.video_container, stretch=1)

        # Score overlay (on video)
        self.score_overlay = OverlayWidget(self.video_container)

        # Info bar
        info_bar = QWidget()
        info_bar.setStyleSheet("background-color: #1a1a2e;")
        info_layout = QVBoxLayout(info_bar)
        info_layout.setContentsMargins(20, 15, 20, 15)

        # Title and score
        self.title_label = QLabel("🎬 Playing Comparison...")
        self.title_label.setFont(QFont("Arial", 20, QFont.Weight.Bold))
        self.title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.title_label.setStyleSheet("color: #4ecca3;")
        info_layout.addWidget(self.title_label)

        self.score_label = QLabel("Current Score: --")
        self.score_label.setFont(QFont("Arial", 16))
        self.score_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.score_label.setStyleSheet("color: #eeeeee;")
        info_layout.addWidget(self.score_label)

        # Progress bar
        self.progress = QProgressBar()
        self.progress.setMinimum(0)
        self.progress.setMaximum(100)
        self.progress.setStyleSheet("""
            QProgressBar {
                border: 2px solid #16213e;
                border-radius: 5px;
                background-color: #0f0f23;
                height: 10px;
            }
            QProgressBar::chunk {
                background-color: #e94560;
            }
        """)
        info_layout.addWidget(self.progress)

        layout.addWidget(info_bar)

        # Button bar
        btn_bar = QWidget()
        btn_bar.setStyleSheet("background-color: #0f0f23;")
        btn_layout = QHBoxLayout(btn_bar)
        btn_layout.setContentsMargins(20, 10, 20, 20)

        self.open_btn = QPushButton("📂 Open Folder")
        self.open_btn.setFont(QFont("Arial", 14))
        self.open_btn.setMinimumSize(150, 45)
        self.open_btn.setStyleSheet("""
            QPushButton {
                background-color: #16213e;
                color: white;
                border: 2px solid #4ecca3;
                border-radius: 8px;
            }
            QPushButton:hover {
                background-color: #0f3460;
            }
        """)
        btn_layout.addWidget(self.open_btn)

        btn_layout.addStretch()

        self.restart_btn = QPushButton("🔄 Dance Again")
        self.restart_btn.setFont(QFont("Arial", 14, QFont.Weight.Bold))
        self.restart_btn.setMinimumSize(150, 45)
        self.restart_btn.setStyleSheet("""
            QPushButton {
                background-color: #e94560;
                color: white;
                border: none;
                border-radius: 8px;
            }
            QPushButton:hover {
                background-color: #ff6b6b;
            }
        """)
        btn_layout.addWidget(self.restart_btn)

        layout.addWidget(btn_bar)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self.score_overlay.setGeometry(self.video_container.geometry())

    def set_results(self, scores: list[float], output_path: Path) -> None:
        """Set results and prepare for playback."""
        self.scores = scores
        self.output_path = output_path
        self._showing_final = False
        self.title_label.setText("🎬 Playing Comparison...")
        self.score_label.setText("Current Score: --")
        self.progress.setValue(0)

    def update_frame(self, frame: np.ndarray, score: float, elapsed: float) -> None:
        """Update display with current frame and score."""
        self.video_view.set_frame(frame)
        self.score_label.setText(f"Current Score: {score:.1f}%")
        self.score_overlay.set_score(score, elapsed)

    def update_progress(self, current: int, total: int) -> None:
        """Update progress bar."""
        if total > 0:
            self.progress.setValue(int(current * 100 / total))

    def show_final_score(self, avg_score: float) -> None:
        """Show final average score after playback."""
        self._showing_final = True
        self.title_label.setText("🎉 Comparison Complete!")
        self.score_label.setText(f"Average Score: {avg_score:.1f}%")
        self.score_overlay.clear()
        self.progress.setValue(100)
        self.playback_finished.emit()



class ProcessingPage(QWidget):
    """Processing page with progress indicator."""

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._init_ui()

    def _init_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # Spinner/text
        self.status_label = QLabel("🔄 Processing...")
        self.status_label.setFont(QFont("Arial", 24))
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.status_label.setStyleSheet("color: #eeeeee;")
        layout.addWidget(self.status_label)

        # Progress bar
        self.progress = QProgressBar()
        self.progress.setMinimum(0)
        self.progress.setMaximum(0)  # Indeterminate
        self.progress.setMinimumWidth(400)
        self.progress.setStyleSheet("""
            QProgressBar {
                border: 2px solid #16213e;
                border-radius: 5px;
                background-color: #0f0f23;
            }
            QProgressBar::chunk {
                background-color: #e94560;
            }
        """)
        layout.addWidget(self.progress, alignment=Qt.AlignmentFlag.AlignCenter)


class MainWindow(QMainWindow):
    """Main application window."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("DanceBattle")
        self.setMinimumSize(1024, 768)
        self.setStyleSheet("background-color: #0f0f23;")

        self._state = AppState.SETUP
        self._pose_cacher: PoseCacher | None = None
        self._camera_worker: CameraWorker | None = None
        self._video_player: VideoPlayerWorker | None = None
        self._pose_load_worker: PoseLoadWorker | None = None
        self._pipeline_worker: PipelineWorker | None = None
        self._result_player: ResultVideoPlayer | None = None
        self._countdown_timer: QTimer | None = None
        self._countdown_value = 3
        self._scores: list[float] = []
        self._start_time = 0.0
        self._output_dir = Path("output")
        self._temp_video_path = self._output_dir / "user_recording.mp4"

        # Audio player for reference video sound
        self._audio_output = QAudioOutput()
        self._audio_player = QMediaPlayer()
        self._audio_player.setAudioOutput(self._audio_output)

        self._init_ui()
        self._connect_signals()

    def _init_ui(self) -> None:
        # Central widget with stacked pages
        self.central = QStackedWidget()
        self.setCentralWidget(self.central)

        # Pages
        self.setup_page = SetupPage()
        self.recording_page = RecordingPage()
        self.processing_page = ProcessingPage()
        self.results_page = ResultsPage()

        self.central.addWidget(self.setup_page)
        self.central.addWidget(self.recording_page)
        self.central.addWidget(self.processing_page)
        self.central.addWidget(self.results_page)

    def _connect_signals(self) -> None:
        self.setup_page.start_btn.clicked.connect(self._on_start_clicked)
        self.setup_page.compare_btn.clicked.connect(self._on_compare_clicked)
        self.recording_page.stop_btn.clicked.connect(self._on_stop_clicked)
        self.results_page.restart_btn.clicked.connect(self._on_restart_clicked)
        self.results_page.open_btn.clicked.connect(self._on_open_folder_clicked)

    def _on_start_clicked(self) -> None:
        """Handle start button: load reference and start countdown."""
        ref_path = self.setup_page.reference_path
        if not ref_path or not ref_path.exists():
            QMessageBox.warning(self, "Error", "Please select a valid reference video.")
            return

        # Pre-compute reference poses in background thread
        self.processing_page.status_label.setText("🎬 Loading reference video...")
        self.central.setCurrentWidget(self.processing_page)

        self._pose_load_worker = PoseLoadWorker(ref_path)
        self._pose_load_worker.finished.connect(self._on_pose_load_finished)
        self._pose_load_worker.error.connect(self._on_pose_load_error)
        self._pose_load_worker.start()

    @Slot(object)
    def _on_pose_load_finished(self, cacher: PoseCacher) -> None:
        """Handle pose loading completion."""
        self._pose_cacher = cacher
        # Switch to recording page and start countdown
        self.central.setCurrentWidget(self.recording_page)
        self._start_countdown()

    @Slot(str)
    def _on_pose_load_error(self, error_msg: str) -> None:
        """Handle pose loading error."""
        QMessageBox.critical(self, "Error", f"Failed to load video: {error_msg}")
        self.central.setCurrentWidget(self.setup_page)

    def _start_countdown(self) -> None:
        """Start the 3-2-1 countdown."""
        self._state = AppState.COUNTDOWN
        self._countdown_value = 3
        self.recording_page.overlay.set_countdown(str(self._countdown_value))
        self.recording_page.stop_btn.setEnabled(False)

        # Start camera preview
        self._start_camera_preview()

        # Countdown timer
        self._countdown_timer = QTimer()
        self._countdown_timer.timeout.connect(self._countdown_tick)
        self._countdown_timer.start(1000)

    def _countdown_tick(self) -> None:
        """Handle countdown timer tick."""
        self._countdown_value -= 1

        if self._countdown_value > 0:
            self.recording_page.overlay.set_countdown(str(self._countdown_value))
        elif self._countdown_value == 0:
            self.recording_page.overlay.set_countdown("GO!")
        else:
            # Countdown finished, start recording
            if self._countdown_timer:
                self._countdown_timer.stop()
            self.recording_page.overlay.clear()
            self._start_video_player()
            self._start_recording()

    def _start_camera_preview(self) -> None:
        """Initialize camera worker for preview."""
        if self._camera_worker is not None:
            self._camera_worker.stop()
            self._camera_worker.wait()

        self._output_dir.mkdir(parents=True, exist_ok=True)

        ref_poses = self._pose_cacher.poses if self._pose_cacher else []
        ref_fps = self._pose_cacher.fps if self._pose_cacher else 30.0

        self._camera_worker = CameraWorker(
            camera_id=0,
            reference_poses=ref_poses,
            reference_fps=ref_fps,
        )
        self._camera_worker.frame_ready.connect(self._on_frame_ready)
        self._camera_worker.error.connect(self._on_camera_error)
        self._camera_worker.start()

    def _start_recording(self) -> None:
        """Start the actual recording phase."""
        self._state = AppState.RECORDING
        self._scores = []
        self._start_time = time.time()
        self.recording_page.stop_btn.setEnabled(True)

        if self._camera_worker:
            self._camera_worker.start_recording(self._temp_video_path)

    @Slot(np.ndarray, object, float)
    def _on_frame_ready(self, frame: np.ndarray, pose: object, score: float) -> None:
        """Handle new frame from camera worker."""
        # Update camera view
        self.recording_page.camera_view.set_frame(frame, pose)

        # Update score overlay during recording
        if self._state == AppState.RECORDING:
            elapsed = time.time() - self._start_time
            self._scores.append(score)
            self.recording_page.overlay.set_score(score, elapsed)



    @Slot(str)
    def _on_camera_error(self, error_msg: str) -> None:
        """Handle camera error."""
        QMessageBox.critical(self, "Camera Error", error_msg)
        self.central.setCurrentWidget(self.setup_page)

    def _on_stop_clicked(self) -> None:
        """Handle stop recording button."""
        if self._camera_worker:
            self._camera_worker.stop_recording()
            self._camera_worker.stop()
            self._camera_worker.wait()

        if self._video_player:
            self._video_player.stop()
            self._video_player.wait()

        # Stop audio playback
        self._audio_player.stop()

        self._state = AppState.PROCESSING
        self.central.setCurrentWidget(self.processing_page)
        self.processing_page.status_label.setText("🎬 Generating comparison video...")

        # Run pipeline in background thread
        ref_path = self.setup_page.reference_path
        self._pipeline_worker = PipelineWorker(
            video_a=ref_path,
            video_b=self._temp_video_path,
            output_dir=self._output_dir,
        )
        self._pipeline_worker.finished.connect(self._on_pipeline_finished)
        self._pipeline_worker.error.connect(self._on_pipeline_error)
        self._pipeline_worker.start()

    def _on_compare_clicked(self) -> None:
        """Handle compare button: skip recording and directly compare videos."""
        user_video = self.setup_page.user_video_path
        ref_path = self.setup_page.reference_path

        if not user_video or not user_video.exists():
            QMessageBox.warning(self, "Error", "Please select a valid user video.")
            return

        if not ref_path or not ref_path.exists():
            QMessageBox.warning(self, "Error", "Please select a valid reference video.")
            return

        self._state = AppState.PROCESSING
        self.central.setCurrentWidget(self.processing_page)
        self.processing_page.status_label.setText("🎬 Comparing videos...")

        # Run pipeline in background thread
        self._pipeline_worker = PipelineWorker(
            video_a=ref_path,
            video_b=user_video,
            output_dir=self._output_dir,
        )
        self._pipeline_worker.finished.connect(self._on_pipeline_finished)
        self._pipeline_worker.error.connect(self._on_pipeline_error)
        self._pipeline_worker.start()

    @Slot(dict)
    def _on_pipeline_finished(self, results: dict) -> None:
        """Handle pipeline completion - start result video playback."""
        self._state = AppState.RESULTS
        self._scores = results.get("scores", [])
        output_video = results["output_video"]

        # Set up results page
        self.results_page.set_results(self._scores, output_video)
        self.central.setCurrentWidget(self.results_page)

        # Start playing the comparison video
        self._start_result_playback(output_video, self._scores)

    @Slot(str)
    def _on_pipeline_error(self, error_msg: str) -> None:
        """Handle pipeline error."""
        QMessageBox.critical(self, "Error", f"Failed to generate video: {error_msg}")
        self.central.setCurrentWidget(self.setup_page)

    def _start_result_playback(self, video_path: Path, scores: list[float]) -> None:
        """Start playing the result video with score overlay."""
        if self._result_player is not None:
            self._result_player.stop()
            self._result_player.wait()

        # Get video FPS
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        cap.release()

        self._result_player = ResultVideoPlayer(
            video_path=video_path,
            scores=scores,
            fps=fps,
        )
        self._result_player.frame_ready.connect(self._on_result_frame_ready)
        self._result_player.progress.connect(self._on_result_progress)
        self._result_player.finished.connect(self._on_result_playback_finished)
        self._result_player.start()

    @Slot(np.ndarray, float, float)
    def _on_result_frame_ready(self, frame: np.ndarray, score: float, elapsed: float) -> None:
        """Handle result video frame."""
        self.results_page.update_frame(frame, score, elapsed)

    @Slot(int, int)
    def _on_result_progress(self, current: int, total: int) -> None:
        """Handle result video progress."""
        self.results_page.update_progress(current, total)

    @Slot(float)
    def _on_result_playback_finished(self, avg_score: float) -> None:
        """Handle result video playback finished."""
        self.results_page.show_final_score(avg_score)

    def _on_restart_clicked(self) -> None:
        """Handle restart button."""
        # Stop result player if running
        if self._result_player is not None:
            self._result_player.stop()
            self._result_player.wait()

        self._state = AppState.SETUP
        self.setup_page.user_video_path = None  # Reset user video selection
        self.central.setCurrentWidget(self.setup_page)

    def _on_open_folder_clicked(self) -> None:
        """Open the output folder in file manager."""
        import subprocess
        import platform

        path = str(self._output_dir.absolute())
        if platform.system() == "Windows":
            subprocess.run(["explorer", path])
        elif platform.system() == "Darwin":
            subprocess.run(["open", path])
        else:
            subprocess.run(["xdg-open", path])

    def _start_video_player(self) -> None:
        """Start the reference video player with audio."""
        ref_path = self.setup_page.reference_path
        if not ref_path or not self._pose_cacher:
            return

        if self._video_player is not None:
            self._video_player.stop()
            self._video_player.wait()

        # Start video frame player (for PiP display)
        self._video_player = VideoPlayerWorker(
            video_path=ref_path,
            poses=self._pose_cacher.poses,
            fps=self._pose_cacher.fps,
        )
        self._video_player.frame_ready.connect(self._on_ref_frame_ready)
        self._video_player.start()
        self._video_player.play()

        # Start audio playback using QMediaPlayer
        self._audio_player.setSource(QUrl.fromLocalFile(str(ref_path.absolute())))
        self._audio_player.play()

    @Slot(np.ndarray, object)
    def _on_ref_frame_ready(self, frame: np.ndarray, pose: object) -> None:
        """Handle new frame from reference video player."""
        self.recording_page.pip_view.set_frame(frame, pose)

    def closeEvent(self, event) -> None:
        """Cleanup on window close."""
        if self._camera_worker:
            self._camera_worker.stop()
            self._camera_worker.wait()
        if self._video_player:
            self._video_player.stop()
            self._video_player.wait()
        if self._result_player:
            self._result_player.stop()
            self._result_player.wait()
        if self._pose_load_worker:
            self._pose_load_worker.wait()
        if self._pipeline_worker:
            self._pipeline_worker.wait()
        event.accept()


def main() -> None:
    """Entry point for the GUI application."""
    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    window = MainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    # Support running directly from IDE without PYTHONPATH setup
    import sys
    from pathlib import Path

    # Add src/ to path for local development
    _project_root = Path(__file__).resolve().parent.parent.parent.parent
    _src_path = _project_root / "src"
    if str(_src_path) not in sys.path:
        sys.path.insert(0, str(_src_path))

    main()
