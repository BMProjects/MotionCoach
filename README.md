# DanceBattle

Audio-aligned dance pose comparison demo. The pipeline extracts audio, aligns two videos, runs MediaPipe pose estimation, scores frame-wise similarity with Procrustes analysis, and exports a side-by-side video annotated with per-frame scores.

## Quickstart
1) Install ffmpeg (moviepy depends on it).  
2) Install dependencies via requirements (YOLO backend also needs torch/torchvision installed for your CPU/CUDA platform):
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```
3) Run the demo with the sample assets:
```bash
python -m dancebattle.cli --video-a assets/chuu.mov --video-b assets/cyves.mov --output-dir output
```
This produces `output/comparison_video.mp4` and prints the detected offset and average similarity.
Defaults already point to the sample assets, so you can also run `python -m dancebattle.cli --output-dir output`.

You can also run the wrapper script (adds `src/` to `PYTHONPATH` for local runs):
```bash
python main.py --video-a assets/chuu.mov --video-b assets/cyves.mov
```

## Project Layout
- `src/dancebattle/` core modules
  - `audio_aligner.py` chroma/onset audio alignment
- `pose_estimation/` MediaPipe pose estimator + YOLO11m-pose estimator
  - `similarity_analysis/` Procrustes similarity scoring
  - `visualizer.py` side-by-side video renderer with skeleton overlay
  - `pipeline.py` end-to-end orchestration
  - `cli.py` argparse CLI entrypoint
- `assets/` sample videos for the demo
- `output/`, `tests/output/` generated artifacts (gitignored)
- `tests/` automated unit tests and diagnostic scripts (visual/chroma/self-alignment)

## Development
- Lint/format: choose your preferred tools (e.g., black/ruff/isort) and add to pre-commit.
- Tests: `pytest` (after `pip install -r requirements.txt`). Synthetic unit tests live in `tests/`，烟雾测试需设置环境变量 RUN_MEDIAPIPE_TEST/RUN_YOLO_TEST。
- 入口：可直接 `python -m dancebattle.cli ...` 或本地 `python main.py ...`（未安装包时会自动添加 `src/`）。
- 版本记录：见 `CHANGELOG.md`，当前计划发布版本 0.1.0。

## 依赖管理
- 依赖统一在 `requirements.txt`，直接 `pip install -r requirements.txt`。
- 高精度 YOLO pose 需额外安装 torch/torchvision（按平台选择 CPU/CUDA wheel），再安装 `ultralytics`（已在 requirements 中列出）。

## 模型与性能
- 默认 pipeline 使用 MediaPipe Pose（BlazePose，`model_complexity=1`）。整体计算量主要来自逐帧姿态推理：音频对齐和 Procrustes 相似度开销很小。桌面 CPU/GPU 在 256–368px 输入下可接近实时；在移动端可跑通但需降低分辨率/FPS，并尽量使用 TFLite GPU delegate。
- 视频解码/重编码的耗时取决于输入分辨率；必要时可下采样或抽帧。
- 高精度版本改为 YOLO11m-pose：CLI 使用 `--pose-backend yolo --yolo-device cuda --yolo-model yolo11m-pose.pt`。该模型对整身关键点精度高但需 GPU；如需更快可降低分辨率或改用 CPU（速度会下降）。
- Pipeline 当前仅在内存中缓存姿态，可按需将关键点序列落盘以复用。

## 许可
- MIT License（见 LICENSE）。

## 致谢 / Acknowledgments
- 灵感来源于 [Mruchus/dance-sync-analysis](https://github.com/Mruchus/dance-sync-analysis)，在此致谢。
