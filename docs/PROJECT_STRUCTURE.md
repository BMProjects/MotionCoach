> 本文描述的是旧版 DanceBattle 舞蹈演示（`src/dancebattle`）。当前 MotionCoach 的整体架构见 [ARCHITECTURE.md](ARCHITECTURE.md)。

# DanceBattle 项目结构

## 目录结构

```
DanceBattle/
├── README.md                 # 项目说明
├── CHANGELOG.md              # 版本历史
├── LICENSE                   # MIT 许可证
├── pyproject.toml / uv.lock   # 依赖（uv 管理）
│
├── main.py                   # CLI 入口 (python main.py)
├── main_gui.py               # GUI 入口 (python main_gui.py)
│
├── assets/                   # 示例视频
│   ├── chuu.mov
│   └── cyves.mov
│
├── docs/                     # 文档
│   └── README.md             # 详细文档和 API 参考
│
├── output/                   # 输出目录 (gitignored)
│
├── tests/                    # 测试
│   ├── test_mediapipe_pipeline.py
│   └── test_yolo_pipeline.py
│
└── src/dancebattle/          # 主包
    ├── __init__.py           # 包入口，重新导出 core API
    ├── cli.py                # 命令行接口
    │
    ├── core/                 # ⭐ 核心算法（平台无关）
    │   ├── __init__.py       # 公共 API 导出
    │   ├── audio_aligner.py  # 音频对齐 (chroma/onset)
    │   ├── pipeline.py       # 端到端处理流程
    │   ├── visualizer.py     # 视频可视化和骨骼绘制
    │   ├── utils.py          # PoseCacher 等工具
    │   │
    │   ├── pose_estimation/  # 姿态估计器
    │   │   ├── __init__.py
    │   │   ├── base_estimator.py      # 抽象基类
    │   │   ├── mediapipe_estimator.py # MediaPipe 实现
    │   │   └── yolo_estimator.py      # YOLO 实现
    │   │
    │   └── similarity_analysis/       # 相似度分析
    │       ├── __init__.py
    │       ├── base_analyzer.py       # 抽象基类
    │       └── procrustes_analyzer.py # Procrustes 实现
    │
    └── gui/                  # 🖥️ PC 演示前端 (PySide6)
        ├── __init__.py
        ├── app.py            # 主窗口和 UI 组件
        └── workers.py        # 后台线程 (摄像头、视频播放)
```

## 算法流程图

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              📥 输入层                                       │
│  ┌─────────────────────┐              ┌─────────────────────┐               │
│  │    参考视频          │              │    用户视频          │               │
│  │   (Reference)       │              │  (User Recording)   │               │
│  └──────────┬──────────┘              └──────────┬──────────┘               │
└─────────────┼────────────────────────────────────┼──────────────────────────┘
              │                                    │
              ▼                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           🎵 音频对齐模块                                    │
│                                                                             │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐          │
│  │   音频提取       │───▶│  Chroma 特征    │───▶│   互相关对齐    │          │
│  │ (MoviePy/FFmpeg)│    │   (librosa)     │    │ (scipy.correlate)│         │
│  └─────────────────┘    └─────────────────┘    └────────┬────────┘          │
│                                                         │                   │
│                                                         ▼                   │
│                                              ┌─────────────────┐            │
│                                              │ 时间偏移量 Δt   │            │
│                                              └────────┬────────┘            │
└───────────────────────────────────────────────────────┼─────────────────────┘
                                                        │
              ┌─────────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           🦴 姿态估计模块                                    │
│                                                                             │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐          │
│  │  MediaPipe Pose │───▶│   逐帧骨骼提取   │───▶│   姿态序列缓存   │         │
│  │  (33 关键点)     │    │                 │    │                 │          │
│  └─────────────────┘    └─────────────────┘    └────────┬────────┘          │
└───────────────────────────────────────────────────────┬─────────────────────┘
                                                        │
              ┌─────────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          📊 相似度计算模块                                   │
│                                                                             │
│  ┌─────────────────┐    ┌─────────────────────┐    ┌─────────────────┐      │
│  │   帧对齐        │───▶│ Procrustes 标准化   │───▶│    残差计算     │      │
│  │  (基于 Δt)      │    │ (平移+缩放+旋转)    │    │  → 相似度分数   │      │
│  └─────────────────┘    └─────────────────────┘    └────────┬────────┘      │
└───────────────────────────────────────────────────────────┬─────────────────┘
                                                            │
              ┌───────────────────┬─────────────────────────┼─────────────────┐
              │                   │                         │                 │
              ▼                   ▼                         ▼                 │
┌─────────────────────────────────────────────────────────────────────────────┐
│                              📤 输出层                                       │
│                                                                             │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐          │
│  │   实时分数显示   │    │   并排对比视频   │    │   骨骼叠加绘制   │         │
│  │                 │    │                 │    │                 │          │
│  └─────────────────┘    └─────────────────┘    └─────────────────┘          │
└─────────────────────────────────────────────────────────────────────────────┘
```

## 模块职责

### `dancebattle.core` - 核心算法
**设计原则**: 平台无关，可移植到移动端

| 模块 | 职责 |
|------|------|
| `audio_aligner.py` | 音频对齐算法 (CQT chroma / onset cross-correlation) |
| `pipeline.py` | 完整处理流程：音频对齐 → 姿态提取 → 相似度计算 → 视频导出 |
| `visualizer.py` | 并排视频渲染、骨骼叠加绘制 |
| `utils.py` | `PoseCacher` - 预计算参考视频姿态用于实时对比 |
| `pose_estimation/` | 姿态估计器：MediaPipe (轻量) / YOLO (高精度) |
| `similarity_analysis/` | 相似度分析：Procrustes 几何形状匹配 |

### `dancebattle.gui` - PC 演示前端
**设计原则**: 仅用于 PC 测试，不用于生产

| 模块 | 职责 |
|------|------|
| `app.py` | PySide6 主窗口、页面切换、状态机 |
| `workers.py` | `CameraWorker` (摄像头+ffmpeg录制)、`VideoPlayerWorker` (参考视频播放) |

### `dancebattle.cli` - 命令行接口
批处理模式，用于对比两个已有视频。

## 运行方式

### 方式 1: 命令行
```bash
# 使用 main.py 入口
python main.py --video-a assets/chuu.mov --video-b assets/cyves.mov

# 或使用模块方式
python -m dancebattle.cli --video-a ref.mp4 --video-b user.mp4
```

### 方式 2: GUI 演示
```bash
# 使用 main_gui.py 入口
python main_gui.py
```

### 方式 3: IDE 直接运行
在 IDE 中直接运行以下文件：
- `src/dancebattle/gui/app.py` - GUI 应用
- `src/dancebattle/cli.py` - CLI (需要命令行参数)
- `tests/test_mediapipe_pipeline.py` - 测试脚本

### 方式 4: 作为库使用
```python
from dancebattle.core import (
    run_pipeline,
    MediaPipePoseEstimator,
    ProcrustesSimilarityAnalyzer,
    PoseCacher,
)

# 完整流程
results = run_pipeline(video_a="ref.mp4", video_b="user.mp4")

# 实时对比
estimator = MediaPipePoseEstimator()
analyzer = ProcrustesSimilarityAnalyzer()
pose = estimator.estimate(frame)
score = analyzer.compare(pose, reference_pose)
```

## 依赖关系

```
dancebattle (主包)
    └── core (核心算法)
            ├── opencv-python
            ├── mediapipe
            ├── numpy
            ├── scipy
            ├── librosa
            ├── moviepy
            └── tqdm

    └── gui (PC 演示) [可选]
            ├── PySide6
            └── ffmpeg (外部)

    └── cli (命令行)
            └── (无额外依赖)
```

## 移动端集成指南

`dancebattle.core` 模块可直接用于移动应用：

1. **姿态估计**: 
   - iOS: 使用 MediaPipe iOS SDK 或 CoreML
   - Android: 使用 MediaPipe Android SDK 或 TFLite

2. **相似度计算**: 
   - `ProcrustesSimilarityAnalyzer` 是纯 NumPy/SciPy
   - 可移植或用其他语言重写

3. **音频对齐**: 
   - librosa 依赖较重
   - 移动端可简化或使用原生音频库

4. **实时流程**:
   ```
   摄像头帧 → 姿态估计 → 与预计算参考对比 → 显示分数
   ```
