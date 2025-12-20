# Changelog

## 0.2.0 (2025-12-20)

### 新增功能
- **PySide6 GUI 演示应用**：交互式舞蹈对比体验
  - 视频会议风格布局（用户全屏 + 参考视频画中画）
  - 3-2-1 倒计时后开始录制
  - 实时相似度分数显示
  - 参考视频音频同步播放（麦克风采集）
  - 对比视频自动播放并显示实时分数
  - "Compare Existing Videos" 按钮支持跳过录制直接对比

### 架构改进
- **核心算法与前端分离**：
  - `dancebattle.core` - 平台无关的核心算法，可用于移动端集成
  - `dancebattle.gui` - PC 演示前端（PySide6）
  - 保持向后兼容，`dancebattle` 包仍导出所有核心 API
  
- **后台线程处理**：
  - `PoseLoadWorker` - 后台加载参考视频姿态
  - `PipelineWorker` - 后台运行对比流程
  - `ResultVideoPlayer` - 后台播放结果视频
  - 消除耗时操作导致的 UI 无响应问题

### 改进
- `extract_audio()` 现在正确处理无音频轨道的视频
- `run_pipeline()` 在无音频时跳过对齐，使用 `time_offset=0`
- 支持 IDE 直接运行 `app.py` 和 `cli.py`

### 依赖
- 新增 `PySide6` 作为 GUI 依赖
- 需要 `ffmpeg` 用于带音频的视频录制

### 文档
- 更新 README.md 和 docs/README.md
- 新增 docs/PROJECT_STRUCTURE.md 项目结构文档

---

## 0.1.0

- 初始公开版本：音频对齐 + MediaPipe/YOLO 姿态估计 + Procrustes 相似度 + 并排视频可视化。
- 提供 CLI 与本地 `main.py` 启动方式；示例素材存放于 `assets/`。
- 添加可选烟雾测试脚本（MediaPipe / YOLO），输出目录被 gitignore 排除。
- 采用 MIT 许可。
