# MotionCoach 文档索引

> 按用途分组，每篇注明状态。时间顺序与可复现的数值见 `ITERATION_LOG.md`。
> 2026-10-23 整理。

## 先读这三篇

| 文档 | 内容 | 状态 |
|---|---|---|
| [ARCHITECTURE.md](ARCHITECTURE.md) | 总体架构：数据契约、处理流程（视频 → 姿态 → 清洗 → 对应 → 指标 → 判定 → 报告）、前端工作台、算法库 | 随实现更新 |
| [ITERATION_LOG.md](ITERATION_LOG.md) | 逐轮记录 I1–I19：问题、改动、测试、结论；§4 当前生产配置；§5 复现命令；§6 已知问题 | 主记录，每轮更新 |
| [EXPERIMENT_PLAN.md](EXPERIMENT_PLAN.md) | 困难动作骨架识别的后期实验计划 E0–E6 与各轮结果（清洗消融、诊断、候选模型、人物框跟踪） | 进行中 |

## 研究与实验（结论已进入生产或计划）

| 文档 | 内容 | 状态 |
|---|---|---|
| [ALIGNMENT_STUDY.md](ALIGNMENT_STUDY.md) | 动作对应方法对比（M0–M3fa），推荐方法的依据 | 完成；M3fa 已上线（I4） |
| [POSE_BENCH.md](POSE_BENCH.md) | 12 个姿态算法的武术分析横评 | 完成（I5、I9）；人物框跟踪（I18）之前的数值 |
| [RECAPTURE_STUDY.md](RECAPTURE_STUDY.md) | 已知答案的翻拍检验：系统误差、测量不确定度判定、重复周期约束 | 完成（I10–I11）；跟踪后的复核见 I18 |

## 设计（部分已实施）

| 文档 | 内容 | 状态 |
|---|---|---|
| [MULTI_PERSON_DESIGN.md](MULTI_PERSON_DESIGN.md) | 多人：目标稳定、身份、筛选、统一时间轴、集体工作台 | 已实施（§7–§12） |
| [SCALE_DESIGN.md](SCALE_DESIGN.md) | 比例尺：相对单位与厘米 | 已实施；厘米只在填写身高时显示（§8） |
| [VIDEO_SPEC.md](VIDEO_SPEC.md) | 接收 / 分析 / 显示三层视频参数 | 已实施（I8–I9） |
| [MOBILE_DESIGN.md](MOBILE_DESIGN.md) | 三种角色、移动端方案、参考包格式 | 部分实施（契约、参考包）；移动端未做 |
| [LLM_COACH_DESIGN.md](LLM_COACH_DESIGN.md) | LLM 辅助建议：调研与接口蓝图 | 设计稿，未实施；建议卡（I16）是其确定性输入 |
| [MOTION_LLM_SURVEY.md](MOTION_LLM_SURVEY.md) | SportsGPT 与骨架序列模型调研 | 调研，供 LLM 设计参考 |

## 交流材料

| 文件 | 内容 |
|---|---|
| `EXPERT_MEMO_PHASE1.html`（仅本地） | 与教练、运动科学专家交流的阶段备忘：架构、结果、问题、计划与需要确认的事项。图表在 `figures/`（入库）；备忘录本身与 `memo_assets/` 中的截图、诊断帧含可识别人物，**不入库**（.gitignore） |
| [index.html](index.html) | 项目展示页（GitHub Pages），图在 `media/` |

## 图表与素材

- `figures/`：实验图表，只含数字，由 `research/memo_figures.py`、`research/align_compare.py`、`research/pose_bench.py` 生成。
- `memo_assets/`：备忘录用的界面截图与诊断帧，仅本地。
- `media/`：展示页与 README 用的演示动图。

## 旧版

| 文档 | 内容 |
|---|---|
| [DANCEBATTLE_LEGACY.md](DANCEBATTLE_LEGACY.md) | 旧版 DanceBattle 舞蹈演示（`src/dancebattle`）的说明 |
| [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md) | 旧版目录结构 |

部署与运维材料（服务器、凭据、转发）不在仓库中，单独存放。
