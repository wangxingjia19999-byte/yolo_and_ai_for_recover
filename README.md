# 康复训练姿态评估系统

> **基于 YOLO 与 AI 大模型的康复训练姿态评估系统**
>
> 宁波大学计算机系《计算机系统综合实习》课程设计 | 236001754 王兴伽

## 系统架构

```
交互层 (Streamlit)
    ├── 标准动作录制界面
    ├── 实时姿态显示界面
    └── 评估报告展示界面
          │
AI 分析层
    ├── LLM 推理引擎 (GPT-4o / Claude / Qwen2.5)
    ├── RAG 康复知识库 (FAISS 向量检索)
    └── 评估提示词模板 + 幻觉抑制机制
          │
姿态检测层 (YOLO-Pose)
    ├── 17 关键点提取
    ├── 关节角度计算
    └── 坐标系归一化
          │
数据层 (SQLite)
    ├── 标准动作库
    └── 康复日志
```

## 快速开始

### 1. 环境要求

- Python >= 3.10
- [uv](https://docs.astral.sh/uv/) (已自动安装于 `.local/bin/uv`)

### 2. 安装依赖

```bash
# 确保 uv 在 PATH 中
export PATH="$HOME/.local/bin:$PATH"

# 同步依赖（自动创建虚拟环境）
uv sync
```

### 3. 配置环境变量

```bash
cp .env.example .env
# 编辑 .env，填入你的 OPENAI_API_KEY
```

### 4. 初始化数据和知识库

```bash
uv run python seed_data.py
```

### 5. 启动服务

**Streamlit 前端（推荐）：**

```bash
uv run streamlit run app.py
```

访问 http://localhost:8501

**FastAPI 后端：**

```bash
uv run python api.py
```

API 文档：http://127.0.0.1:8000/docs

## 核心功能

### 📹 标准动作录制
康复医师在摄像头前完成标准康复动作，系统通过 YOLO-Pose 逐帧提取 17 个关键点，计算关节角度序列，存储为标准动作模板。

### 🏃 患者训练评估
- 实时姿态检测（YOLO-Pose, 17 关键点）
- DTW 帧对齐 + 多维度偏差计算
- 关节角度偏差、关键点空间偏移、动作幅度差异
- 0-100 分整体相似度评分

### 🤖 AI 智能评估
- LLM 推理引擎（GPT-4o / Claude / 本地 Ollama）
- RAG 康复医学知识库（FAISS 向量检索）
- 多层幻觉抑制机制
- 专业评估报告（含评分、诊断、建议、风险提示）

### 📊 康复日志
- 评分趋势折线图
- 关节功能改善雷达图
- 训练频次热力图
- 历史报告检索

## 技术栈

| 模块 | 技术 | 说明 |
|------|------|------|
| 姿态估计 | YOLO11n-Pose | Ultralytics, 17 关键点实时检测 |
| 图像处理 | OpenCV | 视频采集与预处理 |
| 向量数据库 | FAISS | 高性能向量相似度检索 |
| LLM 编排 | OpenAI API / Ollama | 统一模型适配层 |
| 嵌入模型 | text-embedding-3-small | 文本向量化 |
| 前端界面 | Streamlit | 快速构建数据应用 |
| 后端服务 | FastAPI | 异步高性能 REST API |
| 数据库 | SQLite | 轻量级免配置 |
| 数据分析 | Pandas + Plotly | 趋势分析与可视化 |

## 项目结构

```
last_work/
├── pyproject.toml          # 项目配置（uv 依赖管理）
├── .env.example            # 环境变量模板
├── seed_data.py            # 初始化脚本
├── app.py                  # Streamlit 前端入口
├── api.py                  # FastAPI 后端入口
└── src/rehab_pose/
    ├── __init__.py         # 包初始化
    ├── config.py           # 系统配置
    ├── keypoints.py        # 关键点数据结构 & 角度计算
    ├── pose_detector.py    # YOLO 姿态检测模块
    ├── pose_comparator.py  # 姿态对比（DTW + 多维度）
    ├── database.py         # SQLite 数据层
    ├── rag_knowledge_base.py # RAG 知识库（FAISS）
    ├── ai_evaluator.py     # AI 评估引擎
    ├── rehab_logger.py     # 康复日志 & 可视化
    └── visualization.py    # 姿态可视化工具
```
