# 康复训练姿态评估系统

> **基于 YOLO 与 AI 大模型的康复训练姿态评估系统**
>
> 宁波大学计算机系《计算机系统综合实习》课程设计 | 236001754 王兴伽

## 系统架构

```
┌─────────────────────────────────────────────────────────────────┐
│                     交互层 (Streamlit Web App)                    │
│    标准动作录制界面 │ 实时姿态显示界面 │ 评估报告展示界面          │
└──────────────────────────────┬──────────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────────┐
│                   Apache Kafka 消息总线                           │
│                                                                  │
│  ┌──────────┐  ┌──────────┐  ┌──────────────┐  ┌──────────┐    │
│  │ rehab.   │  │ rehab.   │  │ rehab.       │  │ rehab.   │    │
│  │ frames   │  │ poses    │  │ comparisons  │  │ reports  │    │
│  └────┬─────┘  └────┬─────┘  └──────┬───────┘  └────┬─────┘    │
└───────┼──────────────┼───────────────┼───────────────┼──────────┘
        │              │               │               │
┌───────▼──────┐ ┌─────▼──────┐ ┌──────▼──────┐ ┌─────▼─────────┐
│ 视频采集服务  │ │ YOLO推理   │ │ 姿态对比    │ │ AI 评估服务   │
│ (Producer)   │ │ (C→P)      │ │ (C→P)      │ │ (Consumer)    │
│ 多线程:      │ │ GPU/CPU    │ │ DTW对齐     │ │ LLM推理       │
│ 采集线程 +   │ │ 17关键点   │ │ 多维度偏差  │ │ RAG知识库     │
│ 命令监听线程 │ │            │ │             │ │               │
└──────────────┘ └────────────┘ └─────────────┘ └───────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────────┐
│                        数据层                                     │
│              PostgreSQL │ FAISS 向量库 │ 标准动作库               │
└─────────────────────────────────────────────────────────────────┘
```

### Kafka 消息流架构

系统采用 **Apache Kafka** 作为异步消息总线，实现微服务间的解耦通信。共定义 4 个 Topic：

| Topic | 生产者 | 消费者 | 消息内容 |
|-------|--------|--------|----------|
| `rehab.frames` | 视频采集服务 | YOLO 推理服务 | Base64 编码的视频帧（含时间戳、会话ID） |
| `rehab.poses` | YOLO 推理服务 | 姿态对比服务 | 17 关键点坐标 + 置信度 |
| `rehab.comparisons` | 姿态对比服务 | AI 评估服务 | DTW 相似度评分 + 多维度偏差数据 |
| `rehab.reports` | AI 评估服务 | Web 前端 | 完整评估报告（评分、诊断、建议） |

**消息流转流程：**
```
摄像头采集帧 → [rehab.frames] → YOLO提取关键点 → [rehab.poses]
→ DTW姿态对比 → [rehab.comparisons] → LLM生成报告 → [rehab.reports] → 前端展示
```

### 多线程设计

系统在多个层面采用多线程技术实现并发处理：

**1. 视频采集服务 — 双线程架构**
- **主采集线程**：以固定 FPS（默认 15fps）循环读取摄像头帧，Base64 编码后通过 Kafka Producer 异步发送
- **命令监听线程（daemon）**：监听 stdin 控制指令（JSON 格式），接收前端的开始/停止录制命令，与采集线程共享 `session_id` 等全局状态
- 两线程通过全局变量协调：命令线程设置 `session_id`，采集线程根据其是否为空决定是否发送帧

**2. Kafka Consumer — 回调式并发**
- 每个微服务的 Consumer 使用 `poll()` 循环 + 回调函数模式
- YOLO 推理服务在回调中执行模型推理，处理完即释放，天然支持帧级别的并发处理
- 姿态对比服务使用 `defaultdict` 缓存按 `session_id` 分组的帧序列，支持多会话并行

**3. Docker Compose 并行编排**
- 5 个业务服务通过 Docker Compose 并行启动
- 服务间通过 Kafka 异步解耦，无需同步等待上游处理完成
- 使用 `depends_on` 确保 Kafka/PostgreSQL 基础设施就绪后再启动业务服务

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

### 5. Docker 部署（推荐）

系统通过 Docker Compose 编排 7 个容器（2 基础设施 + 5 业务服务），一键启动：

```bash
# 启动全部服务
docker compose up -d

# 查看服务状态
docker compose ps

# 查看日志
docker compose logs -f [service_name]

# 停止全部服务
docker compose down
```

**容器列表：**

| 容器 | 镜像 | 职责 |
|------|------|------|
| rehab-zookeeper | confluentinc/cp-zookeeper:7.5.0 | Kafka 集群协调 |
| rehab-kafka | confluentinc/cp-kafka:7.5.0 | 消息队列（4 Topic） |
| rehab-postgres | postgres:15-bookworm | 数据库 |
| rehab-video-capture | 自建 | 摄像头采集 + 多线程命令监听 |
| rehab-yolo-inference | 自建 | YOLO-Pose 推理（CPU/GPU） |
| rehab-compare-service | 自建 | DTW 姿态对比 |
| rehab-ai-evaluator | 自建 | LLM 评估报告生成 |
| rehab-web-app | 自建 | Streamlit Web 前端 |

### 6. 启动服务（非 Docker）

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
| 消息队列 | Apache Kafka (confluent-kafka) | 异步消息总线，4 Topic 解耦微服务通信 |
| 消息协调 | Apache Zookeeper | Kafka 集群协调（KRaft 前的标配） |
| 向量数据库 | FAISS | 高性能向量相似度检索 |
| LLM 编排 | OpenAI API / Ollama | 统一模型适配层 |
| 嵌入模型 | text-embedding-3-small | 文本向量化 |
| 前端界面 | Streamlit | 快速构建数据应用 |
| 后端服务 | FastAPI | 异步高性能 REST API |
| 数据库 | PostgreSQL | 生产级关系型数据库 |
| 容器编排 | Docker Compose | 多服务并行编排与网络隔离 |
| 并发模型 | Python threading | 多线程采集 + 回调式消息消费 |
| 数据分析 | Pandas + Plotly | 趋势分析与可视化 |

## 项目结构

```
last_work/
├── pyproject.toml              # 项目配置（uv 依赖管理）
├── .env.example                # 环境变量模板
├── docker-compose.yml          # Docker 多服务编排（Kafka + 5 微服务）
├── init-db.sql                 # PostgreSQL 初始化脚本
├── seed_data.py                # 初始化脚本
├── app.py                      # Streamlit 前端入口
├── api.py                      # FastAPI 后端入口
├── deploy.sh                   # 部署脚本
├── docker/
│   ├── video-capture/          # 视频采集微服务（多线程：采集 + 命令监听）
│   │   ├── Dockerfile
│   │   └── main.py
│   ├── yolo-inference/         # YOLO 推理微服务（消费帧 → 输出关键点）
│   │   ├── Dockerfile
│   │   └── main.py
│   ├── compare-service/        # 姿态对比微服务（DTW 对比 + 结果发送）
│   │   ├── Dockerfile
│   │   └── main.py
│   ├── ai-evaluator/           # AI 评估微服务（LLM 生成报告）
│   │   ├── Dockerfile
│   │   └── main.py
│   └── web-app/                # Web 前端微服务
│       ├── Dockerfile
│       └── main.py
└── src/rehab_pose/
    ├── __init__.py             # 包初始化
    ├── config.py               # 系统配置
    ├── keypoints.py            # 关键点数据结构 & 角度计算
    ├── pose_detector.py        # YOLO 姿态检测模块
    ├── pose_comparator.py      # 姿态对比（DTW + 多维度）
    ├── database.py             # PostgreSQL 数据层
    ├── kafka_utils.py          # Kafka Producer/Consumer 封装 + 消息格式定义
    ├── rag_knowledge_base.py   # RAG 知识库（FAISS）
    ├── ai_evaluator.py         # AI 评估引擎
    ├── rehab_logger.py         # 康复日志 & 可视化
    └── visualization.py        # 姿态可视化工具
```
