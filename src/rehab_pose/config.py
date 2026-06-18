"""系统配置模块"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── 路径配置 ─────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data"
RAG_DOCS_DIR = DATA_DIR / "rag_docs"
DB_PATH = DATA_DIR / "rehab.db"
MODEL_DIR = DATA_DIR / "models"
REPORTS_DIR = DATA_DIR / "reports"

for d in [DATA_DIR, RAG_DOCS_DIR, MODEL_DIR, REPORTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── YOLO 姿态检测配置 ──────────────────────────────────────
YOLO_MODEL_NAME = os.getenv("YOLO_MODEL", str(BASE_DIR / "yolo11n-pose.pt"))
YOLO_INPUT_SIZE = (640, 640)
YOLO_CONF_THRESHOLD = 0.5
YOLO_DETECTION_FPS_TARGET = 15

# ── 姿态关键点索引 ─────────────────────────────────────────
# COCO 17关键点定义
KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]

# 人体骨架连接关系（用于可视化）
SKELETON_CONNECTIONS = [
    ("nose", "left_eye"), ("nose", "right_eye"),
    ("left_eye", "left_ear"), ("right_eye", "right_ear"),
    ("left_shoulder", "right_shoulder"),
    ("left_shoulder", "left_elbow"), ("right_shoulder", "right_elbow"),
    ("left_elbow", "left_wrist"), ("right_elbow", "right_wrist"),
    ("left_shoulder", "left_hip"), ("right_shoulder", "right_hip"),
    ("left_hip", "right_hip"),
    ("left_hip", "left_knee"), ("right_hip", "right_knee"),
    ("left_knee", "left_ankle"), ("right_knee", "right_ankle"),
]

# 关键点对应的骨架颜色（左右区分）
SKELETON_COLORS = {
    "left": (255, 0, 0),    # 蓝色
    "right": (0, 0, 255),   # 红色
    "center": (0, 255, 0),  # 绿色
}

# 需要计算角度的关节定义（顶点在中间）
JOINT_ANGLE_DEFS = {
    "left_elbow_angle": ("left_shoulder", "left_elbow", "left_wrist"),
    "right_elbow_angle": ("right_shoulder", "right_elbow", "right_wrist"),
    "left_knee_angle": ("left_hip", "left_knee", "left_ankle"),
    "right_knee_angle": ("right_hip", "right_knee", "right_ankle"),
    "left_shoulder_angle": ("left_elbow", "left_shoulder", "left_hip"),
    "right_shoulder_angle": ("right_elbow", "right_shoulder", "right_hip"),
    "left_hip_angle": ("left_shoulder", "left_hip", "left_knee"),
    "right_hip_angle": ("right_shoulder", "right_hip", "right_knee"),
}

# 关节活动度正常范围 (ROM, 单位: 度)
ROM_NORMAL_RANGES = {
    "left_elbow_angle": (0, 145),
    "right_elbow_angle": (0, 145),
    "left_knee_angle": (0, 135),
    "right_knee_angle": (0, 135),
    "left_shoulder_angle": (0, 180),
    "right_shoulder_angle": (0, 180),
    "left_hip_angle": (0, 120),
    "right_hip_angle": (0, 120),
}

# ── 动作录制配置 ───────────────────────────────────────────
RECORD_FPS = 15
RECORD_DURATION_SEC = 10
RECORD_COUNTDOWN_SEC = 3

# ── RAG 知识库配置 ─────────────────────────────────────────
# 知识库文档分块大小
RAG_CHUNK_SIZE = 500
RAG_CHUNK_OVERLAP = 50
# FAISS 检索 Top-K
RAG_TOP_K = 5
# 相似度阈值 (0-1)
RAG_SIMILARITY_THRESHOLD = 0.7
# 嵌入模型
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

# ── LLM 配置 ───────────────────────────────────────────────
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai")  # openai | ollama
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o")
LLM_API_KEY = os.getenv("OPENAI_API_KEY", "")
LLM_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5")

# ── 姿态对比权重配置 ───────────────────────────────────────
COMPARISON_WEIGHTS = {
    "angle_deviation": 0.40,       # 关节角度偏差权重
    "keypoint_offset": 0.30,       # 关键点空间偏移权重
    "rom_difference": 0.30,        # 动作幅度差异权重
}

# ── 数据库配置 ────────────────────────────────────────────
DATABASE_URL = f"sqlite:///{DB_PATH}"

# ── API 服务配置 ──────────────────────────────────────────
API_HOST = os.getenv("API_HOST", "127.0.0.1")
API_PORT = int(os.getenv("API_PORT", "8000"))
STREAMLIT_PORT = int(os.getenv("STREAMLIT_PORT", "8501"))
