# ══════════════════════════════════════════════════════════════
# YOLO 推理服务 Dockerfile (Mac / 通用)
# ══════════════════════════════════════════════════════════════
# 注意: Mac 上无 NVIDIA GPU，YOLO 将使用 CPU 推理（速度较慢）
# ══════════════════════════════════════════════════════════════

FROM python:3.10-slim

ENV DEBIAN_FRONTEND=noninteractive

# 安装系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 安装 Python 依赖
RUN pip3 install --no-cache-dir -i https://pypi.tuna.tsinghua.edu.cn/simple --trusted-host pypi.tuna.tsinghua.edu.cn \
    ultralytics>=8.0.0 \
    confluent-kafka>=2.3.0 \
    opencv-python>=4.8.0 \
    numpy>=1.24.0 \
    pydantic>=2.5.0 \
    python-dotenv>=1.0.0

WORKDIR /app

# 复制源码
COPY src/ /app/src/

# 复制服务脚本
COPY docker/yolo-inference/main.py /app/main.py

# 模型文件通过 volume 挂载
# docker-compose.yml 中: ./yolo11n-pose.pt:/app/models/yolo11n-pose.pt:ro

CMD ["python3", "main.py"]
