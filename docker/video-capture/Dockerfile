# ══════════════════════════════════════════════════════════════
# 视频采集服务 Dockerfile (Mac / 通用)
# ══════════════════════════════════════════════════════════════

FROM python:3.10-slim

ENV DEBIAN_FRONTEND=noninteractive

# 安装系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 安装 Python 依赖
RUN pip3 install --no-cache-dir -i https://pypi.tuna.tsinghua.edu.cn/simple --trusted-host pypi.tuna.tsinghua.edu.cn \
    confluent-kafka>=2.3.0 \
    opencv-python-headless>=4.8.0 \
    numpy>=1.24.0

WORKDIR /app

# 复制源码
COPY src/ /app/src/

# 复制服务脚本
COPY docker/video-capture/main.py /app/main.py

CMD ["python3", "main.py"]
