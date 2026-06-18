# ══════════════════════════════════════════════════════════════
# 姿态对比服务 Dockerfile (Mac / 通用)
# ══════════════════════════════════════════════════════════════

FROM python:3.10-slim

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir -i https://pypi.tuna.tsinghua.edu.cn/simple --trusted-host pypi.tuna.tsinghua.edu.cn \
    confluent-kafka>=2.3.0 \
    numpy>=1.24.0 \
    scipy>=1.10.0 \
    pydantic>=2.5.0 \
    psycopg2-binary>=2.9.0 \
    python-dotenv>=1.0.0

WORKDIR /app

COPY src/ /app/src/
COPY docker/compare-service/main.py /app/main.py

CMD ["python3", "main.py"]
