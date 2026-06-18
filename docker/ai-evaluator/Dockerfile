# ══════════════════════════════════════════════════════════════
# AI 评估服务 Dockerfile (Mac / 通用)
# ══════════════════════════════════════════════════════════════

FROM python:3.10-slim

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir -i https://pypi.tuna.tsinghua.edu.cn/simple --trusted-host pypi.tuna.tsinghua.edu.cn \
    confluent-kafka>=2.3.0 \
    openai>=1.6.0 \
    httpx>=0.25.0 \
    pydantic>=2.5.0 \
    psycopg2-binary>=2.9.0 \
    python-dotenv>=1.0.0 \
    faiss-cpu>=1.7.4 \
    langchain>=0.1.0 \
    langchain-openai>=0.0.5

WORKDIR /app

COPY src/ /app/src/
COPY docker/ai-evaluator/main.py /app/main.py

CMD ["python3", "main.py"]
