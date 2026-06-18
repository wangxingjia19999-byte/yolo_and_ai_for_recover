# ══════════════════════════════════════════════════════════════
# Web 前端 Dockerfile (Mac / 通用)
# ══════════════════════════════════════════════════════════════

FROM python:3.10-slim

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq-dev \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir -i https://pypi.tuna.tsinghua.edu.cn/simple --trusted-host pypi.tuna.tsinghua.edu.cn \
    confluent-kafka>=2.3.0 \
    streamlit>=1.28.0 \
    plotly>=5.17.0 \
    pandas>=2.0.0 \
    pydantic>=2.5.0 \
    psycopg2-binary>=2.9.0 \
    python-dotenv>=1.0.0

WORKDIR /app

COPY src/ /app/src/
COPY docker/web-app/main.py /app/main.py

EXPOSE 8501

CMD ["streamlit", "run", "main.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
