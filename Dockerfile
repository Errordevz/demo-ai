FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=7860

WORKDIR /app
COPY . .

RUN python -m pip install --upgrade pip && \
    pip install numpy==2.2.5 sentencepiece==0.2.0 && \
    python scripts/fetch_cloud_model.py

EXPOSE 7860
CMD ["python", "server.py"]
