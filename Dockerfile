FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=7860

WORKDIR /app

RUN python -m pip install --upgrade pip && \
    pip install --extra-index-url https://download.pytorch.org/whl/cpu \
      torch==2.8.0+cpu \
      numpy==2.2.5 \
      sentencepiece==0.2.0

COPY . .

RUN python scripts/build_cloud_model.py

CMD ["python","server.py"]
