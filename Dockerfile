FROM python:3.12-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN python -m pip install --upgrade pip && \
    pip install --extra-index-url https://download.pytorch.org/whl/cpu \
      torch==2.8.0+cpu \
      numpy==2.2.5 \
      sentencepiece==0.2.0

COPY . .

RUN python scripts/build_cloud_model.py

FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=7860

WORKDIR /app

RUN python -m pip install --upgrade pip && \
    pip install numpy==2.2.5 sentencepiece==0.2.0

COPY --from=builder /app/api/model /app/api/model
COPY --from=builder /app/api /app/api
COPY --from=builder /app/demo_ai /app/demo_ai
COPY --from=builder /app/index.html /app/index.html
COPY --from=builder /app/styles.css /app/styles.css
COPY --from=builder /app/app.js /app/app.js
COPY --from=builder /app/server.py /app/server.py

EXPOSE 7860

CMD ["python","server.py"]
