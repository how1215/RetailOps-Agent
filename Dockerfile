FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src
COPY policies ./policies

RUN pip install --no-cache-dir . && \
    addgroup --system retailops && \
    adduser --system --ingroup retailops retailops && \
    mkdir -p /app/data /app/traces && \
    chown -R retailops:retailops /app

USER retailops

EXPOSE 8080

CMD ["retailops", "serve", "--host", "0.0.0.0", "--port", "8080"]
