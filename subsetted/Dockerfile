# syntax=docker/dockerfile:1

ARG PYTHON_VERSION=3.9.6
FROM python:${PYTHON_VERSION}-slim as base

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
# Add environment variables for librosa cache
ENV NUMBA_CACHE_DIR=/tmp/numba_cache
ENV XDG_CACHE_HOME=/tmp/cache

WORKDIR /app

ARG UID=10001
RUN adduser \
    --disabled-password \
    --gecos "" \
    --home "/nonexistent" \
    --shell "/sbin/nologin" \
    --no-create-home \
    --uid "${UID}" \
    appuser

RUN --mount=type=cache,target=/root/.cache/pip \
    --mount=type=bind,source=requirements.txt,target=requirements.txt \
    python -m pip install -r requirements.txt

# Create necessary directories and set proper permissions
RUN mkdir -p /app/uploads /tmp/numba_cache /tmp/cache && \
    chown -R appuser:appuser /app/uploads /tmp/numba_cache /tmp/cache && \
    chmod 777 /app/uploads /tmp/numba_cache /tmp/cache

COPY --chown=appuser:appuser . .

USER appuser

EXPOSE 8000

CMD python3 src/app.py
