# gemini-web-mcp — Railway builds and runs this image; /data is the Railway volume.
FROM python:3.11.15-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DATA_DIR=/data

WORKDIR /srv
COPY requirements.txt .
RUN pip install --no-deps -r requirements.txt && pip check

COPY app ./app

# Runs as root because Railway mounts volumes owned by root; everything written to /data is
# owner-only (umask 077). No health route on purpose: only /mcp/<secret> and /i/... exist.
CMD ["python", "-m", "app"]
