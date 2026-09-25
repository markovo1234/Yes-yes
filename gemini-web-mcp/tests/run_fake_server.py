"""Serve the real app with the fake Gemini backend, for MCP Inspector checks that never touch Gemini.

    MCP_PATH_SECRET=... DATA_DIR=/tmp/fake-data PORT=8765 python -m tests.run_fake_server
"""

from __future__ import annotations

import os

import uvicorn

from app.config import Settings
from app.logs import setup_logging
from app.server import build_app
from app.service import ImageService
from app.store import Store
from tests.fakes import FakeBackend


def main() -> None:
    os.umask(0o077)
    setup_logging()
    settings = Settings.from_env()
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    service = ImageService(settings, Store(settings.db_path), FakeBackend(delay=0.2))
    uvicorn.run(build_app(service), host="127.0.0.1", port=settings.port, log_config=None, access_log=False)


if __name__ == "__main__":
    main()
