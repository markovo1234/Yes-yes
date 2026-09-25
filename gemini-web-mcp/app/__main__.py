"""Entry point: `python -m app`."""

from __future__ import annotations

import logging
import os
import sys

import uvicorn

from .backend import WebapiBackend
from .config import ConfigError, Settings
from .logs import setup_logging
from .server import build_app
from .service import ImageService
from .store import Store


def main() -> None:
    os.umask(0o077)  # every file we create on /data (db, images, cookies) is owner-only
    setup_logging()
    log = logging.getLogger("gemini_web_mcp")
    try:
        settings = Settings.from_env()
    except ConfigError as e:
        log.error("configuration error: %s", e)
        sys.exit(2)
    settings.data_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    store = Store(settings.db_path)
    service = ImageService(settings, store, WebapiBackend(settings))
    app = build_app(service)
    log.info("listening on port %d, public base %s", settings.port, settings.public_base_url)
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=settings.port,
        access_log=False,
        log_config=None,
        server_header=False,
        proxy_headers=False,
        timeout_graceful_shutdown=20,
    )


if __name__ == "__main__":
    main()
