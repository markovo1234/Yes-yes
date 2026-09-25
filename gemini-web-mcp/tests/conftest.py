from __future__ import annotations

import base64
import os
import secrets
import socket
import threading
import time
from dataclasses import replace
from pathlib import Path

import pytest
import uvicorn

from app.config import Settings
from app.logs import setup_logging
from app.server import build_app
from app.service import ImageService
from app.store import Store
from tests.fakes import FakeBackend

setup_logging()


def new_secret() -> str:
    return base64.urlsafe_b64encode(secrets.token_bytes(32)).decode().rstrip("=")


def make_settings(data_dir: Path, **overrides) -> Settings:
    base = Settings(
        port=0,
        data_dir=data_dir,
        path_secret=new_secret(),
        public_base_url="https://example.test",
        max_images_per_hour=30,
        session_ttl_hours=24,
        gemini_model=None,
        seed_1psid=None,
        seed_1psidts=None,
    )
    return replace(base, **overrides)


class FakeClock:
    def __init__(self, start: float | None = None):
        self.now = start if start is not None else time.time()

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
def backend() -> FakeBackend:
    return FakeBackend()


@pytest.fixture
def make_service(tmp_path, backend):
    stores: list[Store] = []

    def factory(clock=time.time, **overrides) -> ImageService:
        settings = make_settings(tmp_path / "data", **overrides)
        settings.data_dir.mkdir(parents=True, exist_ok=True)
        store = Store(settings.db_path)
        stores.append(store)
        return ImageService(settings, store, backend, clock=clock)

    yield factory
    for s in stores:
        s.close()


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class LiveServer:
    def __init__(self, service: ImageService, port: int):
        self.service = service
        self.port = port
        self.base = f"http://127.0.0.1:{port}"
        self.mcp_url = f"{self.base}/mcp/{service.settings.path_secret}"


@pytest.fixture
def live(tmp_path, backend):
    """The real ASGI app (gate + MCP + image route) on a local port, with the fake Gemini backend."""
    port = _free_port()
    settings = make_settings(tmp_path / "data", port=port, public_base_url=f"http://127.0.0.1:{port}")
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    store = Store(settings.db_path)
    service = ImageService(settings, store, backend)
    app = build_app(service, warm_up=False)
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_config=None, access_log=False))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    for _ in range(200):
        if server.started:
            break
        time.sleep(0.02)
    assert server.started
    os.environ.setdefault("NO_PROXY", "127.0.0.1,localhost")
    yield LiveServer(service, port)
    server.should_exit = True
    thread.join(10)
    store.close()
