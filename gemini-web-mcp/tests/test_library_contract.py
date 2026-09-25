"""Guards for the gemini_webapi==2.1.1 behaviours app/backend.py relies on. No network."""

from __future__ import annotations

import importlib
import inspect
import time
from importlib.metadata import version

import pytest
from gemini_webapi import ChatSession, GeminiClient
from gemini_webapi.constants import AccountStatus
from gemini_webapi.exceptions import APIError
from gemini_webapi.types import GeneratedImage
from gemini_webapi.utils import running

from app.backend import TrackedChat, _GentleClient

pytestmark = pytest.mark.anyio


def test_pinned_versions():
    assert version("gemini-webapi") == "2.1.1"
    assert version("mcp") == "2.2.0"


def test_features_we_use_exist():
    for name in ("init", "close", "start_chat", "delete_chat", "resolve_model", "list_models", "generate_content"):
        assert callable(getattr(GeminiClient, name))
    assert "full_size" in inspect.signature(GeneratedImage._perform_save).parameters
    assert AccountStatus.UNAUTHENTICATED.value == 1016 and AccountStatus.AVAILABLE.value == 1000
    for module in ("gemini_webapi.utils.rotate_1psidts", "gemini_webapi.client"):
        assert callable(importlib.import_module(module).save_cookies)


async def test_running_decorator_honours_current_retry():
    class Dummy:
        _running = True
        calls = 0

        async def close(self):
            pass

        @running(retry=5)
        async def call(self):
            Dummy.calls += 1
            raise APIError("x")

    with pytest.raises(APIError):
        await Dummy().call(current_retry=0)
    assert Dummy.calls == 1


async def test_gentle_client_sends_each_request_once():
    client = _GentleClient("psid", "psidts")
    client._running = True  # pretend initialised; there is no HTTP session so every request fails
    t0 = time.monotonic()
    with pytest.raises(APIError):
        await client._batch_execute([])
    client._running = True
    with pytest.raises(APIError):
        async for _ in client._generate(prompt="x"):
            pass
    assert time.monotonic() - t0 < 2  # upstream retries would sleep 5 s+ before retrying


def test_tracked_chat_reports_every_cid_once():
    seen: list[str] = []
    client = GeminiClient("psid", "psidts")
    chat = TrackedChat(seen.append, geminiclient=client)
    chat.metadata = ["c_1", "r_1", "rc_1"]
    chat.metadata = ["c_1", "r_2", "rc_2"]
    chat.cid = ""  # the library restores an empty cid after a failed first turn
    chat.metadata = ["c_2", "r_3"]
    assert seen == ["c_1", "c_2"]
    resumed = TrackedChat(seen.append, geminiclient=client, metadata=["c_9", "r_9", "rc_9"])
    assert resumed.cid == "c_9" and seen[-1] == "c_9"
    assert isinstance(resumed, ChatSession)
