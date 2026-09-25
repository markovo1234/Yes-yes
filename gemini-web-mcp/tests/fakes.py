"""A fake Gemini backend. Behaviour is chosen by markers inside the prompt text, e.g. "[fail]"."""

from __future__ import annotations

import asyncio
import io
import itertools
import os
from pathlib import Path
from typing import Any

from PIL import Image

from app.backend import Turn
from app.errors import AuthProblem, Blocked, RateLimited, REFRESH_COOKIES, TransientError


class FakeImage:
    def __init__(self, color=(200, 30, 30), size=(1024, 768), noisy: bool = False, fmt: str = "PNG"):
        self.color, self.size, self.noisy, self.fmt = color, size, noisy, fmt

    def data(self) -> bytes:
        if self.noisy:
            im = Image.frombytes("RGB", self.size, os.urandom(self.size[0] * self.size[1] * 3))
        else:
            im = Image.new("RGB", self.size, self.color)
        buf = io.BytesIO()
        im.save(buf, self.fmt)
        return buf.getvalue()


class FakeBackend:
    default_model: str | None = None

    def __init__(self, delay: float = 0.05):
        self.delay = delay
        self.sent: list[dict[str, Any]] = []
        self.deleted: list[str] = []
        self.fail_delete: set[str] = set()
        self.ensure_calls = 0
        self.ensure_error: Exception | None = None
        self.active = 0
        self.max_active = 0
        self.attempts: dict[str, int] = {}
        self._ids = itertools.count(1)

    async def ensure_ready(self) -> None:
        self.ensure_calls += 1
        if self.ensure_error:
            raise self.ensure_error

    async def send(self, prompt, *, metadata, model, files, on_cid) -> Turn:
        self.sent.append({"prompt": prompt, "metadata": metadata, "model": model, "files": files})
        self.attempts[prompt] = self.attempts.get(prompt, 0) + 1
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        try:
            cid = metadata[0] if metadata else f"c_{next(self._ids)}"
            on_cid(cid)
            delay = 30.0 if "[slow]" in prompt else self.delay
            await asyncio.sleep(delay)
            if "[auth]" in prompt:
                raise AuthProblem(REFRESH_COOKIES)
            if "[rate]" in prompt:
                raise RateLimited("Gemini reports the account's usage limit is reached.")
            if "[blocked]" in prompt:
                raise Blocked("Google temporarily flagged this server.")
            if "[fail]" in prompt or ("[flaky]" in prompt and self.attempts[prompt] == 1):
                raise TransientError("Gemini request failed (APIError).")
            rid = f"r_{next(self._ids)}"
            if "[refuse]" in prompt:
                return Turn("I can't create that image.", [], [cid, rid, "rc"], cid, rid)
            count = 3 if "[multi3]" in prompt else 1
            noisy = "[noisy]" in prompt
            images = [FakeImage(color=(40 * i, 90, 160), noisy=noisy) for i in range(count)]
            return Turn("Here you go!", images, [cid, rid, f"rc_{rid}", None, None, None, None, None, None, "ctx"], cid, rid)
        finally:
            self.active -= 1

    async def save_image(self, handle: FakeImage, dest: Path) -> None:
        dest.write_bytes(handle.data())

    async def delete_chat(self, cid: str) -> None:
        if cid in self.fail_delete:
            raise TransientError("Gemini request failed (APIError).")
        self.deleted.append(cid)

    async def keepalive(self) -> None:
        pass

    async def aclose(self) -> None:
        pass

    def account_note(self) -> str | None:
        return None
