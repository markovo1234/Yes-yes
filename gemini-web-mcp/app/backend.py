"""Gemini web access through gemini_webapi (pinned to 2.1.1).

Everything library-specific lives here so the service logic can be tested with a fake backend.
Library behaviours this module relies on are checked by tests/test_library_contract.py.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from curl_cffi.requests import Cookies
from gemini_webapi import ChatSession, GeminiClient
from gemini_webapi.constants import AccountStatus
from gemini_webapi.exceptions import (
    APIError,
    AuthError,
    GeminiError,
    ModelInvalidError,
    TemporarilyBlockedError,
    UsageLimitExceededError,
)
from gemini_webapi.exceptions import TimeoutError as GeminiTimeout

from .config import Settings
from .cookies import CookieFile, install_library_hooks
from .errors import (
    NO_COOKIES,
    REFRESH_COOKIES,
    AccountProblem,
    AuthProblem,
    Blocked,
    ConfigProblem,
    GeminiProblem,
    RateLimited,
    RequestRejected,
    TransientError,
)

log = logging.getLogger("gemini_web_mcp.gemini")

INIT_RETRY_GAP_SECONDS = 30.0
REQUEST_TIMEOUT_SECONDS = 180.0


@dataclass
class Turn:
    """One Gemini reply."""

    text: str
    images: list[Any]  # opaque handles for Backend.save_image
    metadata: list[Any]  # chat metadata after this turn, enough to continue the chat later
    cid: str
    rid: str


class Backend(Protocol):
    default_model: str | None

    async def ensure_ready(self) -> None: ...

    async def send(
        self,
        prompt: str,
        *,
        metadata: list[Any] | None,
        model: str | None,
        files: list[Path] | None,
        on_cid: Callable[[str], None],
    ) -> Turn: ...

    async def save_image(self, handle: Any, dest: Path) -> None: ...

    async def delete_chat(self, cid: str) -> None: ...

    async def keepalive(self) -> None: ...

    async def aclose(self) -> None: ...

    def account_note(self) -> str | None: ...


class _GentleClient(GeminiClient):
    """GeminiClient whose requests are sent once; the service decides about a second attempt.

    Upstream wraps generation in @running(retry=5) and RPCs in @running(retry=2). Both wrappers
    accept `current_retry`; 0 means a single attempt.
    """

    __slots__ = ()

    def _generate(self, *args: Any, **kwargs: Any):
        kwargs.setdefault("current_retry", 0)
        return super()._generate(*args, **kwargs)

    async def _batch_execute(self, *args: Any, **kwargs: Any):
        kwargs.setdefault("current_retry", 0)
        return await super()._batch_execute(*args, **kwargs)


class TrackedChat(ChatSession):
    """ChatSession that reports every chat id it is assigned, including ones from failed attempts,
    so end_chat can delete every chat this server created."""

    __slots__ = ("_on_cid", "_seen_cids")

    def __init__(self, on_cid: Callable[[str], None], **kwargs: Any):
        object.__setattr__(self, "_on_cid", on_cid)
        object.__setattr__(self, "_seen_cids", set())
        super().__init__(**kwargs)

    def __setattr__(self, name: str, value: Any) -> None:
        super().__setattr__(name, value)
        if name in ("metadata", "cid", "last_output"):
            cid = self.cid
            if cid and cid not in self._seen_cids:
                self._seen_cids.add(cid)
                self._on_cid(cid)


def map_error(e: BaseException) -> GeminiProblem:
    """Translate library/network exceptions. Library messages are only reused where they carry
    no request content (quota and account-status texts)."""
    if isinstance(e, GeminiProblem):
        return e
    if isinstance(e, AuthError):
        return AuthProblem(REFRESH_COOKIES)
    if isinstance(e, UsageLimitExceededError):
        return RateLimited(f"Gemini reports the account's usage limit is reached. {e} Generation is paused for 30 minutes.")
    if isinstance(e, TemporarilyBlockedError):
        return Blocked(
            "Google temporarily flagged or blocked this server (HTTP 429 / abuse protection). "
            "Generation is paused for 30 minutes; do not retry sooner."
        )
    if isinstance(e, ModelInvalidError):
        return RequestRejected(f"Gemini rejected the model for this chat: {e}")
    if isinstance(e, GeminiTimeout | asyncio.TimeoutError):
        return TransientError("Gemini did not answer in time.")
    if isinstance(e, GeminiError) and str(e).startswith("Permission denied. Account status"):
        return AccountProblem(f"Gemini refused access for this account: {e}")
    if isinstance(e, AssertionError):
        return RequestRejected("Empty prompt.")
    if isinstance(e, APIError | GeminiError):
        return TransientError(f"Gemini request failed ({type(e).__name__}).")
    return TransientError(f"Network or protocol error talking to Gemini ({type(e).__name__}).")


def _jar(cookies: list[dict[str, Any]]) -> Cookies:
    jar = Cookies()
    now = time.time()
    for c in cookies:
        expires = c.get("expires")
        if isinstance(expires, int | float) and 0 < expires < now:
            continue
        jar.set(c["name"], c["value"], domain=c.get("domain") or ".google.com", path=c.get("path") or "/", secure=True)
    return jar


class WebapiBackend:
    def __init__(self, settings: Settings):
        self._settings = settings
        self._cookie_file = CookieFile(settings.cookies_path, settings.seed_1psid, settings.seed_1psidts)
        self._client: _GentleClient | None = None
        self._lock = asyncio.Lock()
        self._last_init_failure = -1e9
        self.default_model: str | None = None
        # Safety net: the library's own cookie cache dir. Our hooks replace its writes, so it stays empty.
        lib_dir = settings.data_dir / ".gemini_webapi"
        lib_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
        os.environ["GEMINI_COOKIE_PATH"] = str(lib_dir)
        install_library_hooks(self._persist)

    def _persist(self, jar: Any) -> None:
        if self._cookie_file.save_jar(jar):
            log.info("cookies saved to cookies.json")

    def _running(self) -> bool:
        return self._client is not None and self._client.client is not None

    async def ensure_ready(self) -> None:
        if self._running():
            await self._check_account()
            return
        async with self._lock:
            if self._running():
                await self._check_account()
                return
            if time.monotonic() - self._last_init_failure < INIT_RETRY_GAP_SECONDS:
                raise TransientError("Could not reach Gemini a moment ago; try again in about 30 seconds.")
            if self._client is None:
                cookies, source = self._cookie_file.load()
                if cookies is None:
                    raise AuthProblem(NO_COOKIES)
                client = _GentleClient()
                client.cookies = _jar(cookies)
                self._client = client
                log.info("gemini client created (cookies from %s)", source)
            t0 = time.monotonic()
            try:
                await self._client.init(
                    timeout=REQUEST_TIMEOUT_SECONDS,
                    auto_close=False,
                    auto_refresh=True,
                    refresh_interval=600,
                    verbose=False,
                )
            except Exception as e:
                problem = map_error(e)
                if problem.retryable:
                    self._last_init_failure = time.monotonic()
                    problem = TransientError(f"Could not connect to Gemini ({type(e).__name__}). Try again shortly.")
                log.warning("gemini init failed (%s)", type(e).__name__)
                raise problem from None
            log.info("gemini client ready in %.1fs", time.monotonic() - t0)
            await self._check_account()
            self._resolve_default_model()
            self._persist(self._client.cookies)

    async def _check_account(self) -> None:
        assert self._client is not None
        status = self._client.account_status
        if status == AccountStatus.AVAILABLE:
            return
        log.warning("gemini account status %s", status.name)
        await self._client.close()
        if status == AccountStatus.UNAUTHENTICATED:
            raise AuthProblem(REFRESH_COOKIES)
        raise AccountProblem(f"Gemini refused access for this account: {status.name} - {status.description}")

    def _resolve_default_model(self) -> None:
        name = self._settings.gemini_model
        self.default_model = self._resolve(name).model_name if name else None

    def _resolve(self, name: str | None) -> Any:
        if not name:
            return None
        assert self._client is not None
        try:
            model = self._client.resolve_model(name)
        except ValueError:
            available = ", ".join(m.model_name for m in self._client.list_models() or [])
            raise ConfigProblem(f"GEMINI_MODEL '{name}' is not offered to this account. Available: {available}") from None
        if not model.is_available:
            raise ConfigProblem(f"Model '{name}' is listed but not usable on this account right now.")
        return model

    async def send(
        self,
        prompt: str,
        *,
        metadata: list[Any] | None,
        model: str | None,
        files: list[Path] | None,
        on_cid: Callable[[str], None],
    ) -> Turn:
        assert self._client is not None
        chat = TrackedChat(on_cid, geminiclient=self._client, metadata=list(metadata) if metadata else None,
                           model=self._resolve(model))
        try:
            out = await chat.send_message(prompt, files=[str(f) for f in files] if files else None)
        except Exception as e:
            raise map_error(e) from None
        # The library silently re-initialises after errors; never accept output from a guest session.
        await self._check_account()
        candidate = out.candidates[out.chosen]
        return Turn(
            text=out.text or "",
            images=list(candidate.generated_images),
            metadata=list(chat.metadata),
            cid=chat.cid or "",
            rid=chat.rid or "",
        )

    async def save_image(self, handle: Any, dest: Path) -> None:
        try:
            saved = await handle.save(path=str(dest.parent), filename=dest.name, full_size=True)
        except Exception as e:
            raise map_error(e) from None
        if Path(saved) != dest.resolve():
            raise TransientError("Image download landed in an unexpected place.")

    async def delete_chat(self, cid: str) -> None:
        assert self._client is not None
        try:
            await self._client.delete_chat(cid)
        except Exception as e:
            raise map_error(e) from None

    async def keepalive(self) -> None:
        """Re-open a client the library closed after an error, so cookie refresh keeps running."""
        if self._client is not None and not self._running():
            await self.ensure_ready()

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.close()  # the library saves cookies on close -> our hook

    def account_note(self) -> str | None:
        status = self._client.abuse_status if self._client is not None else None
        if status and not status.get("is_clean", True):
            return f"Gemini reports a restriction flag on the account (status {status.get('status_code')})."
        return None
