"""Google cookie persistence.

/data/cookies.json is the source of truth once it exists. GEMINI_1PSID / GEMINI_1PSIDTS are
only the seed. The file remembers a fingerprint of the seed it grew from, so changing the
seed variables on Railway makes the next boot start from the new seed even if an old file
is still on the volume. Cookie values are never logged.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import logging
import os
from pathlib import Path
from typing import Any

log = logging.getLogger("gemini_web_mcp.cookies")

_AUTH = ("__Secure-1PSID", "__Secure-1PSIDTS")


def seed_fingerprint(psid: str | None, psidts: str | None) -> str | None:
    if not psid:
        return None
    return hashlib.sha256(f"{psid}\n{psidts or ''}".encode()).hexdigest()


def _is_google(domain: str | None) -> bool:
    d = (domain or "").lstrip(".").lower()
    return d == "google.com" or d.endswith(".google.com")


def serialize_jar(jar: Any) -> list[dict[str, Any]]:
    """Mirror gemini_webapi's own cache format: auth cookies + unexpired google.com cookies."""
    out = []
    for c in getattr(jar, "jar", jar):
        if not _is_google(c.domain) or not c.value:
            continue
        if c.name in _AUTH or (c.expires is not None and not c.is_expired()):
            out.append({"name": c.name, "value": c.value, "domain": c.domain, "path": c.path, "expires": c.expires})
    return out


class CookieFile:
    def __init__(self, path: Path, seed_psid: str | None, seed_psidts: str | None):
        self.path = path
        self._seed_psid = seed_psid
        self._seed_psidts = seed_psidts
        self._fingerprint = seed_fingerprint(seed_psid, seed_psidts)

    def load(self) -> tuple[list[dict[str, Any]] | None, str]:
        """Return (cookies, source). source is 'file', 'seed' or 'none'."""
        data = self._read()
        if data is not None:
            stored_fp = data.get("seed_fingerprint")
            if self._fingerprint is None or stored_fp == self._fingerprint:
                cookies = [c for c in data.get("cookies", []) if isinstance(c, dict) and c.get("name") and c.get("value")]
                if any(c["name"] == "__Secure-1PSID" for c in cookies):
                    return cookies, "file"
            else:
                log.info("cookie seed variables changed since cookies.json was written; using the new seed")
        if self._seed_psid:
            cookies = [{"name": "__Secure-1PSID", "value": self._seed_psid, "domain": ".google.com", "path": "/"}]
            if self._seed_psidts:
                cookies.append(
                    {"name": "__Secure-1PSIDTS", "value": self._seed_psidts, "domain": ".google.com", "path": "/"}
                )
            return cookies, "seed"
        return None, "none"

    def _read(self) -> dict[str, Any] | None:
        try:
            raw = self.path.read_text()
        except FileNotFoundError:
            return None
        except OSError as e:
            log.warning("cannot read cookies.json (%s)", type(e).__name__)
            return None
        try:
            data = json.loads(raw)
        except ValueError:
            log.warning("cookies.json is not valid JSON; ignoring it")
            return None
        return data if isinstance(data, dict) else None

    def save_jar(self, jar: Any) -> bool:
        cookies = serialize_jar(jar)
        if not any(c["name"] == "__Secure-1PSID" for c in cookies):
            return False
        self._write({"version": 1, "seed_fingerprint": self._fingerprint, "cookies": cookies})
        return True

    def _write(self, data: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_name(f".{self.path.name}.tmp")
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        try:
            os.write(fd, json.dumps(data).encode())
            os.fsync(fd)
        finally:
            os.close(fd)
        os.chmod(tmp, 0o600)
        os.replace(tmp, self.path)


def install_library_hooks(on_save) -> None:
    """Route every cookie save gemini_webapi makes (after each refresh and on close) to `on_save`.

    The library would otherwise write its own cache file named after the __Secure-1PSID value.
    Pinned to gemini_webapi==2.1.1; tests/test_library_contract.py fails if these hooks move.
    """

    def hook(cookies: Any, verbose: bool = False) -> None:
        try:
            on_save(cookies)
        except Exception as e:  # never let persistence break a refresh
            log.error("saving refreshed cookies failed (%s)", type(e).__name__)

    for module_name in ("gemini_webapi.utils.rotate_1psidts", "gemini_webapi.client"):
        module = importlib.import_module(module_name)
        if not hasattr(module, "save_cookies"):
            raise RuntimeError(f"{module_name}.save_cookies not found; gemini_webapi version changed?")
        module.save_cookies = hook
