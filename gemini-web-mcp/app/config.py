"""Environment configuration. Secrets are validated here but never printed."""

from __future__ import annotations

import base64
import os
import re
from dataclasses import dataclass
from pathlib import Path

_B64URL = re.compile(r"^[A-Za-z0-9_-]+$")


class ConfigError(RuntimeError):
    pass


def _int_env(name: str, default: int, minimum: int = 1) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        raise ConfigError(f"{name} must be an integer") from None
    if value < minimum:
        raise ConfigError(f"{name} must be >= {minimum}")
    return value


def validate_path_secret(secret: str) -> None:
    """MCP_PATH_SECRET must be base64url and decode to at least 32 random bytes."""
    if not secret:
        raise ConfigError("MCP_PATH_SECRET is not set")
    if not _B64URL.match(secret):
        raise ConfigError("MCP_PATH_SECRET must be base64url (A-Z a-z 0-9 _ -, no padding)")
    try:
        raw = base64.urlsafe_b64decode(secret + "=" * (-len(secret) % 4))
    except ValueError:
        raise ConfigError("MCP_PATH_SECRET is not valid base64url") from None
    if len(raw) < 32:
        raise ConfigError("MCP_PATH_SECRET must encode at least 32 random bytes (43+ characters)")


@dataclass(frozen=True)
class Settings:
    port: int
    data_dir: Path
    path_secret: str
    public_base_url: str
    max_images_per_hour: int
    session_ttl_hours: int
    gemini_model: str | None
    seed_1psid: str | None
    seed_1psidts: str | None
    # Tunables that tests shorten; production uses the spec values.
    deadline_seconds: float = 200.0
    max_concurrency: int = 2
    max_attempts: int = 2
    sweep_interval_seconds: float = 1800.0
    result_char_budget: int = 120_000

    @property
    def db_path(self) -> Path:
        return self.data_dir / "state.db"

    @property
    def cookies_path(self) -> Path:
        return self.data_dir / "cookies.json"

    @property
    def images_dir(self) -> Path:
        return self.data_dir / "images"

    @classmethod
    def from_env(cls) -> Settings:
        secret = os.environ.get("MCP_PATH_SECRET", "").strip()
        validate_path_secret(secret)
        port = _int_env("PORT", 8080)

        base = os.environ.get("PUBLIC_BASE_URL", "").strip().rstrip("/")
        if not base:
            domain = os.environ.get("RAILWAY_PUBLIC_DOMAIN", "").strip()
            base = f"https://{domain}" if domain else f"http://localhost:{port}"

        model = os.environ.get("GEMINI_MODEL", "").strip() or None
        return cls(
            port=port,
            data_dir=Path(os.environ.get("DATA_DIR", "/data")),
            path_secret=secret,
            public_base_url=base,
            max_images_per_hour=_int_env("MAX_IMAGES_PER_HOUR", 30),
            session_ttl_hours=_int_env("SESSION_TTL_HOURS", 24),
            gemini_model=model,
            seed_1psid=os.environ.get("GEMINI_1PSID", "").strip() or None,
            seed_1psidts=os.environ.get("GEMINI_1PSIDTS", "").strip() or None,
        )
