from __future__ import annotations

import pytest

from app.config import ConfigError, Settings, validate_path_secret
from tests.conftest import new_secret


def test_secret_must_be_long_base64url():
    validate_path_secret(new_secret())
    for bad in ("", "short", "a" * 42, "x" * 50 + "/", "x" * 50 + "="):
        with pytest.raises(ConfigError):
            validate_path_secret(bad)


def test_settings_from_env(monkeypatch, tmp_path):
    secret = new_secret()
    monkeypatch.setenv("MCP_PATH_SECRET", secret)
    monkeypatch.setenv("PORT", "9123")
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("RAILWAY_PUBLIC_DOMAIN", "gemini-web-mcp.up.railway.app")
    monkeypatch.delenv("PUBLIC_BASE_URL", raising=False)
    s = Settings.from_env()
    assert s.port == 9123 and s.public_base_url == "https://gemini-web-mcp.up.railway.app"
    assert s.max_images_per_hour == 30 and s.session_ttl_hours == 24 and s.deadline_seconds == 200
    assert s.max_concurrency == 2 and s.max_attempts == 2 and s.result_char_budget == 120_000


def test_missing_secret_refuses_to_start(monkeypatch):
    monkeypatch.delenv("MCP_PATH_SECRET", raising=False)
    with pytest.raises(ConfigError) as err:
        Settings.from_env()
    assert "MCP_PATH_SECRET" in str(err.value)
