"""Cookie seed vs /data/cookies.json, persistence on refresh, and the not-signed-in path. No network."""

from __future__ import annotations

import importlib
import json
import stat
from types import SimpleNamespace

import pytest
from curl_cffi.requests import Cookies
from gemini_webapi.constants import AccountStatus

from app import backend as backend_mod
from app.backend import WebapiBackend
from app.cookies import CookieFile
from app.errors import AuthProblem
from tests.conftest import make_settings

pytestmark = pytest.mark.anyio


def _jar(psid: str, psidts: str) -> Cookies:
    jar = Cookies()
    jar.set("__Secure-1PSID", psid, domain=".google.com", path="/", secure=True)
    jar.set("__Secure-1PSIDTS", psidts, domain=".google.com", path="/", secure=True)
    return jar


def _mode(path) -> int:
    return stat.S_IMODE(path.stat().st_mode)


def test_seed_used_when_no_file(tmp_path):
    cookies, source = CookieFile(tmp_path / "cookies.json", "seed-psid", "seed-ts").load()
    assert source == "seed"
    assert {c["name"]: c["value"] for c in cookies} == {"__Secure-1PSID": "seed-psid", "__Secure-1PSIDTS": "seed-ts"}


def test_file_wins_over_seed_and_is_owner_only(tmp_path):
    path = tmp_path / "cookies.json"
    cf = CookieFile(path, "seed-psid", "seed-ts")
    assert cf.save_jar(_jar("seed-psid", "rotated-ts"))
    assert _mode(path) == 0o600
    cookies, source = CookieFile(path, "seed-psid", "seed-ts").load()
    assert source == "file"
    assert {c["name"]: c["value"] for c in cookies}["__Secure-1PSIDTS"] == "rotated-ts"


def test_changed_seed_variables_replace_old_file(tmp_path):
    path = tmp_path / "cookies.json"
    CookieFile(path, "old-psid", "old-ts").save_jar(_jar("old-psid", "rotated"))
    cookies, source = CookieFile(path, "new-psid", "new-ts").load()
    assert source == "seed" and cookies[0]["value"] == "new-psid"


def test_file_used_when_seed_variables_removed(tmp_path):
    path = tmp_path / "cookies.json"
    CookieFile(path, "old-psid", "old-ts").save_jar(_jar("old-psid", "rotated"))
    _, source = CookieFile(path, None, None).load()
    assert source == "file"


def test_no_cookies_at_all(tmp_path):
    assert CookieFile(tmp_path / "cookies.json", None, None).load() == (None, "none")


def _fake_init(status: AccountStatus):
    async def init(self, **kwargs):
        # Stand-in for the network init: the "session" keeps the cookies it was given.
        async def close():
            pass

        self.client = SimpleNamespace(cookies=self._cookies, close=close)
        self.account_status = status

    return init


def _no_close():
    async def close(self, delay: float = 0):
        self.client = None

    return close


async def test_backend_seeds_then_persists_to_data_cookies_json(tmp_path, monkeypatch):
    monkeypatch.setattr(backend_mod._GentleClient, "init", _fake_init(AccountStatus.AVAILABLE))
    settings = make_settings(tmp_path, seed_1psid="seed-psid", seed_1psidts="seed-ts")
    b = WebapiBackend(settings)
    await b.ensure_ready()
    saved = json.loads(settings.cookies_path.read_text())
    assert {c["name"]: c["value"] for c in saved["cookies"]}["__Secure-1PSID"] == "seed-psid"
    assert _mode(settings.cookies_path) == 0o600
    assert list((settings.data_dir / ".gemini_webapi").iterdir()) == []  # library cache never written


async def test_backend_prefers_data_cookies_json(tmp_path, monkeypatch):
    monkeypatch.setattr(backend_mod._GentleClient, "init", _fake_init(AccountStatus.AVAILABLE))
    settings = make_settings(tmp_path, seed_1psid="seed-psid", seed_1psidts="seed-ts")
    CookieFile(settings.cookies_path, "seed-psid", "seed-ts").save_jar(_jar("seed-psid", "rotated-ts"))
    b = WebapiBackend(settings)
    await b.ensure_ready()
    values = {c.name: c.value for c in b._client.cookies.jar}
    assert values["__Secure-1PSIDTS"] == "rotated-ts"


async def test_library_refresh_writes_data_cookies_json(tmp_path):
    settings = make_settings(tmp_path, seed_1psid="seed-psid", seed_1psidts="seed-ts")
    WebapiBackend(settings)  # installs the hooks
    rotate_module = importlib.import_module("gemini_webapi.utils.rotate_1psidts")
    rotate_module.save_cookies(_jar("seed-psid", "after-refresh"))  # what rotate_1psidts() calls
    saved = json.loads(settings.cookies_path.read_text())
    assert {c["name"]: c["value"] for c in saved["cookies"]}["__Secure-1PSIDTS"] == "after-refresh"
    assert _mode(settings.cookies_path) == 0o600
    client_module = importlib.import_module("gemini_webapi.client")
    client_module.save_cookies(_jar("seed-psid", "on-close"))  # what GeminiClient.close() calls
    saved = json.loads(settings.cookies_path.read_text())
    assert {c["name"]: c["value"] for c in saved["cookies"]}["__Secure-1PSIDTS"] == "on-close"


async def test_not_signed_in_gives_refresh_message_and_keeps_file(tmp_path, monkeypatch):
    monkeypatch.setattr(backend_mod._GentleClient, "init", _fake_init(AccountStatus.UNAUTHENTICATED))
    monkeypatch.setattr(backend_mod._GentleClient, "close", _no_close())
    settings = make_settings(tmp_path, seed_1psid="seed-psid", seed_1psidts="seed-ts")
    b = WebapiBackend(settings)
    with pytest.raises(AuthProblem) as err:
        await b.ensure_ready()
    assert "Refresh your cookies" in err.value.message
    assert not settings.cookies_path.exists()  # an unauthenticated session is never persisted


async def test_missing_cookies_is_a_clear_auth_problem(tmp_path):
    b = WebapiBackend(make_settings(tmp_path))
    with pytest.raises(AuthProblem) as err:
        await b.ensure_ready()
    assert "GEMINI_1PSID" in err.value.message
