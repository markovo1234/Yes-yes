"""The real HTTP app on a local port (fake Gemini): routing, 404s, and the MCP protocol end to end."""

from __future__ import annotations

import json
import logging
import re
import urllib.error
import urllib.request

import pytest
from mcp import Client

from mcp_types import ImageContent

pytestmark = pytest.mark.anyio

_NO_PROXY = urllib.request.build_opener(urllib.request.ProxyHandler({}))

TOOLS_CALL = json.dumps(
    {"jsonrpc": "2.0", "id": 1, "method": "tools/call",
     "params": {"name": "generate_images", "arguments": {"prompts": ["x"], "user_request": "x"}}}
).encode()
INIT = json.dumps(
    {"jsonrpc": "2.0", "id": 1, "method": "initialize",
     "params": {"protocolVersion": "2025-06-18", "capabilities": {}, "clientInfo": {"name": "t", "version": "1"}}}
).encode()


def http(method: str, url: str, body: bytes | None = None) -> tuple[int, bytes, dict]:
    req = urllib.request.Request(url, data=body, method=method, headers={
        "Content-Type": "application/json", "Accept": "application/json, text/event-stream",
        "MCP-Protocol-Version": "2025-06-18",
    })
    try:
        with _NO_PROXY.open(req, timeout=30) as resp:
            return resp.status, resp.read(), dict(resp.headers)
    except urllib.error.HTTPError as e:
        return e.code, e.read(), dict(e.headers)


def test_wrong_or_missing_secret_is_404_and_gemini_never_called(live, backend, caplog):
    caplog.set_level(logging.DEBUG)
    secret = live.service.settings.path_secret
    bad_paths = [
        "/mcp", "/mcp/", f"/mcp/{secret[:-1]}", f"/mcp/{secret}x", f"/mcp/{secret}/", f"/mcp/{secret}/extra",
        f"/MCP/{secret}", f"/{secret}", "/mcp/wrong", "/", "/health", "/i/", "/i/abc/def.png", "/favicon.ico",
    ]
    for path in bad_paths:
        for method in ("GET", "POST", "DELETE"):
            status, _, _ = http(method, live.base + path, TOOLS_CALL if method == "POST" else None)
            assert status == 404, (method, path, status)
    assert backend.sent == [] and backend.ensure_calls == 0
    assert secret not in caplog.text


def test_right_secret_reaches_mcp(live):
    status, body, _ = http("POST", live.mcp_url, INIT)
    assert status == 200
    assert json.loads(body)["result"]["serverInfo"]["name"] == "gemini-web-mcp"


@pytest.mark.parametrize("mode", ["legacy", "auto"])  # initialize handshake (2025-xx) and 2026-07-28
async def test_tools_list_is_exactly_the_three_tools(live, mode):
    async with Client(live.mcp_url, mode=mode) as client:
        listing = await client.list_tools()
    tools = {t.name: t for t in listing.tools}
    assert set(tools) == {"generate_images", "edit_image", "end_chat"}
    gen = tools["generate_images"]
    assert len(gen.description) <= 1200
    for phrase in ("Omit session_id on the first call", "Generate an image:", "40-120 words", '"exact" or "raw"',
                   "Afterwards show each link with its final prompt, one line each"):
        assert phrase in gen.description
    assert "Change X to Y. Keep everything else exactly the same." in tools["edit_image"].description
    assert tools["end_chat"].annotations.destructive_hint is True
    assert "end chat / delete everything / wipe" in tools["end_chat"].description
    schema = gen.input_schema
    assert schema["required"] == ["prompts", "user_request"]
    assert schema["properties"]["prompts"]["minItems"] == 1 and schema["properties"]["prompts"]["maxItems"] == 4


async def test_generate_edit_end_over_mcp(live, backend, caplog):
    caplog.set_level(logging.DEBUG)
    async with Client(live.mcp_url, mode="legacy") as client:
        gen = await client.call_tool(
            "generate_images",
            {"prompts": ["Generate an image: SECRET-PROMPT-A", "Generate an image: SECRET-PROMPT-B",
                         "Generate an image: SECRET-PROMPT-C"],
             "user_request": "SECRET-USER-WORDS"},
        )
        assert not gen.is_error
        text = gen.content[0].text
        sid = re.search(r"session_id: ([0-9a-f]{32})", text).group(1)
        links = re.findall(r"link: (\S+)", text)
        ids = re.findall(r"image_id: ([0-9a-f]{16})", text)
        assert len(links) == 3 and len(ids) == 3
        assert sum(isinstance(c, ImageContent) for c in gen.content) == 3
        assert len(gen.model_dump_json(by_alias=True, exclude_none=True)) < 120_000
        for link in links:
            status, body, headers = http("GET", link)
            assert status == 200 and headers["content-type"] == "image/png" and body[:4] == b"\x89PNG"
            assert headers["cache-control"] == "no-store"

        edit = await client.call_tool(
            "edit_image",
            {"session_id": sid, "image_id": ids[0],
             "instruction": "Change SECRET-EDIT to blue. Keep everything else exactly the same."},
        )
        assert not edit.is_error
        new_link = re.search(r"link: (\S+)", edit.content[0].text).group(1)
        assert new_link not in links
        assert http("GET", new_link)[0] == 200 and http("GET", links[0])[0] == 200  # original still works

        end = await client.call_tool("end_chat", {"session_ids": [sid]})
        assert not end.is_error
        assert "deleted 3 Gemini chat(s)" in end.content[0].text

    for link in [*links, new_link]:
        assert http("GET", link)[0] == 404
    assert len(backend.deleted) == 3
    for secret in ("SECRET-PROMPT", "SECRET-USER-WORDS", "SECRET-EDIT", live.service.settings.path_secret):
        assert secret not in caplog.text


async def test_bad_arguments_are_tool_errors(live, backend):
    async with Client(live.mcp_url) as client:
        too_many = await client.call_tool("generate_images", {"prompts": ["a"] * 5, "user_request": "x"})
        unknown = await client.call_tool("generate_images", {"prompts": ["a"], "user_request": "x", "session_id": "nope"})
    assert too_many.is_error and unknown.is_error
    assert "Omit session_id" in unknown.content[0].text
    assert backend.sent == []


def test_image_route_is_strict(live):
    service = live.service
    sid = "a" * 32
    service.store.create_session(sid, 0)
    for path in (f"/i/{sid}/{'b' * 16}.png", f"/i/{sid}/{'b' * 16}.exe", f"/i/{sid.upper()}/{'b' * 16}.png"):
        assert http("GET", live.base + path)[0] == 404
