"""MCP server (Streamable HTTP, stateless) behind a path secret, plus the public image route."""

from __future__ import annotations

import asyncio
import contextlib
import hmac
import logging
import re
import time
from collections.abc import AsyncIterator
from typing import Annotated

from mcp.server.mcpserver import MCPServer
from mcp_types import CallToolResult, ToolAnnotations
from pydantic import Field
from starlette.applications import Starlette
from starlette.responses import FileResponse, PlainTextResponse
from starlette.routing import Route
from starlette.types import Receive, Scope, Send

from . import results
from .images import MIME
from .service import IMAGE_RE as IMAGE_ID_RE
from .service import SESSION_RE, ImageService, UserError

log = logging.getLogger("gemini_web_mcp.http")

MCP_PREFIX = "/mcp/"
IMAGE_PATH_RE = re.compile(r"^/i/([0-9a-f]{32})/([0-9a-f]{16})\.(png|jpg|webp|gif)$")

SERVER_INSTRUCTIONS = (
    "Generates and edits images with the user's Gemini account. Call generate_images without session_id "
    "the first time, then reuse the returned session_id for the whole conversation. Use end_chat only when "
    "the user says end chat / delete everything / wipe."
)

GENERATE_DESCRIPTION = (
    "Generate images with the user's Gemini account. Omit session_id on the first call; reuse the returned "
    "one for the whole conversation.\n"
    "Turn the user's idea into one natural paragraph per image (40-120 words), not a keyword list: subject "
    "and details, action, setting, composition + camera (shot type, angle, lens) for photos or medium + "
    "technique for art, lighting, palette, mood, and the aspect ratio in words. Start each prompt with "
    "\"Generate an image:\". Keep every detail the user gave; add no text, logos or watermarks unless asked; "
    "text to render goes in \"double quotes\" with a described font. Several images of one idea = distinct "
    "prompts varying composition/style/angle, same core subject. If the user says \"exact\" or \"raw\", send "
    "their words unchanged. Afterwards show each link with its final prompt, one line each."
)

EDIT_DESCRIPTION = (
    "Edit an image made earlier in this conversation. Continues that image's Gemini chat so the same picture "
    "is edited; returns a new image_id and link, the original stays. Use the session_id from generate_images. "
    "Write the instruction as: \"Change X to Y. Keep everything else exactly the same.\""
)

END_DESCRIPTION = (
    "Permanently delete sessions: removes every Gemini chat they created from the user's Gemini history and "
    "deletes all their stored images and previews, so their links stop working. Call only when the user says "
    "end chat / delete everything / wipe. Then confirm what was deleted from Gemini and that the links are dead."
)

SessionIdArg = Annotated[
    str,
    Field(description="session_id returned by generate_images (reuse it for the whole conversation)."),
]


def build_mcp(service: ImageService) -> MCPServer:
    mcp = MCPServer("gemini-web-mcp", instructions=SERVER_INSTRUCTIONS, log_level="WARNING")
    budget = service.settings.result_char_budget

    @mcp.tool(
        name="generate_images",
        description=GENERATE_DESCRIPTION,
        annotations=ToolAnnotations(
            title="Generate images (Gemini)", read_only_hint=False, destructive_hint=False,
            idempotent_hint=False, open_world_hint=True,
        ),
    )
    async def generate_images(
        prompts: Annotated[
            list[str],
            Field(min_length=1, max_length=4, description="1-4 final prompts, one per image, each starting with \"Generate an image:\"."),
        ],
        user_request: Annotated[str, Field(description="The user's original words, verbatim.")],
        session_id: Annotated[
            str | None,
            Field(description="Omit on the first call; afterwards reuse the returned session_id for the whole conversation."),
        ] = None,
    ) -> CallToolResult:
        return await _guard(
            "generate_images",
            lambda: _generate(service, prompts, user_request, session_id, budget),
        )

    @mcp.tool(
        name="edit_image",
        description=EDIT_DESCRIPTION,
        annotations=ToolAnnotations(
            title="Edit an image (Gemini)", read_only_hint=False, destructive_hint=False,
            idempotent_hint=False, open_world_hint=True,
        ),
    )
    async def edit_image(
        session_id: SessionIdArg,
        image_id: Annotated[str, Field(description="image_id of the picture to edit, from an earlier result.")],
        instruction: Annotated[
            str, Field(description="\"Change X to Y. Keep everything else exactly the same.\"")
        ],
    ) -> CallToolResult:
        return await _guard("edit_image", lambda: _edit(service, session_id, image_id, instruction, budget))

    @mcp.tool(
        name="end_chat",
        description=END_DESCRIPTION,
        annotations=ToolAnnotations(
            title="End chat: delete Gemini chats and images", read_only_hint=False, destructive_hint=True,
            idempotent_hint=True, open_world_hint=True,
        ),
    )
    async def end_chat(
        session_ids: Annotated[
            list[str], Field(min_length=1, description="Every session_id used in this conversation.")
        ],
    ) -> CallToolResult:
        return await _guard("end_chat", lambda: _end(service, session_ids))

    return mcp


async def _generate(service: ImageService, prompts, user_request, session_id, budget) -> CallToolResult:
    outcome = await service.generate_images(prompts, user_request, session_id)
    return results.generate_result(outcome, budget)


async def _edit(service: ImageService, session_id, image_id, instruction, budget) -> CallToolResult:
    outcome, source = await service.edit_image(session_id, image_id, instruction)
    source_url = service.image_url(source.session_id, source.id, source.ext)
    return results.edit_result(outcome, source, source_url, budget)


async def _end(service: ImageService, session_ids) -> CallToolResult:
    return results.end_result(await service.end_sessions(session_ids))


async def _guard(name: str, run) -> CallToolResult:
    """Tools never raise: errors become tool results, and only the error type is logged."""
    t0 = time.monotonic()
    try:
        result = await run()
    except UserError as e:
        result = results.error_result(str(e))
    except Exception as e:
        log.error("%s crashed (%s)", name, type(e).__name__)
        result = results.error_result(f"Internal server error ({type(e).__name__}).")
    log.info("tool %s done in %.1fs error=%s", name, time.monotonic() - t0, bool(result.is_error))
    return result


_NOT_FOUND = PlainTextResponse("Not Found", status_code=404)


class Gate:
    """Single ASGI entry point: /mcp/{secret}, GET /i/{session}/{image}.{ext}, 404 for everything else."""

    def __init__(self, secret: str, mcp_asgi, service: ImageService):
        self._secret = secret.encode()
        self._mcp = mcp_asgi
        self._service = service

    def _secret_ok(self, candidate: str) -> bool:
        return hmac.compare_digest(candidate.encode("utf-8", "replace"), self._secret)

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        path: str = scope["path"]
        method: str = scope["method"]
        t0 = time.monotonic()
        status = {"code": 0}

        async def send_logged(message) -> None:
            if message["type"] == "http.response.start":
                status["code"] = message["status"]
            await send(message)

        if path.startswith(MCP_PREFIX) and self._secret_ok(path[len(MCP_PREFIX):]):
            kind = "mcp"
            await self._mcp(scope, receive, send_logged)
        elif (m := IMAGE_PATH_RE.match(path)) and method in ("GET", "HEAD"):
            kind = f"image sid={m.group(1)} iid={m.group(2)}"
            await self._image(m.group(1), m.group(2), m.group(3))(scope, receive, send_logged)
        else:
            kind = "other"  # never log the raw path: it may carry a (wrong) secret
            await _NOT_FOUND(scope, receive, send_logged)
        log.info("%s %s -> %d %.0fms", method, kind, status["code"], (time.monotonic() - t0) * 1000)

    def _image(self, sid: str, iid: str, ext: str):
        if not (SESSION_RE.match(sid) and IMAGE_ID_RE.match(iid)):
            return _NOT_FOUND
        record = self._service.store.get_image(sid, iid)
        if record is None or record.ext != ext:
            return _NOT_FOUND
        path = self._service.image_path(sid, iid, ext)
        if not path.is_file():
            return _NOT_FOUND
        return FileResponse(
            path,
            media_type=MIME[ext],
            headers={
                "Cache-Control": "no-store",
                "X-Content-Type-Options": "nosniff",
                "Referrer-Policy": "no-referrer",
                "Content-Disposition": f'inline; filename="{iid}.{ext}"',
            },
        )


def build_app(service: ImageService, *, warm_up: bool = True) -> Starlette:
    mcp = build_mcp(service)
    # Creates the session manager; we mount its ASGI handler behind our own gate.
    mcp.streamable_http_app(stateless_http=True, json_response=True, host="0.0.0.0")
    session_manager = mcp.session_manager
    gate = Gate(service.settings.path_secret, session_manager.handle_request, service)

    @contextlib.asynccontextmanager
    async def lifespan(_app: Starlette) -> AsyncIterator[None]:
        async with session_manager.run():
            tasks = [asyncio.create_task(service.maintenance_loop())]
            if warm_up:
                tasks.append(asyncio.create_task(service.warm_up()))
            try:
                yield
            finally:
                for t in tasks:
                    t.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
                with contextlib.suppress(Exception):
                    await service.backend.aclose()

    # One catch-all route: Gate decides; Starlette only provides lifespan and error handling.
    return Starlette(routes=[Route("/{path:path}", endpoint=gate)], lifespan=lifespan)
