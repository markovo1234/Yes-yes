"""Tool logic: sessions, concurrency, deadlines, attempts, hourly budget, pauses, cleanup.

Logs carry only ids, counts, timings and error types - never prompts, Gemini text or image data.
"""

from __future__ import annotations

import asyncio
import logging
import re
import secrets
import shutil
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .backend import Backend, Turn
from .config import Settings
from .errors import GeminiProblem, TransientError
from .images import detect_ext, make_preview
from .store import ImageRecord, Store

log = logging.getLogger("gemini_web_mcp.service")

SESSION_RE = re.compile(r"^[0-9a-f]{32}$")
IMAGE_RE = re.compile(r"^[0-9a-f]{16}$")
MAX_GEMINI_TEXT = 1500
RETRY_PAUSE_SECONDS = 3.0
MIN_SECONDS_FOR_RETRY = 30.0
# New chats use the configured default model, known only once the backend is connected.
DEFAULT_MODEL = object()


class UserError(Exception):
    """Bad input from the caller; message is shown as a tool error."""


@dataclass
class SavedImage:
    image_id: str
    url: str
    prompt: str
    gemini_text: str
    preview: Path


@dataclass
class Failure:
    label: str  # e.g. "prompt 2"
    prompt: str
    reason: str
    refusal: bool = False


@dataclass
class Outcome:
    session_id: str
    images: list[SavedImage] = field(default_factory=list)
    failures: list[Failure] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


@dataclass
class EndReport:
    session_id: str
    found: bool
    chats_deleted: int = 0
    chat_failures: list[str] = field(default_factory=list)
    images_deleted: int = 0
    session_deleted: bool = False


class Pause:
    """Stops all Gemini traffic after auth/account/rate/abuse problems."""

    def __init__(self, clock: Callable[[], float]):
        self._clock = clock
        self._message: str | None = None
        self._until: float | None = None  # None with a message = until restart

    def trip(self, problem: GeminiProblem) -> None:
        if problem.pause_seconds == 0:
            return
        self._message = problem.message
        self._until = None if problem.pause_seconds is None else self._clock() + problem.pause_seconds
        log.warning("gemini paused (%s)", type(problem).__name__)

    def check(self) -> str | None:
        if self._message is None:
            return None
        if self._until is not None and self._clock() >= self._until:
            self._message = self._until = None
            return None
        if self._until is not None:
            resume = datetime.fromtimestamp(self._until, UTC).strftime("%H:%M UTC")
            return f"{self._message} (Paused until {resume}.)"
        return self._message


class ImageService:
    def __init__(self, settings: Settings, store: Store, backend: Backend, clock: Callable[[], float] = time.time):
        self.settings = settings
        self.store = store
        self.backend = backend
        self.clock = clock
        self.pause = Pause(clock)
        self._slots = asyncio.Semaphore(settings.max_concurrency)
        self._reserved = 0

    # ------------------------------------------------------------------ helpers
    def image_url(self, sid: str, iid: str, ext: str) -> str:
        return f"{self.settings.public_base_url}/i/{sid}/{iid}.{ext}"

    def image_path(self, sid: str, iid: str, ext: str) -> Path:
        return self.settings.images_dir / sid / f"{iid}.{ext}"

    def preview_path(self, sid: str, iid: str) -> Path:
        return self.settings.images_dir / sid / f"{iid}.preview.jpg"

    def _reserve(self) -> bool:
        used = self.store.images_since(self.clock() - 3600)
        if used + self._reserved >= self.settings.max_images_per_hour:
            return False
        self._reserved += 1
        return True

    def _release(self) -> None:
        self._reserved -= 1

    def _budget_message(self) -> str:
        limit = self.settings.max_images_per_hour
        oldest = self.store.oldest_image_since(self.clock() - 3600)
        when = ""
        if oldest is not None:
            when = " The next image is allowed after " + datetime.fromtimestamp(oldest + 3600, UTC).strftime("%H:%M UTC") + "."
        return f"Hourly image limit reached ({limit} images per rolling hour, MAX_IMAGES_PER_HOUR).{when}"

    def budget_line(self) -> str:
        used = self.store.images_since(self.clock() - 3600)
        return f"Hourly budget: {used}/{self.settings.max_images_per_hour} images used."

    def _resolve_session(self, session_id: str | None) -> str:
        now = self.clock()
        if session_id is None or session_id == "":
            sid = secrets.token_hex(16)
            self.store.create_session(sid, now)
            log.info("session created sid=%s", sid)
            return sid
        if not SESSION_RE.match(session_id) or not self.store.session_exists(session_id):
            raise UserError(
                "Unknown session_id (it may have been ended). Omit session_id to start a new session."
            )
        self.store.touch_session(session_id, now)
        return session_id

    def _record_chat(self, sid: str, model: str | None) -> Callable[[str], None]:
        def on_cid(cid: str) -> None:
            self.store.add_chat(sid, cid, model, self.clock())
            log.info("gemini chat tracked sid=%s", sid)

        return on_cid

    async def _save_turn(
        self, sid: str, turn: Turn, prompt: str, user_request: str | None, parent_id: str | None, label: str
    ) -> tuple[list[SavedImage], list[Failure]]:
        saved: list[SavedImage] = []
        failures: list[Failure] = []
        folder = self.settings.images_dir / sid
        folder.mkdir(mode=0o700, parents=True, exist_ok=True)
        text = _clip(turn.text)
        for index, handle in enumerate(turn.images):
            iid = secrets.token_hex(8)
            raw = folder / f"{iid}.download"
            try:
                await self.backend.save_image(handle, raw)
                ext = await asyncio.to_thread(detect_ext, raw)
                final = self.image_path(sid, iid, ext)
                raw.replace(final)
                preview = self.preview_path(sid, iid)
                await asyncio.to_thread(make_preview, final, preview)
            except (GeminiProblem, OSError, ValueError) as e:
                raw.unlink(missing_ok=True)
                log.warning("image download failed sid=%s (%s)", sid, type(e).__name__)
                failures.append(Failure(label, prompt, f"Gemini made an image but downloading it failed ({type(e).__name__})."))
                continue
            self.store.add_image(
                ImageRecord(
                    id=iid, session_id=sid, cid=turn.cid, rid=turn.rid, metadata=turn.metadata, turn_index=index,
                    turn_count=len(turn.images), ext=ext, prompt=prompt, user_request=user_request,
                    parent_id=parent_id, created_at=self.clock(),
                )
            )
            saved.append(SavedImage(iid, self.image_url(sid, iid, ext), prompt, text, preview))
        if turn.cid and turn.rid:
            self.store.set_chat_last_rid(sid, turn.cid, turn.rid)
        return saved, failures

    async def _one_request(
        self,
        *,
        sid: str,
        label: str,
        prompt: str,
        deadline: float,
        metadata: list[Any] | None,
        model: Any,  # model name, None (account default) or DEFAULT_MODEL
        files: list[Path] | None,
        user_request: str | None,
        parent_id: str | None,
    ) -> tuple[list[SavedImage], list[Failure]]:
        """One Gemini request (<= max_attempts attempts) inside a concurrency slot."""
        async with self._slots:
            paused = self.pause.check()
            if paused:
                return [], [Failure(label, prompt, paused)]
            if not self._reserve():
                return [], [Failure(label, prompt, self._budget_message())]
            try:
                last: GeminiProblem | None = None
                made = 0
                for attempt in range(1, self.settings.max_attempts + 1):
                    if attempt > 1:
                        paused = self.pause.check()
                        if paused:
                            return [], [Failure(label, prompt, paused)]
                        if deadline - self.clock() < MIN_SECONDS_FOR_RETRY:
                            break
                        await asyncio.sleep(RETRY_PAUSE_SECONDS)
                    made = attempt
                    t0 = time.monotonic()
                    try:
                        await self.backend.ensure_ready()
                        use_model = self.backend.default_model if model is DEFAULT_MODEL else model
                        turn = await self.backend.send(
                            prompt, metadata=metadata, model=use_model, files=files,
                            on_cid=self._record_chat(sid, use_model),
                        )
                    except GeminiProblem as e:
                        log.warning(
                            "gemini request failed sid=%s %s attempt=%d %.1fs (%s)",
                            sid, label, attempt, time.monotonic() - t0, type(e).__name__,
                        )
                        self.pause.trip(e)
                        if not e.retryable:
                            return [], [Failure(label, prompt, e.message)]
                        last = e
                        continue
                    self.store.record_images(len(turn.images), self.clock())
                    log.info(
                        "gemini request ok sid=%s %s attempt=%d images=%d %.1fs",
                        sid, label, attempt, len(turn.images), time.monotonic() - t0,
                    )
                    if not turn.images:
                        said = _clip(turn.text) or "(no text)"
                        return [], [Failure(label, prompt, f"Gemini returned no image. Gemini said: {said}", refusal=True)]
                    return await self._save_turn(sid, turn, prompt, user_request, parent_id, label)
                reason = last.message if last else "not attempted"
                return [], [Failure(label, prompt, f"Failed after {made} attempt(s): {reason}")]
            finally:
                self._release()

    # ------------------------------------------------------------------ tools
    async def generate_images(self, prompts: list[str], user_request: str, session_id: str | None) -> Outcome:
        if not 1 <= len(prompts) <= 4:
            raise UserError("prompts must contain 1 to 4 prompts.")
        if any(not p.strip() for p in prompts):
            raise UserError("Every prompt must be non-empty.")
        sid = self._resolve_session(session_id)
        outcome = Outcome(sid)
        deadline = self.clock() + self.settings.deadline_seconds
        log.info("generate_images sid=%s prompts=%d", sid, len(prompts))

        tasks = [
            asyncio.create_task(
                self._one_request(
                    sid=sid, label=f"prompt {i}", prompt=p, deadline=deadline, metadata=None,
                    model=DEFAULT_MODEL, files=None, user_request=user_request, parent_id=None,
                )
            )
            for i, p in enumerate(prompts, 1)
        ]
        await _wait_until(tasks, deadline - self.clock())
        for i, (task, prompt) in enumerate(zip(tasks, prompts, strict=True), 1):
            if task.cancelled() or not task.done():
                outcome.failures.append(
                    Failure(f"prompt {i}", prompt, f"Timed out: not finished within {self.settings.deadline_seconds:.0f} s.")
                )
                continue
            if task.exception() is not None:
                log.error("prompt task crashed sid=%s prompt %d (%s)", sid, i, type(task.exception()).__name__)
                outcome.failures.append(Failure(f"prompt {i}", prompt, "Internal error."))
                continue
            images, failures = task.result()
            outcome.images += images
            outcome.failures += failures
        self._add_notes(outcome)
        self.store.touch_session(sid, self.clock())
        return outcome

    async def edit_image(self, session_id: str, image_id: str, instruction: str) -> tuple[Outcome, ImageRecord]:
        if not instruction.strip():
            raise UserError("instruction must be non-empty.")
        sid = self._resolve_session(session_id)
        source = self.store.get_image(sid, image_id) if IMAGE_RE.match(image_id or "") else None
        if source is None:
            raise UserError("Unknown image_id for this session.")
        chat = self.store.get_chat(sid, source.cid)
        model = chat.model if chat else DEFAULT_MODEL
        # Continue the image's own chat. Attach the exact picture when the chat alone is ambiguous:
        # its turn produced several images, or later turns exist in that chat.
        ambiguous = source.turn_count > 1 or bool(chat and chat.last_rid and chat.last_rid != source.rid)
        files = [self.image_path(sid, source.id, source.ext)] if ambiguous else None
        log.info("edit_image sid=%s attach=%s", sid, bool(files))

        outcome = Outcome(sid)
        deadline = self.clock() + self.settings.deadline_seconds
        task = asyncio.create_task(
            self._one_request(
                sid=sid, label="edit", prompt=instruction, deadline=deadline, metadata=source.metadata,
                model=model, files=files, user_request=None, parent_id=source.id,
            )
        )
        await _wait_until([task], deadline - self.clock())
        if task.cancelled() or not task.done():
            outcome.failures.append(
                Failure("edit", instruction, f"Timed out: not finished within {self.settings.deadline_seconds:.0f} s.")
            )
        elif task.exception() is not None:
            log.error("edit task crashed sid=%s (%s)", sid, type(task.exception()).__name__)
            outcome.failures.append(Failure("edit", instruction, "Internal error."))
        else:
            outcome.images, outcome.failures = task.result()
        self._add_notes(outcome)
        self.store.touch_session(sid, self.clock())
        return outcome, source

    def _add_notes(self, outcome: Outcome) -> None:
        if any(f.refusal for f in outcome.failures):
            outcome.notes.append("Refusals are reported as-is. Do not rewrite a prompt to get around a refusal.")
        note = self.backend.account_note()
        if note:
            outcome.notes.append(note)
        outcome.notes.append(self.budget_line())

    async def end_sessions(self, session_ids: list[str], deadline_seconds: float | None = None) -> list[EndReport]:
        deadline = self.clock() + (deadline_seconds or self.settings.deadline_seconds)
        reports = []
        for sid in dict.fromkeys(session_ids):  # dedupe, keep order
            reports.append(await self._end_session(sid, deadline))
        return reports

    async def _end_session(self, sid: str, deadline: float) -> EndReport:
        if not SESSION_RE.match(sid or "") or not self.store.session_exists(sid):
            return EndReport(sid, found=False)
        report = EndReport(sid, found=True)
        # Links die first: files, then records.
        shutil.rmtree(self.settings.images_dir / sid, ignore_errors=True)
        report.images_deleted = self.store.delete_images(sid)

        for chat in self.store.list_chats(sid):
            if self.clock() >= deadline:
                report.chat_failures.append("time limit reached before this chat; run end_chat again")
                continue
            error = await self._delete_chat(chat.cid)
            if error is None:
                self.store.delete_chat(sid, chat.cid)
                report.chats_deleted += 1
            else:
                report.chat_failures.append(error)
        if not report.chat_failures:
            self.store.delete_session(sid)
            report.session_deleted = True
        log.info(
            "session ended sid=%s chats_deleted=%d chat_failures=%d images=%d",
            sid, report.chats_deleted, len(report.chat_failures), report.images_deleted,
        )
        return report

    async def _delete_chat(self, cid: str) -> str | None:
        """Delete one Gemini chat (<= max_attempts attempts). Returns an error message or None."""
        last: GeminiProblem | None = None
        async with self._slots:
            for attempt in range(1, self.settings.max_attempts + 1):
                paused = self.pause.check()
                if paused:
                    return paused
                if attempt > 1:
                    await asyncio.sleep(RETRY_PAUSE_SECONDS)
                try:
                    await self.backend.ensure_ready()
                    await self.backend.delete_chat(cid)
                    return None
                except GeminiProblem as e:
                    log.warning("gemini chat delete failed attempt=%d (%s)", attempt, type(e).__name__)
                    self.pause.trip(e)
                    last = e
                    if not e.retryable:
                        break
        return last.message if last else "not attempted"

    # ------------------------------------------------------------------ background
    async def sweep(self) -> list[EndReport]:
        now = self.clock()
        self.store.prune_image_events(now - 3600)
        idle = self.store.idle_sessions(now - self.settings.session_ttl_hours * 3600)
        if not idle:
            return []
        log.info("ttl sweep: %d idle session(s)", len(idle))
        return await self.end_sessions(idle, deadline_seconds=3600)

    async def maintenance_loop(self) -> None:
        while True:
            try:
                await self.sweep()
                if self.pause.check() is None:
                    await self.backend.keepalive()
            except GeminiProblem as e:
                self.pause.trip(e)
            except Exception as e:  # keep the loop alive
                log.error("maintenance failed (%s)", type(e).__name__)
            await asyncio.sleep(self.settings.sweep_interval_seconds)

    async def warm_up(self) -> None:
        """Connect at boot so the library's cookie refresh starts running."""
        try:
            await self.backend.ensure_ready()
        except GeminiProblem as e:
            self.pause.trip(e)
        except Exception as e:
            log.error("warm-up failed (%s)", type(e).__name__)


async def _wait_until(tasks: list[asyncio.Task], timeout: float) -> None:
    """Wait for tasks until the timeout, then cancel the rest and let them unwind briefly."""
    if not tasks:
        return
    _, pending = await asyncio.wait(tasks, timeout=max(timeout, 0))
    for t in pending:
        t.cancel()
    if pending:
        await asyncio.wait(pending, timeout=5)


def _clip(text: str) -> str:
    text = (text or "").strip()
    return text if len(text) <= MAX_GEMINI_TEXT else text[:MAX_GEMINI_TEXT] + " [...]"


__all__ = ["ImageService", "Outcome", "EndReport", "UserError", "TransientError"]
