"""Tool results: text first, then one JPEG preview per image, always under the character budget."""

from __future__ import annotations

from mcp_types import CallToolResult, ImageContent, TextContent

from .images import SHRINK_LADDER, preview_b64
from .service import EndReport, Outcome
from .store import ImageRecord

# Head-room for the JSON-RPC envelope around the result.
ENVELOPE_MARGIN = 1_000


def _size(result: CallToolResult) -> int:
    return len(result.model_dump_json(by_alias=True, exclude_none=True))


def fit(text: str, previews: list, budget: int) -> CallToolResult:
    """Shrink previews step by step, then drop them from the end, until the result fits."""
    limit = budget - ENVELOPE_MARGIN
    if len(text) > limit // 2:  # cannot happen with capped Gemini text, but never exceed the budget
        text = text[: limit // 2] + "\n[...truncated]"
    smallest: list[str] = []
    for long_side, quality in SHRINK_LADDER:
        data = [preview_b64(p, long_side, quality) for p in previews]
        result = _build(text, data)
        if _size(result) < limit:
            return result
        smallest = data
    kept = list(smallest)
    while kept:
        kept.pop()
        note = f"\n\n({len(previews) - len(kept)} preview(s) omitted to keep the result small; the links above work.)"
        result = _build(text + note, kept)
        if _size(result) < limit:
            return result
    note = f"\n\n({len(previews)} preview(s) omitted to keep the result small; the links above work.)" if previews else ""
    return _build(text + note, [])


def _build(text: str, images_b64: list[str]) -> CallToolResult:
    content: list = [TextContent(type="text", text=text)]
    content += [ImageContent(type="image", data=d, mime_type="image/jpeg") for d in images_b64]
    return CallToolResult(content=content)


def _one_line(text: str) -> str:
    return " ".join(text.split())


def outcome_text(outcome: Outcome, header: list[str] | None = None) -> str:
    lines = [f"session_id: {outcome.session_id}  (reuse this session_id for every later call in this conversation)"]
    lines += header or []
    lines.append("")
    if outcome.images:
        lines.append(f"{len(outcome.images)} image(s):")
        for n, img in enumerate(outcome.images, 1):
            lines.append(f"[{n}] image_id: {img.image_id}")
            lines.append(f"    link: {img.url}")
            lines.append(f"    prompt: {_one_line(img.prompt)}")
            if img.gemini_text:
                lines.append(f"    gemini: {_one_line(img.gemini_text)}")
    else:
        lines.append("No images were produced.")
    if outcome.failures:
        lines.append("")
        lines.append(f"{len(outcome.failures)} failure(s):")
        for f in outcome.failures:
            lines.append(f"- {f.label}: {_one_line(f.reason)}")
            lines.append(f"    prompt: {_one_line(f.prompt)}")
    if outcome.notes:
        lines.append("")
        lines += outcome.notes
    if outcome.images:
        lines.append(f"Previews follow in the same order ({len(outcome.images)}).")
    return "\n".join(lines)


def generate_result(outcome: Outcome, budget: int) -> CallToolResult:
    result = fit(outcome_text(outcome), [img.preview for img in outcome.images], budget)
    result.is_error = not outcome.images
    return result


def edit_result(outcome: Outcome, source: ImageRecord, source_url: str, budget: int) -> CallToolResult:
    header = [f"Edited from image_id {source.id} (unchanged, still at {source_url})."]
    result = fit(outcome_text(outcome, header), [img.preview for img in outcome.images], budget)
    result.is_error = not outcome.images
    return result


def end_result(reports: list[EndReport]) -> CallToolResult:
    lines = ["end_chat results:"]
    totals = {"chats": 0, "failed": 0, "images": 0, "sessions": 0}
    for r in reports:
        if not r.found:
            lines.append(f"- session {r.session_id}: not found (already ended, expired, or never existed). Nothing to delete.")
            continue
        totals["chats"] += r.chats_deleted
        totals["failed"] += len(r.chat_failures)
        totals["images"] += r.images_deleted
        totals["sessions"] += int(r.session_deleted)
        line = (
            f"- session {r.session_id}: deleted {r.chats_deleted} Gemini chat(s) from Gemini history; "
            f"deleted {r.images_deleted} image(s) with their previews"
            + (" - their links now return 404." if r.images_deleted else ".")
        )
        if r.chat_failures:
            reasons = "; ".join(sorted(set(r.chat_failures)))
            line += (
                f" {len(r.chat_failures)} Gemini chat(s) could NOT be deleted and are still in Gemini history: {reasons}"
                " The session record was kept so end_chat can retry them."
            )
        else:
            line += " Session record deleted."
        lines.append(line)
    lines.append(
        f"Totals: {totals['chats']} Gemini chat(s) deleted, {totals['failed']} chat deletion(s) failed, "
        f"{totals['images']} image(s) deleted, {totals['sessions']} session(s) deleted."
    )
    return CallToolResult(content=[TextContent(type="text", text="\n".join(lines))], is_error=totals["failed"] > 0)


def error_result(message: str) -> CallToolResult:
    return CallToolResult(content=[TextContent(type="text", text=message)], is_error=True)
