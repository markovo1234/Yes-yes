"""Result shape and the 120,000-character guard."""

from __future__ import annotations

import base64
import io

import pytest
from PIL import Image

from app import results
from mcp_types import ImageContent, TextContent

pytestmark = pytest.mark.anyio


def _size(r) -> int:
    return len(r.model_dump_json(by_alias=True, exclude_none=True))


def _long_side(block: ImageContent) -> int:
    with Image.open(io.BytesIO(base64.b64decode(block.data))) as im:
        assert im.format == "JPEG"
        return max(im.size)


async def test_text_first_then_one_jpeg_preview_per_image(make_service):
    service = make_service()
    outcome = await service.generate_images(["a", "b"], "x", None)
    r = results.generate_result(outcome, service.settings.result_char_budget)
    assert isinstance(r.content[0], TextContent)
    assert outcome.session_id in r.content[0].text
    for img in outcome.images:
        assert img.url in r.content[0].text and img.image_id in r.content[0].text
    previews = r.content[1:]
    assert len(previews) == 2 and all(isinstance(p, ImageContent) and p.mime_type == "image/jpeg" for p in previews)
    assert all(_long_side(p) == 384 for p in previews)
    assert not r.is_error


async def test_size_guard_shrinks_then_drops_previews(make_service):
    service = make_service()
    # 4 prompts x 3 noisy images = 12 hard-to-compress previews (~50 KB each at 384 px).
    outcome = await service.generate_images([f"p{i} [multi3] [noisy]" for i in range(4)], "x", None)
    assert len(outcome.images) == 12
    r = results.generate_result(outcome, 120_000)
    assert _size(r) < 120_000
    previews = [c for c in r.content if isinstance(c, ImageContent)]
    assert all(_long_side(p) < 384 for p in previews)  # shrunk
    for img in outcome.images:  # every link is still in the text
        assert img.url in r.content[0].text
    tight = results.generate_result(outcome, 12_000)
    assert _size(tight) < 12_000
    kept = [c for c in tight.content if isinstance(c, ImageContent)]
    assert 0 < len(kept) < 12 and f"{12 - len(kept)} preview(s) omitted" in tight.content[0].text


async def test_size_guard_drops_all_previews_if_needed(make_service):
    service = make_service()
    outcome = await service.generate_images(["p [noisy]"], "x", None)
    r = results.generate_result(outcome, 2_500)
    assert _size(r) < 2_500
    assert len(r.content) == 1 and "1 preview(s) omitted" in r.content[0].text


async def test_three_prompt_call_fits(make_service):
    service = make_service()
    outcome = await service.generate_images(["a [noisy]", "b [noisy]", "c [noisy]"], "x", None)
    r = results.generate_result(outcome, 120_000)
    assert len(outcome.images) == 3 and _size(r) < 120_000
