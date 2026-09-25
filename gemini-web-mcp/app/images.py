"""Image files on the volume: format detection, previews, and fitting previews into the result budget."""

from __future__ import annotations

import base64
import io
from pathlib import Path

from PIL import Image

PREVIEW_LONG_SIDE = 384
PREVIEW_QUALITY = 80
# Progressively smaller previews tried when the tool result would be too large.
SHRINK_LADDER = [(384, 80), (320, 72), (256, 65), (192, 60), (128, 55)]

_FORMAT_EXT = {"PNG": "png", "JPEG": "jpg", "WEBP": "webp", "GIF": "gif"}
MIME = {"png": "image/png", "jpg": "image/jpeg", "webp": "image/webp", "gif": "image/gif"}


def detect_ext(path: Path) -> str:
    """Return the canonical extension for a downloaded image; raises ValueError if it is not an image."""
    with Image.open(path) as im:
        fmt = im.format
        im.verify()
    ext = _FORMAT_EXT.get(fmt or "")
    if not ext:
        raise ValueError(f"unsupported image format {fmt}")
    return ext


def _jpeg(im: Image.Image, long_side: int, quality: int) -> bytes:
    im = im.copy()
    im.thumbnail((long_side, long_side), Image.Resampling.LANCZOS)
    if im.mode not in ("RGB", "L"):
        background = Image.new("RGB", im.size, (255, 255, 255))
        rgba = im.convert("RGBA")
        background.paste(rgba, mask=rgba.getchannel("A"))
        im = background
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=quality, optimize=True)
    return buf.getvalue()


def make_preview(source: Path, dest: Path) -> None:
    with Image.open(source) as im:
        data = _jpeg(im, PREVIEW_LONG_SIDE, PREVIEW_QUALITY)
    dest.write_bytes(data)


def preview_b64(preview: Path, long_side: int, quality: int) -> str:
    if long_side >= PREVIEW_LONG_SIDE and quality >= PREVIEW_QUALITY:
        data = preview.read_bytes()
    else:
        with Image.open(preview) as im:
            data = _jpeg(im, long_side, quality)
    return base64.b64encode(data).decode()
