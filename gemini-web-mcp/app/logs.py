"""Logging: ids, timings and error types only. Third-party loggers that could echo request data are muted."""

from __future__ import annotations

import logging
import sys


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        stream=sys.stdout,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        force=True,
    )
    # uvicorn's access log would print /mcp/<secret>; our Gate logs a redacted line instead.
    logging.getLogger("uvicorn.access").disabled = True
    for name in ("uvicorn", "uvicorn.error", "mcp", "httpx2", "httpcore2", "sse_starlette"):
        logging.getLogger(name).setLevel(logging.WARNING)
    # gemini_webapi logs through loguru; some of its messages contain file paths named after the
    # __Secure-1PSID cookie or response snippets. Remove every loguru handler so nothing is emitted.
    from gemini_webapi import logger as gemini_logger

    gemini_logger.remove()
