"""Error taxonomy shared by the Gemini backend and the service.

Messages are shown to the user through Claude; they never contain prompts or cookies.
"""


class GeminiProblem(Exception):
    """Base class. `message` is user-facing."""

    retryable = False
    # Seconds to pause all generation; None = until the process restarts; 0 = do not pause.
    pause_seconds: float | None = 0

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


class AuthProblem(GeminiProblem):
    pause_seconds = None


class AccountProblem(GeminiProblem):
    pause_seconds = None


class ConfigProblem(GeminiProblem):
    pause_seconds = None


class RateLimited(GeminiProblem):
    pause_seconds = 30 * 60


class Blocked(GeminiProblem):
    pause_seconds = 30 * 60


class RequestRejected(GeminiProblem):
    """This request cannot succeed as sent (e.g. invalid model for the chat). No retry, no pause."""


class TransientError(GeminiProblem):
    retryable = True


REFRESH_COOKIES = (
    "Gemini says the saved cookies are not signed in (expired or invalid). Refresh your cookies: copy fresh "
    "__Secure-1PSID and __Secure-1PSIDTS from gemini.google.com and update GEMINI_1PSID / GEMINI_1PSIDTS on "
    "Railway (README -> 'Refreshing cookies'). Generation is paused until the server restarts with new cookies."
)

NO_COOKIES = (
    "No Gemini cookies are configured. Set GEMINI_1PSID and GEMINI_1PSIDTS on Railway (README -> "
    "'Getting the cookies') and redeploy."
)
