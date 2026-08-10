"""Shared helpers for asserting how the model wrappers log failed API calls."""

from mafc.common.logger import logger

# Long enough that an unabbreviated log line would dwarf the error it accompanies.
LONG_PROMPT = "HEAD" + "x" * 20_000 + "TAIL"


def capture_errors(monkeypatch) -> list[str]:
    """Redirect `logger.error` into the returned list for the duration of the test."""
    logged: list[str] = []
    monkeypatch.setattr(logger, "error", lambda *args: logged.append(" ".join(str(a) for a in args)))
    return logged


def assert_abbreviated(logged: list[str]) -> None:
    """Assert the wrapper logged the failure once, with the prompt's middle elided."""
    assert len(logged) == 1
    message = logged[0]

    assert LONG_PROMPT not in message
    assert "chars omitted" in message
    # The identifying edges are still there to reconstruct which call failed.
    assert "HEAD" in message
    assert "TAIL" in message
    assert len(message) < 1_500
