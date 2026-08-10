import logging

import mafc.common.logger  # noqa: F401  -- imported for its library-suppression side effects


def test_scrapemm_per_method_warnings_are_suppressed() -> None:
    """scrapeMM logs a full traceback per failed retrieval method, then falls back to the
    next one, so those warnings are noise rather than outcomes. ScrapeMMRetriever logs the
    actual outcome itself."""
    scrapemm_logger = logging.getLogger("scrapeMM")

    assert not scrapemm_logger.isEnabledFor(logging.WARNING)
    # Conditions that genuinely need attention (read-only ezMM database, full disk,
    # retrieval that could not even be started) are logged above WARNING and must survive.
    assert scrapemm_logger.isEnabledFor(logging.ERROR)
    assert scrapemm_logger.isEnabledFor(logging.CRITICAL)
