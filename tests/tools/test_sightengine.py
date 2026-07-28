import os
from pathlib import Path

import pytest

import config.globals  # noqa: F401  # Triggers .env loading like normal app startup.
from ezmm.common.items import Image, Video
from mafc.tools.media.sightengine import SightengineChecker, SightengineDetectionAction

ASSETS_DIR = Path(__file__).resolve().parents[1] / "assets"


def _skip_if_missing_key(*key_names: str) -> None:
    if not any(os.environ.get(name) for name in key_names):
        pytest.skip(f"Missing API key. Provide one of: {', '.join(key_names)}")


@pytest.mark.integration
@pytest.mark.parametrize(
    "filename, is_video, expect_ai_involved",
    [
        ("ai-generated-city-scene.jpeg", False, True),
        ("Greece.jpeg", False, False),
        ("veritas_q1-2026_36252_ai-generated.mp4", True, True),
        ("veritas_q1-2026_20120_real.mp4", True, False),
    ],
)
def test_live_sightengine_detection(filename: str, is_video: bool, expect_ai_involved: bool) -> None:
    """Hits the real Sightengine API with one known-AI and one known-real sample of
    each media type, so a real regression in scoring/aggregation/verdict logic gets
    caught instead of only ever exercising a mock."""
    _skip_if_missing_key("SIGHTENGINE_API_USER", "sightengine_api_user")
    _skip_if_missing_key("SIGHTENGINE_API_SECRET", "sightengine_api_secret")

    item_cls = Video if is_video else Image
    item = item_cls(file_path=str(ASSETS_DIR / filename))
    checker = SightengineChecker(stores=[], use_cache=False)

    result = checker._perform(SightengineDetectionAction(media=item.reference))

    if result.error is not None:
        pytest.skip(f"Live Sightengine call failed: {result.error}")

    assert result.is_useful(), str(result)
    assert result.ai_involved is expect_ai_involved, str(result)
