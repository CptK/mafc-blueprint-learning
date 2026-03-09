import os
from pathlib import Path

import pytest
import requests
from ezmm import Image
from google.api_core.exceptions import GoogleAPICallError

import config.globals  # noqa: F401  # Triggers .env loading like normal app startup.
from mafc.tools.web_search.common import Query
from mafc.tools.web_search.serper import SerperAPI
from mafc.tools.web_search.google_vision import GoogleVisionAPI


def _skip_if_missing_key(*key_names: str) -> None:
    if not any(os.environ.get(name) for name in key_names):
        pytest.skip(f"Missing API key. Provide one of: {', '.join(key_names)}")


def _call_or_skip(fn):
    try:
        return fn()
    except (requests.RequestException, GoogleAPICallError, TimeoutError, RuntimeError) as exc:
        pytest.skip(f"Live API call skipped due to runtime/provider issue: {exc}")


@pytest.mark.integration
def test_serper_live_api_small_call() -> None:
    _skip_if_missing_key("SERPER_API_KEY", "serper_api_key")

    api = SerperAPI(gl="us", hl="en")
    out = _call_or_skip(lambda: api.search(Query(text="Eiffel Tower", limit=3)))

    assert out is not None
    assert isinstance(out.sources, list)
    assert len(out.sources) > 0
    assert out.sources[0].reference.startswith("http")


@pytest.mark.integration
def test_google_vision_live_api_small_call() -> None:
    creds_file = Path("config/google_service_account_key.json")
    if not creds_file.exists():
        pytest.skip(f"Missing Google service account credentials file: {creds_file.as_posix()}")

    api = GoogleVisionAPI()
    if api.client is None:
        pytest.skip("Google Vision client could not be initialized from available credentials.")

    image_path = Path(__file__).resolve().parents[2] / "assets" / "Paris.avif"
    query = Query(text="landmark", image=Image(file_path=image_path))

    out = _call_or_skip(lambda: api.search(query))

    assert out is not None
    assert isinstance(out.sources, list)
