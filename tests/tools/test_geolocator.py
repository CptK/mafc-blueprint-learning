import builtins
import io
from pathlib import Path

from PIL import Image as PILImage
import pytest
import requests

from mafc.tools.geolocate import Geolocate, GeolocationResults, Geolocator
import mafc.tools.geolocate.geolocate as geolocate_module


class _Response:
    def __init__(self, payload: dict):
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return self._payload


class _ImageWrapper:
    def __init__(self, image):
        self.image = image


def _make_geolocator(monkeypatch, top_k: int = 3) -> Geolocator:
    original_open = builtins.open

    def fake_open(path, *args, **kwargs):
        if path == "default_countries_list.txt":
            return io.StringIO("Germany\nFrance\n")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", fake_open)
    tool = Geolocator(top_k=top_k)
    tool.server_url = "http://geo.test"
    return tool


def test_perform_returns_error_for_invalid_action(monkeypatch) -> None:
    tool = _make_geolocator(monkeypatch)

    result = tool._perform(object())

    assert isinstance(result, GeolocationResults)
    assert result.text == "Invalid action type"
    assert result.most_likely_location == ""
    assert result.top_k_locations == []


def test_perform_returns_error_for_missing_image(monkeypatch) -> None:
    tool = _make_geolocator(monkeypatch)
    action = object.__new__(Geolocate)
    action.image = None

    result = tool._perform(action)

    assert result.text == "Image not found"
    assert result.top_k_locations == []


def test_perform_delegates_to_locate(monkeypatch) -> None:
    tool = _make_geolocator(monkeypatch)
    action = object.__new__(Geolocate)
    pil_image = PILImage.new("RGB", (2, 2), color="white")
    action.image = _ImageWrapper(pil_image)

    captured = {}

    def fake_locate(image, choices=None):
        captured["image"] = image
        captured["choices"] = choices
        return GeolocationResults(text="ok", most_likely_location="Germany", top_k_locations=["Germany"])

    monkeypatch.setattr(tool, "locate", fake_locate)

    result = tool._perform(action)

    assert captured["image"] is pil_image
    assert captured["choices"] is None
    assert result.most_likely_location == "Germany"


def test_locate_posts_payload_and_parses_response(monkeypatch) -> None:
    tool = _make_geolocator(monkeypatch, top_k=5)
    image = PILImage.new("RGB", (1, 1), color="blue")
    captured = {}

    def fake_post(url, json, timeout):
        captured["url"] = url
        captured["json"] = json
        captured["timeout"] = timeout
        return _Response(
            {
                "text": "looks like Germany",
                "most_likely_location": "Germany",
                "top_k_locations": ["Germany", "France"],
            }
        )

    monkeypatch.setattr(geolocate_module.requests, "post", fake_post)

    result = tool.locate(image=image, choices=["Germany", "France"])

    assert captured["url"] == "http://geo.test/geolocate"
    assert captured["timeout"] == 60
    assert captured["json"]["top_k"] == 5
    assert captured["json"]["choices"] == ["Germany", "France"]
    assert isinstance(captured["json"]["image_b64"], str)
    assert len(captured["json"]["image_b64"]) > 0

    assert result.text == "looks like Germany"
    assert result.most_likely_location == "Germany"
    assert result.top_k_locations == ["Germany", "France"]


def test_summarize_returns_multimodal_text(monkeypatch) -> None:
    tool = _make_geolocator(monkeypatch)
    result = GeolocationResults(text="summary text", most_likely_location="Germany", top_k_locations=["Germany"])

    summary = tool._summarize(result)

    assert summary is not None
    assert str(summary)


@pytest.mark.integration
def test_locate_real_service_greece_image() -> None:
    image_path = Path(__file__).resolve().parents[1] / "assets" / "Greece.jpeg"
    image = PILImage.open(image_path)
    tool = Geolocator(top_k=5)

    try:
        result = tool.locate(image=image)
    except requests.RequestException as exc:
        pytest.skip(f"Geolocator service not reachable at {tool.server_url}: {exc}")

    assert result.most_likely_location.strip().lower() == "greece"
