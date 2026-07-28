from __future__ import annotations

from pathlib import Path
from typing import cast

from ezmm import Image, MultimodalSequence, Video
from ezmm.common.registry import item_registry

from mafc.agents.common import AgentSession
from mafc.common.modeling.message import Message
from mafc.common.modeling.model import Model, Response
from mafc.tools.geolocate.geolocate import Geolocate, GeolocationResults, Geolocator
from mafc.tools.media.c2pa_checker import C2PAChecker, C2PAProvenanceResults, CheckC2PAProvenance
from mafc.tools.media.sightengine import (
    SightengineChecker,
    SightengineDetectionAction,
    SightengineDetectionResults,
)
from mafc.tools.media.trufor import DetectTruForManipulation, ManipulationDetectionResults, TruFor
from mafc.tools.tool_result import ToolResult
from mafc.tools.web_search.common import Query, WebSource
from mafc.tools.web_search.google_vision import (
    EXACT_MATCH_NOTE,
    PARTIAL_MATCH_NOTE,
    UNKNOWN_MATCH_NOTE,
    GoogleRisResults,
)
from mafc.tools.web_search.reverse_image_search import ReverseImageSearch, ReverseImageSearchTool

ASSETS_DIR = Path(__file__).resolve().parents[2] / "assets"


class SequencedModel(Model):
    def __init__(self, outputs: list[str]):
        super().__init__(specifier="OPENAI:gpt-5-mini-2025-08-07")
        self.outputs = outputs
        self.calls: list[str] = []

    def _do_generate(self, messages: list[Message]) -> Response:
        self.calls.append("\n".join(f"[{message.role.value}] {message.content}" for message in messages))
        text = self.outputs.pop(0) if self.outputs else ""
        return Response(text=text, total_cost=0.0)


class FailingModel(Model):
    def __init__(self):
        super().__init__(specifier="OPENAI:gpt-5-mini-2025-08-07")

    def _do_generate(self, messages: list[Message]) -> Response:
        raise RuntimeError("model unavailable")


class FakeRisTool(ReverseImageSearchTool):
    def __init__(self, result: ToolResult):
        self._fake_result = result
        self.performed: list[ReverseImageSearch] = []

    def perform(self, action: ReverseImageSearch, summarize: bool = True, **kwargs) -> ToolResult:
        self.performed.append(action)
        return self._fake_result


class FakeGeolocator(Geolocator):
    def __init__(self, result: ToolResult):
        self._fake_result = result
        self.performed: list[Geolocate] = []

    def perform(self, action: Geolocate, summarize: bool = True, **kwargs) -> ToolResult:
        self.performed.append(action)
        return self._fake_result


class FakeC2PAChecker(C2PAChecker):
    def __init__(self, result: ToolResult):
        self._fake_result = result
        self.performed: list[CheckC2PAProvenance] = []

    def perform(self, action: CheckC2PAProvenance, summarize: bool = True, **kwargs) -> ToolResult:
        self.performed.append(action)
        return self._fake_result


class FakeTruFor(TruFor):
    def __init__(self, result: ToolResult):
        self._fake_result = result
        self.performed: list[DetectTruForManipulation] = []

    def perform(self, action: DetectTruForManipulation, summarize: bool = True, **kwargs) -> ToolResult:
        self.performed.append(action)
        return self._fake_result


class FakeSightengine(SightengineChecker):
    def __init__(self, result: ToolResult):
        self._fake_result = result
        self.performed: list[SightengineDetectionAction] = []

    def perform(self, action: SightengineDetectionAction, summarize: bool = True, **kwargs) -> ToolResult:
        self.performed.append(action)
        return self._fake_result


def c2pa_result(media_ref: str, verdict: str = "ai_generated") -> ToolResult:
    return ToolResult(
        raw=C2PAProvenanceResults(ai_generated=True, verdict=verdict, provenance="valid"),
        action=CheckC2PAProvenance(media_ref),
        takeaways=MultimodalSequence(f"C2PA provenance: {verdict}."),
    )


def trufor_result(media_ref: str, score: float = 0.8) -> ToolResult:
    return ToolResult(
        raw=ManipulationDetectionResults(score=score, is_manipulated=score >= 0.5),
        action=DetectTruForManipulation(media_ref),
        takeaways=MultimodalSequence(f"TruFor manipulation score: {score:.2f}."),
    )


def sightengine_result(media_ref: str, ai_score: float = 0.9) -> ToolResult:
    return ToolResult(
        raw=SightengineDetectionResults(ai_generated_score=ai_score),
        action=SightengineDetectionAction(media_ref),
        takeaways=MultimodalSequence(f"SightEngine ai_generated_score: {ai_score:.2f}."),
    )


def make_session(goal: MultimodalSequence) -> AgentSession:
    return AgentSession(id="media-session", goal=goal)


def registered_image() -> Image:
    image = Image(file_path=ASSETS_DIR / "Greece.jpeg")
    item_registry.add_item(image)
    return cast(Image, image)


def registered_video() -> Video:
    video = Video(binary_data=b"video-bytes")
    item_registry.add_item(video)
    return cast(Video, video)


def empty_ris_result(media_ref: str) -> ToolResult:
    return ToolResult(
        raw=GoogleRisResults(sources=[], query=Query(text="seed"), entities={}, best_guess_labels=[]),
        action=ReverseImageSearch(media_ref),
        takeaways=None,
    )


def exact_web_source(url: str, title: str | None = None) -> WebSource:
    """A RIS page Google confirmed as an EXACT copy — promotable to its own evidence."""
    return WebSource(reference=url, title=title, preview=EXACT_MATCH_NOTE)


def partial_web_source(url: str, title: str | None = None) -> WebSource:
    """A RIS page Google confirmed as a PARTIAL (cropped/edited) match."""
    return WebSource(reference=url, title=title, preview=PARTIAL_MATCH_NOTE)


def unknown_web_source(url: str, title: str | None = None) -> WebSource:
    """A merely visually-similar RIS page: no confirmed referent link to the media."""
    return WebSource(reference=url, title=title, preview=UNKNOWN_MATCH_NOTE)


def ris_result_with_sources(media_ref: str, sources: list[WebSource]) -> ToolResult:
    return ToolResult(
        raw=GoogleRisResults(
            sources=sources,
            query=Query(text="seed"),
            entities={},
            best_guess_labels=[],
        ),
        action=ReverseImageSearch(media_ref),
        takeaways=MultimodalSequence("Found matches."),
    )


def geo_result(media_ref: str, location: str = "Greece") -> ToolResult:
    return ToolResult(
        raw=GeolocationResults(
            text=f"Most likely location: {location}",
            most_likely_location=location,
            top_k_locations=[location],
            model_output={"ok": True},
        ),
        action=Geolocate(media_ref),
        takeaways=MultimodalSequence(f"Most likely location: {location}"),
    )
