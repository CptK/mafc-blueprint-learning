from __future__ import annotations

from pathlib import Path

from ezmm import Image, MultimodalSequence, Video
from ezmm.common.registry import item_registry

from mafc.agents.common import AgentSession, AgentStatus
from mafc.agents.media.agent import MediaAgent
from mafc.common.modeling.model import Model, Response
from mafc.common.modeling.prompt import Prompt
from mafc.tools.geolocate.geolocate import Geolocate, GeolocationResults
from mafc.tools.tool_result import ToolResult
from mafc.tools.web_search.common import Query, WebSource
from mafc.tools.web_search.google_vision import GoogleRisResults
from mafc.tools.web_search.reverse_image_search import ReverseImageSearch

ASSETS_DIR = Path(__file__).resolve().parents[1] / "assets"


class SequencedModel(Model):
    def __init__(self, outputs: list[str]):
        super().__init__(specifier="OPENAI:gpt-5-mini-2025-08-07")
        self.outputs = outputs
        self.calls: list[str] = []

    def generate(self, prompt) -> Response:
        self.calls.append(str(prompt))
        text = self.outputs.pop(0) if self.outputs else ""
        return Response(text=text, total_cost=0.0)


class FakeRisTool:
    def __init__(self, result: ToolResult):
        self.result = result
        self.actions: list[ReverseImageSearch] = []

    def perform(self, action: ReverseImageSearch, summarize: bool = True, **kwargs) -> ToolResult:
        self.actions.append(action)
        return self.result


class FakeGeolocator:
    def __init__(self, result: ToolResult):
        self.result = result
        self.actions: list[Geolocate] = []

    def perform(self, action: Geolocate, summarize: bool = True, **kwargs) -> ToolResult:
        self.actions.append(action)
        return self.result


def _make_session(goal: MultimodalSequence) -> AgentSession:
    return AgentSession(id="media-session", goal=goal)


def _registered_image() -> Image:
    image = Image(file_path=ASSETS_DIR / "Greece.jpeg")
    item_registry.add_item(image)
    return image


def _registered_video() -> Video:
    video = Video(binary_data=b"video-bytes")
    item_registry.add_item(video)
    return video


def test_media_agent_runs_geolocate_for_location_question() -> None:
    image = _registered_image()
    model = SequencedModel(outputs=["The image was likely taken in Greece."])
    ris_result = ToolResult(
        raw=GoogleRisResults(sources=[], query=Query(text="seed"), entities={}, best_guess_labels=[]),
        action=ReverseImageSearch(image.reference),
        takeaways=None,
    )
    geo_result = ToolResult(
        raw=GeolocationResults(
            text="Most likely location: Greece",
            most_likely_location="Greece",
            top_k_locations=["Greece", "Italy"],
            model_output={"ok": True},
        ),
        action=Geolocate(image.reference),
        takeaways=MultimodalSequence("Most likely location: Greece"),
    )
    agent = MediaAgent(
        model=model,
        ris_tool=FakeRisTool(ris_result),
        geolocator=FakeGeolocator(geo_result),
    )

    out = agent.run(_make_session(MultimodalSequence("Where was this image taken?", image)))

    assert out.result is not None
    assert out.session.status == AgentStatus.COMPLETED
    assert len(out.evidences) == 1
    assert out.evidences[0].source == image.reference
    assert "Greece" in str(out.result)


def test_media_agent_runs_ris_for_publication_question() -> None:
    image = _registered_image()
    model = SequencedModel(outputs=["The image appeared on example.com."])
    ris_result = ToolResult(
        raw=GoogleRisResults(
            sources=[WebSource(reference="https://example.com/a", title="A")],
            query=Query(text="seed"),
            entities={"Mountain": 0.8},
            best_guess_labels=["Alps"],
        ),
        action=ReverseImageSearch(image.reference),
        takeaways=MultimodalSequence("Reverse image search found a match on example.com."),
    )
    geo_result = ToolResult(
        raw=GeolocationResults(text="unused", most_likely_location="", top_k_locations=[]),
        action=Geolocate(image.reference),
        takeaways=None,
    )
    ris_tool = FakeRisTool(ris_result)
    geolocator = FakeGeolocator(geo_result)
    agent = MediaAgent(model=model, ris_tool=ris_tool, geolocator=geolocator)

    out = agent.run(_make_session(MultimodalSequence("Where was this image published?", image)))

    assert out.result is not None
    assert len(ris_tool.actions) == 1
    assert len(geolocator.actions) == 0
    assert len(out.evidences) == 1
    assert out.evidences[0].source == "https://example.com/a"


def test_media_agent_runs_geolocate_for_video_questions() -> None:
    video = _registered_video()
    model = SequencedModel(
        outputs=["The video was likely taken in Greece, but no publication match was found."]
    )
    ris_result = ToolResult(
        raw=GoogleRisResults(sources=[], query=Query(text="seed"), entities={}, best_guess_labels=[]),
        action=ReverseImageSearch(video.reference),
        takeaways=MultimodalSequence("No reverse matches."),
    )
    geo_result = ToolResult(
        raw=GeolocationResults(
            text="Most likely location: Greece",
            most_likely_location="Greece",
            top_k_locations=["Greece", "Italy"],
            model_output={"ok": True},
        ),
        action=Geolocate(video.reference),
        takeaways=MultimodalSequence("Most likely location: Greece"),
    )
    ris_tool = FakeRisTool(ris_result)
    geolocator = FakeGeolocator(geo_result)
    agent = MediaAgent(
        model=model,
        ris_tool=ris_tool,
        geolocator=geolocator,
    )

    out = agent.run(
        _make_session(MultimodalSequence("Where was this video taken and where was it published?", video))
    )

    assert out.result is not None
    assert len(ris_tool.actions) == 1
    assert len(geolocator.actions) == 1
    assert geolocator.actions[0].media == video
    assert not out.errors
    assert "Greece" in str(out.result)


def test_media_agent_synthesis_handles_models_that_require_prompt_objects() -> None:
    image = _registered_image()

    class PromptOnlyModel(SequencedModel):
        def generate(self, prompt) -> Response:
            assert isinstance(prompt, Prompt)
            return super().generate(prompt)

    model = PromptOnlyModel(outputs=["The image was likely taken in Greece."])
    geo_result = ToolResult(
        raw=GeolocationResults(
            text="Most likely location: Greece",
            most_likely_location="Greece",
            top_k_locations=["Greece", "Italy"],
            model_output={"ok": True},
        ),
        action=Geolocate(image.reference),
        takeaways=MultimodalSequence("Most likely location: Greece"),
    )
    agent = MediaAgent(
        model=model,
        summarization_model=model,
        ris_tool=FakeRisTool(
            ToolResult(
                raw=GoogleRisResults(sources=[], query=Query(text="seed"), entities={}, best_guess_labels=[]),
                action=ReverseImageSearch(image.reference),
                takeaways=None,
            )
        ),
        geolocator=FakeGeolocator(geo_result),
    )

    out = agent.run(_make_session(MultimodalSequence("Where was this image taken?", image)))

    assert out.result is not None
    assert out.session.status == AgentStatus.COMPLETED
    assert "Greece" in str(out.result)


def test_media_agent_returns_only_model_selected_evidences_for_follow_up() -> None:
    image = _registered_image()
    model = SequencedModel(
        outputs=[
            '{"answer":"The image was likely taken in Greece.","relevant_evidence_ids":["ev_1"]}',
            '{"answer":"The image appears on example.com.","relevant_evidence_ids":["ev_2"]}',
        ]
    )
    ris_result = ToolResult(
        raw=GoogleRisResults(
            sources=[WebSource(reference="https://example.com/a", title="A")],
            query=Query(text="seed"),
            entities={"Mountain": 0.8},
            best_guess_labels=["Alps"],
        ),
        action=ReverseImageSearch(image.reference),
        takeaways=MultimodalSequence("Reverse image search found a match on example.com."),
    )
    geo_result = ToolResult(
        raw=GeolocationResults(
            text="Most likely location: Greece",
            most_likely_location="Greece",
            top_k_locations=["Greece", "Italy"],
            model_output={"ok": True},
        ),
        action=Geolocate(image.reference),
        takeaways=MultimodalSequence("Most likely location: Greece"),
    )
    agent = MediaAgent(
        model=model,
        summarization_model=model,
        ris_tool=FakeRisTool(ris_result),
        geolocator=FakeGeolocator(geo_result),
    )
    session = _make_session(MultimodalSequence("Where was this image taken?", image))

    first = agent.run(session)
    assert first.result is not None
    assert len(first.evidences) == 1
    assert first.evidences[0].source == image.reference

    session.goal = MultimodalSequence("Where was this image published?", image)
    second = agent.run(session)

    assert second.result is not None
    assert str(second.result) == "The image appears on example.com."
    assert len(session.evidences) == 2
    assert len(second.evidences) == 1
    assert second.evidences[0].source == "https://example.com/a"


def test_media_agent_falls_back_to_all_evidences_when_synthesis_json_parsing_fails() -> None:
    image = _registered_image()
    model = SequencedModel(outputs=["The image was likely taken in Greece."])
    geo_result = ToolResult(
        raw=GeolocationResults(
            text="Most likely location: Greece",
            most_likely_location="Greece",
            top_k_locations=["Greece", "Italy"],
            model_output={"ok": True},
        ),
        action=Geolocate(image.reference),
        takeaways=MultimodalSequence("Most likely location: Greece"),
    )
    agent = MediaAgent(
        model=model,
        summarization_model=model,
        ris_tool=FakeRisTool(
            ToolResult(
                raw=GoogleRisResults(sources=[], query=Query(text="seed"), entities={}, best_guess_labels=[]),
                action=ReverseImageSearch(image.reference),
                takeaways=None,
            )
        ),
        geolocator=FakeGeolocator(geo_result),
    )

    out = agent.run(_make_session(MultimodalSequence("Where was this image taken?", image)))

    assert out.result is not None
    assert str(out.result) == "The image was likely taken in Greece."
    assert len(out.evidences) == 1
    assert out.evidences[0].source == image.reference
