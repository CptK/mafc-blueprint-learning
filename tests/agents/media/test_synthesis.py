from __future__ import annotations

from ezmm import MultimodalSequence

from mafc.agents.common import AgentStatus
from mafc.agents.media.agent import MediaAgent
from mafc.common.evidence import Evidence
from mafc.common.modeling.message import Message
from mafc.common.modeling.prompt import Prompt
from mafc.common.modeling.model import Response
from mafc.tools.web_search.common import WebSource

from tests.agents.media.helpers import (
    FailingModel,
    FakeGeolocator,
    FakeRisTool,
    SequencedModel,
    empty_ris_result,
    geo_result,
    make_session,
    registered_image,
    ris_result_with_sources,
)


def test_synthesize_from_evidences_returns_plain_string() -> None:
    image = registered_image()
    model = SequencedModel(outputs=['{"answer":"Confirmed in Greece.","relevant_evidence_ids":["ev_1"]}'])
    agent = MediaAgent(
        model=model,
        summarization_model=model,
        ris_tool=FakeRisTool(empty_ris_result(image.reference)),
        geolocator=FakeGeolocator(geo_result(image.reference)),
    )
    from mafc.tools.geolocate.geolocate import Geolocate

    evidences = [
        Evidence(
            raw=MultimodalSequence("Most likely location: Greece"),
            action=Geolocate(image.reference),
            source=image.reference,
            takeaways=MultimodalSequence("Most likely location: Greece"),
        )
    ]

    result = agent.synthesize_from_evidences("Where was this taken?", evidences)

    assert isinstance(result, str)
    assert "Greece" in result


def test_synthesis_handles_models_that_require_prompt_objects() -> None:
    image = registered_image()

    class PromptOnlyModel(SequencedModel):
        def generate(self, messages: list[Message]) -> Response:
            assert len(messages) == 1
            assert isinstance(messages[0].content, Prompt)
            return super().generate(messages)

    model = PromptOnlyModel(
        outputs=[
            '{"tools":["geolocate"]}',
            "The image was likely taken in Greece.",
        ]
    )
    agent = MediaAgent(
        model=model,
        summarization_model=model,
        ris_tool=FakeRisTool(empty_ris_result(image.reference)),
        geolocator=FakeGeolocator(geo_result(image.reference)),
    )

    out = agent.run(make_session(MultimodalSequence("Where was this image taken?", image)))

    assert out.result is not None
    assert out.session.status == AgentStatus.COMPLETED
    assert "Greece" in str(out.result)


def test_returns_only_model_selected_evidences_for_follow_up() -> None:
    image = registered_image()
    model = SequencedModel(
        outputs=[
            '{"tools":["geolocate","reverse_image_search"]}',
            '{"answer":"The image was likely taken in Greece.","relevant_evidence_ids":["ev_1"]}',
            '{"tools":["reverse_image_search"]}',
            '{"answer":"The image appears on example.com.","relevant_evidence_ids":["ev_2"]}',
        ]
    )
    ris_tool = FakeRisTool(
        ris_result_with_sources(image.reference, [WebSource(reference="https://example.com/a", title="A")])
    )
    geolocator = FakeGeolocator(geo_result(image.reference))
    agent = MediaAgent(model=model, summarization_model=model, ris_tool=ris_tool, geolocator=geolocator)
    session = make_session(MultimodalSequence("Where was this image taken?", image))

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


def test_falls_back_to_all_evidences_when_synthesis_json_parsing_fails() -> None:
    image = registered_image()
    model = SequencedModel(
        outputs=[
            '{"tools":["geolocate"]}',
            "The image was likely taken in Greece.",
        ]
    )
    agent = MediaAgent(
        model=model,
        summarization_model=model,
        ris_tool=FakeRisTool(empty_ris_result(image.reference)),
        geolocator=FakeGeolocator(geo_result(image.reference)),
    )

    out = agent.run(make_session(MultimodalSequence("Where was this image taken?", image)))

    assert out.result is not None
    assert str(out.result) == "The image was likely taken in Greece."
    assert len(out.evidences) == 1
    assert out.evidences[0].source == image.reference


def test_synthesis_model_exception_falls_back_to_evidence_blocks() -> None:
    image = registered_image()
    planner_model = SequencedModel(outputs=['{"tools":["geolocate"]}'])
    agent = MediaAgent(
        model=planner_model,
        summarization_model=FailingModel(),
        ris_tool=FakeRisTool(empty_ris_result(image.reference)),
        geolocator=FakeGeolocator(geo_result(image.reference)),
    )

    out = agent.run(make_session(MultimodalSequence("Where was this image taken?", image)))

    # Synthesis fell back to raw evidence blocks joined as text
    assert out.result is not None
    assert out.session.status == AgentStatus.COMPLETED
    assert "Greece" in str(out.result)


def test_empty_synthesis_answer_returns_failed_status() -> None:
    image = registered_image()
    model = SequencedModel(
        outputs=[
            '{"tools":["geolocate"]}',
            '{"answer":"","relevant_evidence_ids":[]}',
        ]
    )
    agent = MediaAgent(
        model=model,
        summarization_model=model,
        ris_tool=FakeRisTool(empty_ris_result(image.reference)),
        geolocator=FakeGeolocator(geo_result(image.reference)),
    )

    out = agent.run(make_session(MultimodalSequence("Where was this image taken?", image)))

    assert out.result is None
    assert out.session.status == AgentStatus.FAILED
