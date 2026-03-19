from __future__ import annotations

from ezmm import MultimodalSequence

from mafc.agents.common import AgentStatus
from mafc.agents.media.agent import MediaAgent
from mafc.tools.web_search.common import WebSource

from tests.agents.media.helpers import (
    FakeGeolocator,
    FakeRisTool,
    SequencedModel,
    empty_ris_result,
    geo_result,
    make_session,
    registered_image,
    registered_video,
    ris_result_with_sources,
)


def _synthesis_answer(text: str, *ev_ids: str) -> str:
    ids = list(ev_ids) if ev_ids else ["ev_1"]
    return f'{{"answer":"{text}","relevant_evidence_ids":{ids}}}'.replace("'", '"')


def test_runs_geolocate_for_location_question() -> None:
    image = registered_image()
    model = SequencedModel(
        outputs=[
            '{"tools":["geolocate"]}',
            "The image was likely taken in Greece.",
        ]
    )
    ris_tool = FakeRisTool(empty_ris_result(image.reference))
    geolocator = FakeGeolocator(geo_result(image.reference))
    agent = MediaAgent(model=model, ris_tool=ris_tool, geolocator=geolocator)

    out = agent.run(make_session(MultimodalSequence("Where was this image taken?", image)))

    assert out.result is not None
    assert out.session.status == AgentStatus.COMPLETED
    assert len(ris_tool.actions) == 0
    assert len(geolocator.actions) == 1
    assert len(out.evidences) == 1
    assert out.evidences[0].source == image.reference
    assert "Greece" in str(out.result)


def test_runs_ris_for_publication_question() -> None:
    image = registered_image()
    model = SequencedModel(
        outputs=[
            '{"tools":["reverse_image_search"]}',
            "The image appeared on example.com.",
        ]
    )
    ris_tool = FakeRisTool(
        ris_result_with_sources(image.reference, [WebSource(reference="https://example.com/a", title="A")])
    )
    geolocator = FakeGeolocator(geo_result(image.reference))
    agent = MediaAgent(model=model, ris_tool=ris_tool, geolocator=geolocator)

    out = agent.run(make_session(MultimodalSequence("Where was this image published?", image)))

    assert out.result is not None
    assert len(ris_tool.actions) == 1
    assert len(geolocator.actions) == 0
    assert len(out.evidences) == 1
    assert out.evidences[0].source == "https://example.com/a"


def test_runs_both_tools_for_video_questions() -> None:
    video = registered_video()
    model = SequencedModel(
        outputs=[
            '{"tools":["reverse_image_search","geolocate"]}',
            "The video was likely taken in Greece, but no publication match was found.",
        ]
    )
    ris_tool = FakeRisTool(empty_ris_result(video.reference))
    geolocator = FakeGeolocator(geo_result(video.reference))
    agent = MediaAgent(model=model, ris_tool=ris_tool, geolocator=geolocator)

    out = agent.run(
        make_session(MultimodalSequence("Where was this video taken and where was it published?", video))
    )

    assert out.result is not None
    assert len(ris_tool.actions) == 1
    assert len(geolocator.actions) == 1
    assert geolocator.actions[0].media == video
    assert not out.errors
    assert "Greece" in str(out.result)


def test_parses_tool_plan_embedded_in_text() -> None:
    image = registered_image()
    model = SequencedModel(
        outputs=[
            'Plan:\n{"tools":["geolocate"]}\nThanks',
            "The image was likely taken in Greece.",
        ]
    )
    ris_tool = FakeRisTool(empty_ris_result(image.reference))
    geolocator = FakeGeolocator(geo_result(image.reference))
    agent = MediaAgent(model=model, summarization_model=model, ris_tool=ris_tool, geolocator=geolocator)

    out = agent.run(make_session(MultimodalSequence("Where was this image taken?", image)))

    assert out.result is not None
    assert len(ris_tool.actions) == 0
    assert len(geolocator.actions) == 1
    assert out.errors == []


def test_repairs_non_json_tool_plan() -> None:
    image = registered_image()
    model = SequencedModel(
        outputs=[
            "I should geolocate this image first.",
            '{"tools":["geolocate"]}',
            "The image was likely taken in Greece.",
        ]
    )
    ris_tool = FakeRisTool(empty_ris_result(image.reference))
    geolocator = FakeGeolocator(geo_result(image.reference))
    agent = MediaAgent(model=model, summarization_model=model, ris_tool=ris_tool, geolocator=geolocator)

    out = agent.run(make_session(MultimodalSequence("Where was this image taken?", image)))

    assert out.result is not None
    assert len(ris_tool.actions) == 0
    assert len(geolocator.actions) == 1
    assert out.errors == []


def test_falls_back_to_both_tools_when_plan_parsing_fails() -> None:
    image = registered_image()
    model = SequencedModel(
        outputs=[
            "not-json",
            "also not-json",
            '{"answer":"Taken in Greece and published on example.com.","relevant_evidence_ids":["ev_1","ev_2"]}',
        ]
    )
    ris_tool = FakeRisTool(
        ris_result_with_sources(image.reference, [WebSource(reference="https://example.com/a", title="A")])
    )
    geolocator = FakeGeolocator(geo_result(image.reference))
    agent = MediaAgent(model=model, summarization_model=model, ris_tool=ris_tool, geolocator=geolocator)

    out = agent.run(make_session(MultimodalSequence("Investigate this image.", image)))

    assert out.result is not None
    assert len(ris_tool.actions) == 1
    assert len(geolocator.actions) == 1
    assert any("Media planner output could not be parsed" in e for e in out.errors)


def test_does_not_readd_existing_evidence_on_second_run() -> None:
    image = registered_image()
    model = SequencedModel(
        outputs=[
            '{"tools":["geolocate"]}',
            '{"answer":"Taken in Greece.","relevant_evidence_ids":["ev_1"]}',
            '{"tools":["geolocate"]}',
            '{"answer":"Still Greece.","relevant_evidence_ids":["ev_1"]}',
        ]
    )
    geolocator = FakeGeolocator(geo_result(image.reference))
    agent = MediaAgent(
        model=model,
        summarization_model=model,
        ris_tool=FakeRisTool(empty_ris_result(image.reference)),
        geolocator=geolocator,
    )
    session = make_session(MultimodalSequence("Where was this taken?", image))

    agent.run(session)
    evidence_count_after_first = len(session.evidences)

    agent.run(session)

    # The second run produces the same evidence object; it must not be appended again.
    assert len(session.evidences) == evidence_count_after_first
