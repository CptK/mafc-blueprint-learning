from __future__ import annotations

from ezmm import MultimodalSequence

from mafc.agents.common import AgentStatus
from mafc.agents.media.agent import MediaAgent

from tests.agents.media.helpers import (
    FakeC2PAChecker,
    FakeGeolocator,
    FakeRisTool,
    FakeSightengine,
    FakeTruFor,
    SequencedModel,
    c2pa_result,
    empty_ris_result,
    geo_result,
    make_session,
    registered_image,
    registered_video,
    exact_web_source,
    ris_result_with_sources,
    sightengine_result,
    trufor_result,
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
    assert len(ris_tool.performed) == 0
    assert len(geolocator.performed) == 1
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
        ris_result_with_sources(image.reference, [exact_web_source("https://example.com/a", title="A")])
    )
    geolocator = FakeGeolocator(geo_result(image.reference))
    agent = MediaAgent(model=model, ris_tool=ris_tool, geolocator=geolocator)

    out = agent.run(make_session(MultimodalSequence("Where was this image published?", image)))

    assert out.result is not None
    assert len(ris_tool.performed) == 1
    assert len(geolocator.performed) == 0
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
    assert len(ris_tool.performed) == 1
    assert len(geolocator.performed) == 1
    assert geolocator.performed[0].media == video
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
    assert len(ris_tool.performed) == 0
    assert len(geolocator.performed) == 1
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
    assert len(ris_tool.performed) == 0
    assert len(geolocator.performed) == 1
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
        ris_result_with_sources(image.reference, [exact_web_source("https://example.com/a", title="A")])
    )
    geolocator = FakeGeolocator(geo_result(image.reference))
    agent = MediaAgent(model=model, summarization_model=model, ris_tool=ris_tool, geolocator=geolocator)

    out = agent.run(make_session(MultimodalSequence("Investigate this image.", image)))

    assert out.result is not None
    assert len(ris_tool.performed) == 1
    assert len(geolocator.performed) == 1
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


# --- assess_authenticity fan-out ---------------------------------------------


def _authenticity_agent(model, image, **checkers):
    return MediaAgent(
        model=model,
        summarization_model=model,
        ris_tool=FakeRisTool(empty_ris_result(image.reference)),
        geolocator=FakeGeolocator(geo_result(image.reference)),
        **checkers,
    )


def test_assess_authenticity_fans_out_to_all_available_detectors() -> None:
    image = registered_image()
    model = SequencedModel(
        outputs=['{"tools":["assess_authenticity"]}', "Likely AI-generated and manipulated."]
    )
    c2pa = FakeC2PAChecker(c2pa_result(image.reference))
    trufor = FakeTruFor(trufor_result(image.reference))
    sightengine = FakeSightengine(sightengine_result(image.reference))
    agent = _authenticity_agent(
        model, image, c2pa_checker=c2pa, trufor_checker=trufor, sightengine_checker=sightengine
    )

    out = agent.run(make_session(MultimodalSequence("Is this image real?", image)))

    assert out.result is not None
    assert out.session.status == AgentStatus.COMPLETED
    # one grouped intent runs all three detectors, each exactly once
    assert len(c2pa.performed) == 1
    assert len(trufor.performed) == 1
    assert len(sightengine.performed) == 1
    # each detector contributes its own evidence (kept separate, not fused)
    assert len(out.evidences) == 3


def test_assess_authenticity_runs_only_wired_detectors() -> None:
    image = registered_image()
    model = SequencedModel(outputs=['{"tools":["assess_authenticity"]}', "No sign of manipulation."])
    trufor = FakeTruFor(trufor_result(image.reference, score=0.1))
    sightengine = FakeSightengine(sightengine_result(image.reference, ai_score=0.05))
    # built without c2pa
    agent = _authenticity_agent(model, image, trufor_checker=trufor, sightengine_checker=sightengine)

    out = agent.run(make_session(MultimodalSequence("Is this image AI-generated?", image)))

    assert out.result is not None
    assert agent.c2pa_checker is None
    assert len(trufor.performed) == 1
    assert len(sightengine.performed) == 1
    assert len(out.evidences) == 2


def test_fanout_order_is_metadata_then_local_then_paid_api() -> None:
    image = registered_image()
    agent = _authenticity_agent(
        SequencedModel(outputs=[]),
        image,
        c2pa_checker=FakeC2PAChecker(c2pa_result(image.reference)),
        trufor_checker=FakeTruFor(trufor_result(image.reference)),
        sightengine_checker=FakeSightengine(sightengine_result(image.reference)),
    )

    names = [name for name, _ in agent._run_authenticity_fanout(image)]

    assert names == ["check_c2pa_provenance", "detect_trufor_manipulation", "sightengine_detection"]


def test_assess_authenticity_not_offered_without_any_detector() -> None:
    from mafc.agents.media.planner import _valid_tools_for

    image = registered_image()
    bare = _authenticity_agent(SequencedModel(outputs=[]), image)
    with_one = _authenticity_agent(
        SequencedModel(outputs=[]), image, trufor_checker=FakeTruFor(trufor_result(image.reference))
    )

    assert "assess_authenticity" not in _valid_tools_for(bare)
    offered = _valid_tools_for(with_one)
    assert "assess_authenticity" in offered
    # the individual detector names are never offered to the planner directly
    assert "detect_trufor_manipulation" not in offered
    assert "sightengine_detection" not in offered
    assert "check_c2pa_provenance" not in offered
