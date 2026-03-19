from __future__ import annotations

from ezmm import MultimodalSequence

from mafc.agents.common import AgentStatus
from mafc.agents.media.agent import MediaAgent

from tests.agents.media.helpers import (
    FakeGeolocator,
    FakeRisTool,
    SequencedModel,
    empty_ris_result,
    geo_result,
    make_session,
    registered_image,
    registered_video,
)


def _synthesis_answer(text: str) -> str:
    return f'{{"answer":"{text}","relevant_evidence_ids":["ev_1"]}}'


def test_stop_signal_aborts_before_execution() -> None:
    image = registered_image()
    model = SequencedModel(outputs=[])
    agent = MediaAgent(
        model=model,
        ris_tool=FakeRisTool(empty_ris_result(image.reference)),
        geolocator=FakeGeolocator(geo_result(image.reference)),
    )
    agent._should_stop = True

    out = agent.run(make_session(MultimodalSequence("Where was this taken?", image)))

    assert out.result is None
    assert out.session.status == AgentStatus.FAILED
    assert any("stopped" in e for e in out.errors)
    assert model.calls == []


def test_empty_instruction_aborts() -> None:
    # Goal has no media so the no-media check is never reached;
    # the empty-instruction check fires first.
    # Tools are never called, so we use the defaults (real tools that stay idle).
    model = SequencedModel(outputs=[])
    agent = MediaAgent(model=model)

    out = agent.run(make_session(MultimodalSequence("   ")))

    assert out.result is None
    assert out.session.status == AgentStatus.FAILED
    assert any("prompt" in e.lower() or "empty" in e.lower() for e in out.errors)
    assert model.calls == []


def test_no_media_items_aborts() -> None:
    model = SequencedModel(outputs=[])
    ris_tool = FakeRisTool(empty_ris_result("<<image:0>>"))
    geolocator = FakeGeolocator(geo_result("<<image:0>>"))
    agent = MediaAgent(model=model, ris_tool=ris_tool, geolocator=geolocator)

    out = agent.run(make_session(MultimodalSequence("Is this claim true?")))

    assert out.result is None
    assert out.session.status == AgentStatus.FAILED
    assert any("image or video" in e for e in out.errors)
    assert ris_tool.performed == []
    assert geolocator.performed == []


def test_multiple_media_items_appends_warning_and_processes_first() -> None:
    image = registered_image()
    video = registered_video()
    model = SequencedModel(
        outputs=[
            '{"tools":["geolocate"]}',
            _synthesis_answer("Taken in Greece."),
        ]
    )
    ris_tool = FakeRisTool(empty_ris_result(image.reference))
    geolocator = FakeGeolocator(geo_result(image.reference))
    agent = MediaAgent(model=model, summarization_model=model, ris_tool=ris_tool, geolocator=geolocator)

    out = agent.run(make_session(MultimodalSequence("Investigate.", image, video)))

    assert out.session.status == AgentStatus.COMPLETED
    assert any("multiple media" in e.lower() for e in out.errors)
    # Only the first item (image) was passed to the geolocator
    assert all(a.media == image for a in geolocator.performed)


def test_no_evidences_collected_returns_failed_status() -> None:
    image = registered_image()
    # The planner always falls back to known tools when parsing fails or tool names
    # are invalid, so "no evidences" is unreachable through the planner alone.
    # Stub _run_selected_tools to return nothing to exercise the code path directly.
    model = SequencedModel(outputs=[])
    agent = MediaAgent(
        model=model,
        ris_tool=FakeRisTool(empty_ris_result(image.reference)),
        geolocator=FakeGeolocator(geo_result(image.reference)),
    )
    agent._run_selected_tools = lambda *args, **kwargs: []  # type: ignore[method-assign]

    out = agent.run(make_session(MultimodalSequence("Where was this taken?", image)))

    assert out.result is None
    assert out.session.status == AgentStatus.FAILED
    assert out.evidences == []
