from __future__ import annotations

import json

from mafc.agents import AgentSession, AgentStatus
from mafc.agents.fact_check.agent import FactCheckAgent
from mafc.common.claim import Claim
from mafc.common.modeling.prompt import Prompt

from tests.agents.fact_check.helpers import (
    FakeWorkerAgent,
    SequencedModel,
    make_registry,
    make_selector,
    registered_image,
)


def test_delegates_to_media_agent_and_finalizes(tmp_path) -> None:
    registry = make_registry(tmp_path)
    planner = SequencedModel(
        outputs=[
            json.dumps(
                {
                    "decision_type": "delegate",
                    "rationale": "Start with media analysis for the image claim.",
                    "tasks": [
                        {
                            "task_id": "media_location",
                            "agent_type": "media",
                            "instruction": "Check where this image was taken.",
                        }
                    ],
                }
            ),
            json.dumps(
                {
                    "next_node_id": "finalize",
                    "rationale": "Media evidence supports the Athens location.",
                    "final_answer": "The image is consistent with Athens.",
                    "check_updates": [
                        {
                            "id": "location_checked",
                            "status": "supported",
                            "reason": "Media evidence supports the location.",
                        }
                    ],
                }
            ),
        ]
    )
    media_agent = FakeWorkerAgent("Likely Athens based on landmarks.", "image://athens")
    web_search_agent = FakeWorkerAgent("unused", "https://unused.example.com")
    agent = FactCheckAgent(
        model=planner,
        blueprint_selector=make_selector(registry),
        delegation_agents={"media": [media_agent], "web_search": [web_search_agent]},
    )
    image = registered_image()
    session = AgentSession(
        id="fact-check:1",
        goal=Prompt(text="Fact-check claim"),
        claim=Claim("This image shows Athens.", image),
    )

    result = agent.run(session)

    assert result.result is not None
    assert result.session.status == AgentStatus.COMPLETED
    assert len(media_agent.calls) == 1
    assert "Blueprint graph:" in planner.calls[0]
    assert "Available sub-agents:" in planner.calls[0]
    assert "media: Fake worker for image://athens" in planner.calls[0]
    assert "Routing decision for node:" in planner.calls[1]
    assert "media_location" in planner.calls[1]
    assert "media_delegation_allowed: True" in planner.calls[0]
    assert image.reference in planner.calls[0]
    assert "Accepted evidence summaries:" in planner.calls[1]
    assert "Likely Athens based on landmarks." in planner.calls[1]
    assert "The image is consistent with Athens." in str(result.result)


def test_delegates_to_web_search_and_finalizes_with_synthesis(tmp_path) -> None:
    registry = make_registry(tmp_path, include_media=False)
    planner = SequencedModel(
        outputs=[
            json.dumps(
                {
                    "decision_type": "delegate",
                    "rationale": "Need web evidence for this text claim.",
                    "tasks": [
                        {
                            "task_id": "web_counterevidence",
                            "agent_type": "web_search",
                            "instruction": "Search for corroborating sources about the claim.",
                        }
                    ],
                }
            ),
            "Claim is disputed based on web sources.",
            "Web evidence suggests the claim is unverified.",
        ]
    )
    media_agent = FakeWorkerAgent("unused", "image://unused")
    web_search_agent = FakeWorkerAgent("Source says the claim is disputed.", "https://example.com/source")
    agent = FactCheckAgent(
        model=planner,
        blueprint_selector=make_selector(registry),
        delegation_agents={"media": [media_agent], "web_search": [web_search_agent]},
    )
    session = AgentSession(
        id="fact-check:2",
        goal=Prompt(text="Fact-check claim"),
        claim=Claim("A politician said X happened in 2024."),
    )

    result = agent.run(session)

    assert result.result is not None
    assert result.session.status == AgentStatus.COMPLETED
    assert len(web_search_agent.calls) == 1
    assert "media_delegation_allowed: False" in planner.calls[0]
    assert "Web evidence suggests the claim is unverified." in str(result.result)


def test_dispatches_to_multiple_workers_for_one_decision(tmp_path) -> None:
    registry = make_registry(tmp_path, include_media=False)
    planner = SequencedModel(
        outputs=[
            json.dumps(
                {
                    "decision_type": "delegate",
                    "rationale": "Fan out to multiple retrieval workers.",
                    "tasks": [
                        {
                            "task_id": "web_source_a",
                            "agent_type": "web_search",
                            "instruction": "Search for corroborating sources.",
                        },
                        {
                            "task_id": "web_source_b",
                            "agent_type": "web_search",
                            "instruction": "Search for counterevidence.",
                        },
                    ],
                }
            ),
            "Multiple sources checked.",
            "Multiple sources were consulted.",
        ]
    )
    web_search_agent_a = FakeWorkerAgent("First source.", "https://example.com/a")
    web_search_agent_b = FakeWorkerAgent("Second source.", "https://example.com/b")
    agent = FactCheckAgent(
        model=planner,
        blueprint_selector=make_selector(registry),
        delegation_agents={"web_search": [web_search_agent_a, web_search_agent_b]},
        n_workers=2,
    )
    session = AgentSession(
        id="fact-check:3",
        goal=Prompt(text="Fact-check claim"),
        claim=Claim("A politician said X happened in 2024."),
    )

    result = agent.run(session)

    assert result.result is not None
    assert result.session.status == AgentStatus.COMPLETED
    assert len(web_search_agent_a.calls) == 1
    assert len(web_search_agent_b.calls) == 1
    assert len(result.evidences) == 2


def test_hallucinated_media_tag_fails_task_gracefully(tmp_path) -> None:
    registry = make_registry(tmp_path)
    planner = SequencedModel(
        outputs=[
            json.dumps(
                {
                    "decision_type": "delegate",
                    "rationale": "Analyze the image.",
                    "tasks": [
                        {
                            "task_id": "media_loc",
                            "agent_type": "media",
                            "instruction": "<image:99999999> Where is this?",
                        }
                    ],
                }
            ),
            json.dumps(
                {
                    "next_node_id": "finalize",
                    "rationale": "No media evidence; cannot determine location.",
                    "final_answer": "Location could not be determined.",
                    "check_updates": [],
                }
            ),
        ]
    )
    media_agent = FakeWorkerAgent("unused", "image://unused")
    agent = FactCheckAgent(
        model=planner,
        blueprint_selector=make_selector(registry),
        delegation_agents={"media": [media_agent]},
    )
    image = registered_image()
    session = AgentSession(
        id="fact-check:bad-tag",
        goal=Prompt(text="Fact-check claim"),
        claim=Claim("This image shows Athens.", image),
    )

    result = agent.run(session)

    assert result.session.status == AgentStatus.COMPLETED
    assert len(media_agent.calls) == 0
    assert any("Failed to build session for task 'media_loc'" in e for e in result.errors)


def test_unknown_agent_type_records_error_and_continues(tmp_path) -> None:
    registry = make_registry(tmp_path, include_media=False)
    planner = SequencedModel(
        outputs=[
            json.dumps(
                {
                    "decision_type": "delegate",
                    "rationale": "Delegate to a nonexistent agent type.",
                    "tasks": [
                        {"task_id": "ghost_task", "agent_type": "ghost_agent", "instruction": "Do something."}
                    ],
                }
            ),
            "No evidence gathered.",
            "",
        ]
    )
    agent = FactCheckAgent(
        model=planner,
        blueprint_selector=make_selector(registry),
        delegation_agents={},
    )
    session = AgentSession(
        id="fact-check:unknown-type",
        goal=Prompt(text="Fact-check claim"),
        claim=Claim("A politician said X happened in 2024."),
    )

    result = agent.run(session)

    assert any("ghost_agent" in e for e in result.errors)
