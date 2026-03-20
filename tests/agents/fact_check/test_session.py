from __future__ import annotations

import json

from ezmm import Image
from ezmm.common.registry import item_registry

from mafc.agents import AgentSession
from mafc.agents.fact_check.agent import FactCheckAgent
from mafc.common.claim import Claim
from mafc.common.modeling.prompt import Prompt

from tests.agents.fact_check.helpers import (
    ASSETS_DIR,
    FakeWorkerAgent,
    SequencedModel,
    make_registry,
    make_selector,
    registered_image,
)


def test_reuses_child_session_for_follow_up_task(tmp_path) -> None:
    registry = make_registry(tmp_path)
    planner = SequencedModel(
        outputs=[
            json.dumps(
                {
                    "decision_type": "delegate",
                    "rationale": "Start with media analysis and immediately follow up.",
                    "tasks": [
                        {
                            "task_id": "media_origin",
                            "agent_type": "media",
                            "instruction": "Investigate the image origin.",
                        },
                        {
                            "task_id": "media_origin_follow_up",
                            "agent_type": "media",
                            "instruction": "Follow up on the earliest publication date.",
                            "follow_up_to": "media_origin",
                        },
                    ],
                }
            ),
            json.dumps(
                {
                    "next_node_id": "finalize",
                    "rationale": "Evidence is sufficient.",
                    "final_answer": "The image appears to predate the claimed event.",
                    "check_updates": [],
                }
            ),
        ]
    )
    media_agent = FakeWorkerAgent("Media evidence.", "image://origin")
    agent = FactCheckAgent(
        model=planner,
        blueprint_selector=make_selector(registry),
        delegation_agents={"media": [media_agent]},
    )
    image = registered_image()
    session = AgentSession(
        id="fact-check:4",
        goal=Prompt(text="Fact-check claim"),
        claim=Claim("This image is from a recent event.", image),
    )

    result = agent.run(session)

    assert result.result is not None
    assert len(media_agent.calls) == 2
    assert media_agent.calls[0].id == media_agent.calls[1].id


def test_media_child_session_receives_tagged_image(tmp_path) -> None:
    registry = make_registry(tmp_path)
    image_a = registered_image()
    image_b = Image(file_path=ASSETS_DIR / "Greece.jpeg")
    item_registry.add_item(image_b)
    planner = SequencedModel(
        outputs=[
            json.dumps(
                {
                    "decision_type": "delegate",
                    "rationale": "Analyze the second image.",
                    "tasks": [
                        {
                            "task_id": "media_b",
                            "agent_type": "media",
                            "instruction": f"{image_b.reference} Where is this?",
                        }
                    ],
                }
            ),
            json.dumps(
                {
                    "next_node_id": "finalize",
                    "rationale": "Got media evidence.",
                    "final_answer": "Second image analyzed.",
                    "check_updates": [{"id": "location_checked", "status": "supported", "reason": "done"}],
                }
            ),
        ]
    )
    media_agent = FakeWorkerAgent("Evidence from second image.", "image://b")
    agent = FactCheckAgent(
        model=planner,
        blueprint_selector=make_selector(registry),
        delegation_agents={"media": [media_agent]},
    )
    claim = Claim("Check these images.", image_a, image_b)
    session = AgentSession(
        id="fact-check:img-tag",
        goal=Prompt(text="Fact-check claim"),
        claim=claim,
    )

    result = agent.run(session)

    assert result.result is not None
    assert len(media_agent.calls) == 1
    child_goal = media_agent.calls[0].goal
    assert len(child_goal.images) == 1
    assert child_goal.images[0].reference == image_b.reference
