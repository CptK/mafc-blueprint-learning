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


def test_prompt_shows_actual_image_references(tmp_path) -> None:
    registry = make_registry(tmp_path)
    planner = SequencedModel(
        outputs=[
            json.dumps(
                {
                    "decision_type": "delegate",
                    "rationale": "Analyze both images.",
                    "tasks": [{"task_id": "media_0", "agent_type": "media", "instruction": "Investigate."}],
                }
            ),
            json.dumps(
                {
                    "next_node_id": "finalize",
                    "rationale": "Done.",
                    "final_answer": "Both images analyzed.",
                    "check_updates": [{"id": "location_checked", "status": "supported", "reason": "done"}],
                }
            ),
        ]
    )
    image_a = registered_image()
    image_b = Image(file_path=ASSETS_DIR / "Greece.jpeg")
    item_registry.add_item(image_b)
    media_agent = FakeWorkerAgent("Evidence.", "image://result")
    agent = FactCheckAgent(
        model=planner,
        blueprint_selector=make_selector(registry),
        delegation_agents={"media": [media_agent]},
    )
    claim = Claim("Check these images.", image_a, image_b)
    session = AgentSession(id="fact-check:refs", goal=Prompt(text="Fact-check claim"), claim=claim)

    agent.run(session)

    assert image_a.reference in planner.calls[0]
    assert image_b.reference in planner.calls[0]
    assert "images: 2" in planner.calls[0]
