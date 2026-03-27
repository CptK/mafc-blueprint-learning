from __future__ import annotations

import json

from mafc.agents import AgentSession
from mafc.agents.fact_check.agent import FactCheckAgent
from mafc.blueprints import BlueprintRegistry, BlueprintSelector
from mafc.common.claim import Claim
from mafc.common.modeling.prompt import Prompt

from tests.agents.fact_check.helpers import (
    FakeWorkerAgent,
    SequencedModel,
    make_registry,
    make_selector,
    registered_image,
)


def test_allows_staying_when_budget_has_layer_slack(tmp_path) -> None:
    registry = make_registry(tmp_path)
    planner = SequencedModel(
        outputs=[
            json.dumps(
                {
                    "decision_type": "delegate",
                    "rationale": "Use slack to stay on the current layer for one more step.",
                    "tasks": [
                        {
                            "task_id": "media_stay",
                            "agent_type": "media",
                            "instruction": "Investigate the image.",
                        }
                    ],
                }
            ),
            "Stayed once, then finalized.",
        ]
    )
    agent = FactCheckAgent(
        model=planner,
        blueprint_selector=make_selector(registry),
        delegation_agents={"media": [FakeWorkerAgent("Media evidence.", "image://stay")]},
    )
    image = registered_image()
    session = AgentSession(
        id="fact-check:5",
        goal=Prompt(text="Fact-check claim"),
        claim=Claim("This image shows Athens.", image),
    )

    result = agent.run(session)

    assert result.result is not None
    assert "stay_allowed: True" in planner.calls[0]
    assert "concise fact-check synthesis" in planner.calls[1]


def test_forces_next_layer_when_budget_has_no_slack(tmp_path) -> None:
    forced_path = tmp_path / "forced.yaml"
    forced_path.write_text(
        """
name: forced_progress
description: Force layer progress when slack is gone.
policy_constraints:
  max_iterations: 1
verification_graph:
  start_node: layer0
  nodes:
    - id: layer0
      type: actions
      actions:
        - action: web_search_agent
      transition:
        - if: continue
          to: layer1
    - id: layer1
      type: synthesis
      transition:
        - if: continue
          to: finalize
""".strip(),
        encoding="utf-8",
    )
    default_path = tmp_path / "default.yaml"
    default_path.write_text(
        """
name: default
description: Catch-all fallback blueprint.
policy_constraints:
  max_iterations: 2
verification_graph:
  start_node: synth
  nodes:
    - id: synth
      type: synthesis
      transition: []
""".strip(),
        encoding="utf-8",
    )
    registry = BlueprintRegistry.from_path(tmp_path)
    selector = BlueprintSelector(
        model=SequencedModel(outputs=[]),
        registry=registry,
        default_blueprint_name="default",
    )
    planner = SequencedModel(
        outputs=[
            json.dumps(
                {
                    "decision_type": "delegate",
                    "rationale": "Try to stay even though there is no slack.",
                    "tasks": [
                        {"task_id": "web_task", "agent_type": "web_search", "instruction": "Find evidence."}
                    ],
                }
            ),
            "Forced advancement worked.",
        ]
    )
    agent = FactCheckAgent(
        model=planner,
        blueprint_selector=selector,
        delegation_agents={"web_search": [FakeWorkerAgent("Web evidence.", "https://example.com/forced")]},
    )
    session = AgentSession(
        id="fact-check:6",
        goal=Prompt(text="Fact-check claim"),
        claim=Claim("A politician said X happened in 2024."),
    )

    result = agent.run(session)

    assert result.result is not None
    assert "stay_allowed: False" in planner.calls[0]
    assert "concise fact-check synthesis" in planner.calls[1]
