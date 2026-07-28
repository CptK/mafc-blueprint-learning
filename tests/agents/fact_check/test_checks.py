from __future__ import annotations

import json
from pathlib import Path

from mafc.agents import AgentSession
from mafc.agents.fact_check.agent import FactCheckAgent
from mafc.blueprints import BlueprintRegistry
from mafc.common.claim import Claim
from mafc.common.modeling.prompt import Prompt

from tests.agents.fact_check.helpers import FakeWorkerAgent, SequencedModel, make_selector

# Marker unique to the standalone check-update prompt, used to count how often the
# agent asked for a check-ledger refresh.
CHECK_PROMPT_MARKER = "maintaining the required-check ledger"

SINGLE_EXIT_CHAIN = """
name: default
description: Catch-all fallback blueprint.
policy_constraints:
  max_iterations: 4
required_checks:
  - id: chain_check
    description: The origin of the material was investigated.
verification_graph:
  start_node: layer0_scope
  nodes:
    - id: layer0_scope
      type: actions
      activates_checks: [chain_check]
      actions:
        - action: web_search_agent
      transition:
        - if: continue
          to: layer1_synthesis
    - id: layer1_synthesis
      type: synthesis
      transition:
        - if: continue
          to: finalize
"""

LONG_SINGLE_EXIT_CHAIN = """
name: default
description: Catch-all fallback blueprint.
policy_constraints:
  max_iterations: 5
required_checks:
  - id: chain_check
    description: The origin of the material was investigated.
verification_graph:
  start_node: layer0_scope
  nodes:
    - id: layer0_scope
      type: actions
      activates_checks: [chain_check]
      actions:
        - action: web_search_agent
      transition:
        - if: continue
          to: layer1_synthesis
    - id: layer1_synthesis
      type: synthesis
      transition:
        - if: continue
          to: layer2_synthesis
    - id: layer2_synthesis
      type: synthesis
      transition:
        - if: continue
          to: finalize
"""

MULTI_EXIT_START = """
name: default
description: Catch-all fallback blueprint.
policy_constraints:
  max_iterations: 4
required_checks:
  - id: chain_check
    description: The origin of the material was investigated.
verification_graph:
  start_node: layer0_scope
  nodes:
    - id: layer0_scope
      type: actions
      activates_checks: [chain_check]
      actions:
        - action: web_search_agent
      transition:
        - if: the origin is established
          to: finalize
        - if: the origin is still open
          to: layer1_synthesis
    - id: layer1_synthesis
      type: synthesis
      transition:
        - if: continue
          to: finalize
"""

CHECKLESS_SINGLE_EXIT_CHAIN = """
name: default
description: Catch-all fallback blueprint.
policy_constraints:
  max_iterations: 4
verification_graph:
  start_node: layer0_scope
  nodes:
    - id: layer0_scope
      type: actions
      actions:
        - action: web_search_agent
      transition:
        - if: continue
          to: layer1_synthesis
    - id: layer1_synthesis
      type: synthesis
      transition:
        - if: continue
          to: finalize
"""

DELEGATE_ONE_TASK = json.dumps(
    {
        "decision_type": "delegate",
        "rationale": "Look for the earliest publication.",
        "tasks": [
            {
                "task_id": "origin_search",
                "agent_type": "web_search_agent",
                "instruction": "Find the earliest publication of the material.",
            }
        ],
    }
)


def make_chain_registry(tmp_path: Path, blueprint_yaml: str) -> BlueprintRegistry:
    """Build a registry holding exactly one blueprint, so selection is rule-based."""
    (tmp_path / "default.yaml").write_text(blueprint_yaml.strip(), encoding="utf-8")
    return BlueprintRegistry.from_path(tmp_path)


def build_agent(
    tmp_path: Path, blueprint_yaml: str, outputs: list[str]
) -> tuple[FactCheckAgent, SequencedModel]:
    """Build a fact-check agent over a one-blueprint registry and a scripted model."""
    planner = SequencedModel(outputs=outputs)
    agent = FactCheckAgent(
        model=planner,
        blueprint_selector=make_selector(make_chain_registry(tmp_path, blueprint_yaml)),
        delegation_agents={"web_search": [FakeWorkerAgent("Origin evidence.", "https://origin.example")]},
    )
    return agent, planner


def run_agent(agent: FactCheckAgent, session_id: str) -> dict:
    """Run the agent on a plain text claim and return the run trace."""
    session = AgentSession(
        id=session_id,
        goal=Prompt(text="Fact-check claim"),
        claim=Claim("The material shows a recent event."),
    )
    result = agent.run(session)
    assert result.trace is not None
    return result.trace


def count_check_prompts(planner: SequencedModel) -> int:
    """Count how many model calls were standalone check-update calls."""
    return sum(1 for call in planner.calls if CHECK_PROMPT_MARKER in call)


def test_check_on_single_exit_chain_gets_resolved(tmp_path) -> None:
    """A check activated on a node with only single-exit successors must still resolve.

    Routing auto-advances here without an LLM call, so before the standalone
    check-update pass existed this check stayed 'unchecked' for the whole run.
    """
    agent, planner = build_agent(
        tmp_path,
        SINGLE_EXIT_CHAIN,
        outputs=[
            DELEGATE_ONE_TASK,
            json.dumps(
                {
                    "check_updates": [
                        {
                            "id": "chain_check",
                            "status": "supported",
                            "reason": "The earliest publication was located.",
                        }
                    ]
                }
            ),
            "Intermediate synthesis.",
            "Final synthesis.",
        ],
    )

    trace = run_agent(agent, "fact-check:single-exit")

    assert trace["summary"]["required_checks"] == {"chain_check": "supported"}
    assert trace["summary"]["required_check_reasons"]["chain_check"] == (
        "The earliest publication was located."
    )
    assert count_check_prompts(planner) == 1
    assert trace["iterations"][0]["check_updates"][0]["id"] == "chain_check"


def test_check_update_not_repeated_while_nothing_changed(tmp_path) -> None:
    """Consecutive single-exit nodes must not each pay for a check-update call."""
    agent, planner = build_agent(
        tmp_path,
        LONG_SINGLE_EXIT_CHAIN,
        outputs=[
            DELEGATE_ONE_TASK,
            json.dumps({"check_updates": []}),  # nothing resolved yet
            "Intermediate synthesis.",
            "Second intermediate synthesis.",
            "Final synthesis.",
        ],
    )

    trace = run_agent(agent, "fact-check:long-chain")

    # Three single-exit nodes, but only the one that saw new evidence asks.
    assert count_check_prompts(planner) == 1
    assert trace["summary"]["required_checks"] == {"chain_check": "unchecked"}


def test_multi_option_node_still_updates_checks_via_routing(tmp_path) -> None:
    """At a node with a real choice the routing call keeps carrying the updates."""
    agent, planner = build_agent(
        tmp_path,
        MULTI_EXIT_START,
        outputs=[
            DELEGATE_ONE_TASK,
            json.dumps(
                {
                    "next_node_id": "finalize",
                    "rationale": "The origin is established.",
                    "final_answer": "The material predates the claimed event.",
                    "check_updates": [
                        {"id": "chain_check", "status": "refuted", "reason": "Older original found."}
                    ],
                }
            ),
        ],
    )

    trace = run_agent(agent, "fact-check:multi-exit")

    assert count_check_prompts(planner) == 0
    assert trace["summary"]["required_checks"] == {"chain_check": "refuted"}


def test_no_check_update_call_without_open_checks(tmp_path) -> None:
    """A blueprint without required checks never pays for a check-update call."""
    agent, planner = build_agent(
        tmp_path,
        CHECKLESS_SINGLE_EXIT_CHAIN,
        outputs=[DELEGATE_ONE_TASK, "Intermediate synthesis.", "Final synthesis."],
    )

    trace = run_agent(agent, "fact-check:no-checks")

    assert count_check_prompts(planner) == 0
    assert trace["summary"]["required_checks"] == {}
