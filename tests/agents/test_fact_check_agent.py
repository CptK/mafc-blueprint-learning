from __future__ import annotations

import json
from pathlib import Path

from ezmm import Image, MultimodalSequence
from ezmm.common.registry import item_registry

from mafc.agents import AgentResult, AgentSession, AgentStatus
from mafc.agents.fact_check.agent import FactCheckAgent
from mafc.blueprints import BlueprintRegistry, BlueprintSelector
from mafc.common.action import Action
from mafc.common.claim import Claim
from mafc.common.evidence import Evidence
from mafc.common.modeling.message import Message
from mafc.common.modeling.model import Model, Response
from mafc.common.modeling.prompt import Prompt

ASSETS_DIR = Path(__file__).resolve().parents[1] / "assets"


class SequencedModel(Model):
    def __init__(self, outputs: list[str]):
        super().__init__(specifier="OPENAI:gpt-5-mini-2025-08-07")
        self.outputs = outputs
        self.calls: list[str] = []

    def generate(self, messages: list[Message]) -> Response:
        self.calls.append("\n".join(f"[{message.role.value}] {message.content}" for message in messages))
        text = self.outputs.pop(0) if self.outputs else ""
        return Response(text=text, total_cost=0.0)


class DummyAction(Action):
    """Simple action used in orchestration tests."""

    name = "dummy_action"

    def __init__(self):
        self._save_parameters(locals())


class FakeWorkerAgent:
    def __init__(self, result_text: str, source: str):
        self.result_text = result_text
        self.source = source
        self.calls: list[AgentSession] = []
        self.description = f"Fake worker for {source}"

    def run(self, session: AgentSession) -> AgentResult:
        self.calls.append(session)
        evidence = Evidence(
            raw=MultimodalSequence(self.result_text),
            action=DummyAction(),
            source=self.source,
            takeaways=MultimodalSequence(self.result_text),
        )
        session.evidences.append(evidence)
        return AgentResult(
            session=session,
            result=MultimodalSequence(self.result_text),
            evidences=[evidence],
            messages=[],
            errors=[],
            status=AgentStatus.COMPLETED,
        )

    def synthesize_from_evidences(self, instruction: str, evidences: list[Evidence]) -> str:
        return self.result_text


def _registered_image() -> Image:
    image = Image(file_path=ASSETS_DIR / "Greece.jpeg")
    item_registry.add_item(image)
    return image


def _make_registry(tmp_path, include_media: bool = True) -> BlueprintRegistry:
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
    if include_media:
        media_path = tmp_path / "media.yaml"
        media_path.write_text(
            """
name: media_location
description: Investigate image location claims.
entry_conditions:
  all:
    - feature: has_image
      op: "=="
      value: true
policy_constraints:
  allowed_actions: [media_agent, web_search_agent]
  max_iterations: 3
required_checks:
  - id: location_checked
    description: The likely location was investigated.
verification_graph:
  start_node: iter1_search
  nodes:
    - id: iter1_search
      type: actions
      actions:
        - action: media_agent
          intent: inspect image
      transition:
        - if: found evidence
          to: verdict_gate
    - id: verdict_gate
      type: gate
      rules:
        support_conditions: [location_checked]
        refute_conditions: []
        if_fail: return unknown
""".strip(),
            encoding="utf-8",
        )
    return BlueprintRegistry.from_path(tmp_path)


def test_fact_check_agent_bootstraps_with_full_blueprint_and_later_reminder(tmp_path) -> None:
    registry = _make_registry(tmp_path)
    selector = BlueprintSelector(
        model=SequencedModel(outputs=[]),
        registry=registry,
        default_blueprint_name="default",
    )
    planner = SequencedModel(
        outputs=[
            """
{
  "decision_type": "delegate",
  "rationale": "Start with media analysis for the image claim.",
  "tasks": [
    {
      "task_id": "media_location",
      "agent_type": "media",
      "instruction": "Check where this image was taken."
    }
  ],
  "target_node_id": "verdict_gate",
  "check_updates": []
}
""".strip(),
            """
{
  "decision_type": "finalize",
  "rationale": "Enough evidence was collected.",
  "final_answer": "The image is consistent with Athens.",
  "check_updates": [{"id":"location_checked","status":"supported","reason":"Media evidence supports the location."}]
}
""".strip(),
        ]
    )
    media_agent = FakeWorkerAgent("Likely Athens based on landmarks.", "image://athens")
    web_search_agent = FakeWorkerAgent("unused", "https://unused.example.com")
    agent = FactCheckAgent(
        model=planner,
        blueprint_selector=selector,
        delegation_agents={
            "media": [media_agent],
            "web_search": [web_search_agent],
        },
    )
    image = _registered_image()
    claim = Claim("This image shows Athens.", image)
    session = AgentSession(id="fact-check:1", goal=Prompt(text="Fact-check claim"), claim=claim)

    result = agent.run(session)

    assert result.result is not None
    assert result.session.status == AgentStatus.COMPLETED
    assert len(media_agent.calls) == 1
    assert "Blueprint graph:" in planner.calls[0]
    assert "Available sub-agents:" in planner.calls[0]
    assert "media: Fake worker for image://athens" in planner.calls[0]
    assert "Blueprint reminder:" in planner.calls[1]
    assert "media_location" in planner.calls[1]
    assert "media_delegation_allowed: True" in planner.calls[0]
    assert "Accepted evidence summaries:" in planner.calls[1]
    assert "Likely Athens based on landmarks." in planner.calls[1]
    assert "The image is consistent with Athens." in str(result.result)


def test_fact_check_agent_can_delegate_web_search_and_finalize_with_synthesis(tmp_path) -> None:
    registry = _make_registry(tmp_path, include_media=False)
    selector = BlueprintSelector(
        model=SequencedModel(outputs=[]),
        registry=registry,
        default_blueprint_name="default",
    )
    planner = SequencedModel(
        outputs=[
            """
{
  "decision_type": "delegate",
  "rationale": "Need web evidence for this text claim.",
  "tasks": [
    {
      "task_id": "web_counterevidence",
      "agent_type": "web_search",
      "instruction": "Search for corroborating sources about the claim."
    }
  ],
  "target_node_id": "synth",
  "check_updates": []
}
""".strip(),
            """
{
  "decision_type": "finalize",
  "rationale": "Sufficient evidence gathered.",
  "instruction": "Summarize the retrieved sources.",
  "check_updates": []
}
""".strip(),
            "Web evidence suggests the claim is unverified.",
        ]
    )
    media_agent = FakeWorkerAgent("unused", "image://unused")
    web_search_agent = FakeWorkerAgent("Source says the claim is disputed.", "https://example.com/source")
    agent = FactCheckAgent(
        model=planner,
        blueprint_selector=selector,
        delegation_agents={
            "media": [media_agent],
            "web_search": [web_search_agent],
        },
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


def test_fact_check_agent_writes_structured_execution_trace(tmp_path) -> None:
    registry = _make_registry(tmp_path)
    selector = BlueprintSelector(
        model=SequencedModel(outputs=[]),
        registry=registry,
        default_blueprint_name="default",
    )
    planner = SequencedModel(
        outputs=[
            """
{
  "decision_type": "delegate",
  "rationale": "Start with media analysis for the image claim.",
  "tasks": [
    {
      "task_id": "media_location",
      "agent_type": "media",
      "instruction": "Check where this image was taken."
    }
  ],
  "target_node_id": "verdict_gate",
  "check_updates": []
}
""".strip(),
            """
{
  "decision_type": "finalize",
  "rationale": "Enough evidence was collected.",
  "final_answer": "The image is consistent with Athens.",
  "check_updates": [{"id":"location_checked","status":"supported","reason":"Media evidence supports the location."}]
}
""".strip(),
        ]
    )
    trace_dir = tmp_path / "traces"
    media_agent = FakeWorkerAgent("Likely Athens based on landmarks.", "image://athens")
    agent = FactCheckAgent(
        model=planner,
        blueprint_selector=selector,
        delegation_agents={"media": [media_agent]},
        trace_dir=trace_dir,
    )
    image = _registered_image()
    session = AgentSession(
        id="fact-check:trace",
        goal=Prompt(text="Fact-check claim"),
        claim=Claim("This image shows Athens.", image),
    )

    result = agent.run(session)

    assert result.result is not None
    trace_path = trace_dir / "fact-check_trace.fact_check_trace.json"
    assert trace_path.exists()

    payload = json.loads(trace_path.read_text(encoding="utf-8"))

    assert payload["agent"] == "FactCheckAgent"
    assert payload["status"] == AgentStatus.COMPLETED.value
    assert payload["blueprint"]["name"] == "media_location"
    assert len(payload["iterations"]) == 2
    assert payload["iterations"][0]["iteration"] == 1
    assert payload["iterations"][0]["node_before"] == "iter1_search"
    assert payload["iterations"][0]["node_after"] == "verdict_gate"
    assert payload["iterations"][0]["planner_messages"][0]["role"] == "system"
    assert payload["iterations"][0]["delegated_tasks"][0]["task_id"] == "media_location"
    assert (
        payload["iterations"][0]["delegated_tasks"][0]["result"]["evidences"][0]["source"] == "image://athens"
    )
    assert payload["iterations"][1]["decision"]["decision_type"] == "finalize"
    assert payload["summary"]["result"]["text"] == "The image is consistent with Athens."
    assert any(event["event_type"] == "planner_prompt" for event in payload["events"])
    assert {"source": "run", "target": "iteration:1", "type": "next"} in payload["flow"]["edges"]
    assert {
        "source": "iteration:1",
        "target": "task:1:media_location",
        "type": "delegates",
    } in payload[
        "flow"
    ]["edges"]


def test_fact_check_agent_can_dispatch_to_multiple_workers_for_one_decision(tmp_path) -> None:
    registry = _make_registry(tmp_path, include_media=False)
    selector = BlueprintSelector(
        model=SequencedModel(outputs=[]),
        registry=registry,
        default_blueprint_name="default",
    )
    planner = SequencedModel(
        outputs=[
            """
{
  "decision_type": "delegate",
  "rationale": "Fan out to multiple retrieval workers.",
  "tasks": [
    {
      "task_id": "web_source_a",
      "agent_type": "web_search",
      "instruction": "Search for corroborating sources about the claim."
    },
    {
      "task_id": "web_source_b",
      "agent_type": "web_search",
      "instruction": "Search for counterevidence about the claim."
    }
  ],
  "target_node_id": "synth",
  "check_updates": []
}
""".strip(),
            """
{
  "decision_type": "finalize",
  "rationale": "Enough evidence gathered.",
  "final_answer": "Multiple sources were consulted.",
  "check_updates": []
}
""".strip(),
        ]
    )
    web_search_agent_a = FakeWorkerAgent("First source.", "https://example.com/a")
    web_search_agent_b = FakeWorkerAgent("Second source.", "https://example.com/b")
    agent = FactCheckAgent(
        model=planner,
        blueprint_selector=selector,
        delegation_agents={
            "web_search": [web_search_agent_a, web_search_agent_b],
        },
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


def test_fact_check_agent_reuses_child_session_for_follow_up_task(tmp_path) -> None:
    registry = _make_registry(tmp_path)
    selector = BlueprintSelector(
        model=SequencedModel(outputs=[]),
        registry=registry,
        default_blueprint_name="default",
    )
    planner = SequencedModel(
        outputs=[
            """
{
  "decision_type": "delegate",
  "rationale": "Start with media analysis.",
  "tasks": [
    {
      "task_id": "media_origin",
      "agent_type": "media",
      "instruction": "Investigate the image origin."
    }
  ],
  "target_node_id": "verdict_gate",
  "check_updates": []
}
""".strip(),
            """
{
  "decision_type": "delegate",
  "rationale": "Ask a follow-up on the same task line.",
  "tasks": [
    {
      "task_id": "media_origin_follow_up",
      "agent_type": "media",
      "instruction": "Follow up on the earliest publication date.",
      "follow_up_to": "media_origin"
    }
  ],
  "target_node_id": "verdict_gate",
  "check_updates": []
}
""".strip(),
            """
{
  "decision_type": "finalize",
  "rationale": "Enough evidence was collected.",
  "final_answer": "The image appears to predate the claimed event.",
  "check_updates": []
}
""".strip(),
        ]
    )
    media_agent = FakeWorkerAgent("Media evidence.", "image://origin")
    agent = FactCheckAgent(
        model=planner,
        blueprint_selector=selector,
        delegation_agents={"media": [media_agent]},
    )
    image = _registered_image()
    session = AgentSession(
        id="fact-check:4",
        goal=Prompt(text="Fact-check claim"),
        claim=Claim("This image is from a recent event.", image),
    )

    result = agent.run(session)

    assert result.result is not None
    assert len(media_agent.calls) == 2
    assert media_agent.calls[0].id == media_agent.calls[1].id


def test_fact_check_agent_allows_staying_when_budget_has_layer_slack(tmp_path) -> None:
    registry = _make_registry(tmp_path)
    selector = BlueprintSelector(
        model=SequencedModel(outputs=[]),
        registry=registry,
        default_blueprint_name="default",
    )
    planner = SequencedModel(
        outputs=[
            """
{
  "decision_type": "delegate",
  "rationale": "Use slack to stay on the current layer for one more step.",
  "tasks": [
    {
      "task_id": "media_stay",
      "agent_type": "media",
      "instruction": "Investigate the image."
    }
  ],
  "check_updates": []
}
""".strip(),
            """
{
  "decision_type": "finalize",
  "rationale": "Enough evidence was collected.",
  "final_answer": "Stayed once, then finalized.",
  "check_updates": []
}
""".strip(),
        ]
    )
    agent = FactCheckAgent(
        model=planner,
        blueprint_selector=selector,
        delegation_agents={"media": [FakeWorkerAgent("Media evidence.", "image://stay")]},
    )
    image = _registered_image()
    session = AgentSession(
        id="fact-check:5",
        goal=Prompt(text="Fact-check claim"),
        claim=Claim("This image shows Athens.", image),
    )

    result = agent.run(session)

    assert result.result is not None
    assert "stay_allowed: True" in planner.calls[0]
    assert "current node: iter1_search" in planner.calls[1]


def test_fact_check_agent_forces_next_layer_when_budget_has_no_slack(tmp_path) -> None:
    forced_path = tmp_path / "forced.yaml"
    forced_path.write_text(
        """
name: forced_progress
description: Force layer progress when slack is gone.
policy_constraints:
  max_iterations: 2
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
          to: layer2
    - id: layer2
      type: gate
      rules:
        support_conditions: []
        refute_conditions: []
        if_fail: return unknown
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
            """
{
  "decision_type": "delegate",
  "rationale": "Try to stay even though there is no slack.",
  "tasks": [
    {
      "task_id": "web_task",
      "agent_type": "web_search",
      "instruction": "Find evidence."
    }
  ],
  "check_updates": []
}
""".strip(),
            """
{
  "decision_type": "finalize",
  "rationale": "Finalize after forced advancement.",
  "final_answer": "Forced advancement worked.",
  "check_updates": []
}
""".strip(),
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
    assert any("auto-advancing to 'layer1'" in error for error in result.errors)
    assert "stay_allowed: False" in planner.calls[0]
    assert "current node: layer1" in planner.calls[1]
