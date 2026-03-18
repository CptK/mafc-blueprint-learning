from __future__ import annotations

import json
from pathlib import Path

from ezmm import Image, MultimodalSequence
from ezmm.common.registry import item_registry

from mafc.agents import AgentResult, AgentSession, AgentStatus
from mafc.agents.fact_check.agent import FactCheckAgent
from mafc.agents.web_search.agent import WebSearchAgent
from mafc.blueprints import BlueprintRegistry, BlueprintSelector
from mafc.common.action import Action
from mafc.common.claim import Claim
from mafc.common.evidence import Evidence
from mafc.common.modeling.message import Message
from mafc.common.modeling.model import Model, Response
from mafc.common.modeling.prompt import Prompt
from mafc.tools.web_search.common import Query, SearchResults, WebSource
from mafc.tools.web_search.integrations.integration import RetrievalIntegration

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


class FakeSearchTool:
    def __init__(self, result_map: dict[str, SearchResults | None]):
        self.result_map = result_map

    def search(self, query: Query):
        return self.result_map.get(query.text or "", SearchResults(sources=[], query=query))


class FakeRetriever(RetrievalIntegration):
    domains = ["*"]

    def __init__(self, payload_by_url: dict[str, str | None]):
        super().__init__()
        self.payload_by_url = payload_by_url

    def _retrieve(self, url: str):
        payload = self.payload_by_url.get(url)
        return None if payload is None else Prompt(text=payload)


def _make_search_result(query_text: str, urls: list[str]) -> SearchResults:
    query = Query(text=query_text)
    sources = [WebSource(reference=url, title=f"T:{url}") for url in urls]
    return SearchResults(sources=sources, query=query)


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
  max_iterations: 4
verification_graph:
  start_node: initial_search
  nodes:
    - id: initial_search
      type: actions
      actions:
        - action: web_search_agent
      transition:
        - if: done
          to: synth
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
            # Iteration 1: action node execution for iter1_search — delegate
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
  ]
}
""".strip(),
            # Iteration 2: LLM routing for verdict_gate — finalize
            """
{
  "next_node_id": "finalize",
  "rationale": "Media evidence supports the Athens location.",
  "final_answer": "The image is consistent with Athens.",
  "check_updates": [{"id": "location_checked", "status": "supported", "reason": "Media evidence supports the location."}]
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
    assert "Routing decision for node:" in planner.calls[1]
    assert "media_location" in planner.calls[1]
    assert "media_delegation_allowed: True" in planner.calls[0]
    assert image.reference in planner.calls[0]
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
            # Iteration 1: action node execution for initial_search — delegate
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
  ]
}
""".strip(),
            # Iteration 2: synthesis node auto-synthesis
            "Claim is disputed based on web sources.",
            # _finalize_run: final synthesis
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
            # Iteration 1: action node execution for iter1_search — delegate
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
  ]
}
""".strip(),
            # Iteration 2: LLM routing for verdict_gate — finalize
            """
{
  "next_node_id": "finalize",
  "rationale": "Evidence is sufficient.",
  "final_answer": "The image is consistent with Athens.",
  "check_updates": []
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
    assert payload["iterations"][1]["routing"]["target_node_id"] == "finalize"
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


def test_fact_check_agent_embeds_web_search_child_trace(tmp_path) -> None:
    registry = _make_registry(tmp_path, include_media=False)
    selector = BlueprintSelector(
        model=SequencedModel(outputs=[]),
        registry=registry,
        default_blueprint_name="default",
    )
    planner = SequencedModel(
        outputs=[
            # Iteration 1: action node execution for initial_search — delegate
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
  ]
}
""".strip(),
            # Iteration 2: synthesis node auto-synthesis
            "Evidence collected.",
            # _finalize_run: final synthesis
            "Multiple sources were consulted.",
        ]
    )
    child_planner = SequencedModel(outputs=['{"queries":["q1"],"done":true}'])
    child_summarizer = SequencedModel(
        outputs=["Summary step 1", "Web evidence suggests the claim is unverified."]
    )
    trace_dir = tmp_path / "traces"
    web_search_agent = WebSearchAgent(
        main_model=child_planner,
        summarization_model=child_summarizer,
        search_tool=FakeSearchTool({"q1": _make_search_result("q1", ["https://example.com/source"])}),
        retriever=FakeRetriever({"https://example.com/source": "Retrieved content"}),
    )
    agent = FactCheckAgent(
        model=planner,
        blueprint_selector=selector,
        delegation_agents={"web_search": [web_search_agent]},
        trace_dir=trace_dir,
    )
    session = AgentSession(
        id="fact-check:web-trace",
        goal=Prompt(text="Fact-check claim"),
        claim=Claim("A politician said X happened in 2024."),
    )

    result = agent.run(session)

    assert result.result is not None
    trace_path = trace_dir / "fact-check_web-trace.fact_check_trace.json"
    payload = json.loads(trace_path.read_text(encoding="utf-8"))
    delegated_task = payload["iterations"][0]["delegated_tasks"][0]
    assert delegated_task["task_id"] == "web_counterevidence"
    assert delegated_task["child_trace"]["agent"] == "WebSearchAgent"
    assert delegated_task["child_trace"]["iterations"][0]["resolved_plan"]["queries"] == ["q1"]
    assert (
        delegated_task["child_trace"]["iterations"][0]["retrievals"][0]["source"]["url"]
        == "https://example.com/source"
    )


def test_fact_check_agent_can_dispatch_to_multiple_workers_for_one_decision(tmp_path) -> None:
    registry = _make_registry(tmp_path, include_media=False)
    selector = BlueprintSelector(
        model=SequencedModel(outputs=[]),
        registry=registry,
        default_blueprint_name="default",
    )
    planner = SequencedModel(
        outputs=[
            # Iteration 1: action node execution for initial_search — delegate 2 tasks
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
  ]
}
""".strip(),
            # Iteration 2: synthesis node auto-synthesis
            "Multiple sources checked.",
            # _finalize_run: final synthesis
            "Multiple sources were consulted.",
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
            # Iteration 1: action node — dispatch both tasks in one delegation
            """
{
  "decision_type": "delegate",
  "rationale": "Start with media analysis and immediately follow up.",
  "tasks": [
    {
      "task_id": "media_origin",
      "agent_type": "media",
      "instruction": "Investigate the image origin."
    },
    {
      "task_id": "media_origin_follow_up",
      "agent_type": "media",
      "instruction": "Follow up on the earliest publication date.",
      "follow_up_to": "media_origin"
    }
  ]
}
""".strip(),
            # Iteration 2: LLM routing for verdict_gate — finalize
            """
{
  "next_node_id": "finalize",
  "rationale": "Evidence is sufficient.",
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
            # Iteration 1: action node — delegate
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
  ]
}
""".strip(),
            # Iteration 2: LLM routing for verdict_gate — finalize
            """
{
  "next_node_id": "finalize",
  "rationale": "Evidence collected.",
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
    assert "Routing decision for node:" in planner.calls[1]


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
            # Iteration 1: layer0 action node — delegate
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
  ]
}
""".strip(),
            # Iteration 2: layer1 synthesis node — auto-synthesis
            "Evidence gathered from web.",
            # _finalize_run: final synthesis
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


def test_fact_check_agent_prompt_shows_actual_image_references(tmp_path) -> None:
    registry = _make_registry(tmp_path)
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
                    "rationale": "Analyze both images.",
                    "tasks": [
                        {"task_id": "media_0", "agent_type": "media", "instruction": "Investigate."},
                    ],
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
    image_a = _registered_image()
    image_b = Image(file_path=ASSETS_DIR / "Greece.jpeg")
    item_registry.add_item(image_b)
    media_agent = FakeWorkerAgent("Evidence.", "image://result")
    agent = FactCheckAgent(
        model=planner,
        blueprint_selector=selector,
        delegation_agents={"media": [media_agent]},
    )
    claim = Claim("Check these images.", image_a, image_b)
    session = AgentSession(id="fact-check:refs", goal=Prompt(text="Fact-check claim"), claim=claim)

    agent.run(session)

    assert image_a.reference in planner.calls[0]
    assert image_b.reference in planner.calls[0]
    assert "images: 2" in planner.calls[0]


def test_fact_check_agent_hallucinates_media_tag_fails_task_gracefully(tmp_path) -> None:
    registry = _make_registry(tmp_path)
    selector = BlueprintSelector(
        model=SequencedModel(outputs=[]),
        registry=registry,
        default_blueprint_name="default",
    )
    planner = SequencedModel(
        outputs=[
            # Iter 1: delegate with a hallucinated image reference that does not exist
            json.dumps(
                {
                    "decision_type": "delegate",
                    "rationale": "Analyze the image.",
                    "tasks": [
                        {
                            "task_id": "media_loc",
                            "agent_type": "media",
                            "instruction": "<image:99999999> Where is this?",
                        },
                    ],
                }
            ),
            # verdict_gate has two routing options → LLM routing call
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
        blueprint_selector=selector,
        delegation_agents={"media": [media_agent]},
    )
    image = _registered_image()
    session = AgentSession(
        id="fact-check:bad-tag",
        goal=Prompt(text="Fact-check claim"),
        claim=Claim("This image shows Athens.", image),
    )

    result = agent.run(session)

    assert result.session.status == AgentStatus.COMPLETED
    assert len(media_agent.calls) == 0
    assert any("Failed to build session for task 'media_loc'" in e for e in result.errors)


def test_fact_check_agent_media_child_session_receives_tagged_image(tmp_path) -> None:
    registry = _make_registry(tmp_path)
    selector = BlueprintSelector(
        model=SequencedModel(outputs=[]),
        registry=registry,
        default_blueprint_name="default",
    )
    image_a = _registered_image()
    image_b = Image(file_path=ASSETS_DIR / "Greece.jpeg")
    item_registry.add_item(image_b)
    planner = SequencedModel(
        outputs=[
            # Instruct to analyze image_b specifically using its actual ezmm reference
            json.dumps(
                {
                    "decision_type": "delegate",
                    "rationale": "Analyze the second image.",
                    "tasks": [
                        {
                            "task_id": "media_b",
                            "agent_type": "media",
                            "instruction": f"{image_b.reference} Where is this?",
                        },
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
        blueprint_selector=selector,
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
