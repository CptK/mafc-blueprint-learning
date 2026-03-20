from __future__ import annotations

import json

from mafc.agents import AgentSession, AgentStatus
from mafc.agents.fact_check.agent import FactCheckAgent
from mafc.agents.web_search.agent import WebSearchAgent
from mafc.common.claim import Claim
from mafc.common.modeling.prompt import Prompt

from tests.agents.fact_check.helpers import (
    FakeRetriever,
    FakeSearchTool,
    FakeWorkerAgent,
    SequencedModel,
    make_registry,
    make_search_result,
    make_selector,
    registered_image,
)


def test_writes_structured_execution_trace(tmp_path) -> None:
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
                    "rationale": "Evidence is sufficient.",
                    "final_answer": "The image is consistent with Athens.",
                    "check_updates": [],
                }
            ),
        ]
    )
    trace_dir = tmp_path / "traces"
    media_agent = FakeWorkerAgent("Likely Athens based on landmarks.", "image://athens")
    agent = FactCheckAgent(
        model=planner,
        blueprint_selector=make_selector(registry),
        delegation_agents={"media": [media_agent]},
        trace_dir=trace_dir,
    )
    image = registered_image()
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
    assert {"source": "iteration:1", "target": "task:1:media_location", "type": "delegates"} in payload[
        "flow"
    ]["edges"]


def test_embeds_web_search_child_trace(tmp_path) -> None:
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
            "Evidence collected.",
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
        search_tool=FakeSearchTool({"q1": make_search_result("q1", ["https://example.com/source"])}),
        retriever=FakeRetriever({"https://example.com/source": "Retrieved content"}),
    )
    agent = FactCheckAgent(
        model=planner,
        blueprint_selector=make_selector(registry),
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
