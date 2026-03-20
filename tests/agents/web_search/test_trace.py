from __future__ import annotations

import json

from mafc.agents.common import AgentStatus
from mafc.agents.web_search.agent import WebSearchAgent

from tests.agents.web_search.helpers import (
    FakeRetriever,
    FakeSearchTool,
    SequencedModel,
    make_search_result,
    make_session,
)


def test_writes_structured_trace(tmp_path) -> None:
    planner = SequencedModel(outputs=['{"queries":["q1"],"done":true}'])
    summarizer = SequencedModel(outputs=["Summary step 1", "Final synthesis"])
    retriever = FakeRetriever({"https://a.example.com": "Retrieved page content A"})
    search_tool = FakeSearchTool({"q1": make_search_result("q1", ["https://a.example.com"])})
    trace_dir = tmp_path / "traces"

    agent = WebSearchAgent(
        main_model=planner,
        summarization_model=summarizer,
        search_tool=search_tool,
        retriever=retriever,
        trace_dir=trace_dir,
    )
    out = agent.run(make_session("Any task"))

    assert out.result is not None
    trace_path = trace_dir / "session_Any_task.web_search_trace.json"
    assert trace_path.exists()

    payload = json.loads(trace_path.read_text(encoding="utf-8"))

    assert payload["agent"] == "WebSearchAgent"
    assert payload["status"] == AgentStatus.COMPLETED.value
    assert len(payload["iterations"]) == 1
    assert payload["iterations"][0]["resolved_plan"]["queries"] == ["q1"]
    assert payload["iterations"][0]["search_results"][0]["query_text"] == "q1"
    assert payload["iterations"][0]["selected_sources"][0]["sources"][0]["url"] == "https://a.example.com"
    assert payload["iterations"][0]["retrievals"][0]["source"]["url"] == "https://a.example.com"
    assert payload["summary"]["result"]["result"]["text"] == "Final synthesis"
    assert any(event["event_type"] == "search_result" for event in payload["events"])
    assert any(event["event_type"] == "retrieval_result" for event in payload["events"])
