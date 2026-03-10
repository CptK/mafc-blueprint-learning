from datetime import date

from mafc.agents.common import AgentSession, AgentStatus
from mafc.agents.web_search_agent import WebSearchAgent
from mafc.common.modeling.model import Model, Response
from mafc.common.modeling.prompt import Prompt
from mafc.tools.web_search.common import Query, SearchResults, WebSource
from mafc.tools.web_search.integrations.integration import RetrievalIntegration


class SequencedModel(Model):
    def __init__(self, outputs: list[str]):
        super().__init__(specifier="OPENAI:gpt-5-mini-2025-08-07")
        self.outputs = outputs
        self.calls: list[str] = []

    def generate(self, prompt: Prompt) -> Response:
        self.calls.append(str(prompt))
        text = self.outputs.pop(0) if self.outputs else ""
        return Response(text=text, total_cost=0.0)


class FakeSearchTool:
    def __init__(self, result_map: dict[str, SearchResults | None]):
        self.result_map = result_map
        self.queries: list[str] = []
        self.query_objects: list[Query] = []

    def search(self, query: Query):
        self.query_objects.append(query)
        self.queries.append(query.text or "")
        return self.result_map.get(query.text or "", SearchResults(sources=[], query=query))


class FakeRetriever(RetrievalIntegration):
    domains = ["*"]

    def __init__(self, payload_by_url: dict[str, str | None]):
        super().__init__()
        self.payload_by_url = payload_by_url
        self.calls: list[str] = []

    def _retrieve(self, url: str):
        self.calls.append(url)
        payload = self.payload_by_url.get(url)
        return None if payload is None else Prompt(text=payload)


def _make_search_result(query_text: str, urls: list[str]) -> SearchResults:
    query = Query(text=query_text)
    sources = [WebSource(reference=url, title=f"T:{url}") for url in urls]
    return SearchResults(sources=sources, query=query)


def _make_session(goal: str) -> AgentSession:
    return AgentSession(id=f"session:{goal}", goal=Prompt(text=goal))


def test_web_search_agent_iterative_loop() -> None:
    planner = SequencedModel(
        outputs=[
            '{"queries":["Samrat Choudhary helicopter fall election campaign"],"done":false}',
            '{"queries":[],"done":true}',
        ]
    )
    summarizer = SequencedModel(outputs=["Summary step 1"])
    retriever = FakeRetriever({"https://a.example.com": "Retrieved page content A"})
    search_tool = FakeSearchTool(
        {
            "Samrat Choudhary helicopter fall election campaign": _make_search_result(
                "Samrat Choudhary helicopter fall election campaign",
                ["https://a.example.com"],
            )
        }
    )

    agent = WebSearchAgent(
        main_model=planner,
        summarization_model=summarizer,
        search_tool=search_tool,
        retriever=retriever,
        max_iterations=4,
    )
    out = agent.run(_make_session("Retrieve incidents during the election campaign of Samrat Choudhary."))

    assert out.result is not None
    assert out.session.status == AgentStatus.COMPLETED
    assert len(out.messages) == 1
    assert "Iteration 1 synthesis:" in str(out.result)
    assert "Summary step 1" in str(out.result)
    assert out.errors == []
    assert search_tool.queries == ["Samrat Choudhary helicopter fall election campaign"]
    assert retriever.calls == ["https://a.example.com"]


def test_web_search_agent_handles_invalid_planner_output() -> None:
    planner = SequencedModel(outputs=["not-json"])
    search_tool = FakeSearchTool({})
    retriever = FakeRetriever({})
    agent = WebSearchAgent(main_model=planner, search_tool=search_tool, retriever=retriever)

    out = agent.run(_make_session("Any task"))
    assert out.result is not None
    assert "Falling back to the original task text" in out.errors[0]
    assert search_tool.queries == ["Any task"]


def test_web_search_agent_collects_query_search_errors() -> None:
    class BrokenSearchTool(FakeSearchTool):
        def search(self, query: Query):
            raise RuntimeError("search down")

    planner = SequencedModel(outputs=['{"queries":["q1"],"done":true}'])
    summarizer = SequencedModel(outputs=["summary"])
    agent = WebSearchAgent(
        main_model=planner,
        summarization_model=summarizer,
        search_tool=BrokenSearchTool({}),
    )

    out = agent.run(_make_session("Any task"))
    assert out.result is None
    assert out.session.status == AgentStatus.FAILED
    assert any("Search failed for query 'q1'" in error for error in out.errors)


def test_web_search_agent_collects_retrieval_errors() -> None:
    planner = SequencedModel(outputs=['{"queries":["q1"],"done":true}'])
    summarizer = SequencedModel(outputs=["summary"])
    search_tool = FakeSearchTool({"q1": _make_search_result("q1", ["https://a.example.com"])})
    retriever = FakeRetriever({"https://a.example.com": None})
    agent = WebSearchAgent(
        main_model=planner,
        summarization_model=summarizer,
        search_tool=search_tool,
        retriever=retriever,
    )

    out = agent.run(_make_session("Any task"))
    assert out.result is not None
    assert out.errors == ["Failed to retrieve content from https://a.example.com"]


def test_web_search_agent_parses_json_embedded_in_text() -> None:
    planner = SequencedModel(outputs=['Plan:\n{"queries":["q1"],"done":true}\nThanks'])
    summarizer = SequencedModel(outputs=["summary"])
    search_tool = FakeSearchTool({"q1": _make_search_result("q1", ["https://a.example.com"])})
    retriever = FakeRetriever({"https://a.example.com": "content"})
    agent = WebSearchAgent(
        main_model=planner,
        summarization_model=summarizer,
        search_tool=search_tool,
        retriever=retriever,
    )

    out = agent.run(_make_session("Any task"))
    assert out.result is not None
    assert out.errors == []
    assert search_tool.queries == ["q1"]


def test_web_search_agent_repairs_non_json_planner_output() -> None:
    planner = SequencedModel(
        outputs=[
            "Ich suche jetzt nach passenden Quellen.",
            '{"queries":["q1"],"done":true}',
        ]
    )
    summarizer = SequencedModel(outputs=["summary"])
    search_tool = FakeSearchTool({"q1": _make_search_result("q1", ["https://a.example.com"])})
    retriever = FakeRetriever({"https://a.example.com": "content"})
    agent = WebSearchAgent(
        main_model=planner,
        summarization_model=summarizer,
        search_tool=search_tool,
        retriever=retriever,
    )

    out = agent.run(_make_session("Any task"))
    assert out.result is not None
    assert out.errors == []
    assert search_tool.queries == ["q1"]


def test_web_search_agent_falls_back_when_summary_is_failure_text() -> None:
    planner = SequencedModel(outputs=['{"queries":["q1"],"done":true}'])
    summarizer = SequencedModel(outputs=["Failed to generate a response."])
    search_tool = FakeSearchTool({"q1": _make_search_result("q1", ["https://a.example.com"])})
    retriever = FakeRetriever({"https://a.example.com": "retrieved content"})
    agent = WebSearchAgent(
        main_model=planner,
        summarization_model=summarizer,
        search_tool=search_tool,
        retriever=retriever,
    )

    out = agent.run(_make_session("Any task"))
    assert out.result is not None
    assert "retrieved content" in str(out.result)


def test_web_search_agent_passes_end_date_to_query() -> None:
    planner = SequencedModel(outputs=['{"queries":["q1"],"done":true}'])
    summarizer = SequencedModel(outputs=["summary"])
    search_tool = FakeSearchTool(
        {
            "q1": SearchResults(
                sources=[
                    WebSource(
                        reference="https://a.example.com",
                        title="A",
                        release_date=date(2024, 1, 10),
                    ),
                    WebSource(
                        reference="https://b.example.com",
                        title="B",
                        release_date=date(2024, 2, 10),
                    ),
                    WebSource(
                        reference="https://c.example.com",
                        title="C",
                        release_date=None,
                    ),
                ],
                query=Query(text="q1"),
            )
        }
    )
    retriever = FakeRetriever(
        {
            "https://a.example.com": "older content",
            "https://b.example.com": "newer content",
            "https://c.example.com": "undated content",
        }
    )
    cutoff = date(2024, 1, 31)
    agent = WebSearchAgent(
        main_model=planner,
        summarization_model=summarizer,
        search_tool=search_tool,
        retriever=retriever,
        latest_allowed_date=cutoff,
    )

    out = agent.run(_make_session("Any task"))

    assert out.result is not None
    assert search_tool.query_objects[0].end_date == cutoff
    assert sorted(retriever.calls) == sorted(
        [
            "https://a.example.com",
            "https://b.example.com",
            "https://c.example.com",
        ]
    )


def test_web_search_agent_filters_sources_with_model_when_many_candidates() -> None:
    planner = SequencedModel(
        outputs=[
            '{"queries":["q1"],"done":true}',
            '{"selected_urls":["https://b.example.com","https://d.example.com"]}',
        ]
    )
    summarizer = SequencedModel(outputs=["summary"])
    urls = [
        "https://a.example.com",
        "https://b.example.com",
        "https://c.example.com",
        "https://d.example.com",
        "https://e.example.com",
        "https://f.example.com",
    ]
    search_tool = FakeSearchTool({"q1": _make_search_result("q1", urls)})
    retriever = FakeRetriever({url: f"content:{url}" for url in urls})
    agent = WebSearchAgent(
        main_model=planner,
        summarization_model=summarizer,
        search_tool=search_tool,
        retriever=retriever,
        max_results_per_query=5,
    )

    out = agent.run(_make_session("Any task"))

    assert out.result is not None
    assert sorted(retriever.calls) == sorted(["https://b.example.com", "https://d.example.com"])


def test_web_search_agent_filters_sources_globally_across_queries() -> None:
    planner = SequencedModel(
        outputs=[
            '{"queries":["q1","q2"],"done":true}',
            '{"selected_urls":["https://c.example.com","https://e.example.com"]}',
        ]
    )
    summarizer = SequencedModel(outputs=["summary"])
    search_tool = FakeSearchTool(
        {
            "q1": _make_search_result(
                "q1",
                [
                    "https://a.example.com",
                    "https://b.example.com",
                    "https://c.example.com",
                ],
            ),
            "q2": _make_search_result(
                "q2",
                [
                    "https://d.example.com",
                    "https://e.example.com",
                    "https://f.example.com",
                ],
            ),
        }
    )
    retriever = FakeRetriever(
        {
            "https://a.example.com": "content:a",
            "https://b.example.com": "content:b",
            "https://c.example.com": "content:c",
            "https://d.example.com": "content:d",
            "https://e.example.com": "content:e",
            "https://f.example.com": "content:f",
        }
    )
    agent = WebSearchAgent(
        main_model=planner,
        summarization_model=summarizer,
        search_tool=search_tool,
        retriever=retriever,
        max_results_per_query=5,
    )

    out = agent.run(_make_session("Any task"))

    assert out.result is not None
    assert sorted(retriever.calls) == sorted(["https://c.example.com", "https://e.example.com"])
