from __future__ import annotations

from mafc.agents.common import AgentSession
from mafc.common.action import Action
from mafc.common.modeling.message import Message
from mafc.common.modeling.model import Model, Response
from mafc.common.modeling.prompt import Prompt
from mafc.tools.web_search.common import Query, SearchResults, WebSource
from mafc.tools.web_search.integrations.integration import RetrievalIntegration


class SequencedModel(Model):
    def __init__(self, outputs: list[str]):
        super().__init__(specifier="OPENAI:gpt-5-mini-2025-08-07")
        self.outputs = outputs
        self.calls: list[str] = []

    def _do_generate(self, messages: list[Message]) -> Response:
        self.calls.append("\n".join(f"[{message.role.value}] {message.content}" for message in messages))
        text = self.outputs.pop(0) if self.outputs else ""
        return Response(text=text, total_cost=0.0)


class FakeSearchTool:
    def __init__(self, result_map: dict[str, SearchResults | None]):
        self.result_map = result_map
        self.queries: list[str] = []
        self.query_objects: list[Query] = []

    def search(self, query: Query) -> SearchResults:
        self.query_objects.append(query)
        self.queries.append(query.text or "")
        result = self.result_map.get(query.text or "") or SearchResults(sources=[], query=query)
        return result


class FakeRetriever(RetrievalIntegration):
    domains = ["*"]

    def __init__(self, payload_by_url: dict[str, str | None]):
        super().__init__()
        self.payload_by_url = payload_by_url
        self.calls: list[str] = []

    def _retrieve(self, url: str) -> Prompt | None:
        self.calls.append(url)
        payload = self.payload_by_url.get(url)
        return None if payload is None else Prompt(text=payload)


class DummyAction(Action):
    name = "dummy"

    def __init__(self):
        self._save_parameters(locals())


def make_search_result(query_text: str, urls: list[str]) -> SearchResults:
    query = Query(text=query_text)
    sources = [WebSource(reference=url, title=f"T:{url}") for url in urls]
    return SearchResults(sources=sources, query=query)


def make_session(goal: str) -> AgentSession:
    return AgentSession(id=f"session:{goal}", goal=Prompt(text=goal))
