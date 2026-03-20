from __future__ import annotations

from pathlib import Path
from typing import cast

from ezmm import Image, MultimodalSequence
from ezmm.common.registry import item_registry

from mafc.agents import AgentResult, AgentSession, AgentStatus
from mafc.agents.agent import Agent
from mafc.blueprints import BlueprintRegistry, BlueprintSelector
from mafc.common.action import Action
from mafc.common.evidence import Evidence
from mafc.common.modeling.message import Message
from mafc.common.modeling.model import Model, Response
from mafc.common.modeling.prompt import Prompt
from mafc.tools.web_search.common import Query, SearchResults, WebSource
from mafc.tools.web_search.integrations.integration import RetrievalIntegration

ASSETS_DIR = Path(__file__).resolve().parents[2] / "assets"


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
    name = "dummy_action"

    def __init__(self):
        self._save_parameters(locals())


class FakeWorkerAgent(Agent):
    name = "FakeWorkerAgent"
    allowed_tools = []

    def __init__(self, result_text: str, source: str):
        self.result_text = result_text
        self.source = source
        self.calls: list[AgentSession] = []
        self.description = f"Fake worker for {source}"

    def run(self, session: AgentSession, trace_scope=None) -> AgentResult:
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

    def search(self, query: Query) -> SearchResults:
        return self.result_map.get(query.text or "") or SearchResults(sources=[], query=query)


class FakeRetriever(RetrievalIntegration):
    domains = ["*"]

    def __init__(self, payload_by_url: dict[str, str | None]):
        super().__init__()
        self.payload_by_url = payload_by_url

    def _retrieve(self, url: str) -> Prompt | None:
        payload = self.payload_by_url.get(url)
        return None if payload is None else Prompt(text=payload)


def make_search_result(query_text: str, urls: list[str]) -> SearchResults:
    query = Query(text=query_text)
    sources = [WebSource(reference=url, title=f"T:{url}") for url in urls]
    return SearchResults(sources=sources, query=query)


def registered_image() -> Image:
    image = Image(file_path=ASSETS_DIR / "Greece.jpeg")
    item_registry.add_item(image)
    return cast(Image, image)


def make_registry(tmp_path: Path, include_media: bool = True) -> BlueprintRegistry:
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


def make_selector(registry: BlueprintRegistry) -> BlueprintSelector:
    return BlueprintSelector(
        model=SequencedModel(outputs=[]),
        registry=registry,
        default_blueprint_name="default",
    )
