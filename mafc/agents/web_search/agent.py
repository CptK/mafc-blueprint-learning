from __future__ import annotations

from datetime import date
from typing import Protocol, cast

from ezmm import MultimodalSequence

from mafc.agents.agent import Agent, AgentResult
from mafc.agents.common import AgentMessage, AgentMessageType, AgentSession
from mafc.tools.tool import Tool
from mafc.tools.web_search.common import Query, SearchResults
from mafc.tools.web_search.google_search import GoogleSearchPlatform
from mafc.tools.web_search.integrations.integration import RetrievalIntegration
from mafc.tools.web_search.integrations.scrapemm_retriever import ScrapeMMRetriever
from mafc.common.modeling.model import Model

from .models import SearchPlanStep, StepQueryPlan
from .planner import plan_step
from .retrieval import collect_observations_for_queries
from .synthesis import synthesize_step


class SearchTool(Protocol):
    """Minimal search interface required by the web-search agent."""

    def search(self, query: Query) -> SearchResults | None:
        pass


class WebSearchAgent(Agent):
    name = "WebSearchAgent"
    description = "An iterative agent that plans search queries with an LLM and synthesizes findings."

    allowed_tools = cast(list[type[Tool]], [GoogleSearchPlatform])

    def __init__(
        self,
        main_model: Model,
        n_workers: int = 1,
        summarization_model: Model | None = None,
        search_tool: SearchTool | None = None,
        retriever: RetrievalIntegration | None = None,
        max_iterations: int = 4,
        max_queries_per_step: int = 5,
        max_results_per_query: int = 5,
        latest_allowed_date: date | None = None,
    ):
        super().__init__(main_model, n_workers=n_workers)
        self.search_tool = search_tool or GoogleSearchPlatform()
        self.retriever = retriever or ScrapeMMRetriever(n_workers=n_workers)
        self.summarization_model = summarization_model or main_model
        self.max_iterations = max_iterations
        self.max_queries_per_step = max_queries_per_step
        self.max_results_per_query = max_results_per_query
        self.latest_allowed_date = latest_allowed_date

    def run(self, session: AgentSession) -> AgentResult:
        self._mark_running(session)
        instruction = str(session.goal).strip()
        if self._should_stop:
            self._mark_failed(session)
            return AgentResult(
                session=session,
                result=None,
                errors=["Agent was stopped before execution started."],
                status=session.status,
            )
        if not instruction:
            self._mark_failed(session)
            return AgentResult(
                session=session,
                result=None,
                errors=["Task prompt is empty."],
                status=session.status,
            )

        errors: list[str] = []
        memory: list[str] = []
        seen_queries: set[str] = set()

        for step in range(1, self.max_iterations + 1):
            if self._should_stop:
                errors.append("Agent execution stopped early by stop signal.")
                break

            step_plan = self._resolve_step_query_plan(
                step=step,
                instruction=instruction,
                memory=memory,
                seen_queries=seen_queries,
                errors=errors,
            )
            if step_plan.should_terminate:
                break

            observations = collect_observations_for_queries(
                agent=self,
                queries=step_plan.queries,
                seen_queries=seen_queries,
                errors=errors,
            )
            if not observations:
                errors.append(f"No observations produced in iteration {step}.")
                break

            synthesis = synthesize_step(self, instruction, observations)
            memory.append(f"Iteration {step} synthesis:\n{synthesis}")

            if step_plan.done:
                break

        if not memory:
            self._mark_failed(session)
            return AgentResult(session=session, result=None, errors=errors, status=session.status)

        final_text = "\n\n".join(memory)
        result_text = MultimodalSequence(final_text)
        result_message = AgentMessage(
            id=f"{session.id}:result",
            session_id=session.id,
            sender=self.name,
            receiver=session.parent_session_id or session.id,
            message_type=AgentMessageType.RESULT,
            content=result_text,
        )
        session.messages.append(result_message)
        self._mark_completed(session)
        return AgentResult(
            session=session,
            result=result_text,
            messages=[result_message],
            evidences=[],
            errors=errors,
            status=session.status,
        )

    def _resolve_step_query_plan(
        self,
        step: int,
        instruction: str,
        memory: list[str],
        seen_queries: set[str],
        errors: list[str],
    ) -> StepQueryPlan:
        plan = plan_step(self, instruction, memory, errors)
        if plan is None:
            if step == 1:
                errors.append(
                    "Planner output could not be parsed in iteration 1. "
                    "Falling back to the original task text as search query."
                )
                plan = SearchPlanStep(queries=[instruction], done=True)
            else:
                errors.append(f"Planner output could not be parsed in iteration {step}.")
                return StepQueryPlan(queries=[], done=False, should_terminate=True)

        if plan.done and not plan.queries:
            return StepQueryPlan(queries=[], done=True, should_terminate=True)

        queries = [q.strip() for q in plan.queries if q and q.strip()]
        queries = [q for q in queries if q not in seen_queries]
        queries = queries[: self.max_queries_per_step]
        if not queries:
            errors.append(f"Planner returned no queries in iteration {step}.")
            return StepQueryPlan(queries=[], done=plan.done, should_terminate=True)

        return StepQueryPlan(queries=queries, done=plan.done, should_terminate=False)
