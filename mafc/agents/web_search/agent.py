from __future__ import annotations

from datetime import date
from typing import cast

from ezmm import MultimodalSequence

from mafc.common.modeling.prompt import Prompt
from mafc.common.modeling.message import Message, MessageRole
from mafc.agents.agent import Agent, AgentResult
from mafc.agents.common import AgentSession
from mafc.tools.tool import Tool
from mafc.tools.web_search.google_search import GoogleSearchPlatform
from mafc.tools.web_search.integrations.integration import RetrievalIntegration
from mafc.tools.web_search.integrations.scrapemm_retriever import ScrapeMMRetriever
from mafc.common.modeling.model import Model
from mafc.utils.parsing import is_failed_model_text

from mafc.agents.web_search.models import IterationOutcome, SearchPlanStep, StepQueryPlan, SearchTool
from mafc.agents.web_search.planner import plan_step
from mafc.agents.web_search.retrieval import (
    execute_search_queries,
    retrieve_query_results,
    select_sources_for_retrieval,
)


class WebSearchAgent(Agent):
    name = "WebSearchAgent"
    description = (
        "Searches the web for external sources and synthesizes findings. "
        "Use it to discover articles, official statements, social posts, webpages, and other online evidence, "
        "including finding where claimed media appears online."
    )

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
        instruction, prior_context, errors, seen_queries, early_result = self._initialize_run(session)
        if early_result:
            return early_result

        for step in range(1, self.max_iterations + 1):
            iteration_outcome = self._execute_iteration(
                session=session,
                step=step,
                instruction=instruction,
                prior_context=prior_context,
                seen_queries=seen_queries,
                errors=errors,
            )
            if iteration_outcome.should_stop:
                break

        return self._finalize_run(
            session=session,
            instruction=instruction,
            errors=errors,
        )

    def synthesize_from_evidences(self, instruction: str, evidences):
        """Synthesize an answer directly from already accepted evidence."""
        evidence_blocks = []
        for evidence in evidences:
            summary = (
                str(evidence.takeaways).strip()
                if evidence.takeaways is not None
                else str(evidence.raw).strip()
            )
            if not summary:
                continue
            evidence_blocks.append(f"Source: {evidence.source}\nSummary: {summary}")

        if not evidence_blocks:
            return ""

        synthesis_prompt = (
            "You are a factual evidence synthesizer.\n"
            "Answer the task using only the accepted evidence provided below.\n"
            "State agreements and disagreements between sources when present.\n"
            "Include concrete facts and source references where possible.\n"
            "Call out important uncertainties or missing evidence.\n\n"
            f"Task:\n{instruction}\n\n"
            f"Accepted evidence:\n{chr(10).join(evidence_blocks)}"
        )
        try:
            synthesis = self.summarization_model.generate(
                [Message(role=MessageRole.USER, content=Prompt(text=synthesis_prompt))]
            ).text.strip()
            if not synthesis or is_failed_model_text(synthesis):
                return "\n\n".join(evidence_blocks)
            return synthesis
        except Exception:
            return "\n\n".join(evidence_blocks)

    def _initialize_run(
        self,
        session: AgentSession,
    ) -> tuple[str, str, list[str], set[str], AgentResult | None]:
        """Prepare one run and return early result on invalid preconditions."""
        self._mark_running(session)
        instruction = str(session.goal).strip()
        prior_context = self.build_prior_context(session)
        if self._should_stop:
            self._mark_failed(session)
            return (
                instruction,
                prior_context,
                [],
                set(),
                AgentResult(
                    session=session,
                    result=None,
                    errors=["Agent was stopped before execution started."],
                    status=session.status,
                ),
            )
        if not instruction:
            self._mark_failed(session)
            return (
                instruction,
                prior_context,
                [],
                set(),
                AgentResult(
                    session=session,
                    result=None,
                    errors=["Task prompt is empty."],
                    status=session.status,
                ),
            )
        return instruction, prior_context, [], set(), None

    def _execute_iteration(
        self,
        session: AgentSession,
        step: int,
        instruction: str,
        prior_context: str,
        seen_queries: set[str],
        errors: list[str],
    ) -> IterationOutcome:
        """Execute one planning and retrieval iteration."""
        if self._should_stop:
            errors.append("Agent execution stopped early by stop signal.")
            return IterationOutcome(should_stop=True)

        step_plan = self._resolve_step_query_plan(
            step=step,
            instruction=instruction,
            prior_context=prior_context,
            seen_queries=seen_queries,
            errors=errors,
        )
        if step_plan.should_terminate:
            return IterationOutcome(should_stop=True)

        candidate_sources = execute_search_queries(
            queries=step_plan.queries,
            seen_queries=seen_queries,
            errors=errors,
            search_tool=self.search_tool,
            n_workers=self.n_workers,
            max_results_per_query=self.max_results_per_query,
            latest_allowed_date=self.latest_allowed_date,
        )
        selected_sources = select_sources_for_retrieval(
            candidate_sources,
            model=self.model,
            max_results_per_query=self.max_results_per_query,
        )
        query_results = retrieve_query_results(
            selected_sources,
            errors=errors,
            model=self.summarization_model,
            retriever=self.retriever,
        )
        if not candidate_sources or not query_results:
            errors.append(f"No observations produced in iteration {step}.")
            return IterationOutcome(should_stop=True)

        new_evidences = [evidence for query_result in query_results for evidence in query_result.evidences]
        if new_evidences:
            session.evidences.extend(new_evidences)
        return IterationOutcome(should_stop=step_plan.done)

    def _finalize_run(
        self,
        session: AgentSession,
        instruction: str,
        errors: list[str],
    ) -> AgentResult:
        """Produce the final run result from the accumulated session evidence."""
        if not session.evidences:
            self._mark_failed(session)
            return AgentResult(
                session=session,
                result=None,
                evidences=list(session.evidences),
                errors=errors,
                status=session.status,
            )

        synthesis = self.synthesize_from_evidences(instruction, session.evidences)
        if not synthesis.strip():
            self._mark_failed(session)
            return AgentResult(
                session=session,
                result=None,
                evidences=list(session.evidences),
                errors=errors,
                status=session.status,
            )

        result_text = MultimodalSequence(synthesis)
        result_message = self.make_result_message(session, result_text, list(session.evidences))
        session.messages.append(result_message)
        self._mark_completed(session)
        return AgentResult(
            session=session,
            result=result_text,
            messages=[result_message],
            evidences=list(session.evidences),
            errors=errors,
            status=session.status,
        )

    def _resolve_step_query_plan(
        self,
        step: int,
        instruction: str,
        prior_context: str,
        seen_queries: set[str],
        errors: list[str],
    ) -> StepQueryPlan:
        plan = plan_step(self, instruction, prior_context, errors)
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
