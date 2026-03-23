from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import cast

from ezmm import MultimodalSequence
from ezmm.common.items import Image, Video

from mafc.common.modeling.message import Message, MessageRole
from mafc.agents.agent import Agent, AgentResult
from mafc.agents.common import AgentSession
from mafc.tools.tool import Tool
from mafc.tools.web_search.google_search import GoogleSearchPlatform
from mafc.tools.web_search.integrations.integration import RetrievalIntegration
from mafc.tools.web_search.integrations.scrapemm_retriever import ScrapeMMRetriever
from mafc.common.modeling.model import Model, Response
from mafc.utils.parsing import extract_json_object, is_failed_model_text
from mafc.utils.media import build_media_json_instruction, deduplicate_media, parse_media_relevance

from mafc.agents.web_search.models import IterationOutcome, SearchPlanStep, StepQueryPlan, SearchTool
from mafc.agents.web_search.planner import plan_step
from mafc.agents.web_search.retrieval import (
    execute_search_queries,
    retrieve_query_results,
    select_sources_for_retrieval,
)
from mafc.agents.web_search.tracing import WebSearchTraceRecorder

_MAX_MEDIA_PER_REQUEST = 50


def _parse_synthesis_response(
    response_text: str, media_items: list[Image | Video]
) -> tuple[str, list[Image | Video]]:
    """Parse the JSON synthesis response and return (synthesis_text, relevant_media)."""
    text = extract_json_object(response_text.strip())
    try:
        payload = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return response_text.strip(), media_items

    synthesis = payload.get("synthesis", "").strip()
    relevant_media = parse_media_relevance(payload.get("media", []), media_items)
    return synthesis, relevant_media


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
        trace_dir: str | Path | None = None,
    ):
        super().__init__(main_model, n_workers=n_workers)
        self.search_tool = search_tool or GoogleSearchPlatform()
        self.retriever = retriever or ScrapeMMRetriever(n_workers=n_workers)
        self.summarization_model = summarization_model or main_model
        self.max_iterations = max_iterations
        self.max_queries_per_step = max_queries_per_step
        self.max_results_per_query = max_results_per_query
        self.latest_allowed_date = latest_allowed_date
        self.trace_dir = trace_dir

    def run(self, session: AgentSession, trace_scope=None) -> AgentResult:
        scope = self._build_trace_scope("web_search_run", session, trace_scope)
        trace = WebSearchTraceRecorder(self.trace_dir, session, self.name, trace_scope=scope)
        instruction, prior_context, errors, seen_queries, early_result = self._initialize_run(session)
        if early_result:
            early_result.trace = trace.trace
            trace.finalize(
                session=session, result=early_result, errors=early_result.errors, seen_queries=seen_queries
            )
            return early_result

        result: AgentResult | None = None
        try:
            for step in range(1, self.max_iterations + 1):
                prior_context = self.build_prior_context(session)
                iteration_outcome = self._execute_iteration(
                    session=session,
                    step=step,
                    instruction=instruction,
                    prior_context=prior_context,
                    seen_queries=seen_queries,
                    errors=errors,
                    trace=trace,
                )
                if iteration_outcome.should_stop:
                    break

            result = self._finalize_run(
                session=session,
                instruction=instruction,
                errors=errors,
                trace=trace,
            )
            return result
        except Exception as exc:
            self._mark_failed(session)
            message = f"{type(exc).__name__}: {exc}"
            errors.append(message)
            trace.record_error(step=None, phase="run", message=message)
            raise
        finally:
            trace.finalize(
                session=session,
                result=result,
                errors=errors,
                seen_queries=seen_queries,
            )

    def synthesize_from_evidences(self, instruction: str, evidences) -> str:
        """Synthesize an answer directly from already accepted evidence."""
        result, _resp = self._synthesize_final(instruction, evidences)
        return str(result).strip() if result is not None else ""

    def _synthesize_final(
        self, instruction: str, evidences
    ) -> tuple[MultimodalSequence | None, "Response | None"]:
        """Synthesize an answer and return (multimodal result, response)."""
        evidence_blocks = []
        all_media: list[Image | Video] = []
        for evidence in evidences:
            if evidence.takeaways is not None:
                summary = str(evidence.takeaways).strip()
                all_media.extend(evidence.takeaways.images)
                all_media.extend(evidence.takeaways.videos)
            else:
                summary = str(evidence.raw).strip()
            if not summary:
                continue
            evidence_blocks.append(f"Source: {evidence.source}\nSummary: {summary}")

        if not evidence_blocks:
            return None, None

        media_instruction = build_media_json_instruction(all_media, context="The evidence")

        synthesis_prompt = (
            "You are a factual evidence synthesizer.\n"
            "Answer the task using only the accepted evidence provided below.\n"
            "State agreements and disagreements between sources when present.\n"
            "Include concrete facts and source references where possible.\n"
            "Call out important uncertainties or missing evidence.\n"
            "Return strict JSON only with schema:\n"
            '  {"synthesis": "..."}\n'
            f"{media_instruction}\n"
            f"Task:\n{instruction}\n\n"
            f"Accepted evidence:\n{chr(10).join(evidence_blocks)}"
        )
        if all_media:
            deduped = deduplicate_media(MultimodalSequence(*all_media))
            all_media = (deduped.images + deduped.videos)[:_MAX_MEDIA_PER_REQUEST]
        content = MultimodalSequence(synthesis_prompt, *all_media)
        try:
            resp = self.summarization_model.generate([Message(role=MessageRole.USER, content=content)])
            synthesis, relevant_media = _parse_synthesis_response(resp.text, all_media)
            if synthesis and is_failed_model_text(synthesis):
                return None, resp
            if not synthesis:
                return MultimodalSequence("\n\n".join(evidence_blocks), *all_media), resp
            return MultimodalSequence(synthesis, *relevant_media), resp
        except Exception:
            return MultimodalSequence("\n\n".join(evidence_blocks), *all_media), None

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
        trace: WebSearchTraceRecorder,
    ) -> IterationOutcome:
        """Execute one planning and retrieval iteration."""
        if self._should_stop:
            message = "Agent execution stopped early by stop signal."
            errors.append(message)
            trace.record_error(step=step, phase="stop_signal", message=message)
            return IterationOutcome(should_stop=True)

        errors_before = len(errors)
        trace.start_iteration(step=step, evidence_count=len(session.evidences), seen_queries=seen_queries)
        step_plan = self._resolve_step_query_plan(
            step=step,
            instruction=instruction,
            prior_context=prior_context,
            seen_queries=seen_queries,
            errors=errors,
            trace=trace,
        )
        if step_plan.should_terminate:
            trace.finish_iteration(
                step=step,
                evidence_count=len(session.evidences),
                seen_queries=seen_queries,
                new_errors=errors[errors_before:],
            )
            return IterationOutcome(should_stop=True)

        candidate_sources = execute_search_queries(
            queries=step_plan.queries,
            seen_queries=seen_queries,
            errors=errors,
            search_tool=self.search_tool,
            n_workers=self.n_workers,
            max_results_per_query=self.max_results_per_query,
            latest_allowed_date=self.latest_allowed_date,
            step=step,
            trace=trace,
        )
        selected_sources = select_sources_for_retrieval(
            candidate_sources,
            model=self.model,
            max_results_per_query=self.max_results_per_query,
            step=step,
            trace=trace,
        )
        seen_urls = {ev.source for ev in session.evidences}
        query_results = retrieve_query_results(
            selected_sources,
            errors=errors,
            model=self.summarization_model,
            retriever=self.retriever,
            seen_urls=seen_urls,
            step=step,
            trace=trace,
            n_workers=self.n_workers,
        )
        if not candidate_sources or not query_results:
            message = f"No observations produced in iteration {step}."
            errors.append(message)
            trace.record_error(step=step, phase="iteration", message=message)
            trace.finish_iteration(
                step=step,
                evidence_count=len(session.evidences),
                seen_queries=seen_queries,
                new_errors=errors[errors_before:],
            )
            return IterationOutcome(should_stop=True)

        new_evidences = [evidence for query_result in query_results for evidence in query_result.evidences]
        if new_evidences:
            session.evidences.extend(new_evidences)
        trace.finish_iteration(
            step=step,
            evidence_count=len(session.evidences),
            seen_queries=seen_queries,
            new_errors=errors[errors_before:],
        )
        return IterationOutcome(should_stop=step_plan.done)

    def _finalize_run(
        self,
        session: AgentSession,
        instruction: str,
        errors: list[str],
        trace: WebSearchTraceRecorder | None = None,
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
                trace=trace.trace if trace is not None else None,
            )

        result_text, synthesis_resp = self._synthesize_final(instruction, session.evidences)
        if trace is not None and synthesis_resp is not None:
            trace.add_usage(synthesis_resp, self.summarization_model.name)
        if trace is not None:
            trace.record_synthesis(
                step=None,
                stage="finalize",
                instruction=instruction,
                answer=str(result_text).strip() if result_text is not None else "",
                evidence_count=len(session.evidences),
            )
        if result_text is None or not str(result_text).strip():
            evidence_lines = []
            for ev in session.evidences:
                summary = str(ev.takeaways).strip() if ev.takeaways is not None else str(ev.raw).strip()
                if summary:
                    evidence_lines.append(f"Source: {ev.source}\n{summary}")
            if not evidence_lines:
                self._mark_failed(session)
                return AgentResult(
                    session=session,
                    result=None,
                    evidences=list(session.evidences),
                    errors=errors,
                    status=session.status,
                    trace=trace.trace if trace is not None else None,
                )
            result_text = MultimodalSequence("\n\n".join(evidence_lines))

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
            trace=trace.trace if trace is not None else None,
        )

    def _resolve_step_query_plan(
        self,
        step: int,
        instruction: str,
        prior_context: str,
        seen_queries: set[str],
        errors: list[str],
        trace: WebSearchTraceRecorder | None = None,
    ) -> StepQueryPlan:
        fallback_used = False
        plan = plan_step(self, instruction, prior_context, errors, step=step, trace=trace)
        if plan is None:
            if step == 1:
                errors.append(
                    "Planner output could not be parsed in iteration 1. "
                    "Falling back to the original task text as search query."
                )
                plan = SearchPlanStep(queries=[instruction], done=True)
                fallback_used = True
            else:
                errors.append(f"Planner output could not be parsed in iteration {step}.")
                if trace is not None:
                    trace.record_error(
                        step=step,
                        phase="planner_parse",
                        message=f"Planner output could not be parsed in iteration {step}.",
                    )
                    trace.record_resolved_plan(
                        step=step,
                        queries=[],
                        done=False,
                        should_terminate=True,
                        fallback_used=False,
                    )
                return StepQueryPlan(queries=[], done=False, should_terminate=True)

        if plan.done and not plan.queries:
            if trace is not None:
                trace.record_resolved_plan(
                    step=step,
                    queries=[],
                    done=True,
                    should_terminate=True,
                    fallback_used=fallback_used,
                )
            return StepQueryPlan(queries=[], done=True, should_terminate=True)

        queries = [q.strip() for q in plan.queries if q and q.strip()]
        queries = [q for q in queries if q not in seen_queries]
        queries = queries[: self.max_queries_per_step]
        if not queries:
            errors.append(f"Planner returned no queries in iteration {step}.")
            if trace is not None:
                trace.record_error(
                    step=step,
                    phase="resolved_plan",
                    message=f"Planner returned no queries in iteration {step}.",
                )
                trace.record_resolved_plan(
                    step=step,
                    queries=[],
                    done=plan.done,
                    should_terminate=True,
                    fallback_used=fallback_used,
                )
            return StepQueryPlan(queries=[], done=plan.done, should_terminate=True)

        if trace is not None:
            trace.record_resolved_plan(
                step=step,
                queries=queries,
                done=plan.done,
                should_terminate=False,
                fallback_used=fallback_used,
            )
        return StepQueryPlan(queries=queries, done=plan.done, should_terminate=False)
