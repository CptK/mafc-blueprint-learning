from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from typing import cast

from ezmm import MultimodalSequence

from mafc.agents.agent import Agent, AgentResult
from mafc.agents.common import AgentMessage, AgentMessageType, AgentSession
from mafc.common.logger import logger
from mafc.common.modeling.model import Model
from mafc.common.modeling.prompt import Prompt
from mafc.tools.tool import Tool
from mafc.tools.web_search.common import Query, Source, WebSource
from mafc.tools.web_search.google_search import GoogleSearchPlatform
from mafc.tools.web_search.integrations.integration import RetrievalIntegration
from mafc.tools.web_search.integrations.scrapemm_retriever import ScrapeMMRetriever


@dataclass
class SearchPlanStep:
    queries: list[str]
    done: bool = False


@dataclass
class StepQueryPlan:
    """Resolved query plan used by the run loop for one iteration."""

    queries: list[str]
    done: bool
    should_terminate: bool = False


@dataclass
class QuerySearchResult:
    """Result container for one query search execution in the worker pool."""

    query_text: str
    sources: Sequence[Source] | None
    errors: list[str]
    mark_seen: bool = False
    stopped: bool = False


@dataclass
class GlobalSourceCandidate:
    """One URL candidate with query context for global relevance filtering."""

    query_text: str
    source: WebSource


class WebSearchAgent(Agent):
    name = "WebSearchAgent"
    description = "An iterative agent that plans search queries with an LLM and synthesizes findings."

    allowed_tools = cast(list[type[Tool]], [GoogleSearchPlatform])

    def __init__(
        self,
        main_model: Model,
        n_workers: int = 1,
        summarization_model: Model | None = None,
        search_tool: GoogleSearchPlatform | None = None,
        retriever: RetrievalIntegration | None = None,
        max_iterations: int = 4,
        max_queries_per_step: int = 5,
        max_results_per_query: int = 5,
        latest_allowed_date: date | None = None,
    ):
        """Initialize the web-search agent and its configurable dependencies.

        Args:
            main_model: Model used for planning and optional fallback summarization.
            summarization_model: Optional dedicated model for step summaries.
            search_tool: Optional search platform implementation.
            retriever: Optional URL retrieval integration.
            max_iterations: Maximum planning/retrieval cycles.
            max_queries_per_step: Maximum planner-proposed queries per cycle.
            max_results_per_query: Maximum sources retrieved per query.
            latest_allowed_date: Optional cutoff date used in querying/filtering.
        """
        super().__init__(main_model, n_workers=n_workers)
        self.search_tool = search_tool or GoogleSearchPlatform()
        self.retriever = retriever or ScrapeMMRetriever(n_workers=n_workers)
        self.summarization_model = summarization_model or main_model
        self.max_iterations = max_iterations
        self.max_queries_per_step = max_queries_per_step
        self.max_results_per_query = max_results_per_query
        self.latest_allowed_date = latest_allowed_date

    def run(self, session: AgentSession) -> AgentResult:
        """Run iterative planning, searching, retrieval, and synthesis for a task.

        Args:
            session: Investigation session containing the goal to investigate.

        Returns:
            AgentResult with aggregated syntheses and collected non-fatal errors.
        """
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

            observations = self._collect_observations_for_queries(
                queries=step_plan.queries,
                seen_queries=seen_queries,
                errors=errors,
            )
            if not observations:
                errors.append(f"No observations produced in iteration {step}.")
                break

            synthesis = self._synthesize_step(instruction, observations)
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
        """Resolve, sanitize, and validate planner queries for one iteration.

        Args:
            step: 1-based run-loop iteration number.
            instruction: Original task instruction text.
            memory: Prior iteration syntheses.
            seen_queries: Set of previously executed query strings.
            errors: Mutable list used to append planning errors.

        Returns:
            StepQueryPlan containing normalized queries and loop-control flags.
        """
        plan = self._plan_step(instruction, memory, errors)
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

    def _collect_observations_for_queries(
        self,
        queries: list[str],
        seen_queries: set[str],
        errors: list[str],
    ) -> list[str]:
        """Search all queries first, then retrieve and format observations.

        Args:
            queries: Sanitized query texts for the current iteration.
            seen_queries: Set that tracks successfully executed query texts.
            errors: Mutable list used to append query/search/retrieval errors.

        Returns:
            List of formatted observation strings, one per handled query.
        """
        if self._should_stop:
            errors.append("Agent execution stopped early by stop signal.")
            return []

        if self.n_workers <= 1:
            results = [self._execute_search_query(query_text) for query_text in queries]
        else:
            max_workers = min(self.n_workers, len(queries))
            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                results = list(pool.map(self._execute_search_query, queries))

        observations: list[str] = []
        candidate_sources: list[tuple[str, Sequence[Source] | None]] = []
        stop_recorded = False
        for result in results:
            if result.errors:
                errors.extend(result.errors)
            if result.stopped and not stop_recorded:
                errors.append("Agent execution stopped early by stop signal.")
                stop_recorded = True
            if result.mark_seen:
                seen_queries.add(result.query_text)
                candidate_sources.append((result.query_text, result.sources))

        selected_sources = self._select_sources_for_retrieval(candidate_sources)
        for query_text, sources in selected_sources:
            if sources is None:
                observations.append(f"Query: {query_text}\nNo results.")
                continue
            observations.append(
                self._retrieve_and_format_observation(
                    query_text=query_text,
                    sources=sources,
                    errors=errors,
                )
            )
        return observations

    def _execute_search_query(self, query_text: str) -> QuerySearchResult:
        """Execute search for one query and return source candidates.

        Args:
            query_text: Query text to execute.

        Returns:
            QuerySearchResult including source candidates, errors, and status flags.
        """
        if self._should_stop:
            return QuerySearchResult(
                query_text=query_text,
                sources=None,
                errors=[],
                mark_seen=False,
                stopped=True,
            )

        query = Query(
            text=query_text,
            limit=self.max_results_per_query,
            end_date=self.latest_allowed_date,
        )
        try:
            result = self.search_tool.search(query)
        except Exception as exc:
            return QuerySearchResult(
                query_text=query_text,
                sources=None,
                errors=[f"Search failed for query '{query_text}': {exc}"],
            )

        return QuerySearchResult(
            query_text=query_text,
            sources=result.sources if result is not None else None,
            errors=[],
            mark_seen=True,
        )

    def _select_sources_for_retrieval(
        self,
        candidates: list[tuple[str, Sequence[Source] | None]],
    ) -> list[tuple[str, Sequence[Source] | None]]:
        """Select relevant sources for retrieval from per-query candidates.

        Args:
            candidates: Per-query source candidates from the search phase.

        Returns:
            Per-query selected sources to retrieve. `None` means no sources for that query.
        """
        per_query_sources: dict[str, list[WebSource] | None] = {}
        global_candidates: list[GlobalSourceCandidate] = []
        for query_text, sources in candidates:
            if sources is None:
                per_query_sources[query_text] = None
                continue
            web_sources = [source for source in sources if isinstance(source, WebSource)]
            per_query_sources[query_text] = web_sources
            for source in web_sources:
                global_candidates.append(GlobalSourceCandidate(query_text=query_text, source=source))

        selected_urls: list[str] = []
        logger.info(
            f"All candidate URLs before filtering:\n{'\n    '.join([c.source.url for c in global_candidates])}"
        )
        if len(global_candidates) > 5:
            max_selected_total = self.max_results_per_query * max(1, len(per_query_sources))
            selected_urls = self._filter_sources_with_model(
                candidates=global_candidates,
                max_selected=max_selected_total,
            )

        selected_by_query: dict[str, list[WebSource]] = {query_text: [] for query_text, _ in candidates}
        if selected_urls:
            url_to_candidates: dict[str, list[GlobalSourceCandidate]] = {}
            for candidate in global_candidates:
                url_to_candidates.setdefault(candidate.source.url, []).append(candidate)
            for url in selected_urls:
                if url not in url_to_candidates or not url_to_candidates[url]:
                    continue
                candidate = url_to_candidates[url].pop(0)
                selected_by_query[candidate.query_text].append(candidate.source)
        else:
            for query_text, sources in per_query_sources.items():
                if sources is None:
                    continue
                selected_by_query[query_text].extend(sources)

        selected: list[tuple[str, Sequence[Source] | None]] = []
        for query_text, _ in candidates:
            sources = per_query_sources.get(query_text)
            if sources is None:
                selected.append((query_text, None))
                continue
            picked = selected_by_query.get(query_text, [])[: self.max_results_per_query]
            selected.append((query_text, picked))
        return selected

    def _filter_sources_with_model(
        self,
        candidates: list[GlobalSourceCandidate],
        max_selected: int,
    ) -> list[str]:
        """Select the most relevant source URLs for a query via model reasoning.

        Args:
            candidates: Global URL candidates with source metadata and query context.
            max_selected: Maximum number of URLs to keep.

        Returns:
            Ordered list of selected URLs. Returns an empty list if parsing fails.
        """
        candidates_payload = []
        for idx, candidate in enumerate(candidates, start=1):
            source = candidate.source
            candidates_payload.append(
                {
                    "id": idx,
                    "query": candidate.query_text,
                    "url": source.url,
                    "title": source.title or "",
                    "snippet": source.preview or "",
                    "release_date": source.release_date.isoformat() if source.release_date else None,
                }
            )

        prompt_payload = {
            "task": "Select globally relevant URLs across all query results.",
            "max_selected": max_selected,
            "candidates": candidates_payload,
        }
        selection_prompt = (
            "You are a source relevance filter for fact-checking.\n"
            "Select the most relevant URLs across all query results from the given candidates.\n"
            "Favor specific, evidence-rich, and directly on-topic sources.\n"
            "Return strict JSON only with schema:\n"
            '{"selected_urls": ["https://..."]}\n\n'
            f"Input JSON:\n{json.dumps(prompt_payload, ensure_ascii=True)}"
        )
        try:
            response_text = self.model.generate(Prompt(text=selection_prompt)).text
        except Exception as exc:
            logger.error(f"[{self.name}] Global source filtering call failed: {exc}")
            return []
        return self._parse_selected_urls(response_text=response_text, max_selected=max_selected)

    def _parse_selected_urls(self, response_text: str, max_selected: int) -> list[str]:
        """Parse selected URL list from model output.

        Args:
            response_text: Raw model output expected to contain strict JSON.
            max_selected: Maximum number of URLs allowed in the final selection.

        Returns:
            Parsed selected URL list, possibly empty on parse/validation failure.
        """
        text = response_text.strip()
        if text.startswith("```"):
            lines = [line for line in text.splitlines() if not line.startswith("```")]
            text = "\n".join(lines).strip()
        text = self._extract_json_object(text)
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            logger.error(f"[{self.name}] Failed to parse source filter output as JSON: {exc}")
            return []

        selected_urls = payload.get("selected_urls")
        if not isinstance(selected_urls, list) or not all(isinstance(url, str) for url in selected_urls):
            logger.error(f"[{self.name}] Invalid source filter output: selected_urls must be list[str].")
            return []
        selected_urls = [url.strip() for url in selected_urls if url.strip()]
        logger.info(f"Model selected URLs:\n{"\n  ".join(selected_urls)}")
        return selected_urls[:max_selected]

    def _plan_step(self, instruction: str, memory: list[str], errors: list[str]) -> SearchPlanStep | None:
        """Generate and parse the next search plan step from model output.

        Args:
            instruction: Original task instruction text.
            memory: Prior iteration syntheses.
            errors: Mutable list used to append planner errors.

        Returns:
            Parsed SearchPlanStep when successful, otherwise None.
        """
        planner_prompt = (
            "You are a web-search planner.\n"
            "Given the task and previous syntheses, propose next search queries.\n"
            "Return strict JSON with keys:\n"
            '- "queries": array of strings\n'
            '- "done": boolean\n\n'
            "Guidelines:\n"
            "- Keep queries specific and evidence-seeking.\n"
            "- If enough evidence is already gathered, set done=true and queries=[].\n"
            f"- If you want to gather more information, set done=false and propose up to {self.max_queries_per_step} new queries.\n"
            "- You have multiple iterations to gather information, so you can search for facts building on previous findings.\n\n"
            f"Task:\n{instruction}\n\n"
            f"Previous syntheses:\n{chr(10).join(memory) if memory else 'None'}\n"
        )
        try:
            response = self.model.generate(Prompt(text=planner_prompt)).text
            logger.info(f"Planner response:\n{response}")
            if self._is_failed_model_text(response):
                return None
            parsed = self._parse_plan(response)
            if parsed is not None:
                return parsed

            # One repair attempt for non-strict outputs.
            repair_prompt = (
                "Convert the following planner response to strict JSON with schema:\n"
                '{"queries": ["..."], "done": false}\n'
                "Only return JSON.\n\n"
                f"Response:\n{response}"
            )
            repaired = self.model.generate(Prompt(text=repair_prompt)).text
            if self._is_failed_model_text(repaired):
                return None
            return self._parse_plan(repaired)
        except Exception as exc:
            logger.error(f"Planner call failed: {exc}")
            errors.append(f"Planner call failed: {exc}")
            return None

    def _synthesize_step(self, instruction: str, observations: list[str]) -> str:
        """Synthesize current observations across sources for the current step.

        Args:
            instruction: Original task instruction text.
            observations: Collected per-query observations for the current iteration.

        Returns:
            A synthesis string produced by the model or fallback text.
        """
        synthesis_prompt = (
            "You are a factual evidence synthesizer.\n"
            "Synthesize the observations across sources, focusing only on information relevant to the task.\n"
            "State agreements and disagreements between sources when present.\n"
            "Include concrete facts and source references where possible.\n"
            "Call out important uncertainties or missing evidence.\n\n"
            f"Task:\n{instruction}\n\n"
            f"Observations:\n{chr(10).join(observations)}"
        )
        try:
            synthesis = self.summarization_model.generate(Prompt(text=synthesis_prompt)).text.strip()
            if not synthesis or self._is_failed_model_text(synthesis):
                return "\n\n".join(observations)
            return synthesis
        except Exception:
            # Fallback keeps the agent useful even if synthesis model call fails.
            return "\n\n".join(observations)

    def _parse_plan(self, response_text: str) -> SearchPlanStep | None:
        """Parse planner output into a validated `SearchPlanStep` object.

        Args:
            response_text: Raw planner model output.

        Returns:
            SearchPlanStep if parsing and schema validation succeed, else None.
        """
        text = response_text.strip()
        if text.startswith("```"):
            lines = [line for line in text.splitlines() if not line.startswith("```")]
            text = "\n".join(lines).strip()
        text = self._extract_json_object(text)
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            logger.error(f"[{self.name}] Failed to parse planner response as JSON: {exc}")
            return None
        queries = payload.get("queries")
        done = payload.get("done", False)
        if not isinstance(queries, list) or not all(isinstance(q, str) for q in queries):
            logger.error(f"[{self.name}] Invalid planner response: queries must be a list of strings.")
            return None
        if not isinstance(done, bool):
            logger.error(f"[{self.name}] Invalid planner response: done must be a boolean.")
            return None
        return SearchPlanStep(queries=queries, done=done)

    def _extract_json_object(self, text: str) -> str:
        """Extract the first JSON object from potentially mixed model output.

        Args:
            text: Raw text that may contain JSON plus extra content.

        Returns:
            Extracted JSON object string if found, otherwise the original text.
        """
        if text.startswith("{") and text.endswith("}"):
            return text
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        return match.group(0).strip() if match else text

    def _is_failed_model_text(self, text: str) -> bool:
        """Return True when the model output matches known failure placeholders.

        Args:
            text: Model output to classify.

        Returns:
            True when text matches known failure/empty sentinel values.
        """
        normalized = text.strip().lower()
        return normalized in {
            "",
            "failed to generate a response.",
            "failed to generate a response",
        }

    def _retrieve_and_format_observation(
        self,
        query_text: str,
        sources: Sequence[Source],
        errors: list[str],
    ) -> str:
        """Retrieve source content in batch and format it into an observation block.

        Following steps are performed:
        1. Limit to max_results_per_query sources.
        2. Retrieve content for all selected urls
        3. Format the observation text with source titles, urls, and content snippets.

        Args:
            query_text: Query string used to produce the given sources.
            sources: Raw source objects returned by the search tool.
            errors: Mutable list used to append retrieval-related errors.

        Returns:
            Formatted multi-line observation text for the query.
        """
        lines = [f"Query: {query_text}"]
        selected_sources = [source for source in sources if isinstance(source, WebSource)]

        # Batch retrieve content for all selected sources, handling retrieval errors gracefully.
        try:
            contents = self.retriever.retrieve_batch([source.url for source in selected_sources])
        except Exception as exc:
            errors.append(f"Batch retrieval failed for query '{query_text}': {exc}")
            logger.error(f"[{self.name}] Batch retrieval failed for query '{query_text}': {exc}")
            contents = [None] * len(selected_sources)

        for source, content in zip(selected_sources, contents):
            if content is None:
                errors.append(f"Failed to retrieve content from {source.url}")
                logger.debug(f"[{self.name}] Failed to retrieve content from {source.url}")
                snippet = source.preview or "No retrieved content."
            else:
                if str(content).strip():
                    snippet = self._summarize_observation(instruction=query_text, observation=str(content))
                else:
                    snippet = "Retrieved content was empty."
            title = source.title or "Untitled"
            lines.append(f"- {title} | {source.url}\n  Content snippet: {snippet}")

        return "\n".join(lines)

    def _summarize_observation(self, instruction: str, observation: str) -> str:
        """Summarize a single observation block with the summarization model.

        Args:
            instruction: Original task instruction text.
            observation: Raw observation text to summarize.

        Returns:
            A summary string produced by the summarization model or the original observation on failure.
        """
        summary_prompt = (
            "You are a factual evidence summarizer.\n"
            "Summarize only information relevant to the task.\n"
            "Include concrete facts and source references where possible.\n\n"
            f"Task:\n{instruction}\n\n"
            f"Observation:\n{observation}"
        )
        try:
            summary = self.summarization_model.generate(Prompt(text=summary_prompt)).text.strip()
            if not summary or self._is_failed_model_text(summary):
                return observation
            return summary
        except Exception:
            return observation
