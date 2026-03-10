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
class QueryObservationResult:
    """Result container for one query execution in the worker pool."""

    query_text: str
    observation: str | None
    errors: list[str]
    mark_seen: bool = False
    stopped: bool = False


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
        max_queries_per_step: int = 3,
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

    def run(self, task: Prompt) -> AgentResult:
        """Run iterative planning, searching, retrieval, and synthesis for a task.

        Args:
            task: Prompt containing the claim/instruction to investigate.

        Returns:
            AgentResult with aggregated syntheses and collected non-fatal errors.
        """
        instruction = str(task).strip()
        if self._should_stop:
            return AgentResult(result=None, errors=["Agent was stopped before execution started."])
        if not instruction:
            return AgentResult(result=None, errors=["Task prompt is empty."])

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
            return AgentResult(result=None, errors=errors)

        final_text = "\n\n".join(memory)
        return AgentResult(result=MultimodalSequence(final_text), errors=errors)

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
        """Search for query results and build observation blocks.

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
            results = [self._execute_query(query_text) for query_text in queries]
        else:
            max_workers = min(self.n_workers, len(queries))
            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                results = list(pool.map(self._execute_query, queries))

        observations: list[str] = []
        stop_recorded = False
        for result in results:
            if result.errors:
                errors.extend(result.errors)
            if result.stopped and not stop_recorded:
                errors.append("Agent execution stopped early by stop signal.")
                stop_recorded = True
            if result.mark_seen:
                seen_queries.add(result.query_text)
            if result.observation is not None:
                observations.append(result.observation)
        return observations

    def _execute_query(self, query_text: str) -> QueryObservationResult:
        """Execute search and retrieval for one query.

        Args:
            query_text: Query text to execute.

        Returns:
            QueryObservationResult including observation text, errors, and status flags.
        """
        if self._should_stop:
            return QueryObservationResult(
                query_text=query_text,
                observation=None,
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
            return QueryObservationResult(
                query_text=query_text,
                observation=None,
                errors=[f"Search failed for query '{query_text}': {exc}"],
            )

        if result is None:
            return QueryObservationResult(
                query_text=query_text,
                observation=f"Query: {query_text}\nNo results.",
                errors=[],
                mark_seen=True,
            )

        query_errors: list[str] = []
        observation = self._retrieve_and_format_observation(
            query_text=query_text,
            sources=result.sources,
            errors=query_errors,
        )
        return QueryObservationResult(
            query_text=query_text,
            observation=observation,
            errors=query_errors,
            mark_seen=True,
        )

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
        web_sources = [source for source in sources if isinstance(source, WebSource)]

        # Limit the number of sources to retrieve based on max_results_per_query.
        selected_sources = web_sources[: self.max_results_per_query]

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
