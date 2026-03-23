from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from collections.abc import Sequence
from datetime import date

from ezmm import MultimodalSequence

from mafc.common.evidence import Evidence
from mafc.common.logger import logger
from mafc.common.modeling.message import Message, MessageRole
from mafc.common.modeling.prompt import Prompt
from mafc.common.modeling.model import Model, Response
from mafc.tools.web_search.common import Query, Source, WebSource
from mafc.tools.web_search.integrations.integration import RetrievalIntegration

from mafc.agents.web_search.actions import InspectWebSource
from mafc.agents.web_search.models import (
    GlobalSourceCandidate,
    QueryInvestigationResult,
    QuerySearchResult,
    SearchTool,
)
from mafc.utils.parsing import extract_json_object, is_failed_model_text, strip_json_fences
from mafc.agents.web_search.synthesis import summarize_observation
from mafc.agents.web_search.tracing import WebSearchTraceRecorder


def execute_search_queries(
    queries: list[str],
    seen_queries: set[str],
    errors: list[str],
    search_tool: SearchTool,
    n_workers: int = 1,
    max_results_per_query: int = 5,
    latest_allowed_date: date | None = None,
    step: int | None = None,
    trace: WebSearchTraceRecorder | None = None,
) -> list[tuple[str, Sequence[Source] | None]]:
    """Execute all search queries and return source candidates per query."""
    if n_workers <= 1:
        results = [
            execute_search_query(query_text, search_tool, max_results_per_query, latest_allowed_date)
            for query_text in queries
        ]
    else:
        max_workers = min(n_workers, len(queries))
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            results = list(
                pool.map(
                    lambda query_text: execute_search_query(
                        query_text,
                        search_tool,
                        max_results_per_query,
                        latest_allowed_date,
                    ),
                    queries,
                )
            )

    candidate_sources: list[tuple[str, Sequence[Source] | None]] = []
    for result in results:
        if result.errors:
            errors.extend(result.errors)
        if result.mark_seen:
            seen_queries.add(result.query_text)
            candidate_sources.append((result.query_text, result.sources))
        if trace is not None and step is not None:
            trace.record_search_result(
                step=step,
                query_text=result.query_text,
                sources=list(result.sources) if result.sources is not None else None,
                errors=result.errors,
                marked_seen=result.mark_seen,
            )
    return candidate_sources


def retrieve_query_results(
    selected_sources: list[tuple[str, Sequence[Source] | None]],
    errors: list[str],
    model: Model,
    retriever: RetrievalIntegration,
    seen_urls: set[str] | None = None,
    step: int | None = None,
    trace: WebSearchTraceRecorder | None = None,
) -> list[QueryInvestigationResult]:
    """Retrieve selected sources and extract structured per-query results."""
    query_results: list[QueryInvestigationResult] = []
    for query_text, sources in selected_sources:
        if sources is None:
            query_results.append(
                QueryInvestigationResult(
                    query_text=query_text,
                    observation_text=f"Query: {query_text}\nNo results.",
                    evidences=[],
                )
            )
            continue
        query_results.append(
            retrieve_and_extract_evidence(
                query_text=query_text,
                sources=sources,
                errors=errors,
                model=model,
                retriever=retriever,
                seen_urls=seen_urls,
                step=step,
                trace=trace,
            )
        )
    return query_results


def execute_search_query(
    query_text: str,
    search_tool: SearchTool,
    max_results_per_query: int = 5,
    latest_allowed_date: date | None = None,
) -> QuerySearchResult:
    """Execute search for one query and return source candidates."""
    query = Query(
        text=query_text,
        limit=max_results_per_query,
        end_date=latest_allowed_date,
    )
    try:
        result = search_tool.search(query)
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


def select_sources_for_retrieval(
    candidates: list[tuple[str, Sequence[Source] | None]],
    model: Model,
    max_results_per_query: int = 5,
    step: int | None = None,
    trace: WebSearchTraceRecorder | None = None,
) -> list[tuple[str, Sequence[Source] | None]]:
    """Select relevant sources for retrieval from per-query candidates."""
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
    selection_prompt: str | None = None
    selection_response: str | None = None
    logger.debug(
        f"All candidate URLs before filtering:\n{'\n    '.join([c.source.url for c in global_candidates])}"
    )
    if len(global_candidates) > 5:
        max_selected_total = max_results_per_query * max(1, len(per_query_sources))
        selected_urls, selection_prompt, selection_response, filter_resp = filter_sources_with_model(
            candidates=global_candidates,
            max_selected=max_selected_total,
            model=model,
        )
        if trace is not None and filter_resp is not None:
            trace.add_usage(filter_resp, model.name)

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
        picked = selected_by_query.get(query_text, [])[:max_results_per_query]
        selected.append((query_text, picked))
    if trace is not None and step is not None:
        trace.record_selected_sources(
            step=step,
            selected_sources=[
                (query_text, list(sources) if sources is not None else None)
                for query_text, sources in selected
            ],
            selection_prompt=selection_prompt,
            selection_response=selection_response,
        )
    return selected


def filter_sources_with_model(
    candidates: list[GlobalSourceCandidate],
    max_selected: int,
    model: Model,
) -> tuple[list[str], str, str, Response | None]:
    """Select the most relevant source URLs for a query via model reasoning.

    Returns (selected_urls, selection_prompt, response_text).
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
        f"You can select up to {max_selected} URLs in total. If there are no good sources, you can select none.\n"
        "Return strict JSON only with schema:\n"
        '{"selected_urls": ["https://..."]}\n\n'
        f"Input JSON:\n{json.dumps(prompt_payload, ensure_ascii=True)}"
    )
    try:
        resp = model.generate([Message(role=MessageRole.USER, content=Prompt(text=selection_prompt))])
        response_text = resp.text
    except Exception as exc:
        logger.error(f"[WebSearch-Agent] Global source filtering call failed: {exc}")
        return [], selection_prompt, "", None
    return (
        parse_selected_urls(response_text=response_text, max_selected=max_selected),
        selection_prompt,
        response_text,
        resp,
    )


def parse_selected_urls(response_text: str, max_selected: int) -> list[str]:
    """Parse selected URL list from model output."""
    text = extract_json_object(strip_json_fences(response_text.strip()))
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        logger.error(f"[WebSearch-Agent] Failed to parse source filter output as JSON: {exc}")
        return []

    selected_urls = payload.get("selected_urls")
    if not isinstance(selected_urls, list) or not all(isinstance(url, str) for url in selected_urls):
        logger.error("[WebSearch-Agent] Invalid source filter output: selected_urls must be list[str].")
        return []
    selected_urls = [url.strip() for url in selected_urls if url.strip()]
    logger.info("Model selected URLs:\n" + "\n  ".join(selected_urls))
    return selected_urls[:max_selected]


def retrieve_and_extract_evidence(
    query_text: str,
    sources: Sequence[Source],
    errors: list[str],
    model: Model,
    retriever: RetrievalIntegration,
    seen_urls: set[str] | None = None,
    step: int | None = None,
    trace: WebSearchTraceRecorder | None = None,
) -> QueryInvestigationResult:
    """Retrieve source content in batch and turn it into observations and evidence."""
    lines = [f"Query: {query_text}"]
    all_web_sources = [source for source in sources if isinstance(source, WebSource)]
    if seen_urls is not None:
        skipped = [s for s in all_web_sources if s.url in seen_urls]
        selected_sources = [s for s in all_web_sources if s.url not in seen_urls]
        if skipped:
            logger.debug(
                f"[WebSearch-Agent] Skipping {len(skipped)} already-retrieved URL(s) for query '{query_text}': "
                + ", ".join(s.url for s in skipped)
            )
    else:
        selected_sources = all_web_sources
    evidences: list[Evidence] = []

    try:
        contents = retriever.retrieve_batch([source.url for source in selected_sources])
    except Exception as exc:
        errors.append(f"Batch retrieval failed for query '{query_text}': {exc}")
        logger.error(f"[WebSearch-Agent] Batch retrieval failed for query '{query_text}': {exc}")
        if trace is not None:
            trace.record_error(
                step=step,
                phase="batch_retrieval",
                message=f"Batch retrieval failed for query '{query_text}': {exc}",
            )
        contents = [None] * len(selected_sources)

    for source, content in zip(selected_sources, contents):
        irrelevant = False
        relevant_media: list = []
        if content is None:
            errors.append(f"Failed to retrieve content from {source.url}")
            logger.debug(f"[WebSearch-Agent] Failed to retrieve content from {source.url}")
            snippet = source.preview or "No retrieved content."
            raw_text = source.preview or ""
            if trace is not None:
                trace.record_error(
                    step=step,
                    phase="source_retrieval",
                    message=f"Failed to retrieve content from {source.url}",
                )
        else:
            content_text = str(content).strip()
            media_items = list(content.images) + list(content.videos)
            if content_text:
                snippet, relevant_media, obs_resp = summarize_observation(
                    model=model,
                    instruction=query_text,
                    observation=content_text,
                    media_items=media_items,
                )
                if trace is not None and obs_resp is not None:
                    trace.add_usage(obs_resp, model.name)
                raw_text = content_text
                if not snippet or is_failed_model_text(snippet):
                    irrelevant = True
                    logger.debug(f"[WebSearch-Agent] No relevant content found at {source.url}")
            else:
                snippet = ""
                raw_text = ""
                relevant_media = []
                irrelevant = True
        title = source.title or "Untitled"
        lines.append(f"- {title} | {source.url}\n  Content snippet: {snippet}")
        evidence = (
            None
            if irrelevant
            else build_evidence_from_source(
                query_text=query_text,
                source=source,
                raw_text=raw_text,
                snippet=snippet,
                preview=source.preview,
                relevant_media=relevant_media,
            )
        )
        if trace is not None and step is not None:
            trace.record_retrieval(
                step=step,
                query_text=query_text,
                source=source,
                retrieved_content=raw_text,
                evidence=evidence,
                irrelevant=irrelevant,
            )
        if seen_urls is not None:
            seen_urls.add(source.url)
        if evidence is not None:
            evidences.append(evidence)

    return QueryInvestigationResult(
        query_text=query_text,
        observation_text="\n".join(lines),
        evidences=evidences,
    )


def build_evidence_from_source(
    query_text: str,
    source: WebSource,
    raw_text: str,
    snippet: str,
    preview: str | None = None,
    relevant_media: list | None = None,
) -> Evidence | None:
    """Build one source-backed evidence item from a retrieved web source."""
    raw_payload = raw_text.strip()
    takeaway_text = snippet.strip()
    if not raw_payload and not takeaway_text:
        return None
    rel_media = relevant_media or []
    return Evidence(
        raw=MultimodalSequence(raw_payload or takeaway_text),
        action=InspectWebSource(
            query_text=query_text,
            source_url=source.url,
            source_title=source.title,
        ),
        source=source.url,
        preview=preview or None,
        takeaways=MultimodalSequence(takeaway_text, *rel_media) if takeaway_text else None,
    )
