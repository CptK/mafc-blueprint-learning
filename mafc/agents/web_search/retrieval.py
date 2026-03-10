from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from collections.abc import Sequence

from mafc.common.logger import logger
from mafc.common.modeling.prompt import Prompt
from mafc.tools.web_search.common import Query, Source, WebSource

from .models import GlobalSourceCandidate, QuerySearchResult
from .parsing import extract_json_object
from .synthesis import summarize_observation


def collect_observations_for_queries(
    agent,
    queries: list[str],
    seen_queries: set[str],
    errors: list[str],
) -> list[str]:
    """Search all queries first, then retrieve and format observations."""
    if agent._should_stop:
        errors.append("Agent execution stopped early by stop signal.")
        return []

    if agent.n_workers <= 1:
        results = [execute_search_query(agent, query_text) for query_text in queries]
    else:
        max_workers = min(agent.n_workers, len(queries))
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            results = list(pool.map(lambda query_text: execute_search_query(agent, query_text), queries))

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

    selected_sources = select_sources_for_retrieval(agent, candidate_sources)
    for query_text, sources in selected_sources:
        if sources is None:
            observations.append(f"Query: {query_text}\nNo results.")
            continue
        observations.append(
            retrieve_and_format_observation(
                agent=agent,
                query_text=query_text,
                sources=sources,
                errors=errors,
            )
        )
    return observations


def execute_search_query(agent, query_text: str) -> QuerySearchResult:
    """Execute search for one query and return source candidates."""
    if agent._should_stop:
        return QuerySearchResult(
            query_text=query_text,
            sources=None,
            errors=[],
            mark_seen=False,
            stopped=True,
        )

    query = Query(
        text=query_text,
        limit=agent.max_results_per_query,
        end_date=agent.latest_allowed_date,
    )
    try:
        result = agent.search_tool.search(query)
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
    agent,
    candidates: list[tuple[str, Sequence[Source] | None]],
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
    logger.info(
        f"All candidate URLs before filtering:\n{'\n    '.join([c.source.url for c in global_candidates])}"
    )
    if len(global_candidates) > 5:
        max_selected_total = agent.max_results_per_query * max(1, len(per_query_sources))
        selected_urls = filter_sources_with_model(
            agent=agent,
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
        picked = selected_by_query.get(query_text, [])[: agent.max_results_per_query]
        selected.append((query_text, picked))
    return selected


def filter_sources_with_model(agent, candidates: list[GlobalSourceCandidate], max_selected: int) -> list[str]:
    """Select the most relevant source URLs for a query via model reasoning."""
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
        response_text = agent.model.generate(Prompt(text=selection_prompt)).text
    except Exception as exc:
        logger.error(f"[{agent.name}] Global source filtering call failed: {exc}")
        return []
    return parse_selected_urls(agent, response_text=response_text, max_selected=max_selected)


def parse_selected_urls(agent, response_text: str, max_selected: int) -> list[str]:
    """Parse selected URL list from model output."""
    text = response_text.strip()
    if text.startswith("```"):
        lines = [line for line in text.splitlines() if not line.startswith("```")]
        text = "\n".join(lines).strip()
    text = extract_json_object(text)
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        logger.error(f"[{agent.name}] Failed to parse source filter output as JSON: {exc}")
        return []

    selected_urls = payload.get("selected_urls")
    if not isinstance(selected_urls, list) or not all(isinstance(url, str) for url in selected_urls):
        logger.error(f"[{agent.name}] Invalid source filter output: selected_urls must be list[str].")
        return []
    selected_urls = [url.strip() for url in selected_urls if url.strip()]
    logger.info("Model selected URLs:\n" + "\n  ".join(selected_urls))
    return selected_urls[:max_selected]


def retrieve_and_format_observation(
    agent,
    query_text: str,
    sources: Sequence[Source],
    errors: list[str],
) -> str:
    """Retrieve source content in batch and format it into an observation block."""
    lines = [f"Query: {query_text}"]
    selected_sources = [source for source in sources if isinstance(source, WebSource)]

    try:
        contents = agent.retriever.retrieve_batch([source.url for source in selected_sources])
    except Exception as exc:
        errors.append(f"Batch retrieval failed for query '{query_text}': {exc}")
        logger.error(f"[{agent.name}] Batch retrieval failed for query '{query_text}': {exc}")
        contents = [None] * len(selected_sources)

    for source, content in zip(selected_sources, contents):
        if content is None:
            errors.append(f"Failed to retrieve content from {source.url}")
            logger.debug(f"[{agent.name}] Failed to retrieve content from {source.url}")
            snippet = source.preview or "No retrieved content."
        else:
            if str(content).strip():
                snippet = summarize_observation(agent, instruction=query_text, observation=str(content))
            else:
                snippet = "Retrieved content was empty."
        title = source.title or "Untitled"
        lines.append(f"- {title} | {source.url}\n  Content snippet: {snippet}")

    return "\n".join(lines)
