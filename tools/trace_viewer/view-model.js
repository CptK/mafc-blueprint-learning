import { EDGE_STYLES, WEB_LAYOUT } from "./constants.js";
import { summarizeText, wrapText } from "./utils.js";

export function buildViewModel(trace) {
  const nodes = [];
  const edges = [];
  const nodeById = {};
  let currentMainY = 80;

  if (trace.agent === "WebSearchAgent") {
    const runId = "websearch-root";
    pushNode(makeNode(runId, "childrun", "WebSearchAgent", buildRunSubtitle(trace), trace, null, false, WEB_LAYOUT.mainX, currentMainY));
    const webLayout = addWebTraceGraph(runId, trace, "root", WEB_LAYOUT.childX, currentMainY + 120);
    if (trace.summary && trace.summary.result && trace.summary.result.result) {
      pushNode(
        makeNode(
          "websearch-result",
          "result",
          "Result",
          summarizeText(trace.summary.result.result.text || "", 120),
          trace.summary.result,
          null,
          false,
          WEB_LAYOUT.mainX,
          webLayout.bottomY + 180
        )
      );
      edges.push(makeEdge(webLayout.lastNodeId || runId, "websearch-result", "finalizes"));
    }
    return { nodes, edges, nodeById };
  }

  pushNode(makeNode("claim", "claim", "Claim", summarizeText(trace.claim && trace.claim.text, 180), trace.claim || {}, null, false, WEB_LAYOUT.mainX, currentMainY));
  currentMainY += WEB_LAYOUT.rowGap;
  pushNode(
    makeNode(
      "blueprint",
      "blueprint",
      "Blueprint",
      trace.blueprint ? trace.blueprint.name : "Unknown blueprint",
      trace.blueprint || {},
      null,
      false,
      WEB_LAYOUT.mainX,
      currentMainY
    )
  );
  edges.push(makeEdge("claim", "blueprint", "selected"));
  currentMainY += WEB_LAYOUT.rowGap;

  let previousMainNodeId = "blueprint";
  for (const iteration of trace.iterations || []) {
    const iterationId = `iteration-${iteration.iteration}`;
    const subtitle = [
      iteration.decision && iteration.decision.decision_type ? `decision: ${iteration.decision.decision_type}` : null,
      `node: ${iteration.node_before} -> ${iteration.node_after}`,
      `evidence: ${iteration.evidence_count_before} -> ${iteration.evidence_count_after}`,
    ]
      .filter(Boolean)
      .join(" | ");
    pushNode(makeNode(iterationId, "iteration", `Iteration ${iteration.iteration}`, subtitle, iteration, null, false, WEB_LAYOUT.mainX, currentMainY));
    edges.push(makeEdge(previousMainNodeId, iterationId, "next"));
    previousMainNodeId = iterationId;

    const tasks = iteration.delegated_tasks || [];
    let localBottomY = currentMainY;
    tasks.forEach((task, index) => {
      const taskId = `task-${iteration.iteration}-${task.task_id}`;
      const taskY = currentMainY + index * 110;
      const taskSubtitle = [
        task.agent_type,
        summarizeText(task.instruction, 90),
        `evidence: ${task.result && Array.isArray(task.result.evidences) ? task.result.evidences.length : 0}`,
        `errors: ${task.result && Array.isArray(task.result.errors) ? task.result.errors.length : 0}`,
      ].join(" | ");
      pushNode(makeNode(taskId, "task", task.task_id, taskSubtitle, task, null, false, WEB_LAYOUT.taskX, taskY));
      edges.push(makeEdge(iterationId, taskId, "delegates"));
      if (index > 0) {
        edges.push(
          makeEdge(`task-${iteration.iteration}-${tasks[index - 1].task_id}`, taskId, "parallel", EDGE_STYLES.parallel)
        );
      }
      if (task.child_trace) {
        const childLayout = addChildTraceNodes(taskId, task.child_trace, WEB_LAYOUT.childX, taskY);
        localBottomY = Math.max(localBottomY, childLayout.bottomY);
      } else {
        localBottomY = Math.max(localBottomY, taskY);
      }
    });
    currentMainY = Math.max(currentMainY + WEB_LAYOUT.rowGap, localBottomY + 220);
  }

  const resultText =
    trace.summary && trace.summary.result && trace.summary.result.text ? trace.summary.result.text : trace.status || "No result";
  pushNode(makeNode("result", "result", "Result", summarizeText(resultText, 180), trace.summary || {}, null, false, WEB_LAYOUT.mainX, currentMainY));
  edges.push(makeEdge(previousMainNodeId, "result", "finalizes"));
  edges.push(makeEdge("blueprint", "result", "constrains"));
  return { nodes, edges, nodeById };

  function pushNode(node) {
    nodes.push(node);
    nodeById[node.id] = node;
  }

  function addChildTraceNodes(taskId, childTrace, x, y) {
    const runId = `${taskId}-child-run`;
    pushNode(makeNode(runId, "childrun", childTrace.agent || "Child Run", buildRunSubtitle(childTrace), childTrace, null, false, x, y));
    edges.push(makeEdge(taskId, runId, "runs"));
    return addWebTraceGraph(runId, childTrace, taskId, x, y + 120);
  }

  function addWebTraceGraph(runId, webTrace, keyPrefix, baseX, baseY) {
    let previousChildNodeId = runId;
    let lastNodeId = runId;
    let bottomY = baseY;
    const childIterations = webTrace.iterations || [];

    childIterations.forEach((childIteration, childIndex) => {
      const iterationY = baseY + childIndex * 520;
      const childIterationId = `${keyPrefix}-${runId}-step-${childIteration.step || childIteration.iteration || "x"}`;
      const querySummary =
        childIteration.resolved_plan && Array.isArray(childIteration.resolved_plan.queries)
          ? childIteration.resolved_plan.queries.join(", ")
          : null;
      const childSubtitle = [
        querySummary ? `queries: ${summarizeText(querySummary, 80)}` : null,
        childIteration.evidence_count_after != null
          ? `evidence: ${childIteration.evidence_count_before} -> ${childIteration.evidence_count_after}`
          : null,
        childIteration.new_errors && childIteration.new_errors.length ? `errors: ${childIteration.new_errors.length}` : null,
      ]
        .filter(Boolean)
        .join(" | ");
      pushNode(
        makeNode(
          childIterationId,
          "childstep",
          `Web Iteration ${childIteration.step || childIteration.iteration || "?"}`,
          childSubtitle,
          childIteration,
          null,
          false,
          baseX,
          iterationY
        )
      );
      edges.push(makeEdge(previousChildNodeId, childIterationId, "next"));
      previousChildNodeId = childIterationId;
      lastNodeId = childIterationId;

      const queryNodeIds = [];
      const candidateBlockIds = [];
      const searchResults = childIteration.search_results || [];
      let deepestCandidateBottomY = iterationY + WEB_LAYOUT.stageGap * 2;

      searchResults.forEach((queryResult, queryIndex) => {
        const queryX = baseX + queryIndex * WEB_LAYOUT.queryGap;
        const queryY = iterationY + WEB_LAYOUT.stageGap;
        const queryId = `${childIterationId}-query-${queryIndex + 1}`;
        queryNodeIds.push(queryId);
        pushNode(
          makeNode(
            queryId,
            "query",
            `Query ${queryIndex + 1}`,
            summarizeText(queryResult.query_text || "Untitled query", 90),
            queryResult,
            null,
            false,
            queryX,
            queryY
          )
        );
        edges.push(makeEdge(childIterationId, queryId, "query"));

        const candidateNodeId = `${queryId}-candidates`;
        candidateBlockIds.push(candidateNodeId);
        const candidateCount = Array.isArray(queryResult.sources) ? queryResult.sources.length : 0;
        pushNode(
          makeContainerNode(
            candidateNodeId,
            "select",
            "URLs from Query",
            `${candidateCount} url${candidateCount === 1 ? "" : "s"}`,
            {
              query_text: queryResult.query_text,
              sources: queryResult.sources || [],
              errors: queryResult.errors || [],
            },
            queryX,
            queryY + WEB_LAYOUT.stageGap
          )
        );
        edges.push(makeEdge(queryId, candidateNodeId, "results", EDGE_STYLES.default, 2, 8));
        deepestCandidateBottomY = Math.max(
          deepestCandidateBottomY,
          candidateBlockBottomY(queryY + WEB_LAYOUT.stageGap, queryResult.sources || [])
        );

        (queryResult.sources || []).forEach((source, sourceIndex) => {
          const sourceId = `${candidateNodeId}-url-${sourceIndex + 1}`;
          pushNode(
            makeChildNode(
              sourceId,
              "retrieval",
              source.title || source.url || source.reference || `URL ${sourceIndex + 1}`,
              summarizeText(source.url || source.reference || "", 80),
              { detailType: "candidate_url", query_text: queryResult.query_text, source },
              candidateNodeId,
              queryX,
              queryY + WEB_LAYOUT.stageGap + 60 + sourceIndex * WEB_LAYOUT.urlGap
            )
          );
          if (sourceIndex > 0) {
            edges.push(makeEdge(`${candidateNodeId}-url-${sourceIndex}`, sourceId, "", EDGE_STYLES.hidden));
          }
        });
      });

      const selectionX = baseX + (Math.max(searchResults.length - 1, 0) * WEB_LAYOUT.queryGap) / 2;
      const selectionY = deepestCandidateBottomY + 90;
      const selectionId = `${childIterationId}-selection`;
      const selectedSources = flattenSelectedSources(childIteration.selected_sources || []);
      pushNode(
        makeNode(
          selectionId,
          "select",
          "Source Selection",
          `${selectedSources.length} selected`,
          {
            detailType: "selection",
            selected_sources: childIteration.selected_sources || [],
            resolved_plan: childIteration.resolved_plan || null,
          },
          null,
          false,
          selectionX,
          selectionY
        )
      );
      if (candidateBlockIds.length > 0) {
        candidateBlockIds.forEach((candidateNodeId) =>
          edges.push(makeEdge(candidateNodeId, selectionId, "propose", EDGE_STYLES.default, 2, 8))
        );
      } else if (queryNodeIds.length > 0) {
        queryNodeIds.forEach((queryId) => edges.push(makeEdge(queryId, selectionId, "propose")));
      } else {
        edges.push(makeEdge(childIterationId, selectionId, "select"));
      }
      lastNodeId = selectionId;

      const retrievals = childIteration.retrievals || [];
      const retrievalStageId = `${childIterationId}-retrieval-stage`;
      const retrievalStageY = selectionY + WEB_LAYOUT.stageGap;
      pushNode(
        makeContainerNode(
          retrievalStageId,
          "select",
          "Retrieved URLs",
          `${retrievals.length} url${retrievals.length === 1 ? "" : "s"}`,
          { detailType: "retrieval_stage", retrievals },
          selectionX,
          retrievalStageY
        )
      );
      edges.push(makeEdge(selectionId, retrievalStageId, "retrieve", EDGE_STYLES.default, 2, 8));
      lastNodeId = retrievalStageId;

      retrievals.forEach((retrieval, retrievalIndex) => {
        const retrievalId = `${retrievalStageId}-url-${retrievalIndex + 1}`;
        const retrievalTitle =
          (retrieval.source && (retrieval.source.title || retrieval.source.url || retrieval.source.reference)) ||
          `retrieval ${retrievalIndex + 1}`;
        const retrievalSubtitle = [
          retrieval.source && retrieval.source.url ? summarizeText(retrieval.source.url, 60) : null,
          retrieval.snippet ? summarizeText(retrieval.snippet, 80) : null,
        ]
          .filter(Boolean)
          .join(" | ");
        pushNode(
          makeChildNode(
            retrievalId,
            "retrieval",
            retrievalTitle,
            retrievalSubtitle,
            { detailType: "retrieval_url", ...retrieval },
            retrievalStageId,
            selectionX,
            retrievalStageY + 60 + retrievalIndex * WEB_LAYOUT.urlGap
          )
        );
        if (retrievalIndex > 0) {
          edges.push(makeEdge(`${retrievalStageId}-url-${retrievalIndex}`, retrievalId, "", EDGE_STYLES.hidden));
        }
      });

      if (childIteration.synthesis && childIteration.synthesis.answer) {
        const synthesisId = `${childIterationId}-synthesis`;
        const synthesisY = retrievalBlockBottomY(retrievalStageY, retrievals) + 80;
        pushNode(
          makeNode(
            synthesisId,
            "result",
            "Iteration Synthesis",
            summarizeText(childIteration.synthesis.answer, 100),
            { detailType: "iteration_synthesis", ...childIteration.synthesis },
            null,
            false,
            selectionX,
            synthesisY
          )
        );
        edges.push(makeEdge(retrievalStageId, synthesisId, "synthesize"));
        lastNodeId = synthesisId;
        bottomY = Math.max(bottomY, synthesisY);
      } else {
        bottomY = Math.max(bottomY, retrievalBlockBottomY(retrievalStageY, retrievals));
      }
    });

    return { lastNodeId, bottomY };
  }

  function flattenSelectedSources(selectedSources) {
    const flat = [];
    selectedSources.forEach((entry) => {
      (entry.sources || []).forEach((source) => flat.push({ query_text: entry.query_text, source }));
    });
    return flat;
  }

  function candidateBlockBottomY(containerY, sources) {
    const count = Array.isArray(sources) ? sources.length : 0;
    return containerY + 110 + Math.max(0, count - 1) * WEB_LAYOUT.urlGap;
  }

  function retrievalBlockBottomY(containerY, retrievals) {
    const count = Array.isArray(retrievals) ? retrievals.length : 0;
    return containerY + 110 + Math.max(0, count - 1) * WEB_LAYOUT.urlGap;
  }
}

function buildRunSubtitle(trace) {
  return [
    trace.status || null,
    trace.summary && trace.summary.evidence_count != null ? `evidence: ${trace.summary.evidence_count}` : null,
  ]
    .filter(Boolean)
    .join(" | ");
}

function makeNode(id, type, title, subtitle, payload, parent = null, isContainer = false, x = 0, y = 0) {
  const titleLines = wrapText(title || "", 18, 2);
  const subtitleLines = wrapText(subtitle || "", 34, 4);
  return {
    id,
    type,
    title,
    subtitle,
    displayLabel: [...titleLines, ...subtitleLines].join("\n"),
    payload,
    parent,
    isContainer,
    x,
    y,
  };
}

function makeContainerNode(id, type, title, subtitle, payload, x = 0, y = 0) {
  return makeNode(id, type, title, subtitle, payload, null, true, x, y);
}

function makeChildNode(id, type, title, subtitle, payload, parent, x = 0, y = 0) {
  return makeNode(id, type, title, subtitle, payload, parent, false, x, y);
}

function makeEdge(source, target, label, edgeStyle = EDGE_STYLES.default, minLen = 1, weight = 1) {
  return { source, target, label, edgeStyle, minLen, weight };
}
