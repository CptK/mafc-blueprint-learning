import { EDGE_STYLES, WEB_LAYOUT } from "./constants.js";
import { summarizeText, wrapText } from "./utils.js";

export function buildViewModel(trace) {
  const nodes = [];
  const edges = [];
  const nodeById = {};
  let currentMainY = 80;

  if (trace.agent === "MediaAgent") {
    const runId = "media-root";
    pushNode(makeNode(runId, "childrun", "MediaAgent", buildRunSubtitle(trace), trace, null, false, WEB_LAYOUT.mainX, currentMainY));
    const mediaLayout = addMediaTraceGraph(runId, trace, "root", WEB_LAYOUT.childX, currentMainY + 120);
    if (trace.summary && trace.summary.result && trace.summary.result.result) {
      pushNode(
        makeNode(
          "media-result",
          "result",
          "Result",
          summarizeText(trace.summary.result.result.text || "", 120),
          trace.summary.result,
          null,
          false,
          WEB_LAYOUT.mainX,
          mediaLayout.bottomY + 180
        )
      );
      edges.push(makeEdge(mediaLayout.lastNodeId || runId, "media-result", "finalizes"));
    }
    return { nodes, edges, nodeById };
  }

  if (trace.agent === "JudgeAgent") {
    const runId = "judge-root";
    pushNode(makeNode(runId, "childrun", "JudgeAgent", buildRunSubtitle(trace), trace, null, false, WEB_LAYOUT.mainX, currentMainY));
    const judgeLayout = addJudgeTraceGraph(runId, trace, "root", WEB_LAYOUT.childX, currentMainY + 120);
    if (trace.summary && trace.summary.result && trace.summary.result.result) {
      pushNode(
        makeNode(
          "judge-result",
          "result",
          "Result",
          summarizeText(trace.summary.result.result.text || "", 120),
          trace.summary.result,
          null,
          false,
          WEB_LAYOUT.mainX,
          judgeLayout.bottomY + 180
        )
      );
      edges.push(makeEdge(judgeLayout.lastNodeId || runId, "judge-result", "finalizes"));
    }
    return { nodes, edges, nodeById };
  }

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
  const blueprintMode = trace.blueprint && trace.blueprint.selection && trace.blueprint.selection.mode;
  const blueprintSubtitle = [
    trace.blueprint ? trace.blueprint.name : "Unknown blueprint",
    blueprintMode ? `(${blueprintMode})` : null,
  ].filter(Boolean).join(" — ");
  pushNode(
    makeNode(
      "blueprint",
      "blueprint",
      "Blueprint",
      blueprintSubtitle,
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
  let pendingReturnEdges = [];
  for (const iteration of trace.iterations || []) {
    const iterationId = `iteration-${iteration.iteration}`;
    const routing = iteration.routing || {};
    const subtitle = [
      iteration.execution_type || null,
      routing.type === "auto"
        ? `auto → ${routing.target_node_id}`
        : routing.type === "llm"
        ? `routed → ${routing.target_node_id}`
        : iteration.decision && iteration.decision.decision_type
        ? `decision: ${iteration.decision.decision_type}`
        : null,
      `node: ${iteration.node_before} -> ${iteration.node_after}`,
      `evidence: ${iteration.evidence_count_before} -> ${iteration.evidence_count_after}`,
    ]
      .filter(Boolean)
      .join(" | ");
    pushNode(makeNode(iterationId, "iteration", `Iteration ${iteration.iteration}`, subtitle, iteration, null, false, WEB_LAYOUT.mainX, currentMainY));
    edges.push(makeEdge(previousMainNodeId, iterationId, "next"));
    for (const returnNodeId of pendingReturnEdges) {
      edges.push(makeEdge(returnNodeId, iterationId, "returns"));
    }
    pendingReturnEdges = [];
    previousMainNodeId = iterationId;

    const tasks = iteration.delegated_tasks || [];
    let localBottomY = currentMainY;
    const childXBase = WEB_LAYOUT.childX - WEB_LAYOUT.taskX;
    let nextTaskX = WEB_LAYOUT.taskX;
    tasks.forEach((task, index) => {
      const taskId = `task-${iteration.iteration}-${task.task_id}`;
      const taskX = nextTaskX;
      const taskY = currentMainY;
      const taskSubtitle = [
        task.agent_type,
        summarizeText(task.instruction, 90),
        `evidence: ${task.result && Array.isArray(task.result.evidences) ? task.result.evidences.length : 0}`,
        `errors: ${task.result && Array.isArray(task.result.errors) ? task.result.errors.length : 0}`,
      ].join(" | ");
      const nodesBeforeColumn = nodes.length;
      pushNode(makeNode(taskId, "task", task.task_id, taskSubtitle, task, null, false, taskX, taskY));
      edges.push(makeEdge(iterationId, taskId, "delegates"));
      if (index > 0) {
        edges.push(
          makeEdge(`task-${iteration.iteration}-${tasks[index - 1].task_id}`, taskId, "parallel", EDGE_STYLES.parallel)
        );
      }
      if (task.child_trace) {
        const childX = taskX + childXBase;
        const childLayout = addChildTraceNodes(taskId, task.child_trace, childX, taskY);
        localBottomY = Math.max(localBottomY, childLayout.bottomY);
        if (childLayout.lastNodeId) {
          pendingReturnEdges.push(childLayout.lastNodeId);
        }
      } else {
        localBottomY = Math.max(localBottomY, taskY);
      }
      const columnRightX = nodes.slice(nodesBeforeColumn).reduce((max, n) => Math.max(max, n.x + 240), taskX);
      nextTaskX = columnRightX + WEB_LAYOUT.taskColumnGap;
    });
    currentMainY = Math.max(currentMainY + WEB_LAYOUT.rowGap, localBottomY + 220);
  }

  const resultText =
    trace.summary && trace.summary.result && trace.summary.result.text ? trace.summary.result.text : trace.status || "No result";
  pushNode(makeNode("result", "result", "Result", summarizeText(resultText, 180), trace.summary || {}, null, false, WEB_LAYOUT.mainX, currentMainY));
  edges.push(makeEdge(previousMainNodeId, "result", "finalizes"));
  for (const returnNodeId of pendingReturnEdges) {
    edges.push(makeEdge(returnNodeId, "result", "returns"));
  }
  edges.push(makeEdge("blueprint", "result", "constrains"));

  let lastMainNodeId = "result";
  let lastMainY = currentMainY;
  if (trace.judge_run) {
    const judgeRunY = currentMainY + WEB_LAYOUT.rowGap;
    const judgeRunId = "judge-run";
    pushNode(makeNode(judgeRunId, "childrun", "JudgeAgent", buildRunSubtitle(trace.judge_run), trace.judge_run, null, false, WEB_LAYOUT.mainX, judgeRunY));
    edges.push(makeEdge("result", judgeRunId, "judges"));
    addJudgeTraceGraph(judgeRunId, trace.judge_run, "judge", WEB_LAYOUT.childX, judgeRunY + 120);
    lastMainNodeId = judgeRunId;
    lastMainY = judgeRunY;
  }

  const summary = trace.summary || {};
  const predictedLabel = (trace.judge_run && trace.judge_run.decision && trace.judge_run.decision.label) || null;
  const trueLabel = summary.true_label || null;
  const correct = predictedLabel && trueLabel ? predictedLabel === trueLabel : null;
  const totalTokens = (summary.total_input_tokens ?? 0) + (summary.total_output_tokens ?? 0);
  const summaryParts = [
    predictedLabel ? `predicted: ${predictedLabel}` : null,
    trueLabel ? `true: ${trueLabel}` : null,
    correct !== null ? (correct ? "correct" : "incorrect") : null,
    summary.runtime_seconds != null ? `${(summary.runtime_seconds / 60).toFixed(1)}min` : null,
    summary.total_cost_usd != null ? `$${summary.total_cost_usd.toFixed(4)}` : null,
    totalTokens > 0 ? `${totalTokens.toLocaleString()} tok` : null,
  ].filter(Boolean).join(" | ");
  const summaryY = lastMainY + WEB_LAYOUT.rowGap;
  pushNode(
    makeNode(
      "run-summary",
      "result",
      "Run Summary",
      summaryParts,
      {
        detailType: "run_summary",
        predicted_label: predictedLabel,
        true_label: trueLabel,
        correct,
        runtime_seconds: summary.runtime_seconds ?? null,
        total_cost_usd: summary.total_cost_usd ?? null,
        total_input_tokens: summary.total_input_tokens ?? null,
        total_output_tokens: summary.total_output_tokens ?? null,
        by_model: summary.by_model ?? {},
        evidence_count: summary.evidence_count ?? null,
        errors: summary.errors || [],
      },
      null,
      false,
      WEB_LAYOUT.mainX,
      summaryY
    )
  );
  edges.push(makeEdge(lastMainNodeId, "run-summary", "summarizes"));

  return { nodes, edges, nodeById };

  function pushNode(node) {
    nodes.push(node);
    nodeById[node.id] = node;
  }

  function addChildTraceNodes(taskId, childTrace, x, y) {
    const runId = `${taskId}-child-run`;
    pushNode(makeNode(runId, "childrun", childTrace.agent || "Child Run", buildRunSubtitle(childTrace), childTrace, null, false, x, y));
    edges.push(makeEdge(taskId, runId, "runs"));

    let layout;
    if (childTrace.agent === "MediaAgent") {
      layout = addMediaTraceGraph(runId, childTrace, taskId, x, y + 120);
    } else if (childTrace.agent === "JudgeAgent") {
      layout = addJudgeTraceGraph(runId, childTrace, taskId, x, y + 120);
    } else {
      layout = addWebTraceGraph(runId, childTrace, taskId, x, y + 120);
    }

    const summary = childTrace.summary || {};
    if (summary.result && summary.result.result && summary.result.result.text) {
      const resultId = `${runId}-result`;
      const resultY = layout.bottomY + 180;
      const isWebSearch = !childTrace.agent || childTrace.agent === "WebSearchAgent";
      pushNode(
        makeNode(
          resultId,
          "result",
          isWebSearch ? "Search Result" : "Result",
          summarizeText(summary.result.result.text, 1),
          {
            detailType: isWebSearch ? "web_search_result" : "child_result",
            answer: summary.result.result.text,
            result: summary.result.result,
            evidences: summary.result.evidences || [],
            errors: summary.errors || [],
          },
          null,
          false,
          x,
          resultY
        )
      );
      edges.push(makeEdge(layout.lastNodeId || runId, resultId, "finalizes"));
      layout.lastNodeId = resultId;
      layout.bottomY = resultY;
    }

    return layout;
  }

  function addWebTraceGraph(runId, webTrace, keyPrefix, baseX, baseY) {
    let previousChildNodeId = runId;
    let lastNodeId = runId;
    let bottomY = baseY;
    const childIterations = webTrace.iterations || [];

    let nextIterationY = baseY;
    childIterations.forEach((childIteration, childIndex) => {
      const iterationY = nextIterationY;
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
            selection_prompt: childIteration.selection_prompt || null,
            selection_response: childIteration.selection_response || null,
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
      const irrelevantCount = retrievals.filter((r) => r.irrelevant).length;
      const retrievalStageId = `${childIterationId}-retrieval-stage`;
      const retrievalStageY = selectionY + WEB_LAYOUT.stageGap;
      const retrievalStageSubtitle = irrelevantCount > 0
        ? `${retrievals.length} url${retrievals.length === 1 ? "" : "s"} · ${irrelevantCount} irrelevant`
        : `${retrievals.length} url${retrievals.length === 1 ? "" : "s"}`;
      pushNode(
        makeContainerNode(
          retrievalStageId,
          "select",
          "Retrieved URLs",
          retrievalStageSubtitle,
          { detailType: "retrieval_stage", retrievals },
          selectionX,
          retrievalStageY
        )
      );
      edges.push(makeEdge(selectionId, retrievalStageId, "retrieve", EDGE_STYLES.default, 2, 8));
      lastNodeId = retrievalStageId;

      const retrievalGridOffsetX = ((Math.min(retrievals.length, WEB_LAYOUT.urlColumns) - 1) * WEB_LAYOUT.urlColumnGap) / 2;
      retrievals.forEach((retrieval, retrievalIndex) => {
        const retrievalId = `${retrievalStageId}-url-${retrievalIndex + 1}`;
        const retrievalTitle =
          (retrieval.source && (retrieval.source.title || retrieval.source.url || retrieval.source.reference)) ||
          `retrieval ${retrievalIndex + 1}`;
        const retrievalSubtitle = [
          retrieval.irrelevant ? "no relevant content" : null,
          retrieval.source && retrieval.source.url ? summarizeText(retrieval.source.url, 60) : null,
          !retrieval.irrelevant && retrieval.evidence && retrieval.evidence.takeaways && retrieval.evidence.takeaways.text
            ? summarizeText(retrieval.evidence.takeaways.text, 80)
            : null,
        ]
          .filter(Boolean)
          .join(" | ");
        const col = retrievalIndex % WEB_LAYOUT.urlColumns;
        const row = Math.floor(retrievalIndex / WEB_LAYOUT.urlColumns);
        pushNode(
          makeChildNode(
            retrievalId,
            retrieval.irrelevant ? "retrieval_irrelevant" : "retrieval",
            retrievalTitle,
            retrievalSubtitle,
            { detailType: "retrieval_url", ...retrieval },
            retrievalStageId,
            selectionX - retrievalGridOffsetX + col * WEB_LAYOUT.urlColumnGap,
            retrievalStageY + 60 + row * WEB_LAYOUT.urlGap
          )
        );
        if (retrievalIndex > 0) {
          edges.push(makeEdge(`${retrievalStageId}-url-${retrievalIndex}`, retrievalId, "", EDGE_STYLES.hidden));
        }
      });

      let thisIterationBottomY;
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
        thisIterationBottomY = synthesisY;
      } else {
        thisIterationBottomY = retrievalBlockBottomY(retrievalStageY, retrievals);
      }
      bottomY = Math.max(bottomY, thisIterationBottomY);
      nextIterationY = thisIterationBottomY + 120;
      previousChildNodeId = retrievalStageId;
    });

    return { lastNodeId, bottomY };
  }

  function addMediaTraceGraph(runId, mediaTrace, keyPrefix, baseX, baseY) {
    let lastNodeId = runId;
    let bottomY = baseY;

    const plannedTools = mediaTrace.planned_tools || [];
    const plannerSubtitle = plannedTools.length > 0 ? `tools: ${plannedTools.join(", ")}` : "no tools planned";
    const plannerId = `${keyPrefix}-${runId}-planner`;
    pushNode(
      makeNode(
        plannerId,
        "childstep",
        "Media Planner",
        plannerSubtitle,
        {
          detailType: "media_planner",
          planner_messages: mediaTrace.planner_messages || [],
          planner_response: mediaTrace.planner_response,
          planned_tools: plannedTools,
          started_at: mediaTrace.started_at,
        },
        null,
        false,
        baseX,
        baseY
      )
    );
    edges.push(makeEdge(runId, plannerId, "plan"));
    lastNodeId = plannerId;

    const toolResults = mediaTrace.tool_results || [];
    let deepestToolBottomY = baseY + WEB_LAYOUT.stageGap;
    const toolNodeIds = [];

    toolResults.forEach((toolResult, index) => {
      const toolX = baseX + index * WEB_LAYOUT.queryGap;
      const toolY = baseY + WEB_LAYOUT.stageGap;
      const toolNodeId = `${keyPrefix}-${runId}-tool-${index + 1}`;
      toolNodeIds.push(toolNodeId);

      const sources = toolResult.sources || [];
      const toolLabel =
        toolResult.tool === "reverse_image_search"
          ? "Reverse Image Search"
          : toolResult.tool === "geolocate"
          ? "Geolocate"
          : toolResult.tool || "Tool";
      const toolSubtitle =
        sources.length > 0 ? `${sources.length} source${sources.length === 1 ? "" : "s"}` : "no sources";

      pushNode(
        makeContainerNode(
          toolNodeId,
          "select",
          toolLabel,
          toolSubtitle,
          { detailType: "media_tool_result", ...toolResult },
          toolX,
          toolY
        )
      );
      edges.push(makeEdge(plannerId, toolNodeId, "run", EDGE_STYLES.default, 2, 8));

      sources.forEach((source, sourceIndex) => {
        const sourceId = `${toolNodeId}-source-${sourceIndex + 1}`;
        pushNode(
          makeChildNode(
            sourceId,
            "retrieval",
            source.title || source.url || source.reference || `Source ${sourceIndex + 1}`,
            summarizeText(source.url || source.reference || "", 80),
            { detailType: "media_source_url", source },
            toolNodeId,
            toolX,
            toolY + 60 + sourceIndex * WEB_LAYOUT.urlGap
          )
        );
        if (sourceIndex > 0) {
          edges.push(makeEdge(`${toolNodeId}-source-${sourceIndex}`, sourceId, "", EDGE_STYLES.hidden));
        }
      });

      if (sources.length === 0 && toolResult.takeaways && toolResult.takeaways.text) {
        const resultChildId = `${toolNodeId}-result`;
        pushNode(
          makeChildNode(
            resultChildId,
            "retrieval",
            "Result",
            summarizeText(toolResult.takeaways.text, 80),
            { detailType: "media_source_url", source: { reference: "", takeaways: toolResult.takeaways } },
            toolNodeId,
            toolX,
            toolY + 60
          )
        );
      }

      const childCount = Math.max(sources.length, sources.length === 0 && toolResult.takeaways ? 1 : 0);
      const thisToolBottomY = toolY + 60 + Math.max(0, childCount - 1) * WEB_LAYOUT.urlGap;
      deepestToolBottomY = Math.max(deepestToolBottomY, thisToolBottomY);
    });

    const synthesisX = baseX + (Math.max(toolResults.length - 1, 0) * WEB_LAYOUT.queryGap) / 2;
    const synthesisY = deepestToolBottomY + 90;
    const synthesis = mediaTrace.synthesis;

    if (synthesis && synthesis.answer) {
      const synthesisId = `${keyPrefix}-${runId}-synthesis`;
      pushNode(
        makeNode(
          synthesisId,
          "result",
          "Media Synthesis",
          summarizeText(synthesis.answer, 100),
          { detailType: "media_synthesis", ...synthesis },
          null,
          false,
          synthesisX,
          synthesisY
        )
      );
      if (toolNodeIds.length > 0) {
        toolNodeIds.forEach((tid) =>
          edges.push(makeEdge(tid, synthesisId, "synthesize", EDGE_STYLES.default, 2, 8))
        );
      } else {
        edges.push(makeEdge(plannerId, synthesisId, "synthesize"));
      }
      lastNodeId = synthesisId;
      bottomY = synthesisY;
    } else {
      bottomY = deepestToolBottomY;
      if (toolNodeIds.length > 0) {
        lastNodeId = toolNodeIds[toolNodeIds.length - 1];
      }
    }

    return { lastNodeId, bottomY };
  }

  function addJudgeTraceGraph(runId, judgeTrace, keyPrefix, baseX, baseY) {
    const decision = judgeTrace.decision || {};
    const summary = judgeTrace.summary || {};
    const label = decision.label || summary.label || null;
    const decisionSubtitle = label ? `label: ${label}` : "no decision";
    const decisionId = `${keyPrefix}-${runId}-decision`;

    pushNode(
      makeNode(
        decisionId,
        "result",
        "Judge Decision",
        decisionSubtitle,
        { detailType: "judge_decision", ...judgeTrace },
        null,
        false,
        baseX,
        baseY
      )
    );
    edges.push(makeEdge(runId, decisionId, "verdict"));

    return { lastNodeId: decisionId, bottomY: baseY };
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
    const rows = Math.ceil(count / WEB_LAYOUT.urlColumns);
    return containerY + 110 + Math.max(0, rows - 1) * WEB_LAYOUT.urlGap;
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
