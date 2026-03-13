import { escapeHtml } from "./utils.js";

export function renderDetail(node) {
  const payload = node.payload || {};
  const detailType = payload.detailType || node.type;

  if (detailType === "query") {
    return `
      <h3>Query</h3>
      <p>${escapeHtml(payload.query_text || "")}</p>
      ${payload.errors && payload.errors.length ? `<p><strong>Errors:</strong> ${escapeHtml(payload.errors.join(" | "))}</p>` : ""}
      ${payload.marked_seen != null ? `<p><strong>Marked seen:</strong> ${payload.marked_seen ? "yes" : "no"}</p>` : ""}
    `;
  }

  if (detailType === "candidate_url") {
    return `
      <h3>${escapeHtml(payload.source.title || payload.source.url || "Candidate URL")}</h3>
      <p><strong>Query:</strong> ${escapeHtml(payload.query_text || "")}</p>
      <p><strong>URL:</strong> ${escapeHtml(payload.source.url || payload.source.reference || "")}</p>
      ${payload.source.release_date ? `<p><strong>Release date:</strong> ${escapeHtml(payload.source.release_date)}</p>` : ""}
      ${payload.source.preview ? `<p><strong>Preview:</strong> ${escapeHtml(payload.source.preview)}</p>` : ""}
    `;
  }

  if (detailType === "retrieval_url") {
    return `
      <h3>${escapeHtml((payload.source && (payload.source.title || payload.source.url)) || "Retrieved URL")}</h3>
      ${payload.irrelevant ? `<p><em>No relevant content found — excluded from evidence.</em></p>` : ""}
      ${payload.query_text ? `<p><strong>Query:</strong> ${escapeHtml(payload.query_text)}</p>` : ""}
      <p><strong>URL:</strong> ${escapeHtml((payload.source && (payload.source.url || payload.source.reference)) || "")}</p>
      ${!payload.irrelevant ? renderCollapsibleText("Preview", payload.evidence && payload.evidence.preview) : ""}
      ${!payload.irrelevant ? renderCollapsibleMultimodal("Takeaways", payload.evidence && payload.evidence.takeaways) : ""}
      ${!payload.irrelevant ? renderCollapsibleMultimodal("Raw", payload.evidence && payload.evidence.raw) : ""}
    `;
  }

  if (detailType === "selection") {
    const entries = (payload.selected_sources || []).map((entry) => {
      const items = (entry.sources || [])
        .map(
          (source) =>
            `<li><strong>${escapeHtml(source.title || source.url || source.reference || "URL")}</strong><br>${escapeHtml(source.url || source.reference || "")}</li>`
        )
        .join("");
      return `<div class="detail-section"><h3>${escapeHtml(entry.query_text || "Query")}</h3><ul>${items || "<li>No URLs selected.</li>"}</ul></div>`;
    });
    return (
      (entries.join("") || "<div class=\"empty\">No selected sources recorded.</div>") +
      renderCollapsibleText("Selection Prompt", payload.selection_prompt) +
      renderCollapsibleText("Model Response", payload.selection_response)
    );
  }

  if (detailType === "retrieval_stage") {
    const items = (payload.retrievals || [])
      .map(
        (entry) =>
          `<li><strong>${escapeHtml((entry.source && (entry.source.title || entry.source.url)) || "URL")}</strong><br>${escapeHtml((entry.source && (entry.source.url || entry.source.reference)) || "")}</li>`
      )
      .join("");
    return `<div class="detail-section"><h3>Retrieval Queue</h3><ul>${items || "<li>No retrievals recorded.</li>"}</ul></div>`;
  }

  if (detailType === "childstep") {
    const duration = formatDuration(payload.started_at, payload.ended_at);
    const evidenceDelta = payload.evidence_count_after - payload.evidence_count_before;
    const errors = payload.new_errors || [];
    const plan = payload.resolved_plan || {};
    const flags = [
      plan.done ? "done" : null,
      plan.should_terminate ? "terminate" : null,
      plan.fallback_used ? "fallback used" : null,
    ].filter(Boolean);

    return `
      <h3>Web Iteration ${escapeHtml(String(payload.step ?? "?"))}</h3>

      <p>
        ${duration ? `<strong>Duration:</strong> ${escapeHtml(duration)} &nbsp;|&nbsp; ` : ""}
        <strong>Evidence:</strong> ${payload.evidence_count_before} → ${payload.evidence_count_after} (+${evidenceDelta})
        ${errors.length ? ` &nbsp;|&nbsp; <strong>Errors:</strong> ${errors.length}` : ""}
        ${flags.length ? ` &nbsp;|&nbsp; ${flags.map(escapeHtml).join(", ")}` : ""}
      </p>

      ${errors.length ? `
        <div class="detail-section">
          <strong>Errors</strong>
          <ul>${errors.map((e) => `<li>${escapeHtml(e)}</li>`).join("")}</ul>
        </div>` : ""}

      ${(payload.planner_messages || []).map((msg, i) =>
        renderCollapsibleText(
          `Prompt — ${escapeHtml(msg.role || `message ${i + 1}`)}`,
          msg.content && msg.content.text
        )
      ).join("")}

      ${renderCollapsibleText("Planner Response", payload.planner_response)}
      ${renderCollapsibleText("Repair Prompt", payload.planner_repair_prompt)}
      ${renderCollapsibleText("Repair Response", payload.planner_repair_response)}
    `;
  }

  if (detailType === "iteration") {
    const duration = formatDuration(payload.started_at, payload.ended_at);
    const evidenceDelta = payload.evidence_count_after - payload.evidence_count_before;
    const errors = payload.new_errors || [];
    const decision = payload.decision || {};
    const checkUpdates = payload.check_updates || [];

    return `
      <h3>Iteration ${escapeHtml(String(payload.iteration ?? "?"))}</h3>

      <p>
        ${duration ? `<strong>Duration:</strong> ${escapeHtml(duration)} &nbsp;|&nbsp; ` : ""}
        <strong>Node:</strong> ${escapeHtml(payload.node_before || "?")} → ${escapeHtml(payload.node_after || "?")}
        &nbsp;|&nbsp; <strong>Evidence:</strong> ${payload.evidence_count_before} → ${payload.evidence_count_after} (+${evidenceDelta})
        ${errors.length ? ` &nbsp;|&nbsp; <strong>Errors:</strong> ${errors.length}` : ""}
      </p>

      ${decision.decision_type ? `
        <div class="detail-section">
          <strong>Decision:</strong> ${escapeHtml(decision.decision_type)}
          ${decision.target_node_id ? ` → ${escapeHtml(decision.target_node_id)}` : ""}
          ${decision.rationale ? `<p>${escapeHtml(decision.rationale)}</p>` : ""}
        </div>` : ""}

      ${checkUpdates.length ? `
        <div class="detail-section">
          <strong>Check updates</strong>
          <ul>${checkUpdates.map((u) => `<li><strong>${escapeHtml(u.id || "")}</strong>: ${escapeHtml(u.status || "")} — ${escapeHtml(u.reason || "")}</li>`).join("")}</ul>
        </div>` : ""}

      ${decision.final_answer ? renderCollapsibleText("Final Answer", decision.final_answer) : ""}

      ${errors.length ? `
        <div class="detail-section">
          <strong>Errors</strong>
          <ul>${errors.map((e) => `<li>${escapeHtml(e)}</li>`).join("")}</ul>
        </div>` : ""}

      ${(payload.planner_messages || []).map((msg, i) =>
        renderCollapsibleText(
          `Prompt — ${escapeHtml(msg.role || `message ${i + 1}`)}`,
          msg.content && msg.content.text
        )
      ).join("")}

      ${renderCollapsibleText("Planner Response", payload.planner_response)}
    `;
  }

  if (detailType === "childrun") {
    const duration = formatDuration(payload.started_at, payload.ended_at);
    const summary = payload.summary || {};
    const errors = summary.errors || [];
    const seenQueries = summary.seen_queries || [];

    return `
      <h3>${escapeHtml(payload.agent || "Web Search Run")}</h3>

      <p>
        ${duration ? `<strong>Duration:</strong> ${escapeHtml(duration)} &nbsp;|&nbsp; ` : ""}
        <strong>Status:</strong> ${escapeHtml(payload.status || "unknown")}
        ${summary.evidence_count != null ? ` &nbsp;|&nbsp; <strong>Evidence:</strong> ${summary.evidence_count}` : ""}
        ${summary.message_count != null ? ` &nbsp;|&nbsp; <strong>Messages:</strong> ${summary.message_count}` : ""}
        ${errors.length ? ` &nbsp;|&nbsp; <strong>Errors:</strong> ${errors.length}` : ""}
      </p>

      ${errors.length ? `
        <div class="detail-section">
          <strong>Errors</strong>
          <ul>${errors.map((e) => `<li>${escapeHtml(e)}</li>`).join("")}</ul>
        </div>` : ""}

      ${seenQueries.length ? `
        <div class="detail-section">
          <strong>Queries issued</strong>
          <ul>${seenQueries.map((q) => `<li>${escapeHtml(q)}</li>`).join("")}</ul>
        </div>` : ""}

      ${renderCollapsibleMultimodal("Goal", payload.goal)}
      ${renderCollapsibleMultimodal("Result", summary.result && summary.result.result)}
    `;
  }

  if (detailType === "task") {
    const result = payload.result || {};
    const errors = result.errors || [];
    const evidenceCount = Array.isArray(result.evidences) ? result.evidences.length : 0;

    return `
      <h3>${escapeHtml(payload.task_id || "Delegated Task")}</h3>

      <p>
        <strong>Agent:</strong> ${escapeHtml(payload.agent_type || "unknown")}
        ${result.status != null ? ` &nbsp;|&nbsp; <strong>Status:</strong> ${escapeHtml(result.status)}` : ""}
        ` + (evidenceCount != null ? ` &nbsp;|&nbsp; <strong>Evidence:</strong> ${evidenceCount}` : ``) + `
        ${result.message_count != null ? ` &nbsp;|&nbsp; <strong>Messages:</strong> ${result.message_count}` : ""}
        ${errors.length ? ` &nbsp;|&nbsp; <strong>Errors:</strong> ${errors.length}` : ""}
      </p>

      ${errors.length ? `
        <div class="detail-section">
          <strong>Errors</strong>
          <ul>${errors.map((e) => `<li>${escapeHtml(e)}</li>`).join("")}</ul>
        </div>` : ""}

      ${renderCollapsibleText("Instruction", payload.instruction)}
      ${renderCollapsibleMultimodal("Result", result.result)}
    `;
  }

  if (detailType === "web_search_result") {
    const errors = payload.errors || [];
    const evidences = payload.evidences || [];
    return `
      <h3>Search Result</h3>
      <p>
        <strong>Evidence:</strong> ${evidences.length}
        ${errors.length ? ` &nbsp;|&nbsp; <strong>Errors:</strong> ${errors.length}` : ""}
      </p>

      ${errors.length ? `
        <div class="detail-section">
          <strong>Errors</strong>
          <ul>${errors.map((e) => `<li>${escapeHtml(e)}</li>`).join("")}</ul>
        </div>` : ""}

      ${renderCollapsibleText("Summary", payload.answer)}

      <div class="evidence-list">
        ${evidences.map((ev, i) => `
          <div class="evidence-card">
            <div class="evidence-header">
              <span class="evidence-index">#${i + 1}</span>
              <span class="evidence-source">${escapeHtml(ev.source || "Unknown source")}</span>
            </div>
            ${ev.preview ? `<p class="evidence-preview">${escapeHtml(ev.preview)}</p>` : ""}
            ${renderCollapsibleMultimodal("Takeaways", ev.takeaways)}
            ${renderCollapsibleMultimodal("Raw", ev.raw)}
          </div>`).join("")}
      </div>
    `;
  }

  if (detailType === "media_planner") {
    return `
      <h3>Media Planner</h3>
      <p><strong>Planned tools:</strong> ${escapeHtml((payload.planned_tools || []).join(", ") || "none")}</p>

      ${(payload.planner_messages || []).map((msg, i) =>
        renderCollapsibleText(
          `Prompt — ${escapeHtml(msg.role || `message ${i + 1}`)}`,
          msg.content && msg.content.text
        )
      ).join("")}

      ${renderCollapsibleText("Planner Response", payload.planner_response)}
    `;
  }

  if (detailType === "media_tool_result") {
    const sources = payload.sources || [];
    const toolLabel =
      payload.tool === "reverse_image_search"
        ? "Reverse Image Search"
        : payload.tool === "geolocate"
        ? "Geolocate"
        : escapeHtml(payload.tool || "Tool");
    return `
      <h3>${toolLabel}</h3>
      <p><strong>Sources found:</strong> ${sources.length}</p>
      ${renderCollapsibleMultimodal("Takeaways", payload.takeaways)}
      ${renderCollapsibleText("Raw", payload.raw_text)}
    `;
  }

  if (detailType === "media_source_url") {
    const source = payload.source || {};
    return `
      <h3>${escapeHtml(source.title || source.url || source.reference || "Source")}</h3>
      ${source.url ? `<p><strong>URL:</strong> ${escapeHtml(source.url)}</p>` : ""}
      ${source.reference && source.reference !== source.url ? `<p><strong>Reference:</strong> ${escapeHtml(source.reference)}</p>` : ""}
      ${renderCollapsibleMultimodal("Takeaways", source.takeaways)}
    `;
  }

  if (detailType === "media_synthesis") {
    return `
      <h3>Media Synthesis</h3>
      <p><strong>Evidence used:</strong> ${payload.evidence_count ?? 0}</p>
      ${renderCollapsibleText("Answer", payload.answer)}
    `;
  }

  if (detailType === "judge_decision") {
    const decision = payload.decision || {};
    const summary = payload.summary || {};
    const label = decision.label || summary.label;
    const justification = decision.justification || summary.justification;
    const evidenceCount = payload.evidence_count ?? summary.evidence_count ?? 0;
    const duration = formatDuration(payload.started_at, payload.ended_at);
    const errors = summary.errors || [];

    return `
      <h3>Judge Decision</h3>
      <p>
        ${duration ? `<strong>Duration:</strong> ${escapeHtml(duration)} &nbsp;|&nbsp; ` : ""}
        ${label ? `<strong>Label:</strong> ${escapeHtml(label)} &nbsp;|&nbsp; ` : ""}
        <strong>Evidence:</strong> ${evidenceCount}
        ${errors.length ? ` &nbsp;|&nbsp; <strong>Errors:</strong> ${errors.length}` : ""}
      </p>

      ${errors.length ? `
        <div class="detail-section">
          <strong>Errors</strong>
          <ul>${errors.map((e) => `<li>${escapeHtml(e)}</li>`).join("")}</ul>
        </div>` : ""}

      ${justification ? renderCollapsibleText("Justification", justification) : ""}

      ${(payload.prompt_messages || []).map((msg, i) =>
        renderCollapsibleText(
          `Prompt — ${escapeHtml(msg.role || `message ${i + 1}`)}`,
          msg.content && msg.content.text
        )
      ).join("")}

      ${renderCollapsibleText("Model Response", payload.model_response)}
      ${renderCollapsibleText("Repair Prompt", payload.repair_prompt)}
      ${renderCollapsibleText("Repair Response", payload.repair_response)}
    `;
  }

  if (detailType === "child_result") {
    const errors = payload.errors || [];
    const evidences = payload.evidences || [];
    return `
      <h3>Result</h3>
      <p>
        <strong>Evidence:</strong> ${evidences.length}
        ${errors.length ? ` &nbsp;|&nbsp; <strong>Errors:</strong> ${errors.length}` : ""}
      </p>

      ${errors.length ? `
        <div class="detail-section">
          <strong>Errors</strong>
          <ul>${errors.map((e) => `<li>${escapeHtml(e)}</li>`).join("")}</ul>
        </div>` : ""}

      ${renderCollapsibleText("Answer", payload.answer)}
    `;
  }

  if (detailType === "iteration_synthesis") {
    return `
      <h3>Iteration Synthesis</h3>
      <p><strong>Stage:</strong> ${escapeHtml(payload.stage || "")}</p>
      ${payload.instruction ? `<p><strong>Instruction:</strong> ${escapeHtml(payload.instruction)}</p>` : ""}
      <p><strong>Answer:</strong> ${escapeHtml(payload.answer || "")}</p>
    `;
  }

  return `<pre>${escapeHtml(JSON.stringify(payload, null, 2))}</pre>`;
}

function formatDuration(startedAt, endedAt) {
  if (!startedAt || !endedAt) return null;
  const ms = new Date(endedAt) - new Date(startedAt);
  if (isNaN(ms) || ms < 0) return null;
  if (ms < 1000) return `${ms}ms`;
  if (ms < 60000) return `${(ms / 1000).toFixed(1)}s`;
  const m = Math.floor(ms / 60000);
  const s = Math.floor((ms % 60000) / 1000);
  return `${m}m ${s}s`;
}

function renderCollapsibleText(label, value) {
  if (!value) return "";
  return `
    <details>
      <summary>${escapeHtml(label)}</summary>
      <pre>${escapeHtml(value)}</pre>
    </details>
  `;
}

function renderCollapsibleMultimodal(label, value) {
  const text = value && value.text ? value.text : "";
  if (!text) {
    return "";
  }
  return `
    <details>
      <summary>${escapeHtml(label)}</summary>
      <pre>${escapeHtml(text)}</pre>
    </details>
  `;
}
