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
      ${payload.query_text ? `<p><strong>Query:</strong> ${escapeHtml(payload.query_text)}</p>` : ""}
      <p><strong>URL:</strong> ${escapeHtml((payload.source && (payload.source.url || payload.source.reference)) || "")}</p>
      ${renderCollapsibleText("Preview", payload.evidence && payload.evidence.preview)}
      ${renderCollapsibleMultimodal("Takeaways", payload.evidence && payload.evidence.takeaways)}
      ${renderCollapsibleMultimodal("Raw", payload.evidence && payload.evidence.raw)}
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
    return entries.join("") || "<div class=\"empty\">No selected sources recorded.</div>";
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
