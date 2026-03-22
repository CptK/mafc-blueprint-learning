import { escapeHtml, humanizeToolName } from "./utils.js";

export function renderDetail(node) {
  const payload = node.payload || {};
  const detailType = payload.detailType || node.type;

  if (detailType === "claim") {
    const images = payload.images || [];
    const videos = payload.videos || [];

    // Annotate claim text: replace <image:N> / <video:N> with styled badges
    const rawText = payload.text || "";
    const annotatedText = escapeHtml(rawText).replace(
      /&lt;(image|video):(\d+)&gt;/g,
      (_, kind, id) =>
        `<span class="media-ref-badge">${kind === "video" ? "▶" : "◉"} ${kind}:${id}</span>`
    );

    const meta = [
      payload.id ? ["ID", payload.id] : null,
      payload.dataset ? ["Dataset", payload.dataset] : null,
      payload.date ? ["Date", payload.date.replace("T00:00:00", "")] : null,
      payload.author ? ["Author", payload.author] : null,
      payload.origin ? ["Origin", payload.origin] : null,
    ].filter(Boolean);

    const mediaHtml = renderMediaGrid(images, videos);

    return `
      <h3>Claim</h3>

      <div class="claim-text">${annotatedText || "<em>No claim text.</em>"}</div>

      ${mediaHtml}

      ${meta.length ? `<div class="claim-meta">${meta.map(([label, val]) =>
        `<span class="claim-meta-item"><span class="claim-meta-label">${escapeHtml(label)}</span>${escapeHtml(String(val))}</span>`
      ).join("")}</div>` : ""}

      ${payload.meta_info ? renderCollapsibleText("Meta Info", typeof payload.meta_info === "string" ? payload.meta_info : JSON.stringify(payload.meta_info, null, 2)) : ""}
    `;
  }

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
      ${renderCollapsibleText("Preview", (payload.evidence && payload.evidence.preview) || (payload.source && payload.source.preview))}
      ${!payload.irrelevant ? renderCollapsibleMultimodal("Takeaways", payload.evidence && payload.evidence.takeaways) : ""}
      ${renderCollapsibleMultimodal("Raw", (payload.evidence && payload.evidence.raw) || (payload.retrieved_content ? { text: payload.retrieved_content } : null))}
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

      ${renderErrorList(errors)}

      ${(payload.planner_messages || []).map((msg, i) =>
        renderCollapsibleMultimodal(
          `Prompt — ${escapeHtml(msg.role || `message ${i + 1}`)}`,
          msg.content,
          { textFirst: true }
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
    const routing = payload.routing || null;
    const routingDecision = (routing && routing.decision) || {};
    const allCheckUpdates = [
      ...checkUpdates,
      ...((routingDecision.check_updates) || []),
    ];

    return `
      <h3>Iteration ${escapeHtml(String(payload.iteration ?? "?"))}</h3>

      <p>
        ${duration ? `<strong>Duration:</strong> ${escapeHtml(duration)} &nbsp;|&nbsp; ` : ""}
        ${payload.execution_type ? `<strong>Type:</strong> ${escapeHtml(payload.execution_type)} &nbsp;|&nbsp; ` : ""}
        <strong>Node:</strong> ${escapeHtml(payload.node_before || "?")} → ${escapeHtml(payload.node_after || "?")}
        &nbsp;|&nbsp; <strong>Evidence:</strong> ${payload.evidence_count_before} → ${payload.evidence_count_after} (+${evidenceDelta})
        ${errors.length ? ` &nbsp;|&nbsp; <strong>Errors:</strong> ${errors.length}` : ""}
      </p>

      ${decision.decision_type ? `
        <div class="detail-section">
          <strong>Execution decision:</strong> ${escapeHtml(decision.decision_type)}
          ${decision.rationale ? `<p>${escapeHtml(decision.rationale)}</p>` : ""}
        </div>` : ""}

      ${routing ? `
        <div class="detail-section">
          <strong>Routing:</strong> ${escapeHtml(routing.type || "unknown")} → ${escapeHtml(routing.target_node_id || "?")}
          ${routingDecision.rationale ? `<p>${escapeHtml(routingDecision.rationale)}</p>` : ""}
        </div>` : ""}

      ${allCheckUpdates.length ? `
        <div class="detail-section">
          <strong>Check updates</strong>
          <ul>${allCheckUpdates.map((u) => `<li><strong>${escapeHtml(u.id || "")}</strong>: ${escapeHtml(u.status || "")} — ${escapeHtml(u.reason || "")}</li>`).join("")}</ul>
        </div>` : ""}

      ${decision.final_answer ? renderCollapsibleText("Final Answer", decision.final_answer) : ""}
      ${routingDecision.final_answer ? renderCollapsibleText("Final Answer (routing)", routingDecision.final_answer) : ""}

      ${renderErrorList(errors)}

      ${(payload.planner_messages || []).map((msg, i) =>
        renderCollapsibleMultimodal(
          `Execution prompt — ${escapeHtml(msg.role || `message ${i + 1}`)}`,
          msg.content,
          { textFirst: true }
        )
      ).join("")}

      ${renderCollapsibleText("Execution response", payload.planner_response)}

      ${routing && routing.type === "llm" ? `
        ${((routing.messages) || []).map((msg, i) =>
          renderCollapsibleMultimodal(
            `Routing prompt — ${escapeHtml(msg.role || `message ${i + 1}`)}`,
            msg.content,
            { textFirst: true }
          )
        ).join("")}
        ${renderCollapsibleText("Routing response", routing.response)}
      ` : ""}
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

      ${renderErrorList(errors)}

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

      ${renderErrorList(errors)}

      ${renderCollapsibleTextWithMedia("Instruction", payload.instruction)}
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

      ${renderErrorList(errors)}

      ${renderCollapsibleMultimodal("Summary", payload.result || (payload.answer ? { text: payload.answer } : null))}

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
        renderCollapsibleMultimodal(
          `Prompt — ${escapeHtml(msg.role || `message ${i + 1}`)}`,
          msg.content,
          { textFirst: true }
        )
      ).join("")}

      ${renderCollapsibleText("Planner Response", payload.planner_response)}
    `;
  }

  if (detailType === "media_tool_result") {
    const sources = payload.sources || [];
    const toolLabel = escapeHtml(humanizeToolName(payload.tool));
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

      ${renderErrorList(errors)}

      ${justification ? renderCollapsibleText("Justification", justification) : ""}

      ${(payload.prompt_messages || []).map((msg, i) =>
        renderCollapsibleMultimodal(
          `Prompt — ${escapeHtml(msg.role || `message ${i + 1}`)}`,
          msg.content,
          { textFirst: true }
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

      ${renderErrorList(errors)}

      ${renderCollapsibleMultimodal("Answer", payload.result || (payload.answer ? { text: payload.answer } : null))}
    `;
  }

  if (detailType === "run_summary") {
    const errors = payload.errors || [];
    const correctIcon = payload.correct === true ? "✓ correct" : payload.correct === false ? "✗ incorrect" : null;
    const totalTokens = (payload.total_input_tokens ?? 0) + (payload.total_output_tokens ?? 0);
    const byModel = payload.by_model || {};
    const modelNames = Object.keys(byModel);
    return `
      <h3>Run Summary</h3>

      <p>
        ${payload.predicted_label ? `<strong>Predicted label:</strong> ${escapeHtml(payload.predicted_label)}` : "<strong>Predicted label:</strong> <em>none</em>"}
        ${payload.true_label ? ` &nbsp;|&nbsp; <strong>True label:</strong> ${escapeHtml(payload.true_label)}` : ""}
        ${correctIcon ? ` &nbsp;|&nbsp; <strong>${escapeHtml(correctIcon)}</strong>` : ""}
      </p>

      <p>
        ${payload.runtime_seconds != null ? `<strong>Runtime:</strong> ${(payload.runtime_seconds / 60).toFixed(1)}min &nbsp;|&nbsp; ` : ""}
        ${payload.evidence_count != null ? `<strong>Evidence:</strong> ${payload.evidence_count}` : ""}
        ${errors.length ? ` &nbsp;|&nbsp; <strong>Errors:</strong> ${errors.length}` : ""}
      </p>

      <div class="detail-section">
        <strong>LLM Usage (total)</strong>
        <table style="margin-top:4px;border-collapse:collapse;width:100%">
          <thead>
            <tr>
              <th style="text-align:left;padding:2px 8px 2px 0">Model</th>
              <th style="text-align:right;padding:2px 8px">Input tok</th>
              <th style="text-align:right;padding:2px 8px">Output tok</th>
              <th style="text-align:right;padding:2px 8px">Total tok</th>
              <th style="text-align:right;padding:2px 0">Cost (USD)</th>
            </tr>
          </thead>
          <tbody>
            ${modelNames.map((name) => {
              const m = byModel[name];
              const mTotal = (m.input_tokens || 0) + (m.output_tokens || 0);
              return `<tr>
                <td style="padding:2px 8px 2px 0">${escapeHtml(name)}</td>
                <td style="text-align:right;padding:2px 8px">${(m.input_tokens || 0).toLocaleString()}</td>
                <td style="text-align:right;padding:2px 8px">${(m.output_tokens || 0).toLocaleString()}</td>
                <td style="text-align:right;padding:2px 8px">${mTotal.toLocaleString()}</td>
                <td style="text-align:right;padding:2px 0">$${(m.cost_usd || 0).toFixed(4)}</td>
              </tr>`;
            }).join("")}
            ${modelNames.length > 1 ? `<tr style="border-top:1px solid #ccc;font-weight:bold">
              <td style="padding:2px 8px 2px 0">Total</td>
              <td style="text-align:right;padding:2px 8px">${(payload.total_input_tokens || 0).toLocaleString()}</td>
              <td style="text-align:right;padding:2px 8px">${(payload.total_output_tokens || 0).toLocaleString()}</td>
              <td style="text-align:right;padding:2px 8px">${totalTokens.toLocaleString()}</td>
              <td style="text-align:right;padding:2px 0">$${(payload.total_cost_usd || 0).toFixed(4)}</td>
            </tr>` : ""}
          </tbody>
        </table>
      </div>

      ${renderErrorList(errors)}
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

  if (detailType === "blueprint") {
    const sel = payload.selection || {};
    const mode = sel.mode || "unknown";
    const modeColors = {
      rule_based: "#2a7a2a",
      llm_tiebreak: "#7a5c00",
      default_fallback: "#7a2a2a",
    };
    const modeColor = modeColors[mode] || "#555";
    const modeBadge = `<span style="display:inline-block;padding:1px 7px;border-radius:3px;font-size:0.85em;font-weight:bold;background:${modeColor};color:#fff">${escapeHtml(mode)}</span>`;

    const allBlueprints = sel.all_blueprints || [];
    const survivingSet = new Set(sel.surviving_blueprints || []);
    const rejectedMap = {};
    (sel.rejected_blueprints || []).forEach((r) => { rejectedMap[r.blueprint_name] = r.reason; });

    const bpRows = allBlueprints.map((name) => {
      const passed = survivingSet.has(name);
      const isSelected = name === payload.name;
      const icon = passed ? "✓" : "✗";
      const iconColor = passed ? "#2a7a2a" : "#aa3333";
      const reason = passed
        ? (isSelected ? "<strong>selected</strong>" : "<em>passed rule filter</em>")
        : escapeHtml(rejectedMap[name] || "");
      return `<tr>
        <td style="padding:2px 6px 2px 0;white-space:nowrap"><span style="color:${iconColor};font-weight:bold">${icon}</span> ${escapeHtml(name)}</td>
        <td style="padding:2px 0;color:#555;font-size:0.9em">${reason}</td>
      </tr>`;
    }).join("");

    const features = sel.claim_features || {};
    const featureRows = Object.entries(features).sort(([a], [b]) => a.localeCompare(b)).map(([k, v]) =>
      `<tr><td style="padding:2px 8px 2px 0;white-space:nowrap">${escapeHtml(k)}</td><td style="color:#555">${escapeHtml(String(v))}</td></tr>`
    ).join("");

    return `
      <h3>Blueprint: ${escapeHtml(payload.name || "unknown")} &nbsp;${modeBadge}</h3>

      ${sel.reason ? `<p>${escapeHtml(sel.reason)}</p>` : ""}

      ${allBlueprints.length ? `
      <details open>
        <summary><strong>Rule filtering</strong> — ${allBlueprints.length} evaluated, ${survivingSet.size} passed</summary>
        <table style="width:100%;border-collapse:collapse;margin-top:6px">
          <tbody>${bpRows}</tbody>
        </table>
      </details>` : ""}

      ${Object.keys(features).length ? `
      <details>
        <summary><strong>Claim features</strong></summary>
        <table style="border-collapse:collapse;margin-top:6px">
          <tbody>${featureRows}</tbody>
        </table>
      </details>` : ""}

      ${renderCollapsibleText("LLM Tiebreak Prompt", sel.llm_prompt)}
      ${renderCollapsibleText("LLM Tiebreak Raw Response", sel.llm_raw_response)}
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

function renderCollapsibleTextWithMedia(label, text) {
  if (!text) return "";
  const images = [];
  const videos = [];
  const re = /<(image|video):(\d+)>/g;
  let m;
  while ((m = re.exec(text)) !== null) {
    const ref = m[0];
    if (m[1] === "image" && !images.includes(ref)) images.push(ref);
    if (m[1] === "video" && !videos.includes(ref)) videos.push(ref);
  }
  const mediaHtml = renderMediaGrid(images, videos);
  return `
    <details>
      <summary>${escapeHtml(label)}</summary>
      ${mediaHtml}
      <pre>${escapeHtml(text)}</pre>
    </details>
  `;
}

function mediaId(ref) {
  const m = String(ref).match(/:(\d+)>/);
  return m ? m[1] : null;
}

function renderMediaGrid(images, videos) {
  const items = [
    ...(images || []).map((ref) => ({ kind: "image", ref })),
    ...(videos || []).map((ref) => ({ kind: "video", ref })),
  ];
  if (!items.length) return "";
  return `<div class="claim-media-grid">${items.map(({ kind, ref }) => {
    const id = mediaId(ref);
    if (!id) return "";
    const url = `/api/media/${kind}/${id}`;
    if (kind === "image") {
      return `<a href="${url}" target="_blank"><img class="claim-media-img" src="${url}" alt="${escapeHtml(ref)}" /></a>`;
    }
    return `<video class="claim-media-video" src="${url}" controls></video>`;
  }).join("")}</div>`;
}

function renderErrorList(errors) {
  if (!errors.length) return "";
  return `
    <div class="detail-section">
      <strong>Errors</strong>
      <ul>${errors.map((e) => `<li>${escapeHtml(e)}</li>`).join("")}</ul>
    </div>`;
}

function renderCollapsibleMultimodal(label, value, { textFirst = false } = {}) {
  const text = (value && value.text) || "";
  const images = (value && value.images) || [];
  const videos = (value && value.videos) || [];
  if (!text && !images.length && !videos.length) return "";
  const mediaHtml = renderMediaGrid(images, videos);
  const textHtml = text ? `<pre>${escapeHtml(text)}</pre>` : "";
  return `
    <details>
      <summary>${escapeHtml(label)}</summary>
      ${textFirst ? textHtml + mediaHtml : mediaHtml + textHtml}
    </details>
  `;
}
