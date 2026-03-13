import { escapeHtml } from "./utils.js";

export function renderDetail(node) {
  const payload = node.payload || {};
  const detailType = payload.detailType || node.type;

  if (detailType === "candidate_url") {
    return `
      <h3>${escapeHtml(payload.source.title || payload.source.url || "Candidate URL")}</h3>
      <p><strong>Query:</strong> ${escapeHtml(payload.query_text || "")}</p>
      <p><strong>URL:</strong> ${escapeHtml(payload.source.url || payload.source.reference || "")}</p>
      ${payload.source.preview ? `<p><strong>Preview:</strong> ${escapeHtml(payload.source.preview)}</p>` : ""}
      ${payload.source.release_date ? `<p><strong>Release date:</strong> ${escapeHtml(payload.source.release_date)}</p>` : ""}
    `;
  }

  if (detailType === "retrieval_url") {
    return `
      <h3>${escapeHtml((payload.source && (payload.source.title || payload.source.url)) || "Retrieved URL")}</h3>
      ${payload.query_text ? `<p><strong>Query:</strong> ${escapeHtml(payload.query_text)}</p>` : ""}
      <p><strong>URL:</strong> ${escapeHtml((payload.source && (payload.source.url || payload.source.reference)) || "")}</p>
      ${payload.snippet ? `<p><strong>Snippet:</strong> ${escapeHtml(payload.snippet)}</p>` : ""}
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
