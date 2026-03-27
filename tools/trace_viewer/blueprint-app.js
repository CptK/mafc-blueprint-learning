import { escapeHtml } from "./utils.js";
import { getCssVar } from "./constants.js";
import { setupDividerDrag } from "./ui.js";

cytoscape.use(cytoscapeDagre);

const selectEl = document.getElementById("blueprintSelect");
const statusEl = document.getElementById("status");
const graphEl = document.getElementById("graph");
const metaEl = document.getElementById("blueprintMeta");
const nodeDetailEl = document.getElementById("nodeDetail");
const appEl = document.querySelector(".app");
const dividerEl = document.getElementById("divider");

let cy = null;
let currentBlueprint = null;

// ── Node colours (mirrors trace viewer CSS vars) ─────────────────────────────

const NODE_COLORS = {
  actions: () => getCssVar("--task"),
  synthesis: () => getCssVar("--iteration"),
  gate: () => getCssVar("--result"),
};

// ── Init ─────────────────────────────────────────────────────────────────────

async function init() {
  try {
    const res = await fetch("/api/blueprints");
    const list = await res.json();
    for (const { name, description } of list) {
      const opt = document.createElement("option");
      opt.value = name;
      opt.textContent = name;
      opt.title = description;
      selectEl.appendChild(opt);
    }
    if (list.length === 0) {
      statusEl.textContent = "No blueprints found.";
    }
  } catch (err) {
    statusEl.textContent = `Failed to load blueprint list: ${err.message}`;
  }
}

selectEl.addEventListener("change", async () => {
  const name = selectEl.value;
  if (!name) return;
  try {
    statusEl.textContent = "Loading…";
    const res = await fetch(`/api/blueprints/${encodeURIComponent(name)}`);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const bp = await res.json();
    currentBlueprint = bp;
    renderBlueprint(bp);
    statusEl.textContent = bp.name;
  } catch (err) {
    statusEl.textContent = `Error: ${err.message}`;
  }
});

// ── Divider drag ──────────────────────────────────────────────────────────────

setupDividerDrag(dividerEl, appEl, () => cy);

// ── Render ────────────────────────────────────────────────────────────────────

function renderBlueprint(bp) {
  renderSidebarMeta(bp);
  renderGraph(bp);
  nodeDetailEl.innerHTML = `<div class="empty">Click a graph node to see details.</div>`;
}

function renderSidebarMeta(bp) {
  const policy = bp.policy_constraints || {};
  const entryAll = (bp.entry_conditions && bp.entry_conditions.all) || [];
  const entryAny = (bp.entry_conditions && bp.entry_conditions.any) || [];
  const checks = bp.required_checks || [];
  const hints = bp.selector_hints || {};
  const positiveExamples = (hints.positive && hints.positive.examples) || [];
  const negativeExamples = (hints.negative && hints.negative.examples) || [];
  const positiveFeatures = (hints.positive && hints.positive.features) || [];
  const negativeFeatures = (hints.negative && hints.negative.features) || [];
  const policyPills = [
    `max ${policy.max_iterations ?? "?"} iterations`,
    ...(policy.allowed_actions || []).map((a) => a),
    policy.require_counterevidence_search ? "counter-evidence required" : null,
  ]
    .filter(Boolean)
    .map((t) => `<span class="pill">${escapeHtml(String(t))}</span>`)
    .join("");

  const conditionHtml = (cond) =>
    `<div class="condition-row">${escapeHtml(cond.feature)} <strong>${escapeHtml(cond.op)}</strong> ${escapeHtml(String(cond.value))}</div>`;

  const entryHtml =
    entryAll.length === 0 && entryAny.length === 0
      ? `<p class="subtle" style="font-size:13px">No entry conditions (fallback blueprint).</p>`
      : `
        ${entryAll.length ? `<div class="condition-label">All required</div>${entryAll.map(conditionHtml).join("")}` : ""}
        ${entryAny.length ? `<div class="condition-label" style="margin-top:8px">At least one of</div>${entryAny.map(conditionHtml).join("")}` : ""}
      `;

  const checksHtml =
    checks.length === 0
      ? `<p class="subtle" style="font-size:13px">None defined.</p>`
      : `<ul class="check-list">${checks
          .map(
            (c) =>
              `<li><span class="check-id">${escapeHtml(c.id)}</span>${escapeHtml(c.description)}</li>`
          )
          .join("")}</ul>`;

  const hintsHtml =
    positiveExamples.length === 0 && negativeExamples.length === 0
      ? ""
      : `
      <details>
        <summary style="font-size:13px;font-weight:600">Selector hints</summary>
        ${
          positiveFeatures.length
            ? `<p style="font-size:12px;margin:8px 0 2px"><strong>Positive features:</strong> ${positiveFeatures.map((f) => `<span class="check-id">${escapeHtml(f)}</span>`).join(" ")}</p>`
            : ""
        }
        ${
          positiveExamples.length
            ? `<p style="font-size:12px;font-weight:600;margin:8px 0 4px">Positive examples</p><ul style="font-size:12px;padding-left:16px;margin:0">${positiveExamples.map((e) => `<li>${escapeHtml(e)}</li>`).join("")}</ul>`
            : ""
        }
        ${
          negativeFeatures.length
            ? `<p style="font-size:12px;margin:10px 0 2px"><strong>Negative features:</strong> ${negativeFeatures.map((f) => `<span class="check-id">${escapeHtml(f)}</span>`).join(" ")}</p>`
            : ""
        }
        ${
          negativeExamples.length
            ? `<p style="font-size:12px;font-weight:600;margin:8px 0 4px">Negative examples</p><ul style="font-size:12px;padding-left:16px;margin:0">${negativeExamples.map((e) => `<li>${escapeHtml(e)}</li>`).join("")}</ul>`
            : ""
        }
      </details>
    `;

  metaEl.innerHTML = `
    <p class="bp-description">${escapeHtml(bp.description || "")}</p>
    <div class="pillbar">${policyPills}</div>

    <div class="bp-section">
      <h3>Entry Conditions</h3>
      ${entryHtml}
    </div>

    <div class="bp-section">
      <h3>Required Checks</h3>
      ${checksHtml}
    </div>

    ${hintsHtml}
  `;
}

// ── Graph ─────────────────────────────────────────────────────────────────────

function buildGraphElements(bp) {
  const graph = bp.verification_graph || {};
  const nodes = graph.nodes || [];
  const startNode = graph.start_node;
  const elements = [];

  for (const node of nodes) {
    const label = humanizeId(node.id) + "\n[" + capitalize(node.type) + "]";
    elements.push({
      data: {
        id: node.id,
        type: node.type,
        label,
        isStart: node.id === startNode ? "true" : "false",
      },
    });
  }

  let edgeIdx = 0;
  for (const node of nodes) {
    for (const t of node.transition || []) {
      elements.push({
        data: {
          id: `e${edgeIdx++}`,
          source: node.id,
          target: t.to,
        },
      });
    }
  }

  return elements;
}

function renderGraph(bp) {
  if (cy) { cy.destroy(); cy = null; }

  const elements = buildGraphElements(bp);
  if (elements.length === 0) return;

  cy = cytoscape({
    container: graphEl,
    elements,
    style: [
      {
        selector: "node",
        style: {
          shape: "round-rectangle",
          width: 200,
          height: "label",
          padding: "18px",
          "background-color": (ele) => (NODE_COLORS[ele.data("type")] || (() => "#ffffff"))(),
          "border-width": 1.5,
          "border-color": "rgba(32, 32, 32, 0.15)",
          label: "data(label)",
          color: "#202020",
          "font-size": 11,
          "font-family": 'Georgia, "Iowan Old Style", "Palatino Linotype", serif',
          "text-wrap": "wrap",
          "text-max-width": 170,
          "text-valign": "center",
          "text-halign": "center",
          "line-height": 1.35,
        },
      },
      {
        selector: 'node[isStart = "true"]',
        style: {
          "border-width": 3,
          "border-color": "#204d48",
        },
      },
      {
        selector: 'node[type = "gate"]',
        style: {
          shape: "round-rectangle",
          "border-width": 2,
          "border-color": "#4a7a5a",
        },
      },
      {
        selector: "node:selected",
        style: {
          "border-width": 3,
          "border-color": "#204d48",
        },
      },
      {
        selector: "edge",
        style: {
          width: 2,
          "line-color": "#927f62",
          "target-arrow-color": "#927f62",
          "target-arrow-shape": "triangle",
          "curve-style": "taxi",
          "taxi-direction": "downward",
          "taxi-turn": 30,
          label: "",
        },
      },
    ],
    layout: {
      name: "dagre",
      rankDir: "TB",
      nodeSep: 60,
      rankSep: 90,
      padding: 50,
      animate: false,
    },
    minZoom: 0.25,
    maxZoom: 3,
    wheelSensitivity: 0.18,
  });

  cy.on("tap", "node", (event) => selectNode(event.target.id()));
  cy.fit(undefined, 50);
}

// ── Node detail ───────────────────────────────────────────────────────────────

function selectNode(nodeId) {
  if (!currentBlueprint) return;
  const graph = currentBlueprint.verification_graph || {};
  const node = (graph.nodes || []).find((n) => n.id === nodeId);
  if (!node) return;

  if (cy) {
    cy.nodes().unselect();
    cy.getElementById(nodeId).select();
  }

  nodeDetailEl.innerHTML = renderNodeDetail(node, graph.start_node);
}

function renderNodeDetail(node, startNode) {
  const isStart = node.id === startNode;
  const startBadge = isStart ? ` <span class="check-id" style="background:var(--accent);color:#fff;border-color:var(--accent)">start</span>` : "";

  if (node.type === "actions") {
    const actionsHtml = (node.actions || [])
      .map(
        (a) => `
        <div class="node-action-card">
          <div class="node-action-name">${escapeHtml(a.action)}</div>
          ${a.intent ? `<div class="node-action-field"><strong>Intent:</strong> ${escapeHtml(a.intent)}</div>` : ""}
          ${a.query_guidance ? `<div class="node-action-field"><strong>Guidance:</strong> ${escapeHtml(a.query_guidance)}</div>` : ""}
        </div>`
      )
      .join("");

    const transitionsHtml = renderTransitions(node.transition);

    return `
      <h3>Actions${startBadge}</h3>
      <p class="subtle" style="font-size:12px;margin-bottom:8px">${escapeHtml(node.id)}</p>
      <div class="node-actions-list">${actionsHtml || "<p class='subtle' style='font-size:13px'>No actions defined.</p>"}</div>
      ${transitionsHtml}
    `;
  }

  if (node.type === "synthesis") {
    return `
      <h3>Synthesis${startBadge}</h3>
      <p class="subtle" style="font-size:12px;margin-bottom:8px">${escapeHtml(node.id)}</p>
      <p style="font-size:13px">Synthesizes accumulated evidence into an intermediate state before proceeding.</p>
      ${renderTransitions(node.transition)}
    `;
  }

  if (node.type === "gate") {
    const rules = node.rules || {};
    const supportChips = (rules.support_conditions || [])
      .map((c) => `<span class="condition-chip">${escapeHtml(c)}</span>`)
      .join("");
    const refuteChips = (rules.refute_conditions || [])
      .map((c) => `<span class="condition-chip refute">${escapeHtml(c)}</span>`)
      .join("");

    return `
      <h3>Gate${startBadge}</h3>
      <p class="subtle" style="font-size:12px;margin-bottom:8px">${escapeHtml(node.id)}</p>
      <div class="gate-conditions">
        <p style="font-size:13px"><strong>Support conditions</strong> (must all be satisfied):</p>
        <div>${supportChips || "<span class='subtle' style='font-size:12px'>none</span>"}</div>
        <p style="font-size:13px;margin-top:10px"><strong>Refute conditions:</strong></p>
        <div>${refuteChips || "<span class='subtle' style='font-size:12px'>none</span>"}</div>
        <p style="font-size:13px;margin-top:10px"><strong>If conditions not met:</strong> ${escapeHtml(rules.if_fail || "—")}</p>
      </div>
    `;
  }

  return `<pre>${escapeHtml(JSON.stringify(node, null, 2))}</pre>`;
}

function renderTransitions(transitions) {
  if (!transitions || transitions.length === 0) return "";
  const rows = transitions
    .map(
      (t) =>
        `<div class="termination-row"><span class="termination-if">${escapeHtml(t.if || "")}</span><span class="termination-return">&rarr; ${escapeHtml(t.to || "")}</span></div>`
    )
    .join("");
  return `<div class="bp-section" style="margin-top:14px"><h3>Transitions</h3>${rows}</div>`;
}

// ── Helpers ───────────────────────────────────────────────────────────────────

function humanizeId(id) {
  return id.replace(/_/g, " ");
}

function capitalize(str) {
  return str.charAt(0).toUpperCase() + str.slice(1);
}

// ── Start ─────────────────────────────────────────────────────────────────────

init();
