import { SAMPLE_TRACE } from "./sample-trace.js";
import { buildViewModel } from "./view-model.js";
import { renderDetail } from "./details.js";
import { renderSummary, createGraph } from "./graph.js";
import { escapeHtml } from "./utils.js";
import { setupDividerDrag } from "./ui.js";

cytoscape.use(cytoscapeDagre);

const graphEl = document.getElementById("graph");
const summaryEl = document.getElementById("summary");
const detailMetaEl = document.getElementById("detailMeta");
const detailBodyEl = document.getElementById("detailBody");
const statusEl = document.getElementById("status");
const fileInputEl = document.getElementById("fileInput");
const loadSampleBtnEl = document.getElementById("loadSampleBtn");
const appEl = document.querySelector(".app");
const dividerEl = document.getElementById("divider");

let cy = null;
let currentModel = null;

fileInputEl.addEventListener("change", async (event) => {
  const file = event.target.files && event.target.files[0];
  if (!file) {
    return;
  }
  try {
    const text = await file.text();
    renderTrace(JSON.parse(text), file.name);
  } catch (error) {
    statusEl.textContent = `Failed to read trace: ${error.message}`;
  }
});

loadSampleBtnEl.addEventListener("click", () => renderTrace(SAMPLE_TRACE, "embedded-example.json"));

setupDividerDrag(dividerEl, appEl, () => cy);

function renderTrace(trace, label) {
  currentModel = buildViewModel(trace);
  renderSummary(summaryEl, trace, label);
  cy = createGraph(graphEl, currentModel, (nodeId) => selectNode(nodeId), cy);
  statusEl.textContent = `Loaded ${label}`;
  if (currentModel.nodes.length > 0) {
    selectNode(currentModel.nodes[0].id);
  }
}

function selectNode(nodeId) {
  const node = currentModel && currentModel.nodeById[nodeId];
  if (!node) {
    detailMetaEl.innerHTML = "";
    detailBodyEl.innerHTML = `<div class="empty">Unknown node.</div>`;
    return;
  }

  const meta = [
    ["Type", node.type],
    ["Node", node.id],
  ];
  if (node.payload && typeof node.payload.iteration === "number") {
    meta.push(["Iteration", String(node.payload.iteration)]);
  }
  if (node.payload && typeof node.payload.step === "number") {
    meta.push(["Step", String(node.payload.step)]);
  }

  detailMetaEl.innerHTML = meta
    .map(([key, value]) => `<span class="pill">${escapeHtml(key)}: ${escapeHtml(value)}</span>`)
    .join("");
  detailBodyEl.innerHTML = renderDetail(node);
}
