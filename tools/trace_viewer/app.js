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
const sampleSelectEl = document.getElementById("sampleSelect");
const appEl = document.querySelector(".app");
const dividerEl = document.getElementById("divider");

const TRACES_ROOT = "../../traces";

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

sampleSelectEl.addEventListener("change", async () => {
  const entry = sampleSelectEl.selectedOptions[0];
  const file = entry && entry.dataset.file;
  if (!file) return;
  try {
    statusEl.textContent = "Loading…";
    const res = await fetch(`${TRACES_ROOT}/${file}`);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    renderTrace(await res.json(), file);
  } catch (err) {
    statusEl.textContent = `Failed to load trace: ${err.message}`;
  }
});

async function loadSampleIndex() {
  try {
    const res = await fetch(`${TRACES_ROOT}/index.json`);
    if (!res.ok) return;
    const samples = await res.json();
    for (const { file, label, blueprint } of samples) {
      const opt = document.createElement("option");
      opt.dataset.file = file;
      opt.textContent = blueprint ? `[${blueprint}] ${label}` : label;
      sampleSelectEl.appendChild(opt);
    }
    if (samples.length > 0) {
      sampleSelectEl.selectedIndex = 1;
      sampleSelectEl.dispatchEvent(new Event("change"));
    }
  } catch (_) {
    // silently ignore — sample index is optional
  }
}

loadSampleIndex();

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
