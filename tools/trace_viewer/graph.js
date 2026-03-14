import { EDGE_STYLES, getColors } from "./constants.js";

const COLORS = getColors();

export function renderSummary(summaryEl, trace, label) {
  const items = [
    ["Trace", label],
    ["Status", trace.status || "unknown"],
    ["Iterations", String((trace.iterations || []).length)],
    ["Evidence", String((trace.summary && trace.summary.evidence_count) || 0)],
    ["Errors", String(((trace.summary && trace.summary.errors) || []).length)],
    ["Agent", trace.agent || "n/a"],
  ];

  summaryEl.innerHTML = items
    .map(
      ([labelText, value]) => `<div class="stat"><span class="label">${labelText}</span><span class="value">${value}</span></div>`
    )
    .join("");
}

export function createGraph(container, model, onSelect, existingCy = null) {
  if (existingCy) {
    existingCy.destroy();
  }

  const cy = cytoscape({
    container,
    elements: [
      ...model.nodes.map((node) => ({
        data: {
          id: node.id,
          type: node.type,
          label: node.displayLabel,
          parent: node.parent || undefined,
          isContainer: node.isContainer ? "true" : "false",
          x: node.x || 0,
          y: node.y || 0,
        },
      })),
      ...model.edges.map((edge, index) => ({
        data: {
          id: `edge-${index}`,
          source: edge.source,
          target: edge.target,
          label: edge.label,
          edgeStyle: edge.edgeStyle || EDGE_STYLES.default,
          minLen: edge.minLen || 1,
          weight: edge.weight || 1,
        },
      })),
    ],
    style: [
      {
        selector: "node",
        style: {
          shape: "round-rectangle",
          width: 240,
          height: "label",
          padding: "16px",
          "background-color": (ele) => COLORS[ele.data("type")] || "#ffffff",
          "border-width": 1.2,
          "border-color": "rgba(32, 32, 32, 0.15)",
          label: "data(label)",
          color: "#202020",
          "font-size": 10,
          "font-family": 'Georgia, "Iowan Old Style", "Palatino Linotype", serif',
          "text-wrap": "wrap",
          "text-max-width": 200,
          "text-valign": "center",
          "text-halign": "center",
          "line-height": 1.2,
        },
      },
      {
        selector: ":parent",
        style: {
          "background-opacity": 0.35,
          "border-width": 1.8,
          "text-valign": "top",
          "text-halign": "center",
          "padding-top": "26px",
          "padding-left": "12px",
          "padding-right": "12px",
          "padding-bottom": "12px",
        },
      },
      {
        selector: 'node[isContainer = "true"]',
        style: {
          width: "label",
          height: "label",
          "font-size": 11,
          "text-wrap": "wrap",
          "text-max-width": 210,
        },
      },
      {
        selector: 'node[type = "iteration"]',
        style: { "border-width": 2.2, "border-color": "#204d48" },
      },
      {
        selector: 'node[type = "childrun"]',
        style: { "border-width": 2, "border-color": "#5f4b7a" },
      },
      {
        selector: 'node[type = "query"]',
        style: { "border-width": 1.6, "border-color": "#667b3c" },
      },
      {
        selector: 'node[type = "select"]',
        style: { "border-width": 1.6, "border-color": "#5d769b" },
      },
      {
        selector: "node:selected",
        style: { "border-width": 3, "border-color": "#204d48" },
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
          "taxi-turn": 28,
          label: "data(label)",
          color: "#6c675d",
          "font-size": 9,
          "text-background-color": "#fffdf8",
          "text-background-opacity": 0.95,
          "text-background-padding": "2px",
          "text-rotation": "autorotate",
        },
      },
      {
        selector: 'edge[edgeStyle = "parallel"]',
        style: { "line-style": "dashed", "target-arrow-shape": "none", opacity: 0.4 },
      },
      {
        selector: 'edge[edgeStyle = "retrieved"]',
        style: { "line-style": "dotted", "target-arrow-shape": "triangle", opacity: 0.6 },
      },
      {
        selector: 'edge[edgeStyle = "hidden"]',
        style: { opacity: 0, width: 0.001, label: "", "target-arrow-shape": "none" },
      },
    ],
    layout: {
      name: "preset",
      positions: (node) => ({ x: node.data("x") || 0, y: node.data("y") || 0 }),
    },
    minZoom: 0.3,
    maxZoom: 2.5,
    wheelSensitivity: 0.18,
  });

  cy.on("tap", "node", (event) => onSelect(event.target.id()));
  cy.fit(undefined, 50);
  return cy;
}
