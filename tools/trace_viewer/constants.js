export const EDGE_STYLES = {
  default: "default",
  parallel: "parallel",
  retrieved: "retrieved",
  hidden: "hidden",
};

export function getCssVar(name) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

export function getColors() {
  return {
    claim: getCssVar("--claim"),
    blueprint: getCssVar("--blueprint"),
    iteration: getCssVar("--iteration"),
    task: getCssVar("--task"),
    result: getCssVar("--result"),
    childrun: getCssVar("--childrun"),
    childstep: getCssVar("--childstep"),
    query: getCssVar("--query"),
    select: getCssVar("--select"),
    retrieval: getCssVar("--retrieval"),
    retrieval_irrelevant: getCssVar("--retrieval_irrelevant"),
  };
}

export const WEB_LAYOUT = {
  mainX: 100,
  taskX: 420,
  childX: 760,
  taskColumnGap: 80,
  rowGap: 150,
  queryGap: 320,
  stageGap: 170,
  urlGap: 110,
  urlColumns: 3,
  urlColumnGap: 280,
};
