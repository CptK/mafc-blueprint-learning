export const SAMPLE_TRACE = {
  trace_version: 1,
  agent: "FactCheckAgent",
  session_id: "veritas-first:123",
  status: "completed",
  claim: { text: "This image shows Athens.", images: ["image://athens"], videos: [] },
  blueprint: { name: "media_location", max_iterations: 3, start_node_id: "iter1_search" },
  iterations: [
    {
      iteration: 1,
      node_before: "iter1_search",
      node_after: "verdict_gate",
      decision: { decision_type: "delegate", rationale: "Start with media analysis." },
      delegated_tasks: [
        {
          task_id: "media_location",
          agent_type: "media",
          instruction: "Check where this image was taken.",
          result: {
            evidences: [{ source: "image://athens", takeaways: { text: "Landmarks match Athens." } }],
            errors: [],
          },
          child_trace: {
            agent: "WebSearchAgent",
            status: "completed",
            summary: { evidence_count: 1 },
            iterations: [
              {
                step: 1,
                resolved_plan: { queries: ["athens image source"], done: true },
                search_results: [
                  {
                    query_text: "athens image source",
                    sources: [
                      { url: "https://example.com/athens", title: "Athens archive", preview: "Archive match" },
                      { url: "https://example.com/athens-2", title: "City page", preview: "Landmarks overview" },
                    ],
                    errors: [],
                    marked_seen: true,
                  },
                ],
                selected_sources: [
                  {
                    query_text: "athens image source",
                    sources: [
                      { url: "https://example.com/athens", title: "Athens archive", preview: "Archive match" },
                    ],
                  },
                ],
                retrievals: [
                  {
                    query_text: "athens image source",
                    source: { url: "https://example.com/athens" },
                    snippet: "Archive source matching the same landmarks.",
                    evidence: {
                      raw: { text: "Long raw page text..." },
                      takeaways: { text: "The page shows the same landmarks and dates the image earlier." },
                    },
                  },
                ],
              },
            ],
          },
        },
      ],
      evidence_count_before: 0,
      evidence_count_after: 1,
      new_errors: [],
    },
    {
      iteration: 2,
      node_before: "verdict_gate",
      node_after: "verdict_gate",
      decision: { decision_type: "finalize", rationale: "Enough evidence was collected." },
      delegated_tasks: [],
      evidence_count_before: 1,
      evidence_count_after: 1,
      new_errors: [],
    },
  ],
  summary: {
    result: { text: "The image is consistent with Athens." },
    errors: [],
    evidence_count: 1,
    action_history: [
      "delegate: Start with media analysis.",
      "task media_location completed with 1 evidences and 0 errors",
      "finalize: Enough evidence was collected.",
    ],
  },
};
