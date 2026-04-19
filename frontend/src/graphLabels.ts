/**
 * LangGraph node ids from ``rag_graph.build_rag_graph`` (including compiled ``__start__`` / ``__end__``).
 * Order matches the pipeline for the step rail (top → bottom).
 */
export const GRAPH_NODE_LABELS: Record<string, string> = {
  __start__: "Graph entry (START)",
  plan: "Planning search queries",
  retrieve: "Retrieving chunks",
  grade: "Grading relevance",
  rewrite: "Rewriting queries",
  broaden: "Broadening search",
  generate: "Generating answer",
  __end__: "Graph exit (END)",
};

export function graphNodeLabel(node: string): string {
  return GRAPH_NODE_LABELS[node] ?? node;
}

/** Summarize routing right after each ``grade`` in an execution path (for UI hints). */
export function describeGradeBranches(path: string[]): string[] {
  const hints: string[] = [];
  for (let i = 0; i < path.length - 1; i++) {
    if (path[i] !== "grade") continue;
    const next = path[i + 1];
    if (next === "rewrite") hints.push("grade → rewrite (re-query)");
    else if (next === "broaden") hints.push("grade → broaden (distance gate)");
    else if (next === "generate") hints.push("grade → generate (answer now)");
  }
  return hints;
}
