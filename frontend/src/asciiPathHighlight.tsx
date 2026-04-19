import { Fragment, type ReactNode } from "react";

/** Same ids as ``GRAPH_NODE_LABELS`` / LangGraph nodes — longest match first for splitting. */
const GRAPH_NODE_IDS = [
  "__start__",
  "__end__",
  "plan",
  "retrieve",
  "grade",
  "rewrite",
  "broaden",
  "generate",
] as const;

const NODE_ID_SET = new Set<string>(GRAPH_NODE_IDS);

function nodeSplitRegex(): RegExp {
  const sorted = [...GRAPH_NODE_IDS].sort((a, b) => b.length - a.length);
  const escaped = sorted.map((id) => id.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"));
  return new RegExp(`(${escaped.join("|")})`, "g");
}

const NODE_SPLIT_RE = nodeSplitRegex();

/** Split ASCII on known node ids and wrap ids that appear on the execution path (or active step). */
export function highlightAsciiForPath(
  ascii: string,
  pathNodes: string[],
  activeStep: string | null
): ReactNode {
  const visited = new Set(pathNodes);
  if (activeStep) visited.add(activeStep);

  const parts = ascii.split(NODE_SPLIT_RE);

  return parts.map((part, i) => {
    if (!NODE_ID_SET.has(part)) {
      return <Fragment key={i}>{part}</Fragment>;
    }
    const isActive = activeStep === part;
    const onPath = visited.has(part);
    if (!isActive && !onPath) {
      return <Fragment key={i}>{part}</Fragment>;
    }
    const cls = [
      "graph-ascii-node",
      isActive ? "graph-ascii-node-active" : "graph-ascii-node-visited",
    ].join(" ");
    return (
      <span key={`${i}-${part}`} className={cls}>
        {part}
      </span>
    );
  });
}
