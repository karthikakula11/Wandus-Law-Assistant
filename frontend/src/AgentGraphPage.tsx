import { useEffect, useId, useMemo, useState } from "react";

import { highlightAsciiForPath } from "./asciiPathHighlight";
import { describeGradeBranches, GRAPH_NODE_LABELS, graphNodeLabel } from "./graphLabels";

import { API } from "./apiBase";

type Props = {
  onBack: () => void;
  /** LangGraph node id currently executing (from SSE ``graph_step``); null when idle. */
  activeStep: string | null;
  /** Ordered node ids for the current or last run (shows which branch after ``grade``). */
  pathNodes: string[];
  /** True while the chat stream is still receiving ``graph_step`` events. */
  pathStreaming: boolean;
  /** Nodes that appeared at least once on the displayed path (for rail highlight). */
  visitedNodeIds: Set<string>;
};

export function AgentGraphPage({
  onBack,
  activeStep,
  pathNodes,
  pathStreaming,
  visitedNodeIds,
}: Props) {
  const gradeBranchHints = describeGradeBranches(pathNodes);
  const pathKey = useMemo(() => pathNodes.join("\0"), [pathNodes]);

  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [asciiArt, setAsciiArt] = useState<string | null>(null);
  const baseId = useId().replace(/:/g, "");

  useEffect(() => {
    let cancelled = false;

    (async () => {
      setLoading(true);
      setError(null);
      setAsciiArt(null);
      try {
        const r = await fetch(`${API}/rag-graph/ascii`);
        if (!r.ok) throw new Error(await r.text());
        const data = (await r.json()) as { ascii: string };
        if (!cancelled) setAsciiArt(data.ascii);
      } catch (e) {
        if (!cancelled) setError(String(e));
      } finally {
        if (!cancelled) setLoading(false);
      }
    })();

    return () => {
      cancelled = true;
    };
  }, [baseId]);

  const asciiHighlighted = useMemo(
    () =>
      asciiArt === null
        ? null
        : highlightAsciiForPath(asciiArt, pathNodes, activeStep),
    [asciiArt, pathKey, activeStep]
  );

  return (
    <div className="app-shell agent-graph-shell">
      <header className="app-header">
        <div className="brand">
          <div className="avatar-wandus" aria-hidden>
            ✦
          </div>
          <div className="brand-text">
            <h1>Agent pipeline</h1>
            <p>LangGraph · agentic RAG</p>
          </div>
        </div>
        <div className="header-actions">
          <button type="button" className="btn-ghost" onClick={onBack}>
            ← Back to chat
          </button>
        </div>
      </header>

      <div className="agent-graph-body">
        <p className="agent-graph-lede">
          The diagram is LangChain <code>Graph.draw_ascii()</code> from your <strong>compiled</strong>{" "}
          LangGraph (same topology as <code>compiled.builder</code>: <code>__start__</code> → nodes →{" "}
          <code>__end__</code>, conditional branches from <code>grade</code>). LangGraph’s{" "}
          <code>get_graph()</code> path is skipped for plain <code>dict</code> state. After a run, node
          names in the diagram that match your path are <strong>highlighted</strong> (mint = visited,
          magenta = current step while streaming). <strong>Path taken</strong> and the step rail list the
          same sequence.
        </p>

        <section className="run-path-section" aria-labelledby="run-path-heading">
          <h2 id="run-path-heading" className="run-path-heading">
            Path taken {pathStreaming ? "(live)" : "(last run)"}
          </h2>
          {pathNodes.length === 0 && !pathStreaming && (
            <p className="run-path-empty">
              Send a message from <strong>Chat</strong> with agentic RAG enabled to record the node
              sequence here.
            </p>
          )}
          {(pathNodes.length > 0 || pathStreaming) && (
            <>
              <div className="run-path-trail" aria-label="Execution order">
                {pathNodes.map((id, i) => (
                  <span key={`${id}-${i}`} className="run-path-step">
                    {i > 0 && (
                      <span className="run-path-arrow" aria-hidden>
                        →
                      </span>
                    )}
                    <span className="run-path-pill">{graphNodeLabel(id)}</span>
                    <code className="run-path-id">{id}</code>
                  </span>
                ))}
                {pathStreaming && <span className="run-path-pending">…</span>}
              </div>
              {gradeBranchHints.length > 0 && (
                <ul className="run-path-branches">
                  {gradeBranchHints.map((line) => (
                    <li key={line}>{line}</li>
                  ))}
                </ul>
              )}
            </>
          )}
        </section>

        {activeStep && (
          <div className="graph-live-banner" role="status" aria-live="polite">
            <span className="graph-live-pulse" aria-hidden />
            <span>
              Now running: <strong>{graphNodeLabel(activeStep)}</strong>{" "}
              <code className="graph-live-code">({activeStep})</code>
            </span>
          </div>
        )}

        <ol className="graph-step-rail" aria-label="Pipeline steps">
          {Object.entries(GRAPH_NODE_LABELS).map(([id, label]) => (
            <li
              key={id}
              className={`graph-step-rail-item ${activeStep === id ? "active" : ""} ${
                visitedNodeIds.has(id) ? "visited" : ""
              }`}
            >
              <span className="graph-step-rail-dot" aria-hidden />
              <span className="graph-step-rail-text">{label}</span>
            </li>
          ))}
        </ol>

        {loading && <p className="agent-graph-status">Loading diagram…</p>}
        {error && (
          <p className="agent-graph-err" role="alert">
            {error}
          </p>
        )}
        {asciiHighlighted !== null && (
          <pre
            className="graph-ascii-pre"
            aria-busy={loading}
            aria-label="Agent graph ASCII diagram"
          >
            {asciiHighlighted}
          </pre>
        )}
      </div>
    </div>
  );
}
