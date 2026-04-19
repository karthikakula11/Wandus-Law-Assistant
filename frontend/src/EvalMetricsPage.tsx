import { useCallback, useEffect, useState } from "react";

import { driftNeedsSetup, fetchDriftData, type DriftData } from "./driftStatus";

type Props = {
  onBack: () => void;
};

export function EvalMetricsPage({ onBack }: Props) {
  const [data, setData] = useState<DriftData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const d = await fetchDriftData();
      if (d === null) {
        throw new Error(
          "Could not load drift. Check the API (port 8000) and run `alembic upgrade head` if the table is missing."
        );
      }
      setData(d);
    } catch (e) {
      setError(String(e));
      setData(null);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void load();
  }, [load]);

  const insufficient =
    data?.message?.toLowerCase().includes("insufficient") ?? false;
  const needsSetupOrData = driftNeedsSetup(data);

  return (
    <div className="app-shell usage-dashboard-shell">
      <header className="app-header">
        <div className="brand">
          <div className="avatar-pintu" aria-hidden>
            ✦
          </div>
          <div className="brand-text">
            <h1>Retrieval drift</h1>
            <p>Confidence signal stability (no gold labels)</p>
          </div>
        </div>
        <div className="header-actions">
          <button type="button" className="btn-ghost" onClick={onBack}>
            ← Back to chat
          </button>
        </div>
      </header>

      <div className="usage-dashboard-body">
        <p className="usage-dashboard-lede">
          Each <strong>document-grounded</strong> reply logs a <strong>retrieval confidence</strong> score
          (from best dense match). We compare the last <strong>7 days</strong> to the previous{" "}
          <strong>7 days</strong> with a <strong>Kolmogorov–Smirnov</strong> test (TalentVibe-style on{" "}
          <strong>live</strong> samples). This measures <em>distribution shift</em>, not legal correctness.
        </p>

        {loading && <p className="usage-dashboard-status">Loading…</p>}
        {error && (
          <p className="usage-dashboard-err" role="alert">
            {error}
          </p>
        )}

        {!loading && !error && data && (
          <>
            <div className="drift-detection-card drift-detection-card--pintu">
              <h2 className="usage-section-title" style={{ marginTop: 0 }}>
                Status
              </h2>
              {needsSetupOrData ? (
                <div className="drift-status drift-status--muted">
                  <span className="drift-status-label">
                    {insufficient ? "Not enough data yet" : "Action or data needed"}
                  </span>
                  <p className="usage-dashboard-note">{data.message}</p>
                  {insufficient ? (
                    <p className="usage-dashboard-note">
                      Ask questions that use your <strong>uploaded PDFs</strong>; we record one sample per
                      grounded answer.
                    </p>
                  ) : null}
                </div>
              ) : (
                <div
                  className={`drift-badge drift-status ${data.drift_detected ? "drift-detected" : "no-drift"}`}
                  role="status"
                  aria-label={data.drift_detected ? "Drift detected" : "No drift"}
                >
                  {data.drift_detected ? "⚠ Drift detected" : "✓ No drift"}
                </div>
              )}

              {!needsSetupOrData && data.detection_method && (
                <p className="usage-dashboard-note drift-method">
                  Method:{" "}
                  <code>{data.detection_method}</code>
                  {data.comparison_type ? (
                    <>
                      {" "}
                      · {data.comparison_type === "split_period" ? "split-window" : "two periods"}
                    </>
                  ) : null}
                </p>
              )}

              {data.message && !needsSetupOrData ? (
                <p className="usage-dashboard-note drift-msg">{data.message}</p>
              ) : null}

              {!needsSetupOrData && (
                <div className="drift-details-grid">
                  {data.p_value != null && (
                    <div>
                      <span className="drift-detail-k">p-value</span>
                      <span className="drift-detail-v">{data.p_value.toFixed(4)}</span>
                    </div>
                  )}
                  {data.ks_statistic != null && (
                    <div>
                      <span className="drift-detail-k">KS statistic</span>
                      <span className="drift-detail-v">{data.ks_statistic.toFixed(4)}</span>
                    </div>
                  )}
                  {data.threshold != null && (
                    <div>
                      <span className="drift-detail-k">Threshold</span>
                      <span className="drift-detail-v">{data.threshold}</span>
                    </div>
                  )}
                  {data.current_period_count != null && (
                    <div>
                      <span className="drift-detail-k">Current window samples</span>
                      <span className="drift-detail-v">{data.current_period_count}</span>
                    </div>
                  )}
                  {data.previous_period_count != null && (
                    <div>
                      <span className="drift-detail-k">Previous window samples</span>
                      <span className="drift-detail-v">{data.previous_period_count}</span>
                    </div>
                  )}
                  {data.current_avg_confidence != null && (
                    <div>
                      <span className="drift-detail-k">Avg confidence (current)</span>
                      <span className="drift-detail-v">
                        {(data.current_avg_confidence * 100).toFixed(1)}%
                      </span>
                    </div>
                  )}
                  {data.previous_avg_confidence != null && (
                    <div>
                      <span className="drift-detail-k">Avg confidence (previous)</span>
                      <span className="drift-detail-v">
                        {(data.previous_avg_confidence * 100).toFixed(1)}%
                      </span>
                    </div>
                  )}
                  {data.confidence_shift != null && (
                    <div>
                      <span className="drift-detail-k">Shift (current − previous)</span>
                      <span
                        className={`drift-detail-v ${
                          data.confidence_shift >= 0 ? "drift-shift-pos" : "drift-shift-neg"
                        }`}
                      >
                        {(data.confidence_shift * 100).toFixed(2)} pp
                      </span>
                    </div>
                  )}
                </div>
              )}
            </div>

            <section className="usage-section" style={{ marginTop: "1.25rem" }}>
              <h2 className="usage-section-title">How to read this</h2>
              <ul className="usage-dashboard-note" style={{ margin: "0.5rem 0 0 1.1rem" }}>
                <li>
                  <strong>No drift</strong> — confidence scores in the recent window look statistically similar
                  to the earlier window (p-value above threshold).
                </li>
                <li>
                  <strong>Drift detected</strong> — the distribution of retrieval confidence changed enough to
                  flag (often after ingest, embedding, or retrieval changes). Investigate; it is not a verdict on
                  hallucinations.
                </li>
                <li>
                  Needs enough <strong>live</strong> samples per window (configurable, default ≥5 each), or
                  split-half of the current window. For demos:{" "}
                  <code>python scripts/seed_drift_demo.py stable|shift</code> from <code>backend/</code>.
                </li>
              </ul>
            </section>

            <p className="usage-dashboard-note" style={{ marginTop: "1rem" }}>
              <button type="button" className="btn-ghost" onClick={() => void load()}>
                Refresh
              </button>
            </p>
          </>
        )}
      </div>
    </div>
  );
}
