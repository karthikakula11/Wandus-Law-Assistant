import { useEffect, useState } from "react";

const API = "/api";

type PricingStatus = {
  source: string;
  models_cached: number;
  langfuse_configured: boolean;
  cost_formula: string;
  last_refresh_at?: string | null;
  last_refresh_error?: string | null;
};

type RecordingHealth = {
  service: string;
  checked_at: string;
  status: string;
  pricing?: PricingStatus;
  checks: {
    langfuse_keys_configured: boolean;
    pricing_catalog_has_models: boolean;
    no_recent_null_cost_rows: boolean;
  };
  recommendations: string[];
  llm_usage_log: {
    total_rows: number;
    rows_last_24h: number;
    rows_last_7d: number;
    null_cost_rows_last_7d: number;
    null_cost_by_model_last_7d: { model: string; rows: number }[];
  };
};

type UsageSummary = {
  rows: number;
  input_tokens: number;
  output_tokens: number;
  total_tokens: number;
  cost_usd_sum: number;
  note?: string;
  pricing?: PricingStatus;
};

type DashboardMeta = {
  tracing_configured: boolean;
  langfuse_host: string;
  open_dashboard_url: string;
};

type Props = {
  onBack: () => void;
};

function fmtUsd(n: number): string {
  return new Intl.NumberFormat(undefined, {
    style: "currency",
    currency: "USD",
    minimumFractionDigits: 2,
    maximumFractionDigits: 6,
  }).format(n);
}

function fmtInt(n: number): string {
  return new Intl.NumberFormat().format(n);
}

export function UsageDashboardPage({ onBack }: Props) {
  const [meta, setMeta] = useState<DashboardMeta | null>(null);
  const [localDb, setLocalDb] = useState<UsageSummary | null>(null);
  const [health, setHealth] = useState<RecordingHealth | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      setLoading(true);
      setError(null);
      try {
        const [dashR, sumR, healthR] = await Promise.all([
          fetch(`${API}/usage/dashboard`),
          fetch(`${API}/monitoring/usage/summary`),
          fetch(`${API}/monitoring/usage/recording-health`),
        ]);
        if (!dashR.ok) {
          const t = await dashR.text();
          throw new Error(t || dashR.statusText);
        }
        const dashJson = (await dashR.json()) as DashboardMeta & { tracing_configured: boolean };
        if (!cancelled) setMeta(dashJson as DashboardMeta);

        if (sumR.ok) {
          const s = (await sumR.json()) as UsageSummary;
          if (!cancelled) setLocalDb(s);
        }
        if (healthR.ok) {
          const h = (await healthR.json()) as RecordingHealth;
          if (!cancelled) setHealth(h);
        }
      } catch (e) {
        if (!cancelled) setError(String(e));
      } finally {
        if (!cancelled) setLoading(false);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  const url = meta?.open_dashboard_url || meta?.langfuse_host || "https://cloud.langfuse.com";

  return (
    <div className="app-shell usage-dashboard-shell">
      <header className="app-header">
        <div className="brand">
          <div className="avatar-wandus" aria-hidden>
            ✦
          </div>
          <div className="brand-text">
            <h1>Usage &amp; cost</h1>
            <p>Tokens and estimated cost from this app</p>
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
          Totals come from <code>llm_usage_log</code> (all chat completions, streaming usage, and embedding
          batches). Estimated USD uses token counts from OpenAI and per-token rates from Langfuse&apos;s{" "}
          <code>GET /api/public/models</code> catalog (refreshed periodically).
        </p>

        {loading && <p className="usage-dashboard-status">Loading…</p>}
        {error && (
          <p className="usage-dashboard-err" role="alert">
            {error}
          </p>
        )}

        {!loading && !error && meta && (
          <div className="usage-dashboard-actions" style={{ marginBottom: "1rem" }}>
            <a
              className="btn-primary usage-open-langfuse"
              href={url}
              target="_blank"
              rel="noopener noreferrer"
            >
              Open Langfuse (tracing)
            </a>
            <span className="usage-meta-range">{url}</span>
          </div>
        )}

        {!loading && !error && meta && (
          <div
            className={`usage-tracing-pill ${meta.tracing_configured ? "usage-tracing-on" : "usage-tracing-off"}`}
          >
            {meta.tracing_configured ? "Langfuse keys configured" : "Langfuse keys not set (tracing & pricing catalog)"}
          </div>
        )}

        {!loading && !error && health && (
          <section className="usage-section" aria-labelledby="recording-health" style={{ marginTop: "1rem" }}>
            <h2 id="recording-health" className="usage-section-title">
              Recording orchestrator
            </h2>
            <p className="usage-dashboard-note">
              <strong>Status:</strong>{" "}
              <span
                style={{
                  fontWeight: 600,
                  color:
                    health.status === "ok"
                      ? "var(--accent-ok, #15803d)"
                      : health.status === "critical"
                        ? "var(--accent-bad, #b91c1c)"
                        : "var(--accent-warn, #a16207)",
                }}
              >
                {health.status}
              </span>
              {" · "}
              Rows (7d): {fmtInt(health.llm_usage_log.rows_last_7d)} · Null cost (7d):{" "}
              {fmtInt(health.llm_usage_log.null_cost_rows_last_7d)}
            </p>
            <ul className="usage-health-checks" style={{ margin: "0.5rem 0 0 1rem" }}>
              <li>
                Langfuse keys: {health.checks.langfuse_keys_configured ? "yes" : "no"}
              </li>
              <li>
                Pricing catalog loaded: {health.checks.pricing_catalog_has_models ? "yes" : "no"}
              </li>
              <li>
                No null-cost rows (7d): {health.checks.no_recent_null_cost_rows ? "yes" : "no"}
              </li>
              {health.pricing?.last_refresh_at ? (
                <li>Last pricing refresh: {health.pricing.last_refresh_at}</li>
              ) : null}
            </ul>
            {health.pricing?.last_refresh_error ? (
              <p className="usage-dashboard-note" style={{ color: "var(--accent-bad, #b91c1c)" }}>
                <strong>Pricing refresh error:</strong> {health.pricing.last_refresh_error}
              </p>
            ) : null}
            {health.recommendations.length > 0 ? (
              <div style={{ marginTop: "0.75rem" }}>
                <strong>Notes</strong>
                <ul style={{ margin: "0.35rem 0 0 1rem" }}>
                  {health.recommendations.map((r, i) => (
                    <li key={i} className="usage-dashboard-note">
                      {r}
                    </li>
                  ))}
                </ul>
              </div>
            ) : null}
          </section>
        )}

        {!loading && !error && localDb && (localDb.rows > 0 || localDb.pricing) && (
          <section className="usage-section" aria-labelledby="usage-local">
            <h2 id="usage-local" className="usage-section-title">
              Totals (<code>llm_usage_log</code>)
            </h2>
            {localDb.rows > 0 ? (
              <div className="usage-stat-grid">
                <div className="usage-stat-card">
                  <div className="usage-stat-label">Input tokens</div>
                  <div className="usage-stat-value">{fmtInt(localDb.input_tokens)}</div>
                </div>
                <div className="usage-stat-card">
                  <div className="usage-stat-label">Output tokens</div>
                  <div className="usage-stat-value">{fmtInt(localDb.output_tokens)}</div>
                </div>
                <div className="usage-stat-card">
                  <div className="usage-stat-label">Estimated cost (USD)</div>
                  <div className="usage-stat-value">{fmtUsd(localDb.cost_usd_sum)}</div>
                </div>
                <div className="usage-stat-card">
                  <div className="usage-stat-label">Logged calls</div>
                  <div className="usage-stat-value">{fmtInt(localDb.rows)}</div>
                </div>
              </div>
            ) : (
              <p className="usage-dashboard-note">No usage rows yet. Totals appear after chat or embed traffic.</p>
            )}
            {localDb.pricing ? (
              <p className="usage-dashboard-note" style={{ marginTop: "0.75rem" }}>
                <strong>Cost basis:</strong> {localDb.pricing.cost_formula}. Langfuse model catalog:{" "}
                <strong>{localDb.pricing.models_cached}</strong> models loaded
                {localDb.pricing.langfuse_configured ? "" : " (set Langfuse keys to load prices)"}.
              </p>
            ) : null}
            {localDb.note ? (
              <p className="usage-dashboard-note" style={{ marginTop: "0.5rem" }}>
                {localDb.note}
              </p>
            ) : null}
          </section>
        )}
      </div>
    </div>
  );
}
