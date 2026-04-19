const API = "/api";

export type DriftData = {
  drift_detected: boolean;
  message?: string | null;
  p_value?: number | null;
  ks_statistic?: number | null;
  threshold?: number;
  window_days?: number;
  detection_method?: string;
  current_period_count?: number;
  previous_period_count?: number;
  current_avg_confidence?: number;
  previous_avg_confidence?: number;
  confidence_shift?: number;
  comparison_type?: string;
};

type DriftResponse = {
  success?: boolean;
  data?: DriftData;
};

export async function fetchDriftData(): Promise<DriftData | null> {
  const r = await fetch(`${API}/eval/drift?days=7&threshold=0.1`);
  if (!r.ok) return null;
  const j = (await r.json()) as DriftResponse;
  return j.data ?? null;
}

export function driftNeedsSetup(data: DriftData | null): boolean {
  if (!data?.message) return false;
  const m = data.message.toLowerCase();
  const insufficient = m.includes("insufficient");
  return (
    insufficient ||
    /insufficient|alembic|table missing|migration|drift check failed|database error/i.test(
      data.message
    )
  );
}
