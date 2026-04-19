-- LLM usage + local cost estimates (run manually against your lawchat DB if not using Alembic)
CREATE TABLE IF NOT EXISTS llm_usage_log (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    model VARCHAR(256) NOT NULL,
    tokens_in INTEGER NOT NULL,
    tokens_out INTEGER NOT NULL,
    cost_usd DOUBLE PRECISION,
    trace_id VARCHAR(64),
    session_id VARCHAR(64),
    route VARCHAR(128),
    metadata JSONB
);

CREATE INDEX IF NOT EXISTS ix_llm_usage_log_trace_id ON llm_usage_log (trace_id);
CREATE INDEX IF NOT EXISTS ix_llm_usage_log_session_id ON llm_usage_log (session_id);
CREATE INDEX IF NOT EXISTS ix_llm_usage_log_created_at ON llm_usage_log (created_at);
