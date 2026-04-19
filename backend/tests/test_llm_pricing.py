"""Langfuse-backed cost estimation."""

from app.monitoring import llm_pricing


def test_compute_cost_matches_versioned_openai_model_name():
    llm_pricing._pricing.clear()
    llm_pricing._pricing["gpt-4o-mini"] = (0.0000015, 0.0000060)
    c = llm_pricing.compute_cost("gpt-4o-mini-2024-07-18", 1_000_000, 500_000)
    assert c is not None
    assert abs(c - (1_000_000 * 0.0000015 + 500_000 * 0.0000060)) < 1e-9


def test_candidate_model_keys_strips_date_suffix():
    keys = llm_pricing._candidate_model_keys("GPT-4o-mini-2024-07-18")
    assert "gpt-4o-mini-2024-07-18" in keys
    assert "gpt-4o-mini" in keys


def test_get_pricing_status_keys():
    s = llm_pricing.get_pricing_status()
    assert s["source"] == "langfuse_get_api_public_models"
    assert "cost_formula" in s
    assert "last_refresh_at" in s
    assert "last_refresh_error" in s
