"""Tests for pricing module."""

from __future__ import annotations

from traqo.pricing import estimate_cost


class TestEstimateCost:
    def test_known_model(self):
        cost = estimate_cost("gpt-4o", input_tokens=1000, output_tokens=500)
        assert cost is not None
        assert cost > 0
        # 1000 * 2.50/1M + 500 * 10.00/1M = 0.0025 + 0.005 = 0.0075
        assert abs(cost - 0.0075) < 1e-9

    def test_unknown_model_returns_none(self):
        cost = estimate_cost(
            "totally-unknown-model", input_tokens=100, output_tokens=50
        )
        assert cost is None

    def test_anthropic_model_with_cache(self):
        cost = estimate_cost(
            "claude-3-5-sonnet-20241022",
            input_tokens=1000,
            output_tokens=500,
            cache_read_tokens=200,
            cache_creation_tokens=100,
        )
        assert cost is not None
        assert cost > 0

    def test_prefix_stripping(self):
        cost_plain = estimate_cost("gpt-4o", input_tokens=1000, output_tokens=500)
        cost_azure = estimate_cost("azure/gpt-4o", input_tokens=1000, output_tokens=500)
        assert cost_plain == cost_azure

    def test_dated_model_direct_match(self):
        """Dated model IDs exist as their own entries from models.dev."""
        cost_plain = estimate_cost("gpt-4o", input_tokens=1000, output_tokens=500)
        cost_dated = estimate_cost(
            "gpt-4o-2024-11-20", input_tokens=1000, output_tokens=500
        )
        assert cost_plain == cost_dated

    def test_zero_tokens(self):
        cost = estimate_cost("gpt-4o", input_tokens=0, output_tokens=0)
        assert cost == 0.0

    def test_gemini_model(self):
        cost = estimate_cost("gemini-2.5-pro", input_tokens=1000, output_tokens=500)
        assert cost is not None
        assert cost > 0

    def test_cache_tokens_with_no_cache_pricing(self):
        """Models without cache pricing: cached tokens reduce input cost but add 0 cache cost."""
        cost_no_cache = estimate_cost("gpt-4", input_tokens=1000, output_tokens=500)
        cost_with_cache = estimate_cost(
            "gpt-4",
            input_tokens=1000,
            output_tokens=500,
            cache_read_tokens=200,
        )
        assert cost_with_cache is not None
        assert cost_no_cache is not None
        # non_cached = 1000 - 200 = 800, cache_read at $0 → cheaper than no-cache
        assert cost_with_cache < cost_no_cache

    def test_empty_string_model(self):
        assert estimate_cost("", input_tokens=100, output_tokens=50) is None

    def test_multiple_prefix_only_first_stripped(self):
        """Only the first matching prefix is stripped."""
        cost = estimate_cost(
            "azure/openai/gpt-4o", input_tokens=1000, output_tokens=500
        )
        assert cost is None  # Double prefix won't resolve

    def test_anthropic_cost_not_double_counted(self):
        """Verify that cache tokens are not double-counted in cost.

        input_tokens is total volume (including cached). The cost formula
        subtracts cache tokens before applying the base input price.
        """
        from traqo.pricing import _PRICING, _normalize_model

        model = "claude-3-5-sonnet-20241022"
        key = _normalize_model(model)
        assert key is not None
        prices = _PRICING[key]

        input_tokens = 1000  # total volume (including cache_read)
        output_tokens = 500
        cache_read_tokens = 200

        cost = estimate_cost(
            model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_read_tokens=cache_read_tokens,
        )
        assert cost is not None

        # non-cached at input rate + output at output rate + cached at cache rate
        non_cached = input_tokens - cache_read_tokens
        expected = (
            non_cached * prices["input"] / 1_000_000
            + output_tokens * prices["output"] / 1_000_000
            + cache_read_tokens * prices["cache_read"] / 1_000_000
        )
        assert abs(cost - expected) < 1e-12

    def test_suffix_stripping_fallback(self):
        """Unknown dated suffix falls back to base model via suffix stripping."""
        # A hypothetical future dated version should fall back to base
        cost_base = estimate_cost("gpt-4o", input_tokens=1000, output_tokens=500)
        cost_future = estimate_cost(
            "gpt-4o-2099-01-01", input_tokens=1000, output_tokens=500
        )
        assert cost_base == cost_future
