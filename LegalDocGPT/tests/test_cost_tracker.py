from utils.cost_tracker import CostTracker, cost_for_tokens, estimate_tokens, MODEL_PRICING


def test_cost_for_tokens_gpt4o_matches_known_rate():
    # 1,000,000 input tokens on gpt-4o should cost exactly the input rate
    cost = cost_for_tokens(input_tokens=1_000_000, output_tokens=0, model="gpt-4o")
    assert cost == MODEL_PRICING["gpt-4o"]["input"]


def test_cost_for_tokens_combines_input_and_output():
    cost = cost_for_tokens(input_tokens=500_000, output_tokens=250_000, model="gpt-4o-mini")
    expected = 0.5 * MODEL_PRICING["gpt-4o-mini"]["input"] + 0.25 * MODEL_PRICING["gpt-4o-mini"]["output"]
    assert round(cost, 6) == round(expected, 6)


def test_unknown_model_falls_back_to_default_pricing():
    cost_unknown = cost_for_tokens(1000, 1000, model="not-a-real-model")
    cost_default = cost_for_tokens(1000, 1000, model="gpt-4o")
    assert cost_unknown == cost_default


def test_estimate_tokens_nonzero_for_nonempty_text():
    assert estimate_tokens("This is a contract clause about liability.") > 0


def test_estimate_tokens_zero_for_empty_string():
    assert estimate_tokens("") == 0


def test_tracker_accumulates_multiple_stages():
    tracker = CostTracker()

    with tracker.track("stage_one", model="gpt-4o") as t:
        t.record_manual(input_tokens=1000, output_tokens=200)

    with tracker.track("stage_two", model="gpt-4o") as t:
        t.record_manual(input_tokens=500, output_tokens=100)

    summary = tracker.summary()
    assert summary["total_input_tokens"] == 1500
    assert summary["total_output_tokens"] == 300
    assert len(summary["by_stage"]) == 2
    assert summary["total_cost_usd"] > 0


def test_tracker_reset_clears_records():
    tracker = CostTracker()
    with tracker.track("stage_one", model="gpt-4o") as t:
        t.record_manual(input_tokens=100, output_tokens=50)

    assert len(tracker.records) == 1
    tracker.reset()
    assert len(tracker.records) == 0
