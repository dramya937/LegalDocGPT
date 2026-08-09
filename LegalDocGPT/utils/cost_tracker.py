"""
cost_tracker.py
----------------
Tracks token usage, cost, and latency for every LLM call in the pipeline,
so the app can report a real per-document cost instead of a guess.

Where possible this reads actual token counts from the OpenAI API response
(response.response_metadata["token_usage"]) rather than estimating with
tiktoken, since the API's own count is authoritative. tiktoken is kept as
a fallback/estimator for places that need a cost *before* making a call
(e.g. the eval script estimating a batch run's cost up front).

Pricing is USD per 1M tokens and should be checked against
https://openai.com/api/pricing/ periodically — prices have dropped
several times (GPT-4 -> GPT-4o -> GPT-4.1). Last verified: 2026-08.
"""

import time
import tiktoken
from dataclasses import dataclass, field
from contextlib import contextmanager

MODEL_PRICING = {
    "gpt-4":        {"input": 30.00, "output": 60.00},   # legacy, 8K context
    "gpt-4o":       {"input": 2.50,  "output": 10.00},   # current default
    "gpt-4o-mini":  {"input": 0.15,  "output": 0.60},    # cheap tier
    "gpt-4.1":      {"input": 2.00,  "output": 8.00},
    "gpt-4.1-mini": {"input": 0.40,  "output": 1.60},
}

DEFAULT_MODEL = "gpt-4o"


def estimate_tokens(text: str, model: str = DEFAULT_MODEL) -> int:
    """Rough token estimate for text that hasn't been sent to the API yet."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(text))


def cost_for_tokens(input_tokens: int, output_tokens: int, model: str = DEFAULT_MODEL) -> float:
    pricing = MODEL_PRICING.get(model, MODEL_PRICING[DEFAULT_MODEL])
    return (
        input_tokens / 1_000_000 * pricing["input"]
        + output_tokens / 1_000_000 * pricing["output"]
    )


@dataclass
class CallRecord:
    stage: str
    model: str
    input_tokens: int
    output_tokens: int
    latency_seconds: float
    cost_usd: float = field(init=False)

    def __post_init__(self):
        self.cost_usd = cost_for_tokens(self.input_tokens, self.output_tokens, self.model)


class CostTracker:
    """Accumulates CallRecords across a single document's pipeline run."""

    def __init__(self):
        self.records: list[CallRecord] = []

    @contextmanager
    def track(self, stage: str, model: str = DEFAULT_MODEL):
        """
        Usage:
            with tracker.track("clause_extraction", model="gpt-4o") as t:
                response = llm.invoke(prompt)
                t.record(response)   # pulls real usage from LangChain response
        """
        start = time.perf_counter()
        state = {"input_tokens": 0, "output_tokens": 0}

        class _Handle:
            def record(_self, response):
                usage = getattr(response, "response_metadata", {}).get("token_usage", {})
                if usage:
                    state["input_tokens"] = usage.get("prompt_tokens", 0)
                    state["output_tokens"] = usage.get("completion_tokens", 0)
                else:
                    # Fallback: response has no usage metadata (e.g. streaming,
                    # or a non-OpenAI-compatible wrapper) — estimate instead.
                    content = getattr(response, "content", str(response))
                    state["output_tokens"] = estimate_tokens(content, model)

            def record_manual(_self, input_tokens, output_tokens):
                state["input_tokens"] = input_tokens
                state["output_tokens"] = output_tokens

        handle = _Handle()
        try:
            yield handle
        finally:
            elapsed = time.perf_counter() - start
            self.records.append(CallRecord(
                stage=stage,
                model=model,
                input_tokens=state["input_tokens"],
                output_tokens=state["output_tokens"],
                latency_seconds=elapsed,
            ))

    def summary(self) -> dict:
        total_cost = sum(r.cost_usd for r in self.records)
        total_latency = sum(r.latency_seconds for r in self.records)
        return {
            "total_cost_usd": round(total_cost, 5),
            "total_latency_seconds": round(total_latency, 2),
            "total_input_tokens": sum(r.input_tokens for r in self.records),
            "total_output_tokens": sum(r.output_tokens for r in self.records),
            "by_stage": [
                {
                    "stage": r.stage,
                    "model": r.model,
                    "input_tokens": r.input_tokens,
                    "output_tokens": r.output_tokens,
                    "cost_usd": round(r.cost_usd, 5),
                    "latency_seconds": round(r.latency_seconds, 2),
                }
                for r in self.records
            ],
        }

    def reset(self):
        self.records = []
