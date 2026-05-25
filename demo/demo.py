"""
demo/demo.py
------------
End-to-end production demo: SemanticCache + QueryRouter + BudgetEnforcer.

Pure Python — no PyTorch, no sentence-transformers, no background threads.
Starts and exits instantly on Windows, Linux, and Mac.

Run:
    python demo/demo.py
"""

from __future__ import annotations

import os
import sys
import time

# ── Path fix: works from any working directory ────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
# ──────────────────────────────────────────────────────────────────────────

from semantic_cache.cache import SemanticCache
from query_router.router import QueryRouter, ModelTier
from token_budget.budget import BudgetEnforcer


# ---------------------------------------------------------------------------
# Simulated LLM call
# Replace with your real API call (OpenAI, Anthropic, etc.)
# Returns (response_text, tokens_used, cost_usd)
# ---------------------------------------------------------------------------

_COST_PER_TOKEN = {
    "gpt-4o-mini": 0.000000165,
    "gpt-4o":      0.000005,
    "gpt-4.5":     0.000015,
}

def simulate_llm_call(query: str, model_id: str, context: str) -> tuple[str, int, float]:
    response = f"[{model_id}] {query[:60]}..."
    tokens   = len(query.split()) * 3 + 80
    cost     = tokens * _COST_PER_TOKEN.get(model_id, 0.000005)
    return response, tokens, cost


# ---------------------------------------------------------------------------
# Demo queries — realistic production mix
# ---------------------------------------------------------------------------

DEMO_QUERIES = [
    # Simple → cheap model
    "What is RAG?",
    "What is a vector database?",
    "Define semantic search.",
    # Standard complexity
    "How does hybrid retrieval differ from pure vector search?",
    "What are the trade-offs between BM25 and dense embeddings?",
    # Complex → expensive model
    (
        "Compare the cost and latency trade-offs of agentic RAG versus "
        "standard retrieval pipelines at 10,000 requests per day. "
        "What architectural decisions minimize cost without degrading quality?"
    ),
    # Repeats → should hit cache (no LLM call)
    "What is RAG?",
    "What is a vector database?",
]


# ---------------------------------------------------------------------------
# Production RAG pipeline
# ---------------------------------------------------------------------------

class ProductionRAGPipeline:
    """Wires SemanticCache + QueryRouter + BudgetEnforcer into one pipeline."""

    def __init__(self):
        self.cache = SemanticCache(
            threshold=0.75,
            ttl_seconds=3600,
            cost_per_llm_call_usd=0.004,
        )
        self.router = QueryRouter(
            simple_threshold=0.25,
            complex_threshold=0.65,
        )
        self.enforcer = BudgetEnforcer(
            hourly_limit_usd=5.0,
            daily_limit_usd=50.0,
            per_request_limit_usd=0.10,
            total_tokens_per_request=4096,
            cooldown_seconds=10.0,
            downgrade_on_breach=True,
        )

    def query(self, user_query: str, retrieved_context: str = "") -> dict:
        t0 = time.perf_counter()

        # ── Step 1: Cache lookup ──────────────────────────────────────
        cached = self.cache.get(user_query)
        if cached is not None:
            return {
                "query":       user_query,
                "response":    cached,
                "source":      "CACHE HIT",
                "model_used":  None,
                "tier":        None,
                "score":       None,
                "tokens_used": 0,
                "cost_usd":    0.0,
                "cost_saved":  self.cache.cost_per_llm_call_usd,
                "pipeline_ms": round((time.perf_counter() - t0) * 1000, 2),
                "downgraded":  False,
            }

        # ── Step 2: Route to model tier ───────────────────────────────
        routing = self.router.route(user_query)
        context = retrieved_context or f"[Context for: {user_query[:40]}]"

        response_text = ""
        actual_tokens = 0
        actual_cost   = 0.0
        source        = "LLM CALL"
        downgraded    = False

        # ── Step 3: Token budget + cost enforcement ───────────────────
        with self.enforcer.request(
            model_tier=routing.tier.value,
            estimated_tokens=500,
        ) as ctx:
            if not ctx.allowed:
                response_text = ctx.fallback_response
                source = "BLOCKED"
            else:
                downgraded    = ctx.downgraded
                effective_tier = ModelTier.SIMPLE if downgraded else routing.tier
                model_id       = self.router.model_map[effective_tier]

                # Reserve in priority order: fixed → history → docs → output
                ctx.budget.reserve("system_prompt", 200)
                ctx.budget.reserve_text("history", "Previous conversation turns...")
                ctx.budget.reserve_text("retrieved_docs", context)
                ctx.budget.reserve("output", min(512, ctx.budget.remaining()))

                response_text, actual_tokens, actual_cost = simulate_llm_call(
                    user_query, model_id, context
                )
                ctx.record_actual(actual_tokens=actual_tokens, cost_usd=actual_cost)

        # ── Step 4: Cache result for future reuse ─────────────────────
        if source == "LLM CALL":
            self.cache.set(user_query, response_text)

        return {
            "query":       user_query,
            "response":    response_text,
            "source":      source,
            "model_used":  routing.model_id,
            "tier":        routing.tier.value,
            "score":       round(routing.score.total, 3),
            "tokens_used": actual_tokens,
            "cost_usd":    round(actual_cost, 6),
            "cost_saved":  round(routing.cost_saved_usd, 6),
            "pipeline_ms": round((time.perf_counter() - t0) * 1000, 2),
            "downgraded":  downgraded,
        }

    def status(self) -> dict:
        return {
            "cache":  self.cache.get_stats(),
            "router": self.router.get_stats(),
            "budget": self.enforcer.status(),
        }


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def main() -> None:
    SEP = "=" * 65
    print(SEP)
    print("RAG Cost Layer — End-to-End Production Demo")
    print(SEP)

    pipeline    = ProductionRAGPipeline()
    total_cost  = 0.0
    total_saved = 0.0

    for i, query in enumerate(DEMO_QUERIES, 1):
        print(f"\n[Query {i:02d}] {query[:72]}")
        r = pipeline.query(query)

        print(f"  Source:  {r['source']}")
        if r["source"] == "LLM CALL":
            print(f"  Tier:    {r['tier']}  (complexity: {r['score']})")
            print(f"  Model:   {r['model_used']}")
            print(f"  Tokens:  {r['tokens_used']}")
            print(f"  Cost:    ${r['cost_usd']:.6f}")
            print(f"  Saved:   ${r['cost_saved']:.6f}  vs always-expensive model")
            if r["downgraded"]:
                print("  ⚠  Downgraded to simple tier by circuit breaker")
        elif r["source"] == "CACHE HIT":
            print(f"  Saved:   ${r['cost_saved']:.4f}  (LLM call avoided)")
        print(f"  Latency: {r['pipeline_ms']} ms")

        total_cost  += r["cost_usd"]
        total_saved += r["cost_saved"]

    # ── Summary ──────────────────────────────────────────────────────────
    print(f"\n{SEP}")
    print("Run Summary")
    print(SEP)
    print(f"  Total cost this run:   ${total_cost:.6f}")
    print(f"  Total saved vs naive:  ${total_saved:.6f}")

    s = pipeline.status()

    print("\nSemantic Cache:")
    for k, v in s["cache"].items():
        print(f"  {k:<32} {v}")

    print("\nQuery Router:")
    for k, v in s["router"].items():
        print(f"  {k:<32} {v}")

    print("\nBudget / Circuit Breaker:")
    ledger = s["budget"]["ledger"]
    print(f"  {'hourly_spend_usd':<32} ${ledger['hourly_spend_usd']}")
    print(f"  {'daily_spend_usd':<32} ${ledger['daily_spend_usd']}")
    print(f"  {'circuit_state':<32} {s['budget']['circuit_breaker']['state']}")

    print(f"\n{SEP}")
    print("Done. Run  benchmarks/run_benchmarks.py  for full cost tables.")
    print(SEP)


if __name__ == "__main__":
    main()
    sys.exit(0)
