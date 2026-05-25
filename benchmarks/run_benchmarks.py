"""
benchmarks/run_benchmarks.py
-----------------------------
Full benchmark suite — generates tables for the article.
Pure Python, no external dependencies, exits instantly.

Run:
    python benchmarks/run_benchmarks.py
"""

from __future__ import annotations

import json
import math
import os
import random
import sys
import time

# ── Path fix ──────────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
# ──────────────────────────────────────────────────────────────────────────

from semantic_cache.cache import SemanticCache
from query_router.router import QueryRouter, ModelTier
from token_budget.budget import BudgetEnforcer, COST_PER_1K_TOKENS


# ---------------------------------------------------------------------------
# Query sets
# ---------------------------------------------------------------------------

SIMPLE_QUERIES = [
    "What is RAG?",
    "What is a vector database?",
    "Define semantic search.",
    "What is a token?",
    "What is cosine similarity?",
    "What does LLM stand for?",
    "List the main RAG components.",
    "What is an embedding?",
    "What is chunking in RAG?",
    "Define hallucination in LLMs.",
]

STANDARD_QUERIES = [
    "How does hybrid retrieval differ from pure vector search?",
    "What are the trade-offs between BM25 and dense embeddings?",
    "Explain how re-ranking improves RAG accuracy.",
    "How should I choose chunk size for my RAG pipeline?",
    "What is Reciprocal Rank Fusion and when should I use it?",
    "How does semantic caching reduce LLM costs in production?",
    "What are the main failure modes of naive RAG systems?",
    "How do I implement memory decay in a conversational RAG system?",
]

COMPLEX_QUERIES = [
    "Compare the cost and latency trade-offs of agentic RAG versus standard retrieval pipelines at 10,000 requests per day.",
    "Analyze the relationship between chunk overlap, retrieval recall, and token budget for legal document search.",
    "Why do RAG systems degrade in multi-tenant environments? Design a monitoring and remediation strategy.",
    "Contrast context engineering for customer support versus code generation RAG systems.",
]


def build_query_mix(n: int, seed: int = 42) -> list[str]:
    """60% simple, 30% standard, 10% complex + 20% repeats."""
    random.seed(seed)
    queries = []
    while len(queries) < int(n * 0.8):
        r = random.random()
        if r < 0.60:
            queries.append(random.choice(SIMPLE_QUERIES))
        elif r < 0.90:
            queries.append(random.choice(STANDARD_QUERIES))
        else:
            queries.append(random.choice(COMPLEX_QUERIES))
    repeat_pool = SIMPLE_QUERIES[:5] * 20
    for i in range(n - len(queries)):
        queries.append(repeat_pool[i % len(repeat_pool)])
    random.shuffle(queries)
    return queries[:n]


# ---------------------------------------------------------------------------
# Cost models
# ---------------------------------------------------------------------------

def naive_cost_usd(n: int, tokens_per_query: int = 800) -> float:
    return n * tokens_per_query * COST_PER_1K_TOKENS["complex"] / 1000


def optimized_cost_usd(
    n: int,
    cache_hit_rate: float = 0.28,
    simple_pct: float = 0.62,
    standard_pct: float = 0.29,
    tokens_per_query: int = 800,
) -> float:
    cached     = int(n * cache_hit_rate)
    uncached   = n - cached
    simple_n   = int(uncached * simple_pct)
    standard_n = int(uncached * standard_pct)
    complex_n  = uncached - simple_n - standard_n
    return round(
        simple_n   * tokens_per_query * COST_PER_1K_TOKENS["simple"]   / 1000 +
        standard_n * tokens_per_query * COST_PER_1K_TOKENS["standard"] / 1000 +
        complex_n  * tokens_per_query * COST_PER_1K_TOKENS["complex"]  / 1000 +
        cached * 0.0001,
        4
    )


# ---------------------------------------------------------------------------
# Benchmark 1: Semantic cache
# ---------------------------------------------------------------------------

def benchmark_cache(n: int = 200) -> dict:
    print(f"  Running cache benchmark ({n} queries)...", flush=True)
    cache = SemanticCache(threshold=0.75, cost_per_llm_call_usd=0.004)
    queries = build_query_mix(n)

    # Pre-populate with 40% of queries
    for q in queries[:int(n * 0.4)]:
        cache.set(q, "Cached RAG response.")

    hit_lat, miss_lat = [], []
    for q in queries:
        t0     = time.perf_counter()
        result = cache.get(q)
        ms     = (time.perf_counter() - t0) * 1000
        if result is not None:
            hit_lat.append(ms)
        else:
            miss_lat.append(ms)
            cache.set(q, "Cached RAG response.")

    stats = cache.get_stats()
    return {
        **stats,
        "avg_hit_latency_ms":  round(sum(hit_lat)  / max(len(hit_lat),  1), 2),
        "avg_miss_latency_ms": round(sum(miss_lat) / max(len(miss_lat), 1), 2),
        "p95_hit_latency_ms":  round(sorted(hit_lat)[int(len(hit_lat) * 0.95)] if hit_lat else 0, 2),
    }


# ---------------------------------------------------------------------------
# Benchmark 2: Query router
# ---------------------------------------------------------------------------

def benchmark_router(n: int = 500) -> dict:
    print(f"  Running router benchmark ({n} queries)...", flush=True)
    router  = QueryRouter()
    queries = build_query_mix(n)
    t0 = time.perf_counter()
    for q in queries:
        router.route(q)
    total_ms = (time.perf_counter() - t0) * 1000
    return {
        **router.get_stats(),
        "total_routing_ms":       round(total_ms, 1),
        "avg_routing_latency_ms": round(total_ms / n, 3),
    }


# ---------------------------------------------------------------------------
# Benchmark 3: Scale comparison
# ---------------------------------------------------------------------------

def benchmark_scale() -> list[dict]:
    print("  Running scale comparison...", flush=True)
    rows = []
    for label, n in [("100 req/day", 100), ("1,000 req/day", 1_000), ("10,000 req/day", 10_000)]:
        naive = naive_cost_usd(n)
        opt   = optimized_cost_usd(n)
        pct   = (naive - opt) / naive * 100 if naive else 0
        rows.append({
            "scale":                   label,
            "naive_per_day_usd":       round(naive, 2),
            "optimized_per_day_usd":   round(opt,   2),
            "saving_per_day_usd":      round(naive - opt, 2),
            "saving_pct":              round(pct, 1),
            "naive_per_month_usd":     round(naive * 30, 2),
            "optimized_per_month_usd": round(opt   * 30, 2),
            "saving_per_month_usd":    round((naive - opt) * 30, 2),
        })
    return rows


# ---------------------------------------------------------------------------
# Benchmark 4: Circuit breaker edge cases
# ---------------------------------------------------------------------------

def benchmark_circuit_breaker() -> dict:
    print("  Running circuit breaker edge cases...", flush=True)

    def run(enforcer, n: int = 10) -> dict:
        allowed = blocked = downgraded = 0
        for _ in range(n):
            with enforcer.request(model_tier="standard", estimated_tokens=400) as ctx:
                if not ctx.allowed:
                    blocked += 1
                elif ctx.downgraded:
                    downgraded += 1
                else:
                    allowed += 1
                    ctx.record_actual(actual_tokens=380, cost_usd=0.0019)
        return {"allowed": allowed, "downgraded": downgraded, "blocked": blocked}

    strict = BudgetEnforcer(
        hourly_limit_usd=0.001, daily_limit_usd=0.01,
        per_request_limit_usd=0.001, cooldown_seconds=30.0,
        downgrade_on_breach=False,
    )
    sensible = BudgetEnforcer(
        hourly_limit_usd=5.0, daily_limit_usd=50.0,
        per_request_limit_usd=0.10, cooldown_seconds=60.0,
        downgrade_on_breach=True,
    )
    return {
        "strict_threshold":   run(strict),
        "sensible_threshold": run(sensible),
        "recommendation": (
            "Set hourly_limit to 2-3x your expected peak — not your average. "
            "Use downgrade_on_breach=True to degrade gracefully instead of blocking users."
        ),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    SEP = "=" * 62
    print(SEP)
    print("RAG Cost Layer — Benchmark Suite")
    print(SEP)

    results = {}

    # 1. Cache
    print("\n[1/4] Semantic Cache")
    cr = benchmark_cache(200)
    results["semantic_cache"] = cr
    print(f"  Hit rate:             {cr['hit_rate_pct']}%")
    print(f"  Avg hit latency:      {cr['avg_hit_latency_ms']} ms")
    print(f"  Avg miss latency:     {cr['avg_miss_latency_ms']} ms")
    print(f"  p95 hit latency:      {cr['p95_hit_latency_ms']} ms")
    print(f"  Cost saved (sample):  ${cr['total_cost_saved_usd']}")

    # 2. Router
    print("\n[2/4] Query Router")
    rr = benchmark_router(500)
    results["query_router"] = rr
    print(f"  Simple:               {rr.get('simple_pct')}%")
    print(f"  Standard:             {rr.get('standard_pct')}%")
    print(f"  Complex:              {rr.get('complex_pct')}%")
    print(f"  Total saved:          ${rr.get('total_saved_usd')}")
    print(f"  Avg routing latency:  {rr.get('avg_routing_latency_ms')} ms")

    # 3. Scale table
    print("\n[3/4] Scale Comparison — Naive vs Optimized")
    sr = benchmark_scale()
    results["scale_comparison"] = sr
    print(f"\n  {'Scale':<18} {'Naive/day':>10} {'Opt/day':>9} {'Saving':>8}  {'Monthly saving':>14}")
    print(f"  {'-'*18} {'-'*10} {'-'*9} {'-'*8}  {'-'*14}")
    for row in sr:
        print(
            f"  {row['scale']:<18}"
            f"  ${row['naive_per_day_usd']:>8.2f}"
            f"  ${row['optimized_per_day_usd']:>7.2f}"
            f"  {row['saving_pct']:>6.1f}%"
            f"  ${row['saving_per_month_usd']:>13.2f}"
        )

    # 4. Circuit breaker
    print("\n[4/4] Circuit Breaker Edge Cases")
    cb = benchmark_circuit_breaker()
    results["circuit_breaker"] = cb
    print(f"  Strict threshold:   {cb['strict_threshold']}")
    print(f"  Sensible threshold: {cb['sensible_threshold']}")
    print(f"  Tip: {cb['recommendation']}")

    # Save
    out = os.path.join(_HERE, "results.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{SEP}")
    print(f"✓ Results saved → {out}")
    print(SEP)


if __name__ == "__main__":
    main()
    sys.exit(0)
