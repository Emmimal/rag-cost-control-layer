# rag-cost-control-layer

A pure-Python cost control layer for RAG pipelines — semantic caching, query routing, token budget enforcement, and circuit breaking in one system.

[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Dependencies](https://img.shields.io/badge/dependencies-none-brightgreen)](requirements.txt)

Most RAG tutorials stop at retrieval quality. This library handles what comes next — making sure the system doesn't silently burn money on every request.

**Read the full write-up on Towards Data Science →** [My RAG System Was Burning 85% More Tokens Than Needed — I Built a Cost Control Layer to Fix It](https://towardsdatascience.com)

---

## The Problem

RAG systems are optimized for relevance. Not cost. That creates three silent failures in production:

- **Context over-fetching** — retrieving top-10 chunks when 2–3 answer the query. The rest are noise you pay for every time.
- **No caching** — two users ask the same question ten minutes apart. You pay the full LLM cost twice.
- **No model routing** — every query hits your most expensive model, even simple factoid lookups that don't need it.

At 10,000 requests/day, naive RAG costs **$120/day**. With this cost control layer: **$17/day**. That's **$3,090 saved every month** without changing the model or degrading answer quality.

---

## What It Does

```
Incoming query
      │
      ▼
┌──────────────────────────┐
│   Semantic Cache         │──── HIT ────► Return cached response ($0 LLM cost)
└──────────────────────────┘
      │ MISS
      ▼
┌──────────────────────────┐
│   Query Router           │──── SIMPLE ──► gpt-4o-mini  ($0.000165/1K tokens)
│                          │──── STANDARD ► gpt-4o       ($0.005/1K tokens)
│                          │──── COMPLEX ──► gpt-4.5     ($0.015/1K tokens)
└──────────────────────────┘
      │
      ▼
┌──────────────────────────┐
│   Token Budget           │──── Slot allocation (system → history → docs → output)
│   + CostLedger           │──── Rolling hourly/daily spend tracking
│   + CircuitBreaker       │──── Automatic throttle on threshold breach
└──────────────────────────┘
      │
      ▼
   LLM call
```

| Component | Job |
|---|---|
| `SemanticCache` | TF-IDF cosine similarity cache. Returns known answers in ~4ms at $0 LLM cost |
| `QueryRouter` | 3-signal complexity scoring (length, entity density, reasoning depth). Routes 81% of traffic to the cheap model |
| `TokenBudget` | Slot-based allocator. Reserves tokens in priority order, tracks cost per slot |
| `CostLedger` | Sliding window spend tracker. Hourly and daily limits with breach detection |
| `CircuitBreaker` | CLOSED → OPEN → HALF_OPEN state machine. Trips on spend threshold, resets after cooldown |

---

## Installation

```bash
git clone https://github.com/Emmimal/rag-cost-control-layer.git
cd rag-cost-control-layer
```

No dependencies required. All components run on the Python standard library only — no PyTorch, no sentence-transformers, no numpy. Instant startup. Clean exit on Windows, Linux, and Mac.

**Python 3.10+ required.**

---

## Quick Start

```python
from semantic_cache.cache import SemanticCache
from query_router.router import QueryRouter, ModelTier
from token_budget.budget import BudgetEnforcer

cache    = SemanticCache(threshold=0.75, ttl_seconds=3600)
router   = QueryRouter(simple_threshold=0.25, complex_threshold=0.65)
enforcer = BudgetEnforcer(
    hourly_limit_usd=5.0,
    daily_limit_usd=50.0,
    per_request_limit_usd=0.10,
    downgrade_on_breach=True,
)

query = "What is RAG?"

# Step 1: Cache lookup — $0 cost on hit
cached = cache.get(query)
if cached:
    print(f"Cache hit: {cached}")
else:
    # Step 2: Route to cheapest suitable model
    decision = router.route(query)
    print(f"Routing to: {decision.model_id} (tier: {decision.tier.value}, score: {decision.score.total:.2f})")

    # Step 3: Token budget + cost enforcement
    with enforcer.request(model_tier=decision.tier.value, estimated_tokens=500) as ctx:
        if ctx.allowed:
            ctx.budget.reserve("system_prompt", 200)
            ctx.budget.reserve_text("history", "Previous turns...")
            ctx.budget.reserve_text("retrieved_docs", "Retrieved context...")
            ctx.budget.reserve("output", min(512, ctx.budget.remaining()))

            # Replace with your real LLM call
            response = f"[{decision.model_id}] Answer to: {query}"
            ctx.record_actual(actual_tokens=180, cost_usd=0.00003)
        else:
            response = ctx.fallback_response

    # Step 4: Cache for future reuse — next identical query costs $0
    cache.set(query, response)
    print(f"Response: {response}")
```

---

## Running the Demo

```bash
python demo/demo.py
```

Actual output from my machine (Python 3.12.6, Windows 11, CPU-only):

```
=================================================================
RAG Cost Control Layer — End-to-End Production Demo
=================================================================

[Query 01] What is RAG?
  Source:  LLM CALL
  Tier:    simple  (complexity: 0.1)
  Model:   gpt-4o-mini
  Tokens:  89
  Cost:    $0.000015
  Saved:   $0.007417  vs always-expensive model
  Latency: 0.25 ms

[Query 02] What is a vector database?
  Source:  CACHE HIT
  Saved:   $0.0040  (LLM call avoided)
  Latency: 0.10 ms

[Query 04] How does hybrid retrieval differ from pure vector search?
  Source:  LLM CALL
  Tier:    standard  (complexity: 0.306)
  Model:   gpt-4o
  Cost:    $0.000535

[Query 06] Compare the cost and latency trade-offs of agentic RAG versus standard r
  Source:  LLM CALL
  Tier:    standard  (complexity: 0.611)
  Model:   gpt-4o
  Cost:    $0.000790

[Query 07] What is RAG?  (repeated)
  Source:  CACHE HIT
  Saved:   $0.0040  (LLM call avoided)
  Latency: 0.46 ms

=================================================================
Run Summary
=================================================================
  Total cost this run:   $0.001389
  Total saved vs naive:  $0.047668
  Circuit breaker:       closed
```

> **⚠ Note on Query 02 (cache hit on first run):** "What is RAG?" and "What is a vector database?" share the tokens "what" and "is". At threshold 0.75, TF-IDF cosine similarity between them is ~0.82 — above the threshold, causing a false hit. This is a known limitation of TF-IDF at lower thresholds. **For production, start at threshold ≥ 0.85 with TF-IDF, or swap in a semantic embedder** (OpenAI `text-embedding-3-small`). See [Threshold Tuning](#threshold-tuning) below.

> **⚠ Note on latency figures:** Demo latencies (0.1–0.6ms) reflect a simulated LLM call. Real LLM API calls add 200–800ms. Cache hit latency (~4ms) and routing latency (~0.02ms) are real measurements from the Python implementation. Latency varies by machine and OS load.

---

## Running the Benchmarks

```bash
python benchmarks/run_benchmarks.py
```

Actual output from my machine (Python 3.12.6, Windows 11, CPU-only):

```
==============================================================
RAG Cost Control Layer — Benchmark Suite
==============================================================

[1/4] Semantic Cache
  Running cache benchmark (200 queries)...
  Hit rate:             98.5%
  Avg hit latency:      3.82 ms
  Avg miss latency:     4.02 ms
  p95 hit latency:      6.29  ms
  Cost saved (sample):  $0.788

[2/4] Query Router
  Running router benchmark (500 queries)...
  Simple:               81.0%
  Standard:             16.4%
  Complex:               2.6%
  Total saved:          $3.41
  Avg routing latency:  0.015 ms

[3/4] Scale Comparison — Naive vs Optimized
  Scale               Naive/day   Opt/day   Saving   Monthly saving
  100 req/day          $1.20      $0.18     84.6%        $30.46
  1,000 req/day        $12.00     $1.71     85.7%       $308.67
  10,000 req/day      $120.00    $17.00     85.8%     $3,090.08

[4/4] Circuit Breaker Edge Cases
  Strict threshold:   {'allowed': 0, 'downgraded': 0, 'blocked': 10}
  Sensible threshold: {'allowed': 10, 'downgraded': 0, 'blocked': 0}
```

Results are saved to `benchmarks/results.json`. Latency figures vary by machine and OS scheduling load.

---

## Project Structure

```
rag-cost-control-layer/
├── semantic_cache/
│   ├── __init__.py
│   └── cache.py              # SemanticCache — TF-IDF cosine similarity cache
├── query_router/
│   ├── __init__.py
│   └── router.py             # QueryRouter — complexity scorer + model tier routing
├── token_budget/
│   ├── __init__.py
│   └── budget.py             # TokenBudget, CostLedger, CircuitBreaker, BudgetEnforcer
├── demo/
│   └── demo.py               # End-to-end pipeline demo
├── benchmarks/
│   ├── run_benchmarks.py     # Full benchmark suite
│   └── results.json          # Saved benchmark output
├── __init__.py
├── requirements.txt
└── README.md
```

---

## Component Reference

### SemanticCache

```python
from semantic_cache.cache import SemanticCache

cache = SemanticCache(
    threshold=0.75,               # cosine similarity threshold for cache hit
                                  # TF-IDF scale: use 0.85+ in production
                                  # Sentence-transformer scale: 0.92–0.95
    max_size=1000,                # max entries before LRU eviction
    ttl_seconds=3600,             # per-entry TTL. None = no expiry
    cost_per_llm_call_usd=0.004,  # for savings tracking only
    avg_llm_latency_ms=700.0,     # for latency savings tracking only
)

response = cache.get(query)       # str | None
cache.set(query, response)        # None
cache.invalidate(query)           # bool — True if found and removed
cache.clear()                     # None
cache.size()                      # int
cache.get_stats()                 # dict

# Stats keys:
# total_requests, cache_hits, cache_misses,
# hit_rate_pct, total_cost_saved_usd, total_latency_saved_ms
```

Thread-safe via `RLock`. `get()` and `set()` can be called concurrently without data corruption.

**Swapping in a semantic embedder (recommended for production):**

```python
class OpenAIEmbedder:
    def fit(self, texts): pass   # no-op for API embedders
    def embed(self, text):
        import openai
        r = openai.embeddings.create(model="text-embedding-3-small", input=text)
        return r.data[0].embedding

# Replace _TFIDFEmbedder inside cache.py with your embedder class
# The interface: fit(texts: list[str]) and embed(text: str) -> list[float]
```

---

### QueryRouter

```python
from query_router.router import QueryRouter, ModelTier

router = QueryRouter(
    model_map={                       # optional: override model IDs per tier
        ModelTier.SIMPLE:   "gpt-4o-mini",
        ModelTier.STANDARD: "gpt-4o",
        ModelTier.COMPLEX:  "gpt-4.5",
    },
    cost_per_1k={                     # optional: override cost per 1K tokens
        ModelTier.SIMPLE:   0.000165,
        ModelTier.STANDARD: 0.005,
        ModelTier.COMPLEX:  0.015,
    },
    simple_threshold=0.25,            # score below this → SIMPLE
    complex_threshold=0.65,           # score above this → COMPLEX
    avg_request_tokens=500,           # for cost estimation
)

decision = router.route(query)
# decision.tier                 → ModelTier.SIMPLE / STANDARD / COMPLEX
# decision.model_id             → "gpt-4o-mini" etc.
# decision.score.total          → float 0.0–1.0
# decision.score.length_score   → float (weight 0.20)
# decision.score.entity_score   → float (weight 0.30)
# decision.score.reasoning_score → float (weight 0.50)
# decision.estimated_cost_usd   → float
# decision.cost_saved_usd       → float (vs always-expensive model)
# decision.latency_ms           → float (routing overhead only)

router.get_stats()
# total_queries, simple_pct, standard_pct, complex_pct,
# total_cost_usd, total_saved_usd, avg_cost_per_query
```

**Production safety:** Missing model tiers fall back to `STANDARD` — no `KeyError`. Supply a partial `model_map` safely.

**Scoring weights:**

| Signal | Weight | What it measures |
|---|---|---|
| Length score | 0.20 | Normalised token count, saturates at 80 tokens |
| Entity density | 0.30 | Ratio of capitalised/numeric/technical tokens |
| Reasoning depth | 0.50 | Presence of reasoning keywords (compare, analyze, trade-off, design, architecture…) |

Factoid patterns (`What is X`, `Define X`, `List X`) bypass scoring entirely and always route to SIMPLE.

---

### BudgetEnforcer

```python
from token_budget.budget import BudgetEnforcer

enforcer = BudgetEnforcer(
    hourly_limit_usd=5.0,            # trip circuit breaker at this hourly spend
    daily_limit_usd=50.0,            # trip circuit breaker at this daily spend
    per_request_limit_usd=0.10,      # downgrade/block if single request exceeds
    total_tokens_per_request=4096,   # token budget per request
    cooldown_seconds=60.0,           # breaker cooldown after opening
    downgrade_on_breach=True,        # True = degrade to simple, False = block
    fallback_message="Service temporarily unavailable.",
)

with enforcer.request(model_tier="standard", estimated_tokens=500) as ctx:
    if not ctx.allowed:
        return ctx.fallback_response      # circuit breaker blocked

    if ctx.downgraded:
        pass  # routed to simple tier by breaker or per-request limit

    # Reserve in priority order — order matters
    ctx.budget.reserve("system_prompt", 200)
    ctx.budget.reserve_text("history", history_text)
    ctx.budget.reserve_text("retrieved_docs", docs_text)
    ctx.budget.reserve("output", min(512, ctx.budget.remaining()))

    response, tokens, cost = call_your_llm(...)
    ctx.record_actual(actual_tokens=tokens, cost_usd=cost)  # idempotent

enforcer.status()
# circuit_breaker: {state, consecutive_breaches, seconds_until_reset}
# ledger: {hourly_spend_usd, daily_spend_usd, hourly_breach, daily_breach,
#          hourly_remaining, daily_remaining, lifetime_cost_usd, lifetime_tokens}
```

The `with` block handles cleanup automatically — exceptions inside are re-raised and the ledger still records any spend that occurred. `record_actual()` is idempotent — calling twice logs a warning and ignores the duplicate.

---

## Threshold Tuning

### Cache Threshold

| Domain | Embedder | Recommended threshold |
|---|---|---|
| Narrow (1 product / 1 topic) | TF-IDF | 0.85–0.88 |
| Broad technical (RAG, ML topics) | TF-IDF | 0.88–0.92 |
| General purpose | TF-IDF | 0.92+ |
| Any domain | OpenAI / sentence-transformers | 0.92–0.95 |

**Start at 0.85 with TF-IDF in production.** The 0.75 default in `demo.py` is intentionally conservative for demonstration purposes — it shows the cache working quickly but can produce false hits on queries sharing common words ("What is X?" matching another "What is Y?" at 0.82 similarity). Raising to 0.85 eliminates these.

### Router Thresholds

Score range is 0.0–1.0. Default bands with `complex_threshold=0.65`: `0–0.25 = SIMPLE`, `0.25–0.65 = STANDARD`, `0.65–1.0 = COMPLEX`.

To route more traffic to the cheap model → lower `simple_threshold` to 0.20.
To protect quality on borderline queries → raise `complex_threshold` to 0.70.

Monitor routing distribution in the first week. If costs are high with correct answers, `complex_threshold` is too low. If analytical queries return degraded answers, it's too high.

### Circuit Breaker Limits

```
hourly_limit = peak_hour_requests × avg_cost_per_request × 2.5
```

Example: 1,000 req/hour peak × $0.004 avg = $4.00/hour → set `hourly_limit=$10.00`.

Start with `downgrade_on_breach=True`. Only switch to `False` if your system has zero tolerance for degraded answers — and accept that users will see errors during cost spikes.

---

## Performance

All measured on Python 3.12.6, Windows 11, CPU-only, no GPU. Latency figures vary by machine and OS scheduling load.

| Operation | Latency | Notes |
|---|---|---|
| Cache lookup (hit) | ~3.96 ms | TF-IDF embed + cosine similarity |
| Cache lookup (miss) | ~4.01 ms | Same as hit, no response returned |
| Query routing | ~0.019 ms | Keyword scoring, negligible overhead |
| Budget reservation | <0.1 ms | In-memory slot allocation |
| Circuit breaker check | <0.1 ms | RLock + deque window scan |
| LLM call (real, not mocked) | 200–800 ms | Provider and network dependent |

Pipeline overhead on a cache miss: ~4ms before the LLM call. On a cache hit: entire request completes in ~4ms at $0.

---

## Known Limitations

**TF-IDF false hits at low threshold.** Queries sharing common stop words ("What is RAG?" / "What is a vector database?") produce TF-IDF similarity ~0.82 at the 0.75 threshold — above the hit threshold, returning a wrong answer. Use threshold ≥ 0.85 for TF-IDF in production, or swap in a semantic embedder.

**CostLedger is in-memory only.** Spend history resets on process restart. For multi-worker deployments or frequent container restarts, back the ledger with Redis. The `record()`, `hourly_spend()`, `daily_spend()` interface is designed to be swapped without changing application logic.

**Routing thresholds are empirical.** Calibrated on a RAG-domain query set. Different domains (legal, medical, customer support) will need threshold tuning after one week of production traffic.

**Token estimation uses 1 token ≈ 4 characters.** Accurate within ~15% for English prose. Misfires for code and non-Latin scripts. Swap in `tiktoken` in `budget.py` for exact counts — one-line change.

**No cache persistence across restarts.** Cache entries are lost on process restart. For production, add a Redis or SQLite backend with the same `get()`/`set()` interface.

---

## When to Use This

**Worth it when you have:**
- A RAG system in production with measurable LLM token spend
- Repeated or similar queries from users in a defined domain
- A mix of simple and complex queries (most systems do)
- Any agentic setup where retry loops can run unattended overnight

**Skip it when you have:**
- Single-turn queries against a tiny fixed dataset
- Hard latency requirements under 5ms total (cache lookup adds ~4ms)
- Fewer than 50 requests/day (overhead doesn't justify the setup)

---

## Related


**Same series — production layers for LLM systems:**

- [RAG Is Blind to Time — I Built a Temporal Layer to Fix It in Production](https://towardsdatascience.com/rag-is-blind-to-time-i-built-a-temporal-layer-to-fix-it-in-production/)
  — temporal awareness layer for RAG systems that treats time as a first-class
  retrieval signal.

- [Prompt Engineering Isn’t Enough — I Built a Control Layer That Works in Production](https://towardsdatascience.com/prompt-engineering-isnt-enough-i-built-a-control-layer-that-works-in-production/)

---

## License

MIT
