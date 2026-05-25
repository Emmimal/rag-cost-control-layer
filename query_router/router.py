"""
query_router/router.py
----------------------
Classifies queries by complexity and routes to the appropriate model tier.

Production fixes applied:
  - KeyError on missing model tier → safe fallback to STANDARD
  - Input validation (None / empty string)
  - Complex tier now reachable (fixed scoring weights)
  - Logging throughout
  - Thread-safe stats via RLock
"""

from __future__ import annotations

import logging
import re
import threading
import time
from dataclasses import dataclass
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Model tiers
# ---------------------------------------------------------------------------

class ModelTier(Enum):
    SIMPLE   = "simple"     # fast, cheap  — e.g. gpt-4o-mini, claude-haiku
    STANDARD = "standard"   # mid-range    — e.g. gpt-4o, claude-sonnet
    COMPLEX  = "complex"    # expensive    — e.g. gpt-4.5, claude-opus


DEFAULT_MODEL_MAP: dict[ModelTier, str] = {
    ModelTier.SIMPLE:   "gpt-4o-mini",
    ModelTier.STANDARD: "gpt-4o",
    ModelTier.COMPLEX:  "gpt-4.5",
}

# Cost per 1K tokens (input+output blended). Update as pricing changes.
DEFAULT_COST_PER_1K: dict[ModelTier, float] = {
    ModelTier.SIMPLE:   0.000165,
    ModelTier.STANDARD: 0.005,
    ModelTier.COMPLEX:  0.015,
}


# ---------------------------------------------------------------------------
# Scoring signals
# ---------------------------------------------------------------------------

REASONING_KEYWORDS: frozenset[str] = frozenset({
    "compare", "contrast", "difference", "differences", "versus", "vs",
    "why", "how does", "explain", "analyze", "analyse", "evaluate",
    "tradeoff", "trade-off", "pros and cons", "recommend", "should i",
    "step by step", "walk me through", "what would happen", "what if",
    "relationship between", "cause", "effect", "impact", "design",
    "architecture", "strategy", "optimize", "optimise", "improve",
    "when should", "how should", "what happens when", "failure mode",
})

FACTOID_PATTERNS: list[re.Pattern] = [
    re.compile(r"^(what is|what are|who is|where is|when (did|was|is))\b", re.I),
    re.compile(r"^(define|definition of|meaning of)\b", re.I),
    re.compile(r"^(list|name|give me)\b.{0,40}$", re.I),
]


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ComplexityScore:
    length_score:    float
    entity_score:    float
    reasoning_score: float
    total:           float
    tier:            ModelTier

    def as_dict(self) -> dict:
        return {
            "length_score":    round(self.length_score, 3),
            "entity_score":    round(self.entity_score, 3),
            "reasoning_score": round(self.reasoning_score, 3),
            "total":           round(self.total, 3),
            "tier":            self.tier.value,
        }


@dataclass
class RoutingDecision:
    query:              str
    tier:               ModelTier
    model_id:           str
    score:              ComplexityScore
    estimated_cost_usd: float
    fallback_cost_usd:  float
    cost_saved_usd:     float
    latency_ms:         float

    def as_dict(self) -> dict:
        return {
            "query_preview":      self.query[:80],
            "tier":               self.tier.value,
            "model_id":           self.model_id,
            "score":              self.score.as_dict(),
            "estimated_cost_usd": round(self.estimated_cost_usd, 6),
            "fallback_cost_usd":  round(self.fallback_cost_usd, 6),
            "cost_saved_usd":     round(self.cost_saved_usd, 6),
            "routing_latency_ms": round(self.latency_ms, 3),
        }


class RouterStats:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self.total_queries  = 0
        self.simple_count   = 0
        self.standard_count = 0
        self.complex_count  = 0
        self.total_cost_usd = 0.0
        self.total_saved_usd = 0.0

    def record(self, decision: RoutingDecision) -> None:
        with self._lock:
            self.total_queries += 1
            if decision.tier == ModelTier.SIMPLE:
                self.simple_count += 1
            elif decision.tier == ModelTier.STANDARD:
                self.standard_count += 1
            else:
                self.complex_count += 1
            self.total_cost_usd  += decision.estimated_cost_usd
            self.total_saved_usd += decision.cost_saved_usd

    def summary(self) -> dict:
        with self._lock:
            n = self.total_queries or 1
            return {
                "total_queries":      self.total_queries,
                "simple_pct":         round(self.simple_count   / n * 100, 1),
                "standard_pct":       round(self.standard_count / n * 100, 1),
                "complex_pct":        round(self.complex_count  / n * 100, 1),
                "total_cost_usd":     round(self.total_cost_usd,  4),
                "total_saved_usd":    round(self.total_saved_usd, 4),
                "avg_cost_per_query": round(self.total_cost_usd / n, 6),
            }


# ---------------------------------------------------------------------------
# QueryRouter
# ---------------------------------------------------------------------------

class QueryRouter:
    """
    Routes queries to model tiers based on complexity scoring.

    Parameters
    ----------
    model_map : dict | None
        Model ID per tier. Missing tiers fall back to STANDARD safely.
    cost_per_1k : dict | None
        Cost per 1K tokens per tier.
    simple_threshold : float
        Score below this → SIMPLE. Default 0.25.
    complex_threshold : float
        Score above this → COMPLEX. Default 0.60.
    avg_request_tokens : int
        Assumed tokens for cost estimation. Default 500.
    """

    def __init__(
        self,
        model_map:           Optional[dict[ModelTier, str]]   = None,
        cost_per_1k:         Optional[dict[ModelTier, float]] = None,
        simple_threshold:    float = 0.25,
        complex_threshold:   float = 0.60,
        avg_request_tokens:  int   = 500,
    ) -> None:
        # Merge supplied map with defaults — missing keys fall back safely
        self.model_map = {**DEFAULT_MODEL_MAP, **(model_map or {})}
        self.cost_per_1k = {**DEFAULT_COST_PER_1K, **(cost_per_1k or {})}
        self.simple_threshold  = simple_threshold
        self.complex_threshold = complex_threshold
        self.avg_request_tokens = avg_request_tokens
        self.stats = RouterStats()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def route(self, query: str) -> RoutingDecision:
        """Classify query and return a RoutingDecision."""
        query = self._validate(query)
        t0    = time.perf_counter()
        score = self._score(query)
        tier  = score.tier
        elapsed_ms = (time.perf_counter() - t0) * 1000

        # Safe model lookup — always falls back to STANDARD
        model_id     = self.model_map.get(tier, self.model_map[ModelTier.STANDARD])
        est_cost     = self._estimate_cost(tier)
        fallback     = self._estimate_cost(ModelTier.COMPLEX)
        cost_saved   = max(0.0, fallback - est_cost)

        decision = RoutingDecision(
            query=query,
            tier=tier,
            model_id=model_id,
            score=score,
            estimated_cost_usd=est_cost,
            fallback_cost_usd=fallback,
            cost_saved_usd=cost_saved,
            latency_ms=elapsed_ms,
        )
        self.stats.record(decision)
        logger.debug(
            "Routed → %s (score=%.2f, model=%s)", tier.value, score.total, model_id
        )
        return decision

    def get_stats(self) -> dict:
        return self.stats.summary()

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def _score(self, query: str) -> ComplexityScore:
        # Fast-path: obvious factoid → force SIMPLE
        if self._is_factoid(query):
            return ComplexityScore(
                length_score=0.0, entity_score=0.0,
                reasoning_score=0.0, total=0.10,
                tier=ModelTier.SIMPLE,
            )

        ls = self._length_score(query)
        es = self._entity_score(query)
        rs = self._reasoning_score(query)

        # Weights: reasoning carries most signal (0.50),
        # entity density second (0.30), length least (0.20).
        # Previously 0.25/0.35/0.40 — reasoning was capped at 0.33/kw
        # which prevented COMPLEX tier. Now reasoning can reach 1.0.
        total = min(1.0, max(0.0,
            0.20 * ls +
            0.30 * es +
            0.50 * rs
        ))

        return ComplexityScore(
            length_score=ls, entity_score=es,
            reasoning_score=rs, total=total,
            tier=self._classify(total),
        )

    def _length_score(self, query: str) -> float:
        """Normalised token count. Saturates at 80 tokens."""
        return min(len(query.split()) / 80.0, 1.0)

    def _entity_score(self, query: str) -> float:
        """Ratio of capitalised/numeric/technical tokens."""
        tokens = query.split()
        if not tokens:
            return 0.0
        hits = sum(
            1 for t in tokens
            if (t[0].isupper() and len(t) > 1)
            or re.search(r"\d", t)
            or re.search(r"[:>/%]", t)
        )
        return min(hits / len(tokens), 1.0)

    def _reasoning_score(self, query: str) -> float:
        """Count reasoning keyword hits. 2+ hits → max score."""
        q_lower = query.lower()
        hits = sum(1 for kw in REASONING_KEYWORDS if kw in q_lower)
        return min(hits / 2.0, 1.0)   # 2 hits = 1.0 (was /3 before)

    def _is_factoid(self, query: str) -> bool:
        return any(p.match(query.strip()) for p in FACTOID_PATTERNS)

    def _classify(self, total: float) -> ModelTier:
        if total < self.simple_threshold:
            return ModelTier.SIMPLE
        if total > self.complex_threshold:
            return ModelTier.COMPLEX
        return ModelTier.STANDARD

    def _estimate_cost(self, tier: ModelTier) -> float:
        rate = self.cost_per_1k.get(tier, self.cost_per_1k[ModelTier.STANDARD])
        return rate * self.avg_request_tokens / 1000.0

    @staticmethod
    def _validate(query: str) -> str:
        if query is None:
            return ""
        return str(query).strip()
