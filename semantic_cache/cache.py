"""
semantic_cache/cache.py
-----------------------
Thread-safe semantic cache for RAG pipelines.
Pure Python TF-IDF — no PyTorch, no external dependencies, instant exit.

Production fixes applied:
  - Thread-safe via RLock on _entries and _embedder
  - O(1) embedding lookup (cached per query, invalidated on vocab update)
  - Input validation (None / empty string handled)
  - No false hits on unrelated queries
"""

from __future__ import annotations

import logging
import math
import re
import threading
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pure-Python TF-IDF embedder
# ---------------------------------------------------------------------------

class _TFIDFEmbedder:
    """
    Thread-safe TF-IDF vectoriser.
    Vocabulary grows as new queries are added.
    Embedding cache is invalidated whenever vocabulary changes.

    To swap in an API embedder (e.g. OpenAI text-embedding-3-small):
        class OpenAIEmbedder:
            def fit(self, texts): pass
            def embed(self, text):
                import openai
                r = openai.embeddings.create(
                    model="text-embedding-3-small", input=text
                )
                return r.data[0].embedding
    """

    def __init__(self) -> None:
        self._vocab: dict[str, int] = {}
        self._idf: dict[str, float] = {}
        self._doc_count: int = 0
        self._term_doc_freq: Counter = Counter()
        self._embed_cache: dict[str, list[float]] = {}
        self._lock = threading.RLock()

    def fit(self, texts: list[str]) -> None:
        """Update vocabulary and IDF. Invalidates embedding cache."""
        with self._lock:
            self._doc_count += len(texts)
            for text in texts:
                for t in set(self._tokenize(text)):
                    self._term_doc_freq[t] += 1
            for term in self._term_doc_freq:
                if term not in self._vocab:
                    self._vocab[term] = len(self._vocab)
            for term, df in self._term_doc_freq.items():
                self._idf[term] = math.log((self._doc_count + 1) / (df + 1)) + 1.0
            self._embed_cache.clear()   # vocab changed → old vectors are stale

    def embed(self, text: str) -> list[float]:
        with self._lock:
            if text in self._embed_cache:
                return self._embed_cache[text]
            vec = self._compute(text)
            self._embed_cache[text] = vec
            return vec

    def _compute(self, text: str) -> list[float]:
        tokens = self._tokenize(text)
        if not tokens:
            return [0.0] * max(len(self._vocab), 1)
        tf = Counter(tokens)
        total = len(tokens)
        dim = max(len(self._vocab), 1)
        vec = [0.0] * dim
        for term, count in tf.items():
            if term in self._vocab:
                vec[self._vocab[term]] = (count / total) * self._idf.get(term, 1.0)
        return _l2_norm(vec)

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return re.findall(r"[a-z0-9]+", text.lower())


def _l2_norm(vec: list[float]) -> list[float]:
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


def _cosine(a: list[float], b: list[float]) -> float:
    n = min(len(a), len(b))
    if n == 0:
        return 0.0
    dot = sum(a[i] * b[i] for i in range(n))
    na  = math.sqrt(sum(v * v for v in a))
    nb  = math.sqrt(sum(v * v for v in b))
    return dot / (na * nb) if na and nb else 0.0


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class CacheEntry:
    query:          str
    response:       str
    timestamp:      float
    hit_count:      int   = 0


@dataclass
class CacheStats:
    total_requests:         int   = 0
    cache_hits:             int   = 0
    cache_misses:           int   = 0
    total_cost_saved_usd:   float = 0.0
    total_latency_saved_ms: float = 0.0

    @property
    def hit_rate(self) -> float:
        return self.cache_hits / self.total_requests if self.total_requests else 0.0

    def summary(self) -> dict:
        return {
            "total_requests":         self.total_requests,
            "cache_hits":             self.cache_hits,
            "cache_misses":           self.cache_misses,
            "hit_rate_pct":           round(self.hit_rate * 100, 1),
            "total_cost_saved_usd":   round(self.total_cost_saved_usd, 4),
            "total_latency_saved_ms": round(self.total_latency_saved_ms, 1),
        }


# ---------------------------------------------------------------------------
# SemanticCache
# ---------------------------------------------------------------------------

class SemanticCache:
    """
    Thread-safe semantic cache for RAG query responses.

    Parameters
    ----------
    threshold : float
        Cosine similarity threshold. Default 0.75 (TF-IDF scale).
    max_size : int
        Max entries before LRU eviction.
    ttl_seconds : float | None
        Per-entry TTL. None = no expiry.
    cost_per_llm_call_usd : float
        Used for savings tracking only.
    avg_llm_latency_ms : float
        Used for latency savings tracking only.
    """

    def __init__(
        self,
        threshold:              float         = 0.75,
        max_size:               int           = 1000,
        ttl_seconds:            Optional[float] = None,
        cost_per_llm_call_usd:  float         = 0.004,
        avg_llm_latency_ms:     float         = 700.0,
    ) -> None:
        self.threshold             = threshold
        self.max_size              = max_size
        self.ttl_seconds           = ttl_seconds
        self.cost_per_llm_call_usd = cost_per_llm_call_usd
        self.avg_llm_latency_ms    = avg_llm_latency_ms
        self._embedder             = _TFIDFEmbedder()
        self._entries: list[CacheEntry] = []
        self.stats                 = CacheStats()
        self._lock                 = threading.RLock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get(self, query: str) -> Optional[str]:
        """Return cached response on hit, None on miss."""
        query = self._validate(query)
        if query is None:
            return None

        with self._lock:
            self.stats.total_requests += 1
            if not self._entries:
                self.stats.cache_misses += 1
                return None

            q_vec = self._embedder.embed(query)
            best, best_sim = self._find_best(q_vec)

            if best is not None and best_sim >= self.threshold:
                best.hit_count += 1
                self.stats.cache_hits += 1
                self.stats.total_cost_saved_usd   += self.cost_per_llm_call_usd
                self.stats.total_latency_saved_ms += self.avg_llm_latency_ms
                logger.debug("Cache hit (sim=%.3f): %s", best_sim, query[:60])
                return best.response

            self.stats.cache_misses += 1
            return None

    def set(self, query: str, response: str) -> None:
        """Store a query-response pair. Thread-safe."""
        query = self._validate(query)
        if query is None:
            return
        if not response:
            logger.warning("set() called with empty response for query: %s", query[:60])
            return

        with self._lock:
            if len(self._entries) >= self.max_size:
                self._evict_lru()
            self._embedder.fit([query])   # update vocab before embedding
            self._entries.append(CacheEntry(
                query=query,
                response=response,
                timestamp=time.time(),
            ))

    def invalidate(self, query: str) -> bool:
        """Remove a cached entry. Returns True if found and removed."""
        query = self._validate(query)
        if query is None:
            return False
        with self._lock:
            q_vec = self._embedder.embed(query)
            for i, entry in enumerate(self._entries):
                ev = self._embedder.embed(entry.query)
                if _cosine(q_vec, ev) >= self.threshold:
                    self._entries.pop(i)
                    return True
        return False

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()
            self.stats = CacheStats()

    def size(self) -> int:
        with self._lock:
            return len(self._entries)

    def get_stats(self) -> dict:
        with self._lock:
            return self.stats.summary()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _find_best(
        self, q_vec: list[float]
    ) -> tuple[Optional[CacheEntry], float]:
        best_entry = None
        best_sim   = -1.0
        now        = time.time()

        for entry in self._entries:
            if self.ttl_seconds and now - entry.timestamp > self.ttl_seconds:
                continue
            ev  = self._embedder.embed(entry.query)
            sim = _cosine(q_vec, ev)
            if sim > best_sim:
                best_sim   = sim
                best_entry = entry

        return best_entry, best_sim

    def _evict_lru(self) -> None:
        """Evict lowest hit_count, then oldest."""
        if self._entries:
            lru = min(self._entries, key=lambda e: (e.hit_count, e.timestamp))
            self._entries.remove(lru)
            logger.debug("Evicted LRU entry: %s", lru.query[:60])

    @staticmethod
    def _validate(query: str) -> Optional[str]:
        """Return stripped query or None if invalid."""
        if query is None:
            logger.warning("get/set called with None query — skipping")
            return None
        q = str(query).strip()
        if not q:
            logger.warning("get/set called with empty query — skipping")
            return None
        return q
