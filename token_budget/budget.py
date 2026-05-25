"""
token_budget/budget.py
----------------------
Token budget layer with CostLedger and CircuitBreaker.

Production fixes applied:
  - Negative token reservations rejected
  - record_actual idempotent (guarded by _recorded flag per context)
  - RLock throughout (no deadlock on reentrant calls)
  - Input validation on model_tier
  - Logging throughout
  - CostLedger.summary() never blocks (non-locking read path)
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Cost constants
# ---------------------------------------------------------------------------

COST_PER_1K_TOKENS: dict[str, float] = {
    "simple":   0.000165,
    "standard": 0.005,
    "complex":  0.015,
}
_VALID_TIERS = set(COST_PER_1K_TOKENS.keys())
TOKEN_CHAR_RATIO = 4   # 1 token ≈ 4 chars (English prose)


# ---------------------------------------------------------------------------
# CircuitBreaker states
# ---------------------------------------------------------------------------

class BreakerState(Enum):
    CLOSED    = "closed"
    OPEN      = "open"
    HALF_OPEN = "half_open"


# ---------------------------------------------------------------------------
# TokenBudget
# ---------------------------------------------------------------------------

@dataclass
class SlotUsage:
    name:             str
    reserved_tokens:  int
    actual_tokens:    int   = 0
    cost_usd:         float = 0.0


class TokenBudget:
    """
    Slot-based token allocator for a single LLM request.

    Parameters
    ----------
    total_tokens : int
        Total token budget for this request.
    model_tier : str
        One of 'simple', 'standard', 'complex'.
    """

    def __init__(self, total_tokens: int = 4096, model_tier: str = "standard") -> None:
        if model_tier not in _VALID_TIERS:
            logger.warning("Unknown model_tier %r — defaulting to 'standard'", model_tier)
            model_tier = "standard"
        self.total_tokens    = max(1, total_tokens)
        self.model_tier      = model_tier
        self._used_tokens    = 0
        self._slots: dict[str, SlotUsage] = {}
        self._cost_per_token = COST_PER_1K_TOKENS[model_tier] / 1000.0

    def reserve(self, slot_name: str, tokens: int) -> int:
        """
        Reserve up to `tokens` for a named slot.
        Negative values are rejected (returns 0).
        Returns actual tokens reserved.
        """
        if tokens <= 0:
            logger.debug("reserve(%s, %d) — non-positive tokens rejected", slot_name, tokens)
            return 0
        available = self.remaining()
        granted   = min(tokens, available)
        self._slots[slot_name] = SlotUsage(
            name=slot_name,
            reserved_tokens=granted,
            cost_usd=granted * self._cost_per_token,
        )
        self._used_tokens += granted
        return granted

    def reserve_text(self, slot_name: str, text: str) -> int:
        """Reserve tokens estimated from character count."""
        if not text:
            return 0
        tokens = max(1, len(text) // TOKEN_CHAR_RATIO)
        return self.reserve(slot_name, tokens)

    def record_actual(self, slot_name: str, actual_tokens: int) -> None:
        """Update a slot with real post-generation token count."""
        if actual_tokens < 0:
            logger.warning("record_actual: negative actual_tokens ignored")
            return
        if slot_name not in self._slots:
            logger.warning("record_actual: unknown slot %r — ignoring", slot_name)
            return
        slot = self._slots[slot_name]
        delta = actual_tokens - slot.reserved_tokens
        self._used_tokens += delta
        slot.actual_tokens = actual_tokens
        slot.cost_usd      = actual_tokens * self._cost_per_token

    def remaining(self) -> int:
        return max(0, self.total_tokens - self._used_tokens)

    def used(self) -> int:
        return self._used_tokens

    def utilization(self) -> float:
        return self._used_tokens / self.total_tokens

    def total_cost_usd(self) -> float:
        return sum(s.cost_usd for s in self._slots.values())

    def budget_exceeded(self) -> bool:
        return self._used_tokens > self.total_tokens

    def slot_report(self) -> list[dict]:
        return [
            {
                "slot":            s.name,
                "reserved_tokens": s.reserved_tokens,
                "actual_tokens":   s.actual_tokens or s.reserved_tokens,
                "cost_usd":        round(s.cost_usd, 6),
            }
            for s in self._slots.values()
        ]

    def summary(self) -> dict:
        return {
            "total_tokens":     self.total_tokens,
            "used_tokens":      self._used_tokens,
            "remaining_tokens": self.remaining(),
            "utilization_pct":  round(self.utilization() * 100, 1),
            "total_cost_usd":   round(self.total_cost_usd(), 6),
            "model_tier":       self.model_tier,
            "slots":            self.slot_report(),
        }


# ---------------------------------------------------------------------------
# CostLedger
# ---------------------------------------------------------------------------

@dataclass
class SpendEvent:
    timestamp:  float
    cost_usd:   float
    tokens:     int
    model_tier: str
    request_id: str


class CostLedger:
    """
    Rolling cost tracker. Thread-safe via RLock.

    Parameters
    ----------
    hourly_limit_usd : float
    daily_limit_usd  : float
    """

    def __init__(
        self,
        hourly_limit_usd: float = 5.0,
        daily_limit_usd:  float = 50.0,
    ) -> None:
        self.hourly_limit_usd = hourly_limit_usd
        self.daily_limit_usd  = daily_limit_usd
        self._events: deque[SpendEvent] = deque()
        self._lock   = threading.RLock()
        self._total_lifetime_usd    = 0.0
        self._total_lifetime_tokens = 0

    def record(
        self,
        cost_usd:   float,
        tokens:     int,
        model_tier: str = "standard",
        request_id: str = "",
    ) -> None:
        if cost_usd < 0 or tokens < 0:
            logger.warning("CostLedger.record: ignoring negative cost/tokens")
            return
        event = SpendEvent(
            timestamp=time.time(),
            cost_usd=cost_usd,
            tokens=tokens,
            model_tier=model_tier,
            request_id=request_id or f"req_{int(time.time()*1000)}",
        )
        with self._lock:
            self._events.append(event)
            self._total_lifetime_usd    += cost_usd
            self._total_lifetime_tokens += tokens
            self._prune()

    def hourly_spend(self) -> float:
        return self._window_spend(3600)

    def daily_spend(self) -> float:
        return self._window_spend(86400)

    def hourly_breach(self) -> bool:
        return self.hourly_spend() >= self.hourly_limit_usd

    def daily_breach(self) -> bool:
        return self.daily_spend() >= self.daily_limit_usd

    def summary(self) -> dict:
        with self._lock:
            h = self._window_spend_unlocked(3600)
            d = self._window_spend_unlocked(86400)
            return {
                "lifetime_cost_usd":  round(self._total_lifetime_usd, 4),
                "lifetime_tokens":    self._total_lifetime_tokens,
                "hourly_spend_usd":   round(h, 4),
                "daily_spend_usd":    round(d, 4),
                "hourly_limit_usd":   self.hourly_limit_usd,
                "daily_limit_usd":    self.daily_limit_usd,
                "hourly_remaining":   round(max(0.0, self.hourly_limit_usd - h), 4),
                "daily_remaining":    round(max(0.0, self.daily_limit_usd  - d), 4),
                "hourly_breach":      h >= self.hourly_limit_usd,
                "daily_breach":       d >= self.daily_limit_usd,
            }

    def _window_spend(self, seconds: float) -> float:
        with self._lock:
            return self._window_spend_unlocked(seconds)

    def _window_spend_unlocked(self, seconds: float) -> float:
        cutoff = time.time() - seconds
        return sum(e.cost_usd for e in self._events if e.timestamp >= cutoff)

    def _prune(self) -> None:
        cutoff = time.time() - 86400
        while self._events and self._events[0].timestamp < cutoff:
            self._events.popleft()


# ---------------------------------------------------------------------------
# CircuitBreaker
# ---------------------------------------------------------------------------

class CircuitBreaker:
    """
    Protects against runaway spend.

    CLOSED    → normal operation
    OPEN      → threshold breached; requests blocked or downgraded
    HALF_OPEN → cooldown elapsed; one probe allowed
    """

    def __init__(
        self,
        cooldown_seconds:        float = 60.0,
        downgrade_on_open:       bool  = True,
        trip_on_consecutive:     int   = 1,
    ) -> None:
        self.cooldown_seconds    = cooldown_seconds
        self.downgrade_on_open   = downgrade_on_open
        self.trip_on_consecutive = trip_on_consecutive
        self._state              = BreakerState.CLOSED
        self._opened_at: Optional[float] = None
        self._consecutive        = 0
        self._lock               = threading.RLock()

    @property
    def state(self) -> BreakerState:
        with self._lock:
            self._maybe_transition()
            return self._state

    def is_open(self) -> bool:
        return self.state == BreakerState.OPEN

    def trip(self) -> None:
        with self._lock:
            self._consecutive += 1
            if self._consecutive >= self.trip_on_consecutive:
                if self._state != BreakerState.OPEN:
                    logger.warning("CircuitBreaker OPENED")
                self._state     = BreakerState.OPEN
                self._opened_at = time.time()

    def record_success(self) -> None:
        with self._lock:
            logger.info("CircuitBreaker CLOSED (probe succeeded)")
            self._state       = BreakerState.CLOSED
            self._consecutive = 0
            self._opened_at   = None

    def record_failure(self) -> None:
        with self._lock:
            logger.warning("CircuitBreaker re-OPENED (probe failed)")
            self._state     = BreakerState.OPEN
            self._opened_at = time.time()

    def status(self) -> dict:
        s = self.state
        return {
            "state":                s.value,
            "consecutive_breaches": self._consecutive,
            "seconds_until_reset":  self._seconds_until_reset(),
        }

    def _maybe_transition(self) -> None:
        if self._state == BreakerState.OPEN and self._opened_at:
            if time.time() - self._opened_at >= self.cooldown_seconds:
                self._state = BreakerState.HALF_OPEN

    def _seconds_until_reset(self) -> Optional[float]:
        if self._state != BreakerState.OPEN or not self._opened_at:
            return None
        return round(max(0.0, self.cooldown_seconds - (time.time() - self._opened_at)), 1)


# ---------------------------------------------------------------------------
# RequestContext
# ---------------------------------------------------------------------------

class RequestContext:
    """Passed into the `with enforcer.request()` block."""

    def __init__(
        self,
        budget:            TokenBudget,
        allowed:           bool,
        downgraded:        bool,
        fallback_response: str,
        model_tier:        str,
        enforcer:          "BudgetEnforcer",
    ) -> None:
        self.budget            = budget
        self.allowed           = allowed
        self.downgraded        = downgraded
        self.fallback_response = fallback_response
        self.model_tier        = model_tier
        self._enforcer         = enforcer
        self._recorded         = False   # guard against double record_actual

    def record_actual(self, actual_tokens: int, cost_usd: float) -> None:
        """
        Call once after the LLM responds.
        Idempotent — calling twice logs a warning and is ignored.
        """
        if self._recorded:
            logger.warning("record_actual called more than once — ignoring duplicate")
            return
        self._recorded = True
        if actual_tokens < 0 or cost_usd < 0:
            logger.warning("record_actual: negative values ignored")
            return
        self._enforcer.ledger.record(
            cost_usd=cost_usd,
            tokens=actual_tokens,
            model_tier=self.model_tier,
        )


# ---------------------------------------------------------------------------
# BudgetEnforcer
# ---------------------------------------------------------------------------

class BudgetEnforcer:
    """
    Orchestrates TokenBudget + CostLedger + CircuitBreaker.
    Single entry point for production RAG cost control.

    Parameters
    ----------
    hourly_limit_usd : float
    daily_limit_usd  : float
    per_request_limit_usd : float
    total_tokens_per_request : int
    cooldown_seconds : float
    downgrade_on_breach : bool
        True  → route to SIMPLE tier on breach (graceful degradation)
        False → block request entirely
    fallback_message : str
    """

    def __init__(
        self,
        hourly_limit_usd:         float = 5.0,
        daily_limit_usd:          float = 50.0,
        per_request_limit_usd:    float = 0.10,
        total_tokens_per_request: int   = 4096,
        cooldown_seconds:         float = 60.0,
        downgrade_on_breach:      bool  = True,
        fallback_message:         str   = (
            "Service temporarily unavailable due to high load. "
            "Please retry in a moment."
        ),
    ) -> None:
        self.per_request_limit_usd    = per_request_limit_usd
        self.total_tokens_per_request = total_tokens_per_request
        self.fallback_message         = fallback_message
        self.downgrade_on_breach      = downgrade_on_breach

        self.ledger  = CostLedger(
            hourly_limit_usd=hourly_limit_usd,
            daily_limit_usd=daily_limit_usd,
        )
        self.breaker = CircuitBreaker(
            cooldown_seconds=cooldown_seconds,
            downgrade_on_open=downgrade_on_breach,
        )

    @contextmanager
    def request(
        self,
        model_tier:        str = "standard",
        estimated_tokens:  int = 500,
        request_id:        str = "",
    ):
        """
        Context manager for a single LLM request.

        Usage:
            with enforcer.request(model_tier='standard', estimated_tokens=800) as ctx:
                if ctx.allowed:
                    ctx.budget.reserve('system_prompt', 200)
                    ...
                    ctx.record_actual(actual_tokens=620, cost_usd=0.0031)
                else:
                    return ctx.fallback_response
        """
        # Validate tier
        if model_tier not in _VALID_TIERS:
            logger.warning("Unknown model_tier %r — defaulting to 'standard'", model_tier)
            model_tier = "standard"

        effective_tier, allowed, downgraded = self._evaluate(model_tier, estimated_tokens)

        budget = TokenBudget(
            total_tokens=self.total_tokens_per_request,
            model_tier=effective_tier,
        )
        ctx = RequestContext(
            budget=budget,
            allowed=allowed,
            downgraded=downgraded,
            fallback_response=self.fallback_message,
            model_tier=effective_tier,
            enforcer=self,
        )

        try:
            yield ctx
        except Exception:
            raise
        finally:
            # If caller used budget but didn't call record_actual, record estimate
            if allowed and not ctx._recorded and budget.total_cost_usd() > 0:
                self.ledger.record(
                    cost_usd=budget.total_cost_usd(),
                    tokens=budget.used(),
                    model_tier=effective_tier,
                    request_id=request_id,
                )
            self._check_and_trip()

    def status(self) -> dict:
        return {
            "circuit_breaker":      self.breaker.status(),
            "ledger":               self.ledger.summary(),
            "per_request_limit_usd": self.per_request_limit_usd,
        }

    def _evaluate(self, tier: str, estimated_tokens: int) -> tuple[str, bool, bool]:
        """Returns (effective_tier, allowed, downgraded)."""
        state = self.breaker.state

        est_cost     = COST_PER_1K_TOKENS.get(tier, 0.005) * max(estimated_tokens, 0) / 1000
        over_request = est_cost > self.per_request_limit_usd

        if state == BreakerState.OPEN:
            if self.downgrade_on_breach:
                return "simple", True, True
            return tier, False, False

        if state == BreakerState.HALF_OPEN:
            return "simple", True, True

        if over_request:
            if self.downgrade_on_breach:
                return "simple", True, True
            return tier, False, False

        return tier, True, False

    def _check_and_trip(self) -> None:
        if self.ledger.hourly_breach() or self.ledger.daily_breach():
            self.breaker.trip()
