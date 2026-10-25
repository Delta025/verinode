"""
Shared rate limiter for Gemini API calls.

Enforces per-minute (RPM) rate limiting and optional cost-based or
request-count budgets with exponential backoff on 429 errors.
Designed to be shared between the fact extractor and contradiction
checker so they draw from the same budget.

Thread-safe: uses a threading.Lock for concurrent access.
"""

import logging
import time
import threading
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


class DailyLimitReached(Exception):
    """Raised when the Gemini daily request budget is exhausted."""


class BudgetExhausted(Exception):
    """Raised when the cost budget has been reached."""


@dataclass
class RateLimiterStats:
    """Cumulative usage statistics for logging and resume."""
    total_requests: int = 0
    total_retries: int = 0
    total_429_errors: int = 0
    daily_requests: int = 0
    daily_limit_reached: bool = False
    budget_exhausted: bool = False
    estimated_cost_usd: float = 0.0
    start_time: float = field(default_factory=time.time)


class GeminiRateLimiter:
    """
    Thread-safe rate limiter for Gemini API calls.

    Args:
        rpm: Requests per minute (60 is conservative for paid tier
            with 2K RPM actual limit).
        rpd: Requests per day.  0 means unlimited (no daily cap).
        max_retries: Max retries per call on 429 before giving up.
        daily_budget_reserve: Stop early at ``rpd - reserve``
            to leave headroom.  Only applies when rpd > 0.
        max_cost_usd: Maximum estimated cost in USD before stopping.
            0.0 means no cost limit.  Based on approximate
            Gemini 2.0 Flash pricing (~$0.075 per 1K requests).
        cost_per_request: Estimated cost per API call in USD.
            Default is based on observed Gemini 2.0 Flash pricing.
    """

    # Approximate cost per request for Gemini 2.0 Flash
    # Based on observed $0.025 / 11000 requests = ~$0.0000023
    # Using conservative 10x multiplier to account for larger prompts
    DEFAULT_COST_PER_REQUEST = 0.000025

    def __init__(
        self,
        rpm: int = 60,
        rpd: int = 0,
        max_retries: int = 5,
        daily_budget_reserve: int = 50,
        max_cost_usd: float = 1.0,
        cost_per_request: float | None = None,
    ):
        self.rpm = rpm
        self.rpd = rpd
        self.max_retries = max_retries
        self.daily_budget_reserve = daily_budget_reserve
        self.max_cost_usd = max_cost_usd
        self.cost_per_request = (
            cost_per_request
            if cost_per_request is not None
            else self.DEFAULT_COST_PER_REQUEST
        )

        self._lock = threading.Lock()
        self._request_times: list[float] = []
        self._daily_count = 0
        self._stats = RateLimiterStats()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def stats(self) -> RateLimiterStats:
        return self._stats

    @property
    def daily_remaining(self) -> int:
        effective_limit = self.rpd - self.daily_budget_reserve
        return max(0, effective_limit - self._daily_count)

    def wait_for_slot(self) -> bool:
        """
        Block until a request slot is available.

        Returns:
            True if a slot was acquired, False if any budget
            (daily count or cost) is exhausted.
        """
        with self._lock:
            # Check cost budget
            if self.max_cost_usd > 0:
                estimated = self._daily_count * self.cost_per_request
                if estimated >= self.max_cost_usd:
                    self._stats.budget_exhausted = True
                    self._stats.estimated_cost_usd = estimated
                    logger.warning(
                        "Cost budget reached (~$%.4f / $%.2f limit, "
                        "%d requests). Stopping.",
                        estimated,
                        self.max_cost_usd,
                        self._daily_count,
                    )
                    return False

            # Check daily limit (only if rpd > 0)
            if self.rpd > 0:
                effective_limit = self.rpd - self.daily_budget_reserve
                if self._daily_count >= effective_limit:
                    self._stats.daily_limit_reached = True
                    logger.warning(
                        "Daily limit approaching (%d/%d used, reserve=%d). "
                        "Stopping.",
                        self._daily_count,
                        self.rpd,
                        self.daily_budget_reserve,
                    )
                    return False

            # Enforce per-minute rate
            now = time.time()
            window_start = now - 60.0
            self._request_times = [
                t for t in self._request_times if t > window_start
            ]

            if len(self._request_times) >= self.rpm:
                oldest = self._request_times[0]
                sleep_time = 60.0 - (now - oldest) + 0.5  # +0.5s safety buffer
                logger.info("Rate limit: sleeping %.1fs", sleep_time)
                # Release lock while sleeping so other threads aren't blocked
                self._lock.release()
                try:
                    time.sleep(sleep_time)
                finally:
                    self._lock.acquire()
                now = time.time()
                self._request_times = [
                    t for t in self._request_times if t > now - 60.0
                ]

            self._request_times.append(now)
            self._daily_count += 1
            self._stats.total_requests += 1
            self._stats.daily_requests = self._daily_count
            self._stats.estimated_cost_usd = (
                self._daily_count * self.cost_per_request
            )
            return True

    def handle_429(self, attempt: int) -> bool:
        """
        Handle a 429 rate-limit response with exponential backoff.

        Args:
            attempt: Zero-based retry attempt number.

        Returns:
            True if should retry, False if max retries exceeded.
        """
        self._stats.total_429_errors += 1
        self._stats.total_retries += 1

        if attempt >= self.max_retries:
            logger.error("Max retries (%d) exceeded on 429", self.max_retries)
            return False

        backoff = min(2 ** attempt * 6.0, 120.0)  # 6s, 12s, 24s, 48s, 96s
        logger.warning(
            "429 received, backing off %.1fs (attempt %d/%d)",
            backoff,
            attempt + 1,
            self.max_retries,
        )
        time.sleep(backoff)
        return True

    def log_progress(self) -> str:
        """Return a human-readable progress string."""
        elapsed = time.time() - self._stats.start_time
        hours = elapsed / 3600
        cost = self._daily_count * self.cost_per_request
        rpd_str = str(self.rpd) if self.rpd > 0 else "unlimited"
        budget_str = f"${self.max_cost_usd:.2f}" if self.max_cost_usd > 0 else "unlimited"
        return (
            f"Requests: {self._daily_count} (RPD: {rpd_str}) | "
            f"Est. cost: ${cost:.4f} / {budget_str} | "
            f"{self._stats.total_retries} retries, "
            f"{self._stats.total_429_errors} 429s | "
            f"{hours:.1f}h elapsed"
        )
