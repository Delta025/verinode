"""Tests for src.core.rate_limiter."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from unittest.mock import MagicMock, patch

import pytest

from core.rate_limiter import (
    BudgetExhausted,
    DailyLimitReached,
    GeminiRateLimiter,
    RateLimiterStats,
)

# ------------------------------------------------------------------
# Construction
# ------------------------------------------------------------------


class TestConstruction:
    def test_default_params(self):
        limiter = GeminiRateLimiter()
        assert limiter.rpm == 60
        assert limiter.rpd == 0
        assert limiter.max_retries == 5
        assert limiter.daily_budget_reserve == 50
        assert limiter.max_cost_usd == 1.0
        assert limiter.cost_per_request == GeminiRateLimiter.DEFAULT_COST_PER_REQUEST

    def test_custom_params(self):
        limiter = GeminiRateLimiter(
            rpm=10,
            rpd=1500,
            max_retries=3,
            daily_budget_reserve=100,
            max_cost_usd=5.0,
            cost_per_request=0.001,
        )
        assert limiter.rpm == 10
        assert limiter.rpd == 1500
        assert limiter.max_retries == 3
        assert limiter.daily_budget_reserve == 100
        assert limiter.max_cost_usd == 5.0
        assert limiter.cost_per_request == 0.001

    def test_stats_initialized(self):
        limiter = GeminiRateLimiter()
        s = limiter.stats
        assert isinstance(s, RateLimiterStats)
        assert s.total_requests == 0
        assert s.total_retries == 0
        assert s.total_429_errors == 0
        assert s.daily_requests == 0
        assert s.daily_limit_reached is False
        assert s.budget_exhausted is False


# ------------------------------------------------------------------
# wait_for_slot
# ------------------------------------------------------------------


class TestWaitForSlot:
    @patch("core.rate_limiter.time")
    def test_under_rpm_limit_returns_true_no_wait(self, mock_time):
        """When under RPM limit, returns True without sleeping."""
        mock_time.time.return_value = 1000.0
        limiter = GeminiRateLimiter(rpm=10, rpd=0, max_cost_usd=0.0)

        result = limiter.wait_for_slot()

        assert result is True
        mock_time.sleep.assert_not_called()

    @patch("core.rate_limiter.time")
    def test_at_rpm_limit_sleeps_then_returns_true(self, mock_time):
        """When at RPM limit, sleeps until a slot opens, then returns True."""
        base = 1000.0
        call_count = [0]

        def fake_time():
            call_count[0] += 1
            # After sleep, advance time by 61 seconds so window clears
            if call_count[0] > 10:
                return base + 61.0
            return base

        mock_time.time.side_effect = fake_time
        mock_time.sleep = MagicMock()

        limiter = GeminiRateLimiter(rpm=5, rpd=0, max_cost_usd=0.0)
        # Fill up the window with 5 timestamps all at base time
        limiter._request_times = [base] * 5

        result = limiter.wait_for_slot()

        assert result is True
        mock_time.sleep.assert_called_once()
        # Sleep time = 60.0 - (now - oldest) + 0.5 = 60.0 - 0 + 0.5 = 60.5
        sleep_arg = mock_time.sleep.call_args[0][0]
        assert sleep_arg == pytest.approx(60.5, abs=0.1)

    @patch("core.rate_limiter.time")
    def test_at_rpd_limit_returns_false(self, mock_time):
        """When daily count reaches effective limit, returns False."""
        mock_time.time.return_value = 1000.0
        limiter = GeminiRateLimiter(rpm=60, rpd=100, daily_budget_reserve=10, max_cost_usd=0.0)
        # Effective limit = 100 - 10 = 90
        limiter._daily_count = 90

        result = limiter.wait_for_slot()

        assert result is False
        assert limiter.stats.daily_limit_reached is True

    @patch("core.rate_limiter.time")
    def test_cost_budget_exhausted_returns_false(self, mock_time):
        """When estimated cost >= max_cost_usd, returns False."""
        mock_time.time.return_value = 1000.0
        limiter = GeminiRateLimiter(rpm=60, rpd=0, max_cost_usd=0.01, cost_per_request=0.001)
        # 10 requests * 0.001 = 0.01 >= 0.01 limit
        limiter._daily_count = 10

        result = limiter.wait_for_slot()

        assert result is False
        assert limiter.stats.budget_exhausted is True
        assert limiter.stats.estimated_cost_usd == pytest.approx(0.01)

    @patch("core.rate_limiter.time")
    def test_rpd_disabled_no_daily_limit(self, mock_time):
        """When rpd=0, daily limit is never enforced."""
        mock_time.time.return_value = 1000.0
        limiter = GeminiRateLimiter(rpm=60, rpd=0, max_cost_usd=0.0)
        # Even with huge daily count, should not be blocked
        limiter._daily_count = 999999

        result = limiter.wait_for_slot()

        assert result is True
        assert limiter.stats.daily_limit_reached is False

    @patch("core.rate_limiter.time")
    def test_cost_budget_disabled_when_zero(self, mock_time):
        """When max_cost_usd=0.0, cost budget is never enforced."""
        mock_time.time.return_value = 1000.0
        limiter = GeminiRateLimiter(rpm=60, rpd=0, max_cost_usd=0.0)
        limiter._daily_count = 999999

        result = limiter.wait_for_slot()

        assert result is True
        assert limiter.stats.budget_exhausted is False


# ------------------------------------------------------------------
# Sliding window cleanup
# ------------------------------------------------------------------


class TestSlidingWindow:
    @patch("core.rate_limiter.time")
    def test_old_timestamps_removed(self, mock_time):
        """Timestamps older than 60 seconds are pruned from the window."""
        mock_time.time.return_value = 1100.0
        limiter = GeminiRateLimiter(rpm=10, rpd=0, max_cost_usd=0.0)
        # These are all > 60 seconds old relative to now=1100
        limiter._request_times = [1000.0, 1010.0, 1020.0]

        limiter.wait_for_slot()

        # Old timestamps should be pruned; only the new one remains
        assert len(limiter._request_times) == 1
        assert limiter._request_times[0] == 1100.0

    @patch("core.rate_limiter.time")
    def test_recent_timestamps_kept(self, mock_time):
        """Timestamps within the 60-second window are preserved."""
        mock_time.time.return_value = 1050.0
        limiter = GeminiRateLimiter(rpm=10, rpd=0, max_cost_usd=0.0)
        # 1000 is 50s old (within 60s window), 990 is out
        limiter._request_times = [990.0, 1000.0, 1010.0]

        limiter.wait_for_slot()

        # 990.0 should be pruned (> 60s old), 1000.0 and 1010.0 kept, plus new at 1050.0
        assert 990.0 not in limiter._request_times
        assert 1000.0 in limiter._request_times
        assert 1010.0 in limiter._request_times
        assert 1050.0 in limiter._request_times


# ------------------------------------------------------------------
# handle_429
# ------------------------------------------------------------------


class TestHandle429:
    @patch("core.rate_limiter.time")
    def test_first_attempt_sleeps_returns_true(self, mock_time):
        """First attempt (0) should sleep and return True."""
        mock_time.sleep = MagicMock()
        limiter = GeminiRateLimiter(max_retries=5)

        result = limiter.handle_429(attempt=0)

        assert result is True
        mock_time.sleep.assert_called_once()
        # Backoff = min(2^0 * 6, 120) = 6.0
        assert mock_time.sleep.call_args[0][0] == pytest.approx(6.0)

    @patch("core.rate_limiter.time")
    def test_at_max_retries_returns_false(self, mock_time):
        """When attempt >= max_retries, returns False without sleeping."""
        mock_time.sleep = MagicMock()
        limiter = GeminiRateLimiter(max_retries=3)

        result = limiter.handle_429(attempt=3)

        assert result is False
        mock_time.sleep.assert_not_called()

    @patch("core.rate_limiter.time")
    def test_exponential_backoff_sequence(self, mock_time):
        """Backoff increases exponentially: 6, 12, 24, 48, 96."""
        mock_time.sleep = MagicMock()
        limiter = GeminiRateLimiter(max_retries=5)

        expected_backoffs = [6.0, 12.0, 24.0, 48.0, 96.0]
        for attempt, expected in enumerate(expected_backoffs):
            limiter.handle_429(attempt=attempt)
            actual = mock_time.sleep.call_args[0][0]
            assert actual == pytest.approx(expected), f"Attempt {attempt}: expected {expected}, got {actual}"

    @patch("core.rate_limiter.time")
    def test_backoff_capped_at_120(self, mock_time):
        """Backoff is capped at 120 seconds regardless of attempt number."""
        mock_time.sleep = MagicMock()
        limiter = GeminiRateLimiter(max_retries=20)

        # attempt=5: 2^5 * 6 = 192 -> capped to 120
        limiter.handle_429(attempt=5)
        assert mock_time.sleep.call_args[0][0] == pytest.approx(120.0)

    @patch("core.rate_limiter.time")
    def test_handle_429_increments_stats(self, mock_time):
        """Each handle_429 call increments retry and 429 counters."""
        mock_time.sleep = MagicMock()
        limiter = GeminiRateLimiter(max_retries=5)

        limiter.handle_429(attempt=0)
        limiter.handle_429(attempt=1)

        assert limiter.stats.total_429_errors == 2
        assert limiter.stats.total_retries == 2

    @patch("core.rate_limiter.time")
    def test_handle_429_at_max_still_increments_stats(self, mock_time):
        """Even when max_retries exceeded, stats are incremented."""
        mock_time.sleep = MagicMock()
        limiter = GeminiRateLimiter(max_retries=2)

        result = limiter.handle_429(attempt=2)

        assert result is False
        assert limiter.stats.total_429_errors == 1
        assert limiter.stats.total_retries == 1


# ------------------------------------------------------------------
# Stats tracking
# ------------------------------------------------------------------


class TestStatsTracking:
    @patch("core.rate_limiter.time")
    def test_total_requests_increments(self, mock_time):
        """Each wait_for_slot call increments total_requests."""
        mock_time.time.return_value = 1000.0
        limiter = GeminiRateLimiter(rpm=60, rpd=0, max_cost_usd=0.0)

        limiter.wait_for_slot()
        limiter.wait_for_slot()
        limiter.wait_for_slot()

        assert limiter.stats.total_requests == 3
        assert limiter.stats.daily_requests == 3

    @patch("core.rate_limiter.time")
    def test_daily_remaining_decreases(self, mock_time):
        """daily_remaining decreases with each request."""
        mock_time.time.return_value = 1000.0
        limiter = GeminiRateLimiter(rpm=60, rpd=100, daily_budget_reserve=10, max_cost_usd=0.0)
        # Effective limit = 90, so daily_remaining starts at 90
        assert limiter.daily_remaining == 90

        limiter.wait_for_slot()
        assert limiter.daily_remaining == 89

    @patch("core.rate_limiter.time")
    def test_estimated_cost_tracks(self, mock_time):
        """Estimated cost updates after each slot acquisition."""
        mock_time.time.return_value = 1000.0
        limiter = GeminiRateLimiter(rpm=60, rpd=0, max_cost_usd=0.0, cost_per_request=0.01)

        limiter.wait_for_slot()
        assert limiter.stats.estimated_cost_usd == pytest.approx(0.01)

        limiter.wait_for_slot()
        assert limiter.stats.estimated_cost_usd == pytest.approx(0.02)


# ------------------------------------------------------------------
# log_progress
# ------------------------------------------------------------------


class TestLogProgress:
    @patch("core.rate_limiter.time")
    def test_log_progress_format(self, mock_time):
        """log_progress returns a human-readable string."""
        mock_time.time.side_effect = [100.0, 200.0]  # init start_time, then log call
        limiter = GeminiRateLimiter(rpm=60, rpd=1500, max_cost_usd=2.0)
        limiter._daily_count = 42

        progress = limiter.log_progress()

        assert "42" in progress
        assert "1500" in progress
        assert "$2.00" in progress

    @patch("core.rate_limiter.time")
    def test_log_progress_unlimited_rpd(self, mock_time):
        """When rpd=0, log shows 'unlimited' for RPD."""
        mock_time.time.side_effect = [100.0, 200.0]
        limiter = GeminiRateLimiter(rpm=60, rpd=0, max_cost_usd=0.0)

        progress = limiter.log_progress()

        assert "unlimited" in progress


# ------------------------------------------------------------------
# Exception classes
# ------------------------------------------------------------------


class TestExceptions:
    def test_daily_limit_reached_is_exception(self):
        assert issubclass(DailyLimitReached, Exception)

    def test_budget_exhausted_is_exception(self):
        assert issubclass(BudgetExhausted, Exception)

    def test_daily_limit_raised(self):
        with pytest.raises(DailyLimitReached):
            raise DailyLimitReached("daily limit hit")

    def test_budget_exhausted_raised(self):
        with pytest.raises(BudgetExhausted):
            raise BudgetExhausted("budget hit")
