import os
import sys
import time
from unittest.mock import MagicMock, patch
import pytest

# mock OUTPUT_DB_URL so server.py can be imported without crashing
os.environ["OUTPUT_DB_URL"] = "postgresql://mock"

sys.modules["psycopg2"] = MagicMock()
sys.modules["psycopg2.extras"] = MagicMock()
sys.modules["mcp"] = MagicMock()
sys.modules["mcp.server"] = MagicMock()
sys.modules["mcp.server.fastmcp"] = MagicMock()

from server import EmailRateLimiter

class TestEmailRateLimiter:
    def setup_method(self, method):
        self.mock_time = 1000.0
        self.patcher = patch('time.monotonic', side_effect=lambda: self.mock_time)
        self.patcher.start()

    def teardown_method(self, method):
        self.patcher.stop()

    def test_acquire_burst_limit(self):
        limiter = EmailRateLimiter(max_emails_per_minute=10, max_burst=15)

        # We should be able to acquire 15 tokens immediately
        for _ in range(15):
            assert limiter.acquire() is True

        # The 16th should fail
        assert limiter.acquire() is False

    def test_refill_over_time(self):
        limiter = EmailRateLimiter(max_emails_per_minute=60, max_burst=60)

        # Exhaust all 60 burst tokens
        for _ in range(60):
            limiter.acquire()

        assert limiter.acquire() is False

        # Rate is 60/min = 1/sec
        # Advance time by 0.5 seconds, shouldn't be enough for a token
        self.mock_time += 0.5
        assert limiter.acquire() is False

        # Advance time by another 0.5 seconds (total 1.0s), should get 1 token
        self.mock_time += 0.5
        assert limiter.acquire() is True
        assert limiter.acquire() is False

        # Advance time by 10 seconds, should get 10 tokens
        self.mock_time += 10.0
        for _ in range(10):
            assert limiter.acquire() is True
        assert limiter.acquire() is False

    def test_wait_time(self):
        limiter = EmailRateLimiter(max_emails_per_minute=60, max_burst=5)

        # wait_time should be 0.0 when tokens are available
        assert limiter.wait_time() == 0.0

        # Exhaust tokens
        for _ in range(5):
            limiter.acquire()

        # After exhausting, we need 1 full token. Rate is 1 token / second.
        # Calculation: wait_time = deficit / refill_rate
        # deficit = 1.0 - 0.0 = 1.0
        # refill_rate = 60 / 60.0 = 1.0
        # wait_time = 1.0 / 1.0 = 1.0
        assert limiter.wait_time() == 1.0

        # Advance time by 0.5 seconds
        self.mock_time += 0.5
        # We now have 0.5 tokens, need 0.5 more.
        # deficit = 1.0 - 0.5 = 0.5
        # wait_time = 0.5 / 1.0 = 0.5
        assert limiter.wait_time() == 0.5

        # Advance time by 0.5 seconds
        self.mock_time += 0.5
        # We have 1.0 token, wait time is 0.0
        assert limiter.wait_time() == 0.0

    def test_max_burst_cap(self):
        limiter = EmailRateLimiter(max_emails_per_minute=10, max_burst=15)

        # Advance time by a lot (e.g., 1 hour = 3600 seconds)
        # 10 emails/min = 600 emails in an hour
        self.mock_time += 3600.0

        # It should cap at max_burst (15)
        for _ in range(15):
            assert limiter.acquire() is True

        # 16th should fail, proving it didn't accumulate more than max_burst
        assert limiter.acquire() is False
