"""
Test Rate Limiting
"""
import pytest
import time
from src.security.rate_limiter import RateLimiter


class TestRateLimiter:
    """Test rate limiting"""
    
    def test_rate_limit_allows_initial_requests(self):
        """Test that initial requests are allowed"""
        limiter = RateLimiter()
        
        for i in range(5):
            assert limiter.is_allowed("test_user", max_requests=10, window_seconds=60)
    
    def test_rate_limit_blocks_excess_requests(self):
        """Test that excess requests are blocked"""
        limiter = RateLimiter()
        
        # Make max requests
        for i in range(5):
            limiter.is_allowed("test_user2", max_requests=5, window_seconds=60)
        
        # Next request should be blocked
        assert not limiter.is_allowed("test_user2", max_requests=5, window_seconds=60)
    
    def test_rate_limit_resets_over_time(self):
        """Test that rate limit resets"""
        limiter = RateLimiter()
        
        # Use up tokens
        for i in range(3):
            limiter.is_allowed("test_user3", max_requests=3, window_seconds=2)
        
        # Should be blocked
        assert not limiter.is_allowed("test_user3", max_requests=3, window_seconds=2)
        
        # Wait for reset
        time.sleep(3)
        
        # Should be allowed again
        assert limiter.is_allowed("test_user3", max_requests=3, window_seconds=2)
    
    def test_get_remaining(self):
        """Test getting remaining requests"""
        limiter = RateLimiter()
        
        # Initial remaining should be max
        remaining = limiter.get_remaining("test_user4", max_requests=10, window_seconds=60)
        assert remaining == 10
        
        # After one request
        limiter.is_allowed("test_user4", max_requests=10, window_seconds=60)
        remaining = limiter.get_remaining("test_user4", max_requests=10, window_seconds=60)
        assert remaining < 10
    
    def test_reset(self):
        """Test resetting rate limit"""
        limiter = RateLimiter()
        
        # Use some requests
        for i in range(5):
            limiter.is_allowed("test_user5", max_requests=10, window_seconds=60)
        
        # Reset
        limiter.reset("test_user5")
        
        # Should have full quota again
        remaining = limiter.get_remaining("test_user5", max_requests=10, window_seconds=60)
        assert remaining == 10
    
    def test_cleanup_old_buckets(self):
        """Test cleanup of old buckets"""
        limiter = RateLimiter()
        
        # Create some buckets
        limiter.is_allowed("user1", max_requests=10, window_seconds=60)
        limiter.is_allowed("user2", max_requests=10, window_seconds=60)
        
        assert len(limiter.buckets) == 2
        
        # Cleanup (with very short max age for testing)
        time.sleep(1)
        limiter.cleanup_old_buckets(max_age_seconds=0)
        
        # Buckets should be cleaned
        assert len(limiter.buckets) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])