"""
Rate Limiter - Prevent abuse and DoS attacks
Implements token bucket and sliding window algorithms
"""
import time
from typing import Dict, Optional
from collections import defaultdict
import threading


class RateLimiter:
    """
    Rate limiter using token bucket algorithm
    """
    
    def __init__(self):
        """Initialize rate limiter"""
        self.buckets: Dict[str, Dict] = defaultdict(dict)
        self.lock = threading.Lock()
    
    def is_allowed(
        self,
        identifier: str,
        max_requests: int = 100,
        window_seconds: int = 3600,
        burst: Optional[int] = None
    ) -> bool:
        """
        Check if request is allowed
        
        Args:
            identifier: Unique identifier (e.g., IP address, user ID)
            max_requests: Maximum requests allowed in window
            window_seconds: Time window in seconds
            burst: Maximum burst size (defaults to max_requests)
            
        Returns:
            True if request is allowed, False otherwise
        """
        burst = burst or max_requests
        
        with self.lock:
            current_time = time.time()
            
            # Get or create bucket
            if identifier not in self.buckets:
                self.buckets[identifier] = {
                    'tokens': burst,
                    'last_update': current_time
                }
            
            bucket = self.buckets[identifier]
            
            # Calculate tokens to add
            time_passed = current_time - bucket['last_update']
            tokens_to_add = time_passed * (max_requests / window_seconds)
            
            # Update tokens
            bucket['tokens'] = min(burst, bucket['tokens'] + tokens_to_add)
            bucket['last_update'] = current_time
            
            # Check if request is allowed
            if bucket['tokens'] >= 1:
                bucket['tokens'] -= 1
                return True
            else:
                return False
    
    def get_remaining(
        self,
        identifier: str,
        max_requests: int = 100,
        window_seconds: int = 3600
    ) -> int:
        """
        Get remaining requests for identifier
        
        Args:
            identifier: Unique identifier
            max_requests: Maximum requests allowed
            window_seconds: Time window in seconds
            
        Returns:
            Number of remaining requests
        """
        with self.lock:
            if identifier not in self.buckets:
                return max_requests
            
            bucket = self.buckets[identifier]
            current_time = time.time()
            
            # Calculate current tokens
            time_passed = current_time - bucket['last_update']
            tokens_to_add = time_passed * (max_requests / window_seconds)
            current_tokens = min(max_requests, bucket['tokens'] + tokens_to_add)
            
            return int(current_tokens)
    
    def reset(self, identifier: str):
        """
        Reset rate limit for identifier
        
        Args:
            identifier: Unique identifier to reset
        """
        with self.lock:
            if identifier in self.buckets:
                del self.buckets[identifier]
    
    def cleanup_old_buckets(self, max_age_seconds: int = 7200):
        """
        Clean up old bucket entries
        
        Args:
            max_age_seconds: Maximum age of buckets to keep
        """
        with self.lock:
            current_time = time.time()
            to_delete = []
            
            for identifier, bucket in self.buckets.items():
                if current_time - bucket['last_update'] > max_age_seconds:
                    to_delete.append(identifier)
            
            for identifier in to_delete:
                del self.buckets[identifier]


# Global rate limiter instance
rate_limiter = RateLimiter()


# Decorator for rate limiting functions
def rate_limit(max_requests: int = 100, window_seconds: int = 3600):
    """
    Decorator to rate limit function calls
    
    Args:
        max_requests: Maximum requests allowed
        window_seconds: Time window in seconds
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            # Use function name as identifier
            # In production, use user ID or IP address
            identifier = func.__name__
            
            if not rate_limiter.is_allowed(identifier, max_requests, window_seconds):
                raise Exception(f"Rate limit exceeded. Max {max_requests} requests per {window_seconds}s")
            
            return func(*args, **kwargs)
        return wrapper
    return decorator


# Test
if __name__ == "__main__":
    limiter = RateLimiter()
    
    # Test rate limiting
    identifier = "test_user"
    max_requests = 5
    window = 10  # 10 seconds
    
    print(f"Testing rate limiter: {max_requests} requests per {window}s")
    
    for i in range(10):
        allowed = limiter.is_allowed(identifier, max_requests, window)
        remaining = limiter.get_remaining(identifier, max_requests, window)
        
        if allowed:
            print(f"✅ Request {i+1} allowed. Remaining: {remaining}")
        else:
            print(f"❌ Request {i+1} blocked. Remaining: {remaining}")
        
        time.sleep(0.5)