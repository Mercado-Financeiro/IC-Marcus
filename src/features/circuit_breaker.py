"""
Circuit Breaker pattern for resilient feature processing.
Prevents cascading failures and provides graceful degradation.
"""
import time
import logging
from typing import Callable, Any, Optional, Dict
from enum import Enum
from functools import wraps
import traceback

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"  # Normal operation
    OPEN = "open"      # Failures exceeded threshold, blocking calls
    HALF_OPEN = "half_open"  # Testing if service recovered


class CircuitBreaker:
    """
    Circuit breaker for protecting feature processing operations.
    
    States:
    - CLOSED: Normal operation, calls pass through
    - OPEN: Too many failures, calls are blocked
    - HALF_OPEN: Testing recovery with limited calls
    """
    
    def __init__(self,
                 failure_threshold: int = 5,
                 recovery_timeout: int = 60,
                 expected_exception: type = Exception,
                 name: str = "CircuitBreaker"):
        """
        Initialize circuit breaker.
        
        Args:
            failure_threshold: Number of failures before opening circuit
            recovery_timeout: Seconds to wait before attempting recovery
            expected_exception: Exception type to catch
            name: Name for logging
        """
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        self.name = name
        
        # State tracking
        self.failure_count = 0
        self.last_failure_time = None
        self.state = CircuitState.CLOSED
        self.success_count = 0
        
        # Statistics
        self.stats = {
            'total_calls': 0,
            'successful_calls': 0,
            'failed_calls': 0,
            'rejected_calls': 0,
            'state_changes': []
        }
    
    def call(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute function with circuit breaker protection.
        
        Args:
            func: Function to execute
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Function result or fallback value
            
        Raises:
            Exception: If circuit is open or function fails
        """
        self.stats['total_calls'] += 1
        
        # Check circuit state
        if self.state == CircuitState.OPEN:
            if self._should_attempt_reset():
                self._transition_to_half_open()
            else:
                self.stats['rejected_calls'] += 1
                raise Exception(f"Circuit breaker {self.name} is OPEN")
        
        try:
            # Execute function
            result = func(*args, **kwargs)
            self._on_success()
            return result
            
        except self.expected_exception as e:
            self._on_failure()
            raise e
    
    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt reset."""
        return (self.last_failure_time and 
                time.time() - self.last_failure_time >= self.recovery_timeout)
    
    def _on_success(self):
        """Handle successful call."""
        self.stats['successful_calls'] += 1
        self.failure_count = 0
        
        if self.state == CircuitState.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= 3:  # Require 3 successes to fully close
                self._transition_to_closed()
    
    def _on_failure(self):
        """Handle failed call."""
        self.stats['failed_calls'] += 1
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.state == CircuitState.HALF_OPEN:
            self._transition_to_open()
        elif self.failure_count >= self.failure_threshold:
            self._transition_to_open()
    
    def _transition_to_closed(self):
        """Transition to CLOSED state."""
        old_state = self.state
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self._log_transition(old_state, self.state)
    
    def _transition_to_open(self):
        """Transition to OPEN state."""
        old_state = self.state
        self.state = CircuitState.OPEN
        self.last_failure_time = time.time()
        self.success_count = 0
        self._log_transition(old_state, self.state)
    
    def _transition_to_half_open(self):
        """Transition to HALF_OPEN state."""
        old_state = self.state
        self.state = CircuitState.HALF_OPEN
        self.success_count = 0
        self._log_transition(old_state, self.state)
    
    def _log_transition(self, from_state: CircuitState, to_state: CircuitState):
        """Log state transition."""
        self.stats['state_changes'].append({
            'from': from_state.value,
            'to': to_state.value,
            'timestamp': time.time()
        })
        logger.info(f"Circuit breaker {self.name}: {from_state.value} -> {to_state.value}")
    
    def reset(self):
        """Manually reset circuit breaker."""
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.last_failure_time = None
        self.success_count = 0
        logger.info(f"Circuit breaker {self.name} manually reset")
    
    def get_state(self) -> str:
        """Get current circuit state."""
        return self.state.value
    
    def get_stats(self) -> Dict:
        """Get circuit breaker statistics."""
        return {
            **self.stats,
            'current_state': self.state.value,
            'failure_count': self.failure_count
        }


def circuit_breaker(failure_threshold: int = 5,
                    recovery_timeout: int = 60,
                    expected_exception: type = Exception,
                    fallback: Optional[Callable] = None):
    """
    Decorator for applying circuit breaker pattern to functions.
    
    Args:
        failure_threshold: Number of failures before opening circuit
        recovery_timeout: Seconds to wait before attempting recovery
        expected_exception: Exception type to catch
        fallback: Optional fallback function
        
    Returns:
        Decorated function
    """
    def decorator(func):
        # Create circuit breaker for this function
        breaker = CircuitBreaker(
            failure_threshold=failure_threshold,
            recovery_timeout=recovery_timeout,
            expected_exception=expected_exception,
            name=func.__name__
        )
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return breaker.call(func, *args, **kwargs)
            except Exception as e:
                if fallback:
                    logger.warning(f"Circuit breaker {func.__name__} triggered, using fallback")
                    return fallback(*args, **kwargs)
                raise e
        
        # Attach breaker for inspection
        wrapper.circuit_breaker = breaker
        return wrapper
    
    return decorator


class FeatureProcessingBreaker:
    """
    Specialized circuit breaker for feature processing pipeline.
    """
    
    def __init__(self):
        """Initialize feature processing circuit breakers."""
        self.breakers = {
            'correlation': CircuitBreaker(
                failure_threshold=3,
                recovery_timeout=30,
                name='correlation_filter'
            ),
            'mutual_info': CircuitBreaker(
                failure_threshold=5,
                recovery_timeout=60,
                name='mutual_info_filter'
            ),
            'vif': CircuitBreaker(
                failure_threshold=2,
                recovery_timeout=120,
                name='vif_filter'
            ),
            'zombie': CircuitBreaker(
                failure_threshold=10,
                recovery_timeout=30,
                name='zombie_filter'
            )
        }
    
    def protected_call(self, breaker_name: str, func: Callable, 
                      fallback: Optional[Callable] = None, *args, **kwargs) -> Any:
        """
        Execute function with circuit breaker protection.
        
        Args:
            breaker_name: Name of breaker to use
            func: Function to execute
            fallback: Optional fallback function
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Function result or fallback result
        """
        if breaker_name not in self.breakers:
            # No breaker, execute directly
            return func(*args, **kwargs)
        
        breaker = self.breakers[breaker_name]
        
        try:
            return breaker.call(func, *args, **kwargs)
        except Exception as e:
            if fallback:
                logger.warning(f"Using fallback for {breaker_name}: {str(e)}")
                return fallback(*args, **kwargs)
            logger.error(f"Circuit breaker {breaker_name} failed with no fallback: {str(e)}")
            raise e
    
    def get_all_stats(self) -> Dict[str, Dict]:
        """Get statistics for all circuit breakers."""
        return {name: breaker.get_stats() for name, breaker in self.breakers.items()}
    
    def reset_all(self):
        """Reset all circuit breakers."""
        for breaker in self.breakers.values():
            breaker.reset()


# Example usage for feature filtering
@circuit_breaker(failure_threshold=3, recovery_timeout=30)
def risky_correlation_calculation(X, threshold):
    """Example of protected correlation calculation."""
    # This would be the actual correlation calculation
    # that might fail due to memory or computation issues
    pass


def safe_fallback_correlation(X, threshold):
    """Fallback for correlation calculation."""
    # Simple fallback that returns basic filtering
    return list(X.columns)[:100]  # Keep first 100 features