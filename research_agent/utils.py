import time
from functools import wraps
from typing import Callable, TypeVar

from .config import settings

T = TypeVar("T")


def log(message: str) -> None:
    print(f"[research-agent] {message}", flush=True)


def retry_call(operation: Callable[[], T], label: str, max_retry: int | None = None) -> T:
    """Retry one operation with the shared bounded retry policy."""
    last_error = None
    max_retries = max_retry if max_retry is not None else getattr(settings, "max_retries", 3)
    for attempt in range(1, max_retries + 1):
        try:
            return operation()
        except Exception as exc:
            last_error = exc
            log(f"retry.failed | operation={label} | attempt={attempt}/{max_retries} | error={exc!r}")
            if attempt < max_retries:
                delay_seconds = max(0.0, getattr(settings, "retry_delay_seconds", 1.0))
                log(f"retry.waiting | operation={label} | delay_seconds={delay_seconds}")
                time.sleep(delay_seconds)
    raise last_error


def log_function(function):
    """Log start, completion, and failure for any wrapped function."""
    @wraps(function)
    def wrapped(*args, **kwargs):
        started = time.perf_counter()
        function_name = function.__name__
        log(f"function.started | name={function_name}")
        try:
            result = function(*args, **kwargs)
        except Exception as exc:
            elapsed = time.perf_counter() - started
            log(f"function.failed | name={function_name} | elapsed_seconds={elapsed:.2f} | error={exc!r}")
            raise
        elapsed = time.perf_counter() - started
        log(f"function.completed | name={function_name} | elapsed_seconds={elapsed:.2f}")
        return result
    return wrapped
