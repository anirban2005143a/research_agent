import time
from contextlib import contextmanager
from typing import Callable, Iterator, TypeVar

from .config import settings

T = TypeVar("T")


def log(message: str) -> None:
    print(f"[research-agent] {message}", flush=True)


def retry_call(operation: Callable[[], T], label: str) -> T:
    """Retry one operation with the shared bounded retry policy."""
    last_error = None
    max_retries = getattr(settings, "max_retries", 3)
    for attempt in range(1, max_retries + 1):
        try:
            return operation()
        except Exception as exc:
            last_error = exc
            log(f"RETRY {label} | attempt={attempt}/{max_retries} | error={exc!r}")
            if attempt < max_retries:
                time.sleep(min(2 ** (attempt - 1), 8))
    raise last_error


@contextmanager
def timed(label: str) -> Iterator[None]:
    started = time.perf_counter()
    log(f"START {label}")
    try:
        yield
    finally:
        elapsed = time.perf_counter() - started
        log(f"END {label} | elapsed={elapsed:.2f}s")