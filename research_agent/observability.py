import time
from contextlib import contextmanager
from typing import Iterator


def log(message: str) -> None:
    print(f"[research-agent] {message}", flush=True)


@contextmanager
def timed(label: str) -> Iterator[None]:
    started = time.perf_counter()
    log(f"START {label}")
    try:
        yield
    finally:
        elapsed = time.perf_counter() - started
        log(f"END {label} | elapsed={elapsed:.2f}s")