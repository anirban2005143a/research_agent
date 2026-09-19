import time
from contextvars import ContextVar
from functools import wraps
from typing import Callable, TypeVar

from .config import settings

T = TypeVar("T")
_log_scope: ContextVar[str | None] = ContextVar("log_scope", default=None)
_log_context: ContextVar[dict[str, object]] = ContextVar("log_context", default={})


def _scoped_event(message: str) -> str:
    scope = _log_scope.get()
    if scope and not message.startswith("graph."):
        message = f"{scope}.{message}"
    context = _log_context.get()
    if context:
        context_text = " | ".join(f"{key}={value!r}" for key, value in context.items())
        message = f"{message} | {context_text}"
    return message


def log(message: str) -> None:
    print(f"[research-agent] {_scoped_event(message)}", flush=True)


def retry_call(operation: Callable[[], T], label: str, max_retry: int | None = None) -> T:
    """Retry one operation with the shared bounded retry policy."""
    last_error = None
    max_retries = max_retry if max_retry is not None else getattr(settings, "max_retries", 3)
    for attempt in range(1, max_retries + 1):
        try:
            return operation()
        except Exception as exc:
            last_error = exc
            log(f"retry.attempt_failed | operation={label} | attempt={attempt}/{max_retries} | error={exc!r}")
            if attempt < max_retries:
                delay_seconds = max(0.0, getattr(settings, "retry_delay_seconds", 1.0))
                log(f"retry.waiting_before_next_attempt | operation={label} | delay_seconds={delay_seconds}")
                time.sleep(delay_seconds)
    raise last_error


def log_function(function):
    """Log a function with its graph scope and active research-task context."""
    @wraps(function)
    def wrapped(*args, **kwargs):
        started = time.perf_counter()
        function_name = function.__name__
        scope = (
            f"graph.{function_name}"
            if function.__module__.endswith(".nodes")
            else f"{function.__module__.rsplit('.', 1)[-1]}.{function_name}"
        )
        state = next(
            (argument for argument in (*args, *kwargs.values()) if isinstance(argument, dict)),
            {},
        )
        tasks = state.get("tasks", [])
        task_index = state.get("current_task_index")
        context = {}
        if tasks and isinstance(task_index, int) and task_index < len(tasks):
            context = {
                "task_number": task_index + 1,
                "task_total": len(tasks),
                "task": str(tasks[task_index]),
            }
        scope_token = _log_scope.set(scope)
        context_token = _log_context.set(context)
        log("started")
        try:
            result = function(*args, **kwargs)
        except Exception as exc:
            elapsed = time.perf_counter() - started
            log(f"failed | elapsed_seconds={elapsed:.2f} | error={exc!r}")
            _log_context.reset(context_token)
            _log_scope.reset(scope_token)
            raise
        elapsed = time.perf_counter() - started
        log(f"completed | elapsed_seconds={elapsed:.2f}")
        _log_context.reset(context_token)
        _log_scope.reset(scope_token)
        return result
    return wrapped
