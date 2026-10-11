"""Fail fast when a remote host can't be reached (broken DNS, offline laptop).

On a network whose DNS servers stop answering, every lookup blocks for about
10 seconds, and SDK retries multiply that: one chat turn took 44 seconds
before falling back. ``ensure_reachable(host)`` resolves the host with a short
timeout and remembers a failure for a cool-down period, so the first failing
turn costs a few seconds and the following turns cost nothing.

    ensure_reachable("api.openai.com")   # raises HostUnreachable quickly
    mark_down("api.openai.com")          # after a connection error elsewhere

Only DNS is probed. A host that resolves but refuses connections is handled
by the caller's own timeouts and by ``mark_down``.
"""

from __future__ import annotations

import logging
import socket
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout

log = logging.getLogger(__name__)


class HostUnreachable(ConnectionError):
    """A network failure: callers treat it like any other connection error."""


_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="dns-probe")
_lock = threading.Lock()
_down_until: dict[str, float] = {}


def _defaults() -> tuple[float, float]:
    from core.config import get_settings

    settings = get_settings()
    return settings.net_dns_timeout_s, settings.net_unreachable_cooldown_s


def mark_down(host: str, cooldown: float | None = None) -> None:
    if not host:
        return
    if cooldown is None:
        cooldown = _defaults()[1]
    with _lock:
        _down_until[host] = time.monotonic() + cooldown
    log.warning("%s is unreachable; skipping it for %.0fs", host, cooldown)


def mark_up(host: str) -> None:
    with _lock:
        _down_until.pop(host, None)


def reset() -> None:
    """Forget all failures (tests)."""
    with _lock:
        _down_until.clear()


def ensure_reachable(host: str, *, timeout: float | None = None, cooldown: float | None = None) -> None:
    """Raise HostUnreachable if ``host`` failed recently or doesn't resolve in time."""
    if not host:
        return
    default_timeout, default_cooldown = _defaults()
    timeout = default_timeout if timeout is None else timeout
    cooldown = default_cooldown if cooldown is None else cooldown

    with _lock:
        remaining = _down_until.get(host, 0.0) - time.monotonic()
    if remaining > 0:
        raise HostUnreachable(f"{host} was unreachable moments ago; trying again in {remaining:.0f}s")

    future = _executor.submit(socket.getaddrinfo, host, 443, 0, socket.SOCK_STREAM)
    try:
        future.result(timeout=timeout)
    except FutureTimeout:
        mark_down(host, cooldown)
        raise HostUnreachable(
            f"DNS lookup for {host} timed out after {timeout:g}s (check this computer's network or DNS settings)"
        ) from None
    except OSError as error:
        mark_down(host, cooldown)
        raise HostUnreachable(f"DNS lookup for {host} failed: {error}") from error
