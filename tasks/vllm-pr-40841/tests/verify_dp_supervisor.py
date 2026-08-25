#!/usr/bin/env python3
"""Production node-local DP supervisor contract.

The verifier substitutes only heavyweight model serving. It still exercises
the production supervisor, spawn context, HTTP aggregation, process death,
signal forwarding, and socket cleanup with two real child processes.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import os
import signal
import socket
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.error import HTTPError, URLError
from urllib.request import urlopen


def reserve_ports(count: int = 3) -> tuple[int, ...]:
    for first in range(23100, 32000 - count):
        sockets: list[socket.socket] = []
        try:
            for port in range(first, first + count):
                sock = socket.socket()
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                sock.bind(("127.0.0.1", port))
                sockets.append(sock)
            return tuple(range(first, first + count))
        except OSError:
            pass
        finally:
            for sock in sockets:
                sock.close()
    raise RuntimeError("could not reserve contiguous loopback ports")


def child_server(child_args: argparse.Namespace, _env: dict[str, str]) -> None:
    healthy = False
    stopping = False

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            nonlocal healthy
            if self.path == "/health":
                self.send_response(200 if healthy else 503)
            elif self.path == "/set_healthy":
                healthy = True
                self.send_response(200)
            else:
                self.send_response(404)
            self.end_headers()

        def log_message(self, _format: str, *_args: object) -> None:
            return

    def stop(_signum: int, _frame: object) -> None:
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    server = ThreadingHTTPServer(("127.0.0.1", child_args.port), Handler)
    server.timeout = 0.1
    while not stopping:
        server.handle_request()
    server.server_close()


def status(port: int, path: str = "/health") -> int:
    try:
        with urlopen(f"http://127.0.0.1:{port}{path}", timeout=0.5) as response:
            return response.status
    except HTTPError as exc:
        return exc.code
    except URLError:
        return -1


async def wait_status(port: int, expected: int, timeout: float = 10.0) -> None:
    deadline = time.monotonic() + timeout
    observed = -1
    while time.monotonic() < deadline:
        observed = await asyncio.to_thread(status, port)
        if observed == expected:
            return
        await asyncio.sleep(0.05)
    raise AssertionError(f"port {port}: expected {expected}, observed {observed}")


async def wait_closed(port: int, timeout: float = 10.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if await asyncio.to_thread(status, port) == -1:
            return
        await asyncio.sleep(0.05)
    raise AssertionError(f"port {port} remained reachable")


def args(child_port: int, supervisor_port: int) -> argparse.Namespace:
    return argparse.Namespace(
        host="127.0.0.1",
        port=child_port,
        data_parallel_multi_port_external_lb=True,
        data_parallel_supervisor_port=supervisor_port,
        dp_supervisor_probe_interval_s=0.1,
        dp_supervisor_probe_timeout_s=0.2,
        dp_supervisor_probe_failure_threshold=1,
        data_parallel_size=2,
        data_parallel_size_local=2,
        data_parallel_start_rank=0,
        data_parallel_rank=None,
        data_parallel_external_lb=False,
        data_parallel_hybrid_lb=False,
        api_server_count=None,
        headless=False,
        grpc=False,
        uds=None,
        ssl_keyfile=None,
        ssl_certfile=None,
        ssl_ca_certs=None,
        node_rank=0,
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
        uvicorn_log_level="warning",
        shutdown_timeout=0.0,
    )


def verify_helpers(dp_sup: object, first: int, supervisor_port: int) -> None:
    base = args(first, supervisor_port)
    child0 = dp_sup._build_vllm_dp_server_args(base, 0)
    child1 = dp_sup._build_vllm_dp_server_args(base, 1)
    assert (child0.port, child1.port) == (first, first + 1)
    assert (child0.data_parallel_rank, child1.data_parallel_rank) == (0, 1)
    assert child0.data_parallel_external_lb is True
    assert child0.data_parallel_multi_port_external_lb is False

    invalid = args(first, first + 1)
    try:
        dp_sup.validate_multi_port_external_lb_args(invalid)
    except ValueError:
        pass
    else:
        raise AssertionError("overlapping supervisor/child ports were accepted")


async def exercise(dp_sup: object) -> None:
    first, second, supervisor_port = reserve_ports()
    verify_helpers(dp_sup, first, supervisor_port)
    dp_sup._run_vllm_dp_server = child_server
    dp_sup._build_vllm_dp_server_env = lambda _args, _rank: {}

    supervisor = dp_sup.DPSupervisor(args(first, supervisor_port))
    task = asyncio.create_task(supervisor.run())
    try:
        await wait_status(supervisor_port, 503)
        await wait_status(first, 503)
        await wait_status(second, 503)
        assert await asyncio.to_thread(status, first, "/set_healthy") == 200
        await asyncio.sleep(0.15)
        assert await asyncio.to_thread(status, supervisor_port) == 503
        assert await asyncio.to_thread(status, second, "/set_healthy") == 200
        await wait_status(supervisor_port, 200)
        assert await asyncio.to_thread(status, supervisor_port, "/ready") == 200
        assert await asyncio.to_thread(status, supervisor_port, "/readyz") == 200

        victim, sibling = supervisor._processes
        assert victim.pid and sibling.pid and victim.pid != sibling.pid
        os.kill(victim.pid, signal.SIGKILL)
        await asyncio.wait_for(task, timeout=10.0)
        assert not victim.is_alive()
        assert not sibling.is_alive(), "surviving child was orphaned"
        await wait_closed(first)
        await wait_closed(second)
        await wait_closed(supervisor_port)
    finally:
        if not task.done():
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
        for process in getattr(supervisor, "_processes", []):
            if process.is_alive() and process.pid is not None:
                with contextlib.suppress(ProcessLookupError):
                    os.kill(process.pid, signal.SIGKILL)
                process.join(timeout=2)


def main() -> int:
    try:
        import vllm.entrypoints.openai.dp_supervisor as dp_sup
    except ModuleNotFoundError as exc:
        print(f"FAIL: production supervisor module is absent: {exc}", file=sys.stderr)
        return 1
    asyncio.run(exercise(dp_sup))
    print("PASS: production supervisor handled readiness, child crash, shutdown, and port cleanup")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
