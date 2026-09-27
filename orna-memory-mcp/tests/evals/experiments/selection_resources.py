"""Одновременные RSS процесса/harness и host swap с incremental raw samples."""

import json
import os
import subprocess
from itertools import pairwise
from threading import Event, Thread
from time import perf_counter

from tests.evals.experiments.qwen_runtime.worker import _swap_bytes

GIB = 1024**3


def summarize_resources(samples, error=None):
    rss = [s["rss_bytes"] for s in samples if s["rss_bytes"] is not None]
    worker = [s["worker_rss_bytes"] for s in samples if s["worker_rss_bytes"] is not None]
    swap = [s["swap_used_bytes"] for s in samples if s["swap_used_bytes"] is not None]
    gap = max(
        (b["elapsed_seconds"] - a["elapsed_seconds"] for a, b in pairwise(samples)),
        default=0,
    )
    healthy = len(samples) >= 2 and gap <= 1 and error is None
    first = samples[0]["swap_used_bytes"] if samples else None
    growth = max(0, max(swap) - first) if first is not None and swap else None
    return {
        "host_swap_role": "diagnostic_only",
        "sample_count": len(samples),
        "max_gap_seconds": gap,
        "sampling_healthy": healthy,
        "aggregate_sampling_healthy": healthy and gap <= 0.1,
        "sampling_error": error,
        "aggregate_peak_rss_bytes": max(rss) if rss else None,
        "worker_peak_rss_bytes": max(worker) if worker else None,
        "aggregate_rss_pass": False
        if rss and max(rss) > 8 * GIB
        else (True if healthy and gap <= 0.1 and len(rss) == len(samples) else None),
        "worker_rss_pass": False
        if worker and max(worker) > 6 * GIB
        else (True if healthy and len(worker) == len(samples) else None),
        "swap_peak_growth_bytes": growth,
        "swap_no_growth": False
        if growth is not None and growth > 0
        else (True if healthy and len(swap) == len(samples) else None),
    }


class SessionMonitor:
    def __init__(self, output, worker=lambda: None):
        self.output, self.worker = output, worker
        self.samples = []
        self.started = perf_counter()
        self.phase = "provisioning"
        self.deadline = None
        self.query_deadline = None
        self.error = None
        self.stop = Event()
        self.thread = Thread(target=self._collect, daemon=True)

    def _sample(self):
        probe = subprocess.run(
            ["ps", "-axo", "pid=,ppid=,rss="], capture_output=True, text=True, timeout=1, check=True
        )
        table = {
            int(p): (int(parent), int(rss) * 1024)
            for p, parent, rss in (line.split() for line in probe.stdout.splitlines())
        }
        pids = {os.getpid()}
        while more := {p for p, (parent, _) in table.items() if parent in pids} - pids:
            pids |= more
        process = self.worker()
        pid = process.pid if process is not None else None
        row = {
            "elapsed_seconds": perf_counter() - self.started,
            "phase": self.phase,
            "swap_used_bytes": _swap_bytes(),
            "rss_bytes": sum(table[p][1] for p in pids if p in table),
            "worker_rss_bytes": table.get(pid, (0, 0))[1],
            "processes": {str(p): table[p][1] for p in sorted(pids) if p in table},
        }
        self.samples.append(row)
        self.stream.write(json.dumps(row) + "\n")
        self.stream.flush()
        try:
            self.check()
        except RuntimeError:
            if process is not None and process.poll() is None:
                process.kill()
            raise

    def _collect(self):
        try:
            while not self.stop.wait(0.025):
                self._sample()
        except Exception as exc:
            self.error = f"resource monitor: {type(exc).__name__}"

    def check(self):
        if self.error:
            raise RuntimeError(self.error)
        if self.samples:
            row = self.samples[-1]
            if row["rss_bytes"] is not None and row["rss_bytes"] > 8 * GIB:
                raise RuntimeError("aggregate RSS budget exceeded")
            if row["worker_rss_bytes"] is not None and row["worker_rss_bytes"] > 6 * GIB:
                raise RuntimeError("worker RSS budget exceeded")
        if self.deadline is not None and perf_counter() > self.deadline:
            raise RuntimeError("quality run time budget exceeded")
        if self.query_deadline is not None and perf_counter() > self.query_deadline:
            raise RuntimeError("query time budget exceeded")

    def __enter__(self):
        self.stream = self.output.open("x")
        try:
            self._sample()
            self.thread.start()
        except BaseException:
            self.stream.close()
            raise
        return self

    def __exit__(self, *_):
        self.stop.set()
        self.thread.join(timeout=3)
        self.stream.close()

    def summary(self):
        return summarize_resources(self.samples, self.error)


def next_sample_delay(deadline, now):
    """Absolute cadence: probe duration не прибавляется к следующей паузе."""
    deadline += 0.025
    return deadline, max(0, deadline - now)


def _process_collect(output, parent_pid, state, stop, ready):
    started = perf_counter()
    deadline = started
    try:
        with output.open("x") as stream:
            while True:
                final = stop.is_set()
                probe_started = perf_counter()
                probe = subprocess.run(
                    ["ps", "-axo", "pid=,ppid=,rss="],
                    capture_output=True,
                    text=True,
                    timeout=1,
                    check=True,
                )
                table = {
                    int(p): (int(parent), int(rss) * 1024)
                    for p, parent, rss in (line.split() for line in probe.stdout.splitlines())
                }
                sampled_at = perf_counter()
                pids = {parent_pid}
                while more := {p for p, (parent, _) in table.items() if parent in pids} - pids:
                    pids |= more
                # Include monitor overhead conservatively, as part of owned tree.
                worker_pid = state["worker"].value
                row = {
                    "elapsed_seconds": sampled_at - started,
                    "phase": state["phase"].value.decode(),
                    "swap_used_bytes": _swap_bytes(),
                    "rss_bytes": sum(table[p][1] for p in pids if p in table)
                    if parent_pid in table
                    else None,
                    "worker_rss_bytes": table.get(worker_pid, (0, None))[1] if worker_pid else 0,
                    "processes": {str(p): table[p][1] for p in sorted(pids) if p in table},
                    "probe_seconds": perf_counter() - probe_started,
                    "final_sample": final,
                }
                stream.write(json.dumps(row) + "\n")
                stream.flush()
                reason = None
                if row["rss_bytes"] is not None and row["rss_bytes"] > 8 * GIB:
                    reason = "aggregate RSS budget exceeded"
                if row["worker_rss_bytes"] is not None and row["worker_rss_bytes"] > 6 * GIB:
                    reason = "worker RSS budget exceeded"
                for key in ("deadline", "query_deadline"):
                    if state[key].value and perf_counter() > state[key].value:
                        reason = key + " exceeded"
                if reason:
                    state["error"].value = reason.encode()
                    for owned_pid in (worker_pid, state["embedding_worker"].value):
                        if owned_pid:
                            try:
                                os.kill(owned_pid, 9)
                            except ProcessLookupError:
                                pass
                    ready.set()
                    return
                ready.set()
                if final:
                    return
                deadline, delay = next_sample_delay(deadline, perf_counter())
                stop.wait(delay)
    except Exception as exc:
        state["error"].value = ("resource monitor: " + type(exc).__name__).encode()
        ready.set()


class ProcessSessionMonitor:
    """Отдельный spawn process: E5/GIL не блокируют sampling/hard-stop."""

    def __init__(self, output, worker=lambda: None):
        import multiprocessing

        self.output, self.worker = output, worker
        ctx = multiprocessing.get_context("spawn")
        self.state = {
            "worker": ctx.Value("i", 0),
            "embedding_worker": ctx.Value("i", 0),
            "deadline": ctx.Value("d", 0),
            "query_deadline": ctx.Value("d", 0),
            "phase": ctx.Array("c", 128),
            "error": ctx.Array("c", 128),
        }
        self.phase = "provisioning"
        self.stop, self.ready = ctx.Event(), ctx.Event()
        self.process = ctx.Process(
            target=_process_collect,
            args=(output, os.getpid(), self.state, self.stop, self.ready),
            daemon=True,
        )

    @property
    def phase(self):
        return self.state["phase"].value.decode()

    @phase.setter
    def phase(self, value):
        self.state["phase"].value = value.encode()

    @property
    def deadline(self):
        return self.state["deadline"].value or None

    @deadline.setter
    def deadline(self, value):
        self.state["deadline"].value = value or 0

    @property
    def query_deadline(self):
        return self.state["query_deadline"].value or None

    @query_deadline.setter
    def query_deadline(self, value):
        self.state["query_deadline"].value = value or 0

    def check(self):
        process = self.worker()
        self.state["worker"].value = (
            process.pid if process is not None and process.poll() is None else 0
        )
        error = self.state["error"].value.decode()
        if error:
            raise RuntimeError(error)
        if self.process.pid is not None and not self.process.is_alive() and not self.stop.is_set():
            raise RuntimeError("resource collector exited unexpectedly")
        for key in ("deadline", "query_deadline"):
            if self.state[key].value and perf_counter() > self.state[key].value:
                raise RuntimeError(key + " exceeded")

    def __enter__(self):
        if self.output.exists():
            raise FileExistsError(self.output)
        self.process.start()
        try:
            if not self.ready.wait(10):
                raise RuntimeError("resource collector startup timeout")
            self.check()
        except BaseException:
            self.__exit__()
            raise
        return self

    def __exit__(self, *_):
        self.state["worker"].value = 0
        self.stop.set()
        self.process.join(timeout=3)
        if self.process.is_alive():
            self.state["error"].value = b"resource collector shutdown timeout"
            self.process.kill()
            self.process.join()

    def summary(self):
        rows = []
        error = self.state["error"].value.decode() or None
        if self.output.exists():
            for line in self.output.read_text().splitlines():
                try:
                    rows.append(json.loads(line))
                except ValueError:
                    error = error or "truncated resource sample"
                    break
        if rows and not rows[-1].get("final_sample"):
            error = error or "missing final sample"
        result = summarize_resources(rows, error)
        result["measurement_version"] = "process-sampler-v1"
        result["scope"] = (
            "owned harness/E5/Qwen/monitor tree; host swap; provisioning through worker exit"
        )
        return result
