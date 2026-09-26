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
