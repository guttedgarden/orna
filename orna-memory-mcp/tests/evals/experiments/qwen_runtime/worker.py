"""Standalone offline Qwen reranker worker и ограниченный P2.5-03 smoke.

Model-card wire format: Qwen/Qwen3-Reranker-0.6B@e61197ed (Apache-2.0).
Реализация формата и score самостоятельная, без import production Orna.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import re
import resource
import signal
import subprocess
import sys
from contextlib import contextmanager
from itertools import pairwise
from pathlib import Path
from threading import Event, Lock, Thread
from time import perf_counter, sleep

CONTROL_SECONDS = 10
SAMPLE_INTERVAL_SECONDS = 0.1
MAX_SAMPLE_GAP_SECONDS = 1.0

MODEL = "Qwen/Qwen3-Reranker-0.6B"
REVISION = "e61197ed45024b0ed8a2d74b80b4d909f1255473"
INSTRUCTION = (
    "Given a software engineering query, retrieve memory records that provide a supported "
    "answer, a concrete constraint, or a useful partial clue. Topic overlap alone is not relevant."
)
PREFIX = (
    "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query "
    'and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n'
    "<|im_start|>user\n"
)
SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
PINS = {
    "torch": "2.7.1",
    "transformers": "4.51.3",
    "tokenizers": "0.21.1",
    "huggingface-hub": "0.30.2",
    "safetensors": "0.5.3",
    "numpy": "2.2.6",
    "filelock": "3.18.0",
    "fsspec": "2025.3.2",
    "jinja2": "3.1.6",
    "markupsafe": "3.0.2",
    "sympy": "1.14.0",
    "mpmath": "1.3.0",
    "networkx": "3.4.2",
    "typing-extensions": "4.13.2",
    "setuptools": "80.9.0",
    "packaging": "25.0",
    "pyyaml": "6.0.2",
    "regex": "2024.11.6",
    "requests": "2.32.3",
    "charset-normalizer": "3.4.2",
    "idna": "3.10",
    "urllib3": "2.4.0",
    "certifi": "2025.4.26",
    "tqdm": "4.67.1",
}


class WorkerError(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_environment(args) -> tuple[Path, dict]:
    if sys.version_info[:3] != (3, 12, 10):
        raise WorkerError("Python 3.12.10 required")
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        raise WorkerError("native macOS arm64 required")
    for name, expected in PINS.items():
        if importlib.metadata.version(name) != expected:
            raise WorkerError(f"package pin mismatch: {name}")
    if {key: os.environ.get(key) for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")} != {
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
    }:
        raise WorkerError("HF_HUB_OFFLINE=1 and TRANSFORMERS_OFFLINE=1 required")
    if os.environ.get("TOKENIZERS_PARALLELISM") != "false":
        raise WorkerError("TOKENIZERS_PARALLELISM=false required")
    if os.environ.get("OMP_NUM_THREADS") != "6" or os.environ.get("MKL_NUM_THREADS") != "6":
        raise WorkerError("OMP/MKL_NUM_THREADS=6 required")
    if (args.model, args.revision, args.device, args.dtype, args.attention) != (
        MODEL,
        REVISION,
        "cpu",
        "float32",
        "eager",
    ):
        raise WorkerError("model/revision/device/dtype/attention pin mismatch")
    if (
        args.max_length,
        args.batch_size,
        args.threads,
        args.interop_threads,
        args.instruction_id,
        args.seed,
    ) != (2048, 1, 6, 1, "selection-instruct-v1", 0):
        raise WorkerError("input/runtime pin mismatch")
    root = Path(args.cache_dir)
    manifest_path = root / "qwen3-reranker-0.6b-sha256.json"
    if not manifest_path.is_file():
        raise WorkerError("pinned SHA-256 cache manifest missing")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["model"] != MODEL or manifest["revision"] != REVISION:
        raise WorkerError("cache manifest revision mismatch")
    snapshot = root / "models--Qwen--Qwen3-Reranker-0.6B" / "snapshots" / REVISION
    required = {
        "config.json",
        "model.safetensors",
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
        "merges.txt",
        "generation_config.json",
        "chat_template.jinja",
    }
    if set(manifest["files"]) != required:
        raise WorkerError("incomplete or unexpected cache manifest")
    for name, evidence in manifest["files"].items():
        path = snapshot / name
        if (
            not path.is_file()
            or path.stat().st_size != evidence["bytes"]
            or _sha256(path) != evidence["sha256"]
        ):
            raise WorkerError(f"pinned cache missing or hash mismatch: {name}")
    return snapshot, manifest


def _swap_bytes() -> int | None:
    try:
        probe = subprocess.run(
            ["sysctl", "-n", "vm.swapusage"],
            capture_output=True,
            text=True,
            check=False,
            timeout=1,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if probe.returncode:
        return None
    match = re.search(r"used\s*=\s*(\d+(?:\.\d+)?)([KMG])(?:\s|$)", probe.stdout)
    if not match:
        return None
    value = float(match.group(1)) * {"K": 1024, "M": 1024**2, "G": 1024**3}[match.group(2)]
    return round(value) if math.isfinite(value) else None


def _resource_summary(samples: list[dict], sampling_error: str | None = None) -> dict:
    before, after = samples[0]["swap_used_bytes"], samples[-1]["swap_used_bytes"]
    known = [
        sample["swap_used_bytes"] for sample in samples if sample["swap_used_bytes"] is not None
    ]
    growth = None if before is None else max(0, max(known) - before)
    max_gap = max(
        (b["elapsed_seconds"] - a["elapsed_seconds"] for a, b in pairwise(samples)), default=0
    )
    healthy = len(samples) >= 2 and max_gap <= MAX_SAMPLE_GAP_SECONDS and sampling_error is None
    complete = healthy and len(known) == len(samples)
    no_growth = False if growth is not None and growth > 0 else (True if complete else None)
    return {
        "swap_before_bytes": before,
        "swap_after_bytes": after,
        "swap_delta_bytes": None if before is None or after is None else after - before,
        "swap_peak_growth_bytes": growth,
        "swap_no_growth": no_growth,
        "worker_peak_rss_bytes": max(sample["worker_peak_rss_bytes"] for sample in samples),
        "max_sample_gap_seconds": max_gap,
        "sampling_healthy": healthy,
        "sampling_error": sampling_error,
        "samples": samples,
    }


class ResourceMonitor:
    """Host swap и накопленный worker peak RSS; выборка не даёт process attribution."""

    def __init__(self, phase: str):
        self.phase = phase
        self.samples: list[dict] = []
        self.started = perf_counter()
        self.lock = Lock()
        self.stopped = Event()
        self.sampling_error: str | None = None
        self.thread = Thread(target=self._collect, name="qwen-resource-monitor", daemon=True)

    def _sample(self):
        # Вызывается под lock, чтобы phase и порядок samples оставались согласованными.
        started = perf_counter()
        swap = _swap_bytes()
        self.samples.append(
            {
                "elapsed_seconds": perf_counter() - self.started,
                "probe_seconds": perf_counter() - started,
                "phase": self.phase,
                "swap_used_bytes": swap,
                "worker_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            }
        )

    def _collect(self):
        try:
            while not self.stopped.wait(SAMPLE_INTERVAL_SECONDS):
                with self.lock:
                    self._sample()
        except Exception as exc:
            # Сбой фонового sampler не должен превращаться в успешное наблюдение.
            self.sampling_error = type(exc).__name__

    def mark(self, phase: str):
        with self.lock:
            self.phase = phase
            self._sample()

    def __enter__(self):
        self.mark(self.phase)
        self.thread.start()
        return self

    def __exit__(self, *_):
        self.stopped.set()
        self.thread.join()
        self.mark(self.phase)

    def summary(self) -> dict:
        return _resource_summary(self.samples, self.sampling_error)


@contextmanager
def _timeout(seconds: int):
    def interrupt(_number, _frame):
        raise TimeoutError(f"query exceeded {seconds} s")

    previous = signal.signal(signal.SIGALRM, interrupt)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


def _stable_score(yes: float, no: float) -> float:
    if not math.isfinite(yes) or not math.isfinite(no) or not math.isfinite(yes - no):
        raise WorkerError("non-finite logits")
    difference = yes - no
    return (
        1 / (1 + math.exp(-difference))
        if difference >= 0
        else math.exp(difference) / (1 + math.exp(difference))
    )


class Reranker:
    def __init__(self, snapshot: Path, args):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        torch.manual_seed(0)
        torch.set_num_threads(6)
        torch.set_num_interop_threads(1)
        torch.use_deterministic_algorithms(True)
        self.tokenizer = AutoTokenizer.from_pretrained(
            snapshot, local_files_only=True, trust_remote_code=False, padding_side="left"
        )
        self.model = (
            AutoModelForCausalLM.from_pretrained(
                snapshot,
                local_files_only=True,
                trust_remote_code=False,
                torch_dtype=torch.float32,
                attn_implementation="eager",
            )
            .to("cpu")
            .eval()
        )
        self.model.config.use_cache = False
        self.prefix_ids = self.tokenizer.encode(PREFIX, add_special_tokens=False)
        self.suffix_ids = self.tokenizer.encode(SUFFIX, add_special_tokens=False)
        yes_ids = self.tokenizer.encode("yes", add_special_tokens=False)
        no_ids = self.tokenizer.encode("no", add_special_tokens=False)
        if len(yes_ids) != 1 or len(no_ids) != 1 or yes_ids == no_ids:
            raise WorkerError("yes/no must be distinct single token IDs")
        self.yes_id, self.no_id = yes_ids[0], no_ids[0]
        self.max_length = args.max_length
        self.query_timeout = args.query_timeout_seconds

    def prepare(self, query: str, content: str) -> dict:
        if not isinstance(query, str) or not query or not isinstance(content, str) or not content:
            raise WorkerError("nonempty query/content required")
        header = f"<Instruct>: {INSTRUCTION}\n<Query>: {query}\n<Document>: "
        body = header + content
        header_ids = self.tokenizer.encode(header, add_special_tokens=False)
        body_ids = self.tokenizer.encode(body, add_special_tokens=False)
        allowance = self.max_length - len(self.prefix_ids) - len(self.suffix_ids)
        if len(header_ids) > allowance or (
            len(body_ids) > allowance
            and not self.tokenizer.decode(body_ids[:allowance]).startswith(header)
        ):
            raise WorkerError("instruction/query/document header cannot be preserved")
        used_body = body_ids[:allowance]
        ids = self.prefix_ids + used_body + self.suffix_ids
        removed = len(body_ids) - len(used_body)
        return {
            "input_ids": ids,
            "diagnostics": {
                "input_sha256": hashlib.sha256((PREFIX + body + SUFFIX).encode()).hexdigest(),
                "body_tokens_full": len(body_ids),
                "body_tokens_kept": len(used_body),
                "tokens_full": len(self.prefix_ids) + len(body_ids) + len(self.suffix_ids),
                "tokens_used": len(ids),
                "tokens_removed": removed,
                "document_truncated": removed > 0,
                "prefix_tokens": len(self.prefix_ids),
                "suffix_tokens": len(self.suffix_ids),
                "header_tokens": len(header_ids),
                "suffix_preserved": ids[-len(self.suffix_ids) :] == self.suffix_ids,
                "input_ids": ids,
            },
        }

    def score(self, prepared: list[dict], *, full_logits: bool = False) -> list[dict]:
        if not prepared or len(prepared) > 2:
            raise WorkerError("invalid batch cardinality")
        encoded = self.tokenizer.pad(
            {"input_ids": [item["input_ids"] for item in prepared]},
            padding=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        with _timeout(self.query_timeout), self.torch.inference_mode():
            outputs = self.model(
                input_ids=encoded["input_ids"],
                attention_mask=encoded["attention_mask"],
                use_cache=False,
                logits_to_keep=0 if full_logits else 1,
            )
        logits = outputs.logits[:, -1, :]
        if logits.shape[0] != len(prepared):
            raise WorkerError("model logit cardinality mismatch")
        results = []
        for row in logits:
            yes, no = float(row[self.yes_id]), float(row[self.no_id])
            results.append(
                {
                    "z_yes": yes,
                    "z_no": no,
                    "logit_difference": yes - no,
                    "score": _stable_score(yes, no),
                }
            )
        return results


def _score_request(runtime: Reranker, request: dict) -> dict:
    candidates = request["candidates"]
    if len({item["id"] for item in candidates}) != len(candidates):
        raise WorkerError("duplicate candidate ID")
    if not candidates:
        return {"case_id": request["case_id"], "results": []}
    results = []
    for item in candidates:
        prepared = runtime.prepare(request["query"], item["content"])
        result = runtime.score([prepared])[0]
        results.append({"id": item["id"], **result, "diagnostics": prepared["diagnostics"]})
    return {"case_id": request["case_id"], "results": results}


def _smoke(
    runtime: Reranker, args, manifest: dict, load_seconds: float, monitor: ResourceMonitor
) -> dict:
    good = "The demo service listens on port 8123."
    unrelated = "The garden has three apple trees."
    queries = {
        "en": "Which port does the demo service use?",
        "ru": "На каком порту работает демонстрационный сервис?",
    }
    evidence = {}
    query_seconds = []
    for language, query in queries.items():
        monitor.mark(f"{language}/prepare")
        prepared = [runtime.prepare(query, content) for content in (good, unrelated)]
        repeats = []
        for repeat in range(args.repeats):
            monitor.mark(f"{language}/batch1-repeat-{repeat + 1}")
            started = perf_counter()
            pair = [runtime.score([item])[0] for item in prepared]
            query_seconds.append(perf_counter() - started)
            if not pair[0]["score"] > pair[1]["score"]:
                raise WorkerError(f"{language}: good <= unrelated")
            repeats.append(pair)
        if any(repeat != repeats[0] for repeat in repeats[1:]):
            raise WorkerError(f"{language}: repeated scores differ")
        monitor.mark(f"{language}/batch2")
        started = perf_counter()
        batched = runtime.score(prepared)
        query_seconds.append(perf_counter() - started)
        monitor.mark(f"{language}/full-logits")
        started = perf_counter()
        full = [runtime.score([item], full_logits=True)[0] for item in prepared]
        query_seconds.append(perf_counter() - started)
        for one, two, all_logits in zip(repeats[0], batched, full, strict=True):
            for key in ("z_yes", "z_no", "logit_difference", "score"):
                if not math.isclose(
                    one[key], two[key], rel_tol=1e-5, abs_tol=1e-5
                ) or not math.isclose(one[key], all_logits[key], rel_tol=1e-5, abs_tol=1e-5):
                    raise WorkerError(f"{language}: batch/full logits parity failed")
            sigmoid = runtime.torch.sigmoid(
                runtime.torch.tensor(one["logit_difference"], dtype=runtime.torch.float64)
            ).item()
            if not math.isclose(one["score"], sigmoid, rel_tol=1e-7, abs_tol=1e-7):
                raise WorkerError(f"{language}: sigmoid parity failed")
        if not batched[0]["score"] > batched[1]["score"] or not full[0]["score"] > full[1]["score"]:
            raise WorkerError(f"{language}: batch/full ordering failed")
        evidence[language] = {
            "query": query,
            "documents": [good, unrelated],
            "inputs": prepared,
            "batch1_repeats": repeats,
            "batch2": batched,
            "full_logits": full,
        }
    monitor.mark("overlong/prepare")
    tail_fact = "The demo service listens on port 8123."
    long_doc = ("Unrelated garden observations. " * 1500) + tail_fact
    overlong = runtime.prepare(queries["en"], long_doc)
    if (
        not overlong["diagnostics"]["document_truncated"]
        or not overlong["diagnostics"]["suffix_preserved"]
    ):
        raise WorkerError("overlong diagnostics/suffix failed")
    if tail_fact in runtime.tokenizer.decode(overlong["input_ids"]):
        raise WorkerError("overlong tail unexpectedly preserved")
    monitor.mark("overlong/score")
    started = perf_counter()
    overlong_score = runtime.score([overlong])[0]
    query_seconds.append(perf_counter() - started)
    evidence["overlong"] = {
        "query": queries["en"],
        "document": long_doc,
        "tail_fact": tail_fact,
        "input": overlong,
        "score": overlong_score,
    }
    monitor.mark("finalize")
    return {
        "model": MODEL,
        "revision": REVISION,
        "instruction_id": "selection-instruct-v1",
        "instruction": INSTRUCTION,
        "prefix": PREFIX,
        "suffix": SUFFIX,
        "yes_token_id": runtime.yes_id,
        "no_token_id": runtime.no_id,
        "packages": {name: importlib.metadata.version(name) for name in PINS},
        "python": platform.python_version(),
        "os": platform.platform(),
        "cache_manifest": manifest,
        "cache_manifest_sha256": _sha256(Path(args.cache_dir) / "qwen3-reranker-0.6b-sha256.json"),
        "worker_sha256": _sha256(Path(__file__)),
        "lock_sha256": _sha256(Path(__file__).with_name("uv.lock")),
        "command": " ".join(sys.argv),
        "runtime": {
            "device": "cpu",
            "dtype": "float32",
            "attention": "eager",
            "batch_size_measurement": 1,
            "max_length": 2048,
            "seed": 0,
            "threads": 6,
            "interop_threads": 1,
        },
        "resources": {
            "load_seconds": load_seconds,
            "query_seconds": query_seconds,
        },
        "evidence": evidence,
    }


def _run_smoke(snapshot: Path, args, manifest: dict) -> dict:
    with ResourceMonitor("control/no-model") as control:
        sleep(CONTROL_SECONDS)
    with ResourceMonitor("load") as measurement:
        started = perf_counter()
        runtime = Reranker(snapshot, args)
        load_seconds = perf_counter() - started
        evidence = _smoke(runtime, args, manifest, load_seconds, measurement)
    control_summary, measured = control.summary(), measurement.summary()
    resources = evidence["resources"]
    resources.update(
        {
            "protocol": "p2-5-03-resources-v2",
            "control_seconds_requested": CONTROL_SECONDS,
            "sample_interval_seconds_requested": SAMPLE_INTERVAL_SECONDS,
            "max_sample_gap_seconds_allowed": MAX_SAMPLE_GAP_SECONDS,
            "swap_scope": "host",
            "host_swap_role": "diagnostic_only",
            "resource_policy": "host-swap-diagnostic-v1",
            "rss_scope": "worker-lifetime-high-water-mark",
            "control": control_summary,
            "measurement": measured,
            **{
                key: measured[key]
                for key in (
                    "worker_peak_rss_bytes",
                    "swap_before_bytes",
                    "swap_after_bytes",
                    "swap_delta_bytes",
                    "swap_peak_growth_bytes",
                )
            },
        }
    )
    evidence["gates"] = {
        "load_seconds_le_120": load_seconds <= 120,
        "each_query_seconds_le_180": all(s <= 180 for s in resources["query_seconds"]),
        "worker_peak_rss_le_6_gib": measured["worker_peak_rss_bytes"] <= 6 * 1024**3,
    }
    return evidence


def _session(snapshot, args, manifest):
    """Одна загрузка модели на stdin JSONL session; EOF завершает процесс."""
    started = perf_counter()
    with _timeout(120):
        runtime = Reranker(snapshot, args)
    metadata = {
        "ready": True,
        "pid": os.getpid(),
        "load_seconds": perf_counter() - started,
        "worker_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "model": MODEL,
        "revision": REVISION,
        "instruction": INSTRUCTION,
        "instruction_id": args.instruction_id,
        "prefix": PREFIX,
        "suffix": SUFFIX,
        "yes_token_id": runtime.yes_id,
        "no_token_id": runtime.no_id,
        "python": platform.python_version(),
        "packages": PINS,
        "cache_manifest": manifest,
        "worker_sha256": _sha256(Path(__file__)),
        "lock_sha256": _sha256(Path(__file__).with_name("uv.lock")),
        "runtime": {
            k: getattr(args, k)
            for k in (
                "device",
                "dtype",
                "attention",
                "max_length",
                "batch_size",
                "threads",
                "interop_threads",
                "seed",
                "query_timeout_seconds",
            )
        },
    }
    print(json.dumps(metadata), flush=True)
    for line in sys.stdin:
        request = json.loads(line)
        response = _score_request(runtime, request)
        response.update(
            pid=os.getpid(),
            worker_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        )
        print(json.dumps(response, ensure_ascii=False), flush=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("operation", choices=("smoke", "score", "session"))
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--dtype", required=True)
    parser.add_argument("--attention", required=True)
    parser.add_argument("--max-length", type=int, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--threads", type=int, required=True)
    parser.add_argument("--interop-threads", type=int, required=True)
    parser.add_argument("--instruction-id", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--repeats", type=int, required=True)
    parser.add_argument("--query-timeout-seconds", type=int, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        if args.query_timeout_seconds != 180 or args.repeats != 3:
            raise WorkerError("smoke timeout/repeat pin mismatch")
        if args.operation == "smoke" and (args.output is None or args.output.exists()):
            raise WorkerError("new --output directory required")
        snapshot, manifest = verify_environment(args)
        if args.operation == "session":
            _session(snapshot, args, manifest)
            return 0
        if args.operation == "score":
            runtime = Reranker(snapshot, args)
            request = json.loads(sys.stdin.read())
            print(json.dumps(_score_request(runtime, request), ensure_ascii=False))
            return 0
        evidence = _run_smoke(snapshot, args, manifest)
        args.output.mkdir(parents=True, exist_ok=False)
        (args.output / "run.json").write_text(
            json.dumps(evidence, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        if not all(value is True for value in evidence["gates"].values()):
            raise WorkerError(f"smoke resource gate failed/unknown: {evidence['gates']}")
        print(
            json.dumps(
                {
                    "artifact": str(args.output / "run.json"),
                    "gates": evidence["gates"],
                }
            )
        )
        return 0
    except Exception as exc:
        print(f"P2.5-03 worker failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
