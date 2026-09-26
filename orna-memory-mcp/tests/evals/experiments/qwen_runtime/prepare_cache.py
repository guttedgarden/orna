"""Отдельный provisioning pinned model cache; никогда не вызывается worker smoke."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from huggingface_hub import snapshot_download

MODEL = "Qwen/Qwen3-Reranker-0.6B"
REVISION = "e61197ed45024b0ed8a2d74b80b4d909f1255473"
FILES = (
    "config.json",
    "generation_config.json",
    "model.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "chat_template.jinja",
)


def main() -> None:
    root = Path(__file__).resolve().parents[5] / "data/models/phase-2-5/hub"
    root.mkdir(parents=True, exist_ok=True)
    snapshot = Path(
        snapshot_download(
            repo_id=MODEL,
            revision=REVISION,
            cache_dir=root,
            allow_patterns=list(FILES),
            local_files_only=False,
            max_workers=1,
        )
    )
    if snapshot.name != REVISION:
        raise RuntimeError("unexpected model revision")
    entries = {}
    for name in FILES:
        path = snapshot / name
        if not path.is_file():
            raise RuntimeError(f"missing model file: {name}")
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        entries[name] = {"sha256": digest.hexdigest(), "bytes": path.stat().st_size}
    manifest = {"model": MODEL, "revision": REVISION, "files": entries}
    path = root / "qwen3-reranker-0.6b-sha256.json"
    if path.exists():
        raise RuntimeError("cache manifest exists; refusing overwrite")
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(path)


if __name__ == "__main__":
    main()
