"""Preload and verify the pinned FastEmbed model snapshot."""

import json
from pathlib import Path
from typing import Any, Literal

from huggingface_hub import snapshot_download

from app.config import Settings, settings
from app.embedding_profile import ACTIVE_EMBEDDING_PROFILE, EmbeddingProfile
from app.embeddings import EmbeddingService

MARKER_FILENAME = "orna-embedding-cache.json"


def _expected_manifest(profile: EmbeddingProfile) -> dict[str, Any]:
    return {
        "profile_version": profile.version,
        "model_name": profile.model_name,
        "source_repository": profile.source_repository,
        "source_revision": profile.source_revision,
        "required_files": list(profile.required_files),
    }


def _marker_path(cache_dir: Path) -> Path:
    return cache_dir / MARKER_FILENAME


def is_model_cache_ready(
    cache_dir: Path,
    profile: EmbeddingProfile = ACTIVE_EMBEDDING_PROFILE,
) -> bool:
    """Проверяет marker, pinned snapshot directory и обязательные model files."""
    try:
        manifest = json.loads(_marker_path(cache_dir).read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return False

    if manifest != _expected_manifest(profile):
        return False

    snapshot_path = profile.snapshot_path(cache_dir)
    return all(
        (snapshot_path / relative_path).is_file()
        and (snapshot_path / relative_path).resolve().is_file()
        for relative_path in profile.required_files
    )


def preload_model(app_settings: Settings | None = None) -> Literal["cached", "downloaded"]:
    """Скачивает pinned snapshot при необходимости и подтверждает его inference."""
    configured_settings = app_settings or settings
    profile = ACTIVE_EMBEDDING_PROFILE
    cache_dir = configured_settings.embedding_cache_dir
    cache_dir.mkdir(parents=True, exist_ok=True)

    if is_model_cache_ready(cache_dir, profile):
        return "cached"

    downloaded_path = Path(
        snapshot_download(
            repo_id=profile.source_repository,
            revision=profile.source_revision,
            cache_dir=cache_dir,
            allow_patterns=list(profile.required_files),
        )
    )
    expected_path = profile.snapshot_path(cache_dir)
    if downloaded_path.resolve() != expected_path.resolve():
        raise RuntimeError(
            "downloaded model snapshot does not match the pinned revision path: "
            f"expected {expected_path}, got {downloaded_path}"
        )

    verification_settings = configured_settings.model_copy(
        update={"embedding_local_files_only": True}
    )
    EmbeddingService(verification_settings).embed_memory("Orna model cache warmup")

    manifest = json.dumps(
        _expected_manifest(profile),
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    )
    marker_path = _marker_path(cache_dir)
    temporary_marker = marker_path.with_suffix(".tmp")
    temporary_marker.write_text(f"{manifest}\n", encoding="utf-8")
    temporary_marker.replace(marker_path)
    return "downloaded"


def main() -> None:
    """CLI entry point для Compose init service."""
    result = preload_model()
    print(f"Embedding model cache is ready ({result}).")


if __name__ == "__main__":
    main()
