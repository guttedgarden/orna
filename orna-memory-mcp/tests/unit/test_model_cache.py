import json
from unittest.mock import MagicMock, patch

from app.config import Settings
from app.embedding_profile import ACTIVE_EMBEDDING_PROFILE
from app.model_cache import MARKER_FILENAME, is_model_cache_ready, preload_model


def _create_snapshot(cache_dir):
    profile = ACTIVE_EMBEDDING_PROFILE
    snapshot_path = profile.snapshot_path(cache_dir)
    snapshot_path.mkdir(parents=True)
    for relative_path in profile.required_files:
        (snapshot_path / relative_path).write_text("test", encoding="utf-8")
    return snapshot_path


def test_preload_downloads_pinned_snapshot_and_writes_marker(tmp_path):
    settings = Settings(embedding_cache_dir=tmp_path, _env_file=None)
    snapshot_path = ACTIVE_EMBEDDING_PROFILE.snapshot_path(tmp_path)

    def fake_download(**_kwargs):
        return str(_create_snapshot(tmp_path))

    embedding_service = MagicMock()
    embedding_service.embed_memory.return_value = [1.0] * 1024
    with (
        patch("app.model_cache.snapshot_download", side_effect=fake_download) as download,
        patch("app.model_cache.EmbeddingService", return_value=embedding_service) as service,
    ):
        result = preload_model(settings)

    assert result == "downloaded"
    download.assert_called_once_with(
        repo_id=ACTIVE_EMBEDDING_PROFILE.source_repository,
        revision=ACTIVE_EMBEDDING_PROFILE.source_revision,
        cache_dir=tmp_path,
        allow_patterns=list(ACTIVE_EMBEDDING_PROFILE.required_files),
    )
    service.assert_called_once()
    assert service.call_args.args[0].embedding_local_files_only is True
    embedding_service.embed_memory.assert_called_once_with("Orna model cache warmup")
    assert snapshot_path.is_dir()
    assert is_model_cache_ready(tmp_path)


def test_preload_reuses_complete_cache_without_network_or_model_load(tmp_path):
    _create_snapshot(tmp_path)
    profile = ACTIVE_EMBEDDING_PROFILE
    marker = {
        "profile_version": profile.version,
        "model_name": profile.model_name,
        "source_repository": profile.source_repository,
        "source_revision": profile.source_revision,
        "required_files": list(profile.required_files),
    }
    (tmp_path / MARKER_FILENAME).write_text(
        f"{json.dumps(marker, sort_keys=True)}\n",
        encoding="utf-8",
    )

    with (
        patch("app.model_cache.snapshot_download") as download,
        patch("app.model_cache.EmbeddingService") as service,
    ):
        result = preload_model(Settings(embedding_cache_dir=tmp_path, _env_file=None))

    assert result == "cached"
    download.assert_not_called()
    service.assert_not_called()


def test_cache_with_missing_model_file_is_not_ready(tmp_path):
    _create_snapshot(tmp_path)
    profile = ACTIVE_EMBEDDING_PROFILE
    marker = {
        "profile_version": profile.version,
        "model_name": profile.model_name,
        "source_repository": profile.source_repository,
        "source_revision": profile.source_revision,
        "required_files": list(profile.required_files),
    }
    (tmp_path / MARKER_FILENAME).write_text(json.dumps(marker), encoding="utf-8")
    (profile.snapshot_path(tmp_path) / profile.required_files[0]).unlink()

    assert is_model_cache_ready(tmp_path) is False
