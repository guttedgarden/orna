"""Phase-1 safety checks for persisted agent-controlled memory fields."""

import re
from collections.abc import Iterable

from fastembed.common.preprocessor_utils import load_tokenizer

from app.config import Settings
from app.embedding_profile import ACTIVE_EMBEDDING_PROFILE, EmbeddingProfile
from app.embeddings import ModelCacheMissingError, prepare_memory_text

MEMORY_TOO_LONG_MESSAGE = "Memory is too long (> 512 tokens); summarize it into a durable claim."
PROBABLE_SECRET_MESSAGE = "Memory contains a probable secret and was not stored."


class MemorySafetyError(ValueError):
    """Публично безопасный отказ Write Safety без отражения входных данных."""


class MemoryTooLongError(MemorySafetyError):
    """Canonical passage превышает контекст активного E5 profile."""


class ProbableSecretError(MemorySafetyError):
    """Persisted agent-controlled field содержит probable credential."""


class E5LengthGuard:
    """Проверяет длину через отдельный tokenizer pinned E5 snapshot без inference."""

    def __init__(
        self,
        settings: Settings,
        profile: EmbeddingProfile = ACTIVE_EMBEDDING_PROFILE,
    ) -> None:
        if settings.embedding_model != profile.model_name:
            raise ValueError("configured model does not match the active embedding profile")
        if settings.embedding_profile_version != profile.version:
            raise ValueError("configured version does not match the active embedding profile")

        snapshot_path = profile.snapshot_path(settings.embedding_cache_dir)
        if not snapshot_path.is_dir():
            raise ModelCacheMissingError(
                "pinned embedding snapshot is missing; run "
                "`python -m app.model_cache` before starting the service"
            )

        tokenizer, _special_tokens = load_tokenizer(snapshot_path)
        truncation = tokenizer.truncation
        if truncation is None or truncation.get("max_length") != profile.max_input_tokens:
            raise ValueError("pinned tokenizer context does not match the active embedding profile")

        # Это отдельный validation tokenizer. Runtime tokenizer FastEmbed не передаётся
        # guard-у и никогда не мутируется; validation instance остаётся untruncated навсегда.
        tokenizer.no_truncation()
        self.__tokenizer = tokenizer
        self.__max_input_tokens = profile.max_input_tokens

    def validate(self, prepared_content: str) -> None:
        """Отклоняет passage, если исходные input IDs не помещаются в E5 context."""
        input_ids = self.__tokenizer.encode(prepared_content).ids
        if len(input_ids) > self.__max_input_tokens:
            raise MemoryTooLongError(MEMORY_TOO_LONG_MESSAGE)


class SecretScanner:
    """Не позволяет сохранить явные секреты доступа в полях, управляемых агентом."""

    _patterns = (
        re.compile(r"-----BEGIN (?:[A-Z0-9][A-Z0-9 ]* )?PRIVATE KEY-----"),
        re.compile(
            r"(?i)\bauthorization\s*:\s*bearer[ \t]+"
            r"[A-Za-z0-9._~+/=-]{8,}(?![A-Za-z0-9._~+/=-])"
        ),
        re.compile(
            r"(?i)\bbearer[ \t]+(?=[A-Za-z0-9._~+/=-]{8,}(?![A-Za-z0-9._~+/=-]))"
            r"(?=[A-Za-z0-9._~+/=-]*[0-9._~+/=-])[A-Za-z0-9._~+/=-]{8,}"
            r"(?![A-Za-z0-9._~+/=-])"
        ),
        re.compile(
            r"(?<![A-Za-z0-9_-])eyJ[A-Za-z0-9_-]*\.[A-Za-z0-9_-]+\."
            r"[A-Za-z0-9_-]+(?![A-Za-z0-9_-])"
        ),
        re.compile(r"(?<![A-Za-z0-9_-])sk-(?!learn\b)[A-Za-z0-9_-]{16,}(?![A-Za-z0-9_-])"),
        re.compile(r"(?<![A-Za-z0-9_])ghp_[A-Za-z0-9]{20,}(?![A-Za-z0-9_])"),
        re.compile(r"(?<![A-Za-z0-9_-])glpat-[A-Za-z0-9_-]{20,}(?![A-Za-z0-9_-])"),
        re.compile(
            r"(?i)(?<![A-Za-z0-9_])[\"']?(?:password|passwd|pwd|api[ _-]?key)[\"']?"
            r"\s*(?:=|:)\s*"
            r"(?:\"[^\"\s]+\"|'[^'\s]+'|[^\s,;]+)"
        ),
    )

    def validate(self, values: Iterable[str]) -> None:
        """Проверяет строки без возврата match или rejected value наружу."""
        for value in values:
            if any(pattern.search(value) is not None for pattern in self._patterns):
                raise ProbableSecretError(PROBABLE_SECRET_MESSAGE)


class MemoryWriteSafety:
    """Закрепляет порядок secret scan -> untruncated E5 length guard."""

    def __init__(self, length_guard: E5LengthGuard, scanner: SecretScanner | None = None) -> None:
        self._length_guard = length_guard
        self._scanner = scanner or SecretScanner()

    def validate(
        self,
        *,
        content: str,
        memory_type: str,
        tags: list[str],
        identifiers: list[str],
    ) -> None:
        """Проверяет каждый agent-controlled persisted string до inference/INSERT."""
        self._scanner.validate((content, memory_type, *tags, *identifiers))
        prepared_content = prepare_memory_text(content)
        self._length_guard.validate(prepared_content)
