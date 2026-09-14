"""Generation policy for persistent memory identifiers."""

from threading import Lock
from uuid import UUID

from uuid6 import uuid7 as _uuid7

_uuid7_lock = Lock()


def new_memory_id() -> UUID:
    """Создаёт UUIDv7; временной порядок не заменяет поле created_at."""
    # uuid6 keeps unsynchronized process-local monotonic state; блокировка
    # сохраняет его корректность, если генератор вызовут не только из event loop.
    with _uuid7_lock:
        return UUID(int=_uuid7().int)
