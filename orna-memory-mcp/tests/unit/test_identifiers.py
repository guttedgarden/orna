from uuid import UUID

from app.identifiers import new_memory_id


def test_new_memory_id_returns_monotonic_stdlib_uuid7():
    first = new_memory_id()
    second = new_memory_id()

    assert type(first) is UUID
    assert type(second) is UUID
    assert first.version == 7
    assert second.version == 7
    assert first < second
