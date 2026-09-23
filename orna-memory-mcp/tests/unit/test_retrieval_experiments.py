"""Контрактные проверки eval-only SQL и fusion helpers."""

from app.normalizer import normalize_query_to_lexical_groups
from tests.evals.experiments.ann import _index_names
from tests.evals.experiments.lexical import _query_sql
from tests.evals.experiments.trigram import _eligible_identifier_query, _fuse_three


def test_fts_ablation_uses_matching_document_and_query_configs() -> None:
    groups = normalize_query_to_lexical_groups("ResponseProviderExecutor")
    simple, simple_values = _query_sql("simple", groups)
    russian, russian_values = _query_sql("russian", groups)
    dual, dual_values = _query_sql("dual", groups)

    assert "m.lexical_text @@ q.value" in simple
    assert "plainto_tsquery('simple'" in simple
    assert "to_tsvector('russian', m.lexical_source) @@ q.value" in russian
    assert "plainto_tsquery('russian'" in russian
    assert "l.code_simple @@ q.code OR l.natural_ru @@ q.natural" in dual
    assert "plainto_tsquery('simple'" in dual
    assert "plainto_tsquery('russian'" in dual
    assert len(dual_values) == len(simple_values) + len(russian_values)


def test_plan_index_names_traverse_nested_explain() -> None:
    assert _index_names(
        {"Node Type": "Limit", "Plans": [{"Node Type": "Index Scan", "Index Name": "hnsw"}]}
    ) == ["hnsw"]


def test_three_channel_rrf_deduplicates_one_record_per_channel() -> None:
    ranks = _fuse_three(
        ["dense", "shared"],
        ["lexical", "shared"],
        ["shared", "shared"],
        {"dense": (1, 1), "lexical": (2, 2), "shared": (3, 3)},
    )
    assert ranks[0] == "shared"
    assert len(ranks) == 3


def test_trigram_gate_is_syntax_only_and_excludes_natural_negative_queries() -> None:
    assert _eligible_identifier_query("response provider executr")
    assert _eligible_identifier_query("ResponseProviderExecutor")
    assert not _eligible_identifier_query("Which Redis cluster stores Orna sessions?")
    assert not _eligible_identifier_query("Какую MySQL таблицу использует memory_search?")
