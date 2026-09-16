"""Unit-контракт deterministic retrieval metrics."""

import math

import pytest

from tests.evals.metrics import (
    EvaluationContractError,
    MetricCase,
    evaluate_rankings,
)


def test_evaluate_rankings_matches_hand_calculated_graded_example() -> None:
    summary = evaluate_rankings(
        (
            MetricCase(
                case_id="positive",
                relevance={"a": 2, "b": 1},
                ranking=("x", "a"),
                slices=("en", "exact_identifier"),
                forbidden=frozenset(),
                allowed_result_keys=frozenset({"a", "b", "x"}),
            ),
            MetricCase(
                case_id="negative-hit",
                relevance={},
                ranking=("x",),
                slices=("en", "negative"),
                forbidden=frozenset(),
                allowed_result_keys=frozenset({"a", "b", "x"}),
            ),
            MetricCase(
                case_id="negative-empty",
                relevance={},
                ranking=(),
                slices=("ru", "negative"),
                forbidden=frozenset(),
                allowed_result_keys=frozenset({"a", "b", "x"}),
            ),
        ),
        known_keys=frozenset({"a", "b", "x"}),
        cutoff=5,
    )

    expected_ndcg = (3 / math.log2(3)) / (3 + 1 / math.log2(3))
    assert summary.cutoff == 5
    assert summary.aggregate.recall_at_k == 0.5
    assert summary.aggregate.mrr_at_k == 0.5
    assert summary.aggregate.ndcg_at_k == pytest.approx(expected_ndcg)
    assert summary.aggregate.false_positive_at_k == 0.5
    assert summary.aggregate.abstention_accuracy == 0.5
    assert summary.aggregate.positive_queries == 1
    assert summary.aggregate.negative_queries == 2


def test_cutoff_excludes_relevant_result_below_top_five() -> None:
    summary = evaluate_rankings(
        (
            MetricCase(
                case_id="cutoff",
                relevance={"relevant": 2},
                ranking=("a", "b", "c", "d", "e", "relevant"),
                slices=("mixed",),
                forbidden=frozenset(),
                allowed_result_keys=frozenset({"a", "b", "c", "d", "e", "relevant"}),
            ),
        ),
        known_keys=frozenset({"a", "b", "c", "d", "e", "relevant"}),
        cutoff=5,
    )

    assert summary.aggregate.recall_at_k == 0.0
    assert summary.aggregate.mrr_at_k == 0.0
    assert summary.aggregate.ndcg_at_k == 0.0
    assert summary.aggregate.top1_accuracy == 0.0


def test_positive_empty_ranking_is_a_miss_and_missing_negative_denominator_is_unavailable() -> None:
    summary = evaluate_rankings(
        (
            MetricCase(
                case_id="miss",
                relevance={"relevant": 1},
                ranking=(),
                slices=("ru", "cross_lingual", "ru_to_en"),
                forbidden=frozenset(),
                allowed_result_keys=frozenset({"relevant"}),
            ),
        ),
        known_keys=frozenset({"relevant"}),
    )

    assert summary.aggregate.recall_at_k == 0.0
    assert summary.aggregate.mrr_at_k == 0.0
    assert summary.aggregate.ndcg_at_k == 0.0
    assert summary.aggregate.false_positive_at_k is None
    assert summary.aggregate.abstention_accuracy is None


def test_macro_average_gives_each_positive_query_equal_weight() -> None:
    summary = evaluate_rankings(
        (
            MetricCase(
                case_id="two-labels",
                relevance={"a": 2, "b": 1},
                ranking=("a",),
                slices=("en",),
                forbidden=frozenset(),
                allowed_result_keys=frozenset({"a", "b", "c"}),
            ),
            MetricCase(
                case_id="one-label-miss",
                relevance={"c": 2},
                ranking=(),
                slices=("ru",),
                forbidden=frozenset(),
                allowed_result_keys=frozenset({"a", "b", "c"}),
            ),
        ),
        known_keys=frozenset({"a", "b", "c"}),
    )

    assert summary.aggregate.recall_at_k == 0.25
    assert summary.aggregate.mrr_at_k == 0.5
    assert summary.aggregate.top1_accuracy == 0.5


def test_slice_metrics_are_stable_and_keep_language_and_direction_subsets() -> None:
    summary = evaluate_rankings(
        (
            MetricCase(
                case_id="ru-to-en",
                relevance={"a": 2},
                ranking=("a",),
                slices=("ru", "cross_lingual", "ru_to_en", "exact_identifier"),
                forbidden=frozenset(),
                allowed_result_keys=frozenset({"a", "b"}),
            ),
            MetricCase(
                case_id="en-to-ru",
                relevance={"b": 2},
                ranking=(),
                slices=("en", "cross_lingual", "en_to_ru"),
                forbidden=frozenset(),
                allowed_result_keys=frozenset({"a", "b"}),
            ),
        ),
        known_keys=frozenset({"a", "b"}),
    )

    assert list(summary.slices) == sorted(summary.slices)
    assert summary.slices["ru"].recall_at_k == 1.0
    assert summary.slices["en"].recall_at_k == 0.0
    assert summary.slices["cross_lingual"].recall_at_k == 0.5
    assert summary.slices["ru_to_en"].recall_at_k == 1.0
    assert summary.slices["en_to_ru"].recall_at_k == 0.0
    assert summary.slices["exact_identifier"].top1_accuracy == 1.0


@pytest.mark.parametrize(
    ("ranking", "known_keys", "allowed_keys", "forbidden", "message"),
    [
        (("a", "a"), frozenset({"a"}), frozenset({"a"}), frozenset(), "duplicate"),
        (("unknown",), frozenset({"a"}), frozenset({"a"}), frozenset(), "unknown"),
        (("hidden",), frozenset({"a", "hidden"}), frozenset({"a"}), frozenset(), "not visible"),
        (
            ("old",),
            frozenset({"a", "old"}),
            frozenset({"a", "old"}),
            frozenset({"old"}),
            "forbidden",
        ),
    ],
)
def test_result_contract_rejects_invalid_rankings(
    ranking: tuple[str, ...],
    known_keys: frozenset[str],
    allowed_keys: frozenset[str],
    forbidden: frozenset[str],
    message: str,
) -> None:
    case = MetricCase(
        case_id="invalid",
        relevance={"a": 2},
        ranking=ranking,
        slices=("en",),
        forbidden=forbidden,
        allowed_result_keys=allowed_keys,
    )

    with pytest.raises(EvaluationContractError, match=message):
        evaluate_rankings((case,), known_keys=known_keys)


def test_run_without_positive_queries_is_invalid() -> None:
    case = MetricCase(
        case_id="negative-only",
        relevance={},
        ranking=(),
        slices=("negative",),
        forbidden=frozenset(),
        allowed_result_keys=frozenset({"a"}),
    )

    with pytest.raises(EvaluationContractError, match="positive"):
        evaluate_rankings((case,), known_keys=frozenset({"a"}))


def test_negative_forbidden_label_is_measured_as_false_positive() -> None:
    summary = evaluate_rankings(
        (
            MetricCase(
                case_id="positive-control",
                relevance={"answer": 2},
                ranking=("answer",),
                slices=("en",),
                forbidden=frozenset(),
                allowed_result_keys=frozenset({"answer", "hard-negative"}),
            ),
            MetricCase(
                case_id="negative-hard-negative",
                relevance={},
                ranking=("hard-negative",),
                slices=("en", "negative"),
                forbidden=frozenset({"hard-negative"}),
                allowed_result_keys=frozenset({"answer", "hard-negative"}),
            ),
        ),
        known_keys=frozenset({"answer", "hard-negative"}),
    )

    assert summary.aggregate.false_positive_at_k == 1.0
    assert summary.aggregate.abstention_accuracy == 0.0


def test_cutoff_must_be_positive() -> None:
    with pytest.raises(EvaluationContractError, match="cutoff"):
        evaluate_rankings((), known_keys=frozenset(), cutoff=0)
