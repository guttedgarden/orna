"""Чистые deterministic retrieval metrics без зависимостей от БД и embeddings."""

from collections.abc import Mapping
from dataclasses import dataclass
from math import fsum, log2
from types import MappingProxyType


class EvaluationContractError(ValueError):
    """Ranking либо labels нарушают контракт evaluation run."""


@dataclass(frozen=True, slots=True)
class MetricCase:
    """Один query с labels, ranking и уже вычисленной областью видимости."""

    case_id: str
    relevance: Mapping[str, int]
    ranking: tuple[str, ...]
    slices: tuple[str, ...]
    forbidden: frozenset[str]
    allowed_result_keys: frozenset[str]


@dataclass(frozen=True, slots=True)
class MetricAggregate:
    """Macro metrics одного набора queries."""

    query_count: int
    positive_queries: int
    negative_queries: int
    recall_at_k: float | None
    mrr_at_k: float | None
    ndcg_at_k: float | None
    false_positive_at_k: float | None
    abstention_accuracy: float | None
    top1_accuracy: float | None


@dataclass(frozen=True, slots=True)
class MetricSummary:
    """Общие и slice metrics с явным cutoff."""

    cutoff: int
    aggregate: MetricAggregate
    slices: Mapping[str, MetricAggregate]


@dataclass(frozen=True, slots=True)
class _PerQuery:
    positive: bool
    recall: float | None
    reciprocal_rank: float | None
    ndcg: float | None
    false_positive: float | None
    abstention: float | None
    top1: float | None


def _validate_case(case: MetricCase, known_keys: frozenset[str]) -> None:
    if not case.case_id:
        raise EvaluationContractError("case_id must not be empty")
    if len(case.ranking) != len(set(case.ranking)):
        raise EvaluationContractError(f"query {case.case_id} returned a duplicate result key")

    unknown_labels = set(case.relevance) - known_keys
    if unknown_labels:
        raise EvaluationContractError(
            f"query {case.case_id} has unknown relevance key {sorted(unknown_labels)[0]}"
        )
    invalid_grades = {grade for grade in case.relevance.values() if grade not in (1, 2)}
    if invalid_grades:
        raise EvaluationContractError(f"query {case.case_id} has invalid relevance grade")

    for key in case.ranking:
        if key not in known_keys:
            raise EvaluationContractError(f"query {case.case_id} returned unknown key {key}")
        # В negative fixtures forbidden перечисляет видимые hard negatives. Такой hit
        # является измеряемым false positive; для positive cases forbidden сохраняет
        # строгую lifecycle/visibility семантику и прерывает run.
        if case.relevance and key in case.forbidden:
            raise EvaluationContractError(f"query {case.case_id} returned forbidden key {key}")
        if key not in case.allowed_result_keys:
            raise EvaluationContractError(
                f"query {case.case_id} returned key {key} that is not visible or active"
            )


def _dcg(grades: tuple[int, ...]) -> float:
    return fsum((2**grade - 1) / log2(rank + 1) for rank, grade in enumerate(grades, 1))


def _measure(case: MetricCase, cutoff: int) -> _PerQuery:
    top_k = case.ranking[:cutoff]
    if not case.relevance:
        has_result = bool(top_k)
        return _PerQuery(
            positive=False,
            recall=None,
            reciprocal_rank=None,
            ndcg=None,
            false_positive=float(has_result),
            abstention=float(not has_result),
            top1=None,
        )

    relevant = set(case.relevance)
    retrieved_relevant = sum(key in relevant for key in top_k)
    reciprocal_rank = 0.0
    for rank, key in enumerate(top_k, 1):
        if key in relevant:
            reciprocal_rank = 1 / rank
            break

    actual_grades = tuple(case.relevance.get(key, 0) for key in top_k)
    ideal_grades = tuple(sorted(case.relevance.values(), reverse=True)[:cutoff])
    ideal_dcg = _dcg(ideal_grades)
    if ideal_dcg == 0:  # Защита от программного обхода schema validator-а.
        raise EvaluationContractError(f"query {case.case_id} has no positive relevance grade")

    return _PerQuery(
        positive=True,
        recall=retrieved_relevant / len(relevant),
        reciprocal_rank=reciprocal_rank,
        ndcg=_dcg(actual_grades) / ideal_dcg,
        false_positive=None,
        abstention=None,
        top1=float(bool(top_k) and top_k[0] in relevant),
    )


def _mean(values: list[float]) -> float | None:
    return fsum(values) / len(values) if values else None


def _aggregate(measured: tuple[_PerQuery, ...], *, require_positive: bool) -> MetricAggregate:
    positives = tuple(result for result in measured if result.positive)
    negatives = tuple(result for result in measured if not result.positive)
    if require_positive and not positives:
        raise EvaluationContractError("evaluation run requires at least one positive query")

    return MetricAggregate(
        query_count=len(measured),
        positive_queries=len(positives),
        negative_queries=len(negatives),
        recall_at_k=_mean([result.recall for result in positives if result.recall is not None]),
        mrr_at_k=_mean(
            [result.reciprocal_rank for result in positives if result.reciprocal_rank is not None]
        ),
        ndcg_at_k=_mean([result.ndcg for result in positives if result.ndcg is not None]),
        false_positive_at_k=_mean(
            [result.false_positive for result in negatives if result.false_positive is not None]
        ),
        abstention_accuracy=_mean(
            [result.abstention for result in negatives if result.abstention is not None]
        ),
        top1_accuracy=_mean([result.top1 for result in positives if result.top1 is not None]),
    )


def evaluate_rankings(
    cases: tuple[MetricCase, ...],
    *,
    known_keys: frozenset[str],
    cutoff: int = 5,
) -> MetricSummary:
    """Валидирует rankings и считает macro metrics и deterministic slices."""

    if cutoff < 1:
        raise EvaluationContractError("cutoff must be >= 1")

    measured_by_case: dict[str, _PerQuery] = {}
    case_by_id: dict[str, MetricCase] = {}
    for case in cases:
        if case.case_id in case_by_id:
            raise EvaluationContractError(f"duplicate case_id {case.case_id}")
        _validate_case(case, known_keys)
        case_by_id[case.case_id] = case
        measured_by_case[case.case_id] = _measure(case, cutoff)

    aggregate = _aggregate(tuple(measured_by_case.values()), require_positive=True)
    slice_names = sorted({slice_name for case in cases for slice_name in case.slices})
    slices = {
        slice_name: _aggregate(
            tuple(measured_by_case[case.case_id] for case in cases if slice_name in case.slices),
            require_positive=False,
        )
        for slice_name in slice_names
    }
    return MetricSummary(
        cutoff=cutoff,
        aggregate=aggregate,
        slices=MappingProxyType(slices),
    )
