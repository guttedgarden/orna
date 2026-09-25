"""Защита от ложного выбора порога, теряющего полезную память."""

from pathlib import Path

import pytest

from tests.evals.experiments.abstention import (
    filter_ranking,
    load_dev_cases,
    load_validation_cases,
    regression_cases,
    select_candidate,
    validation_outcome,
)


def test_ood_only_gain_does_not_support_near_topic_fix() -> None:
    baseline = {"negative_hits": 10, "near_topic_negative_hits": 8}
    trial = {"negative_hits": 8, "near_topic_negative_hits": 8, "regressions": []}
    assert validation_outcome(baseline, trial) == {
        "numeric_gate_passed": True,
        "near_topic_gain": False,
        "supports_near_topic_fix": False,
        "runtime_promotion": False,
    }


def test_loaders_never_open_old_holdout_or_validation_during_dev(monkeypatch) -> None:
    original = Path.open
    validation = False

    def guarded(path, *args, **kwargs):
        assert path.name != "memory_holdout.jsonl"
        assert validation or not path.name.startswith("abstention_validation")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded)
    dataset, queries = load_dev_cases()
    assert len(queries) == 48
    validation = True
    validated, holdout = load_validation_cases()
    assert set(q.case_id for q in queries).isdisjoint(q.case_id for q in holdout)
    assert {k for q in queries for k in q.relevance}.isdisjoint(
        k for q in holdout for k in q.relevance
    )
    assert len(validated.corpus) > len(dataset.corpus)


def test_identical_vector_is_kept_at_maximum_allowed_threshold() -> None:
    assert filter_ranking(
        [{"key": "same", "similarity": 1.0, "lexical": False}], "cosine", 1.0
    ) == ["same"]


def test_old_dev_gain_alone_does_not_qualify_for_selection() -> None:
    trial = {
        "rule": "cosine",
        "threshold": 0.8,
        "regressions": [],
        "negative_hits": 2,
        "new_negative_hits": 2,
        "returned": 7,
    }
    assert select_candidate([trial], baseline_negative_hits=3, baseline_new_negative_hits=2) is None


def test_cosine_filter_preserves_order_and_includes_boundary() -> None:
    rows = [
        {"key": "weak", "similarity": 0.70, "lexical": False},
        {"key": "edge", "similarity": 0.80, "lexical": False},
        {"key": "strong", "similarity": 0.90, "lexical": False},
        {"key": "lexical", "similarity": None, "lexical": True},
    ]
    assert filter_ranking(rows, "cosine", 0.80) == ["edge", "strong"]
    assert filter_ranking(rows, "cosine_or_lexical", 0.80) == ["edge", "strong", "lexical"]
    assert filter_ranking(rows, "lexical", None) == ["lexical"]
    assert filter_ranking(rows, "cosine", 1.0) == []


def test_filter_rejects_invalid_scores_and_rules() -> None:
    with pytest.raises(ValueError):
        filter_ranking(
            [{"key": "bad", "similarity": float("nan"), "lexical": False}], "cosine", 0.8
        )
    with pytest.raises(ValueError):
        filter_ranking([], "unknown", 0.8)
    with pytest.raises(ValueError):
        filter_ranking([], "cosine", None)


def test_regressions_compare_each_relevant_record_not_only_macro_recall() -> None:
    cases = [
        {"case_id": "multi", "relevance": {"a": 2, "b": 1}, "baseline": ["a", "b"]},
        {"case_id": "other", "relevance": {"c": 2}, "baseline": ["c"]},
    ]
    assert regression_cases(cases, {"multi": ["a"], "other": ["c"]}) == ["multi"]
    assert regression_cases(cases, {"multi": ["a", "b"], "other": ["c"]}) == []


def test_successful_top1_cannot_be_replaced_with_weaker_relevant_record() -> None:
    cases = [{"case_id": "id", "relevance": {"a": 2, "b": 1}, "baseline": ["a", "b"]}]
    assert regression_cases(cases, {"id": ["b", "a"]}) == ["id"]


def test_no_candidate_without_negative_gain_and_zero_regressions() -> None:
    trials = [
        {"rule": "cosine", "threshold": 0.8, "regressions": [], "negative_hits": 3, "returned": 9},
        {
            "rule": "cosine",
            "threshold": 0.9,
            "regressions": ["positive"],
            "negative_hits": 0,
            "returned": 2,
        },
    ]
    assert select_candidate(trials, baseline_negative_hits=3) is None
    trials.append(
        {
            "rule": "cosine_or_lexical",
            "threshold": 0.85,
            "regressions": [],
            "negative_hits": 2,
            "returned": 7,
        }
    )
    assert select_candidate(trials, baseline_negative_hits=3) == trials[-1]
