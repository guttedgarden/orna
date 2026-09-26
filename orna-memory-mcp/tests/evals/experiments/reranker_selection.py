"""Чистый replay dev scores: gates и точный threshold tie-break, без inference."""

from dataclasses import asdict
from math import isclose, isfinite

from tests.evals.metrics import MetricCase, evaluate_rankings


def candidate_admitted(threshold, gates):
    return threshold is not None and bool(gates) and all(v is True for v in gates.values())


def ranking(sample, threshold, cutoff=5):
    # Python stable sort сохраняет исходный pool order при равенстве scores.
    rows = sorted(sample["scores"], key=lambda x: -x["score"])
    return [r["key"] for r in rows if threshold is None or r["score"] >= threshold][:cutoff]


def summarize(queries, rankings):
    known = frozenset(k for q in queries for k in (*q["relevance"], *rankings[q["case_id"]]))
    metrics = evaluate_rankings(
        tuple(
            MetricCase(
                case_id=q["case_id"],
                relevance=q["relevance"],
                ranking=tuple(rankings[q["case_id"]]),
                slices=tuple(q["slices"]),
                forbidden=frozenset(q.get("forbidden", [])),
                allowed_result_keys=frozenset(q.get("allowed", known)),
            )
            for q in queries
        ),
        known_keys=known,
    )
    result = {
        "aggregate": asdict(metrics.aggregate),
        "slices": {k: asdict(v) for k, v in metrics.slices.items()},
        "rankings": rankings,
    }
    for name in ("near_topic", "ood", "negative"):
        subset = [
            q for q in queries if not q["relevance"] and (name == "negative" or name in q["slices"])
        ]
        result[name] = {
            "hits": sum(bool(rankings[q["case_id"]]) for q in subset),
            "total": len(subset),
        }
    result["returned_records"] = sum(len(r) for r in rankings.values())
    result["grade1_hits"] = sum(
        q["relevance"].get(k) == 1 for q in queries for k in rankings[q["case_id"]]
    )
    result["irrelevant_positive_records"] = sum(
        k not in q["relevance"] for q in queries if q["relevance"] for k in rankings[q["case_id"]]
    )
    return result


def preservation(queries, baseline, rankings):
    regressions, new_hits, fixed = [], [], []
    for q in queries:
        cid, relevant = q["case_id"], set(q["relevance"])
        before, after = baseline[cid], rankings[cid]
        if relevant:
            lost = sorted((set(before) & relevant) - set(after))
            top1_changed = bool(
                before and before[0] in relevant and (not after or after[0] != before[0])
            )
            if lost or top1_changed:
                regressions.append(
                    {"case_id": cid, "lost_relevant": lost, "top1_changed": top1_changed}
                )
        elif not before and after:
            new_hits.append(cid)
        elif before and not after and "near_topic" in q["slices"]:
            fixed.append(cid)
    return regressions, new_hits, fixed


def compare(queries, baseline, samples, threshold):
    rankings = {s["case_id"]: ranking(s, threshold) for s in samples}
    result = summarize(queries, rankings)
    regressions, new_hits, fixed = preservation(queries, baseline["rankings"], rankings)
    n = result["near_topic"]["total"]
    gain = (baseline["near_topic"]["hits"] - result["near_topic"]["hits"]) / n if n else None
    b, m = baseline["aggregate"], result["aggregate"]
    result.update(
        regressions=regressions,
        new_negative_hits=new_hits,
        fixed_near_topic=fixed,
        near_topic_gain=gain,
        preservation_pass=not regressions,
        quality_pass=(
            not regressions
            and not new_hits
            and n >= 10
            and gain >= 0.20
            and len(fixed) >= 2
            and m["mrr_at_k"] >= b["mrr_at_k"]
            and m["ndcg_at_k"] >= b["ndcg_at_k"]
        ),
    )
    return result


def analyze_repeats(queries, repeats):
    expected = {q["case_id"] for q in queries}
    if not repeats or any(
        len(r) != len(expected) or {s["case_id"] for s in r} != expected for r in repeats
    ):
        raise ValueError("incomplete or duplicate query coverage")
    for repeat in repeats:
        for s in repeat:
            if [x["key"] for x in s["pool"]] != [x["key"] for x in s["scores"]]:
                raise ValueError("score/pool mapping mismatch")
            if len({x["key"] for x in s["scores"]}) != len(s["scores"]):
                raise ValueError("duplicate score")
            if any(not isfinite(x["score"]) or not 0 <= x["score"] <= 1 for x in s["scores"]):
                raise ValueError("invalid score")
    by_id = [{s["case_id"]: s for s in repeat} for repeat in repeats]
    stable_pool = all(
        r[c]["pool"] == by_id[0][c]["pool"]
        and r[c]["baseline"] == by_id[0][c]["baseline"]
        and r[c].get("channels") == by_id[0][c].get("channels")
        for r in by_id
        for c in expected
    )
    stable_scores = all(
        len(r[c]["scores"]) == len(by_id[0][c]["scores"])
        and all(
            a["key"] == b["key"] and isclose(a["score"], b["score"], abs_tol=1e-6, rel_tol=1e-5)
            for a, b in zip(r[c]["scores"], by_id[0][c]["scores"], strict=True)
        )
        for r in by_id
        for c in expected
    )
    stable_rankings = all(
        ranking(r[c], None, 20) == ranking(by_id[0][c], None, 20) for r in by_id for c in expected
    )
    baselines = [summarize(queries, {s["case_id"]: s["baseline"] for s in r}) for r in repeats]
    reranked = [compare(queries, b, r, None) for b, r in zip(baselines, repeats, strict=True)]
    thresholds = sorted({0.0, 1.0, *(x["score"] for r in repeats for s in r for x in s["scores"])})
    trials = []
    for t in thresholds:
        measured = [compare(queries, b, r, t) for b, r in zip(baselines, repeats, strict=True)]
        decisions = [
            [[(x["key"], x["score"] >= t) for x in r[c]["scores"]] for c in sorted(expected)]
            for r in by_id
        ]
        stable_decisions = all(d == decisions[0] for d in decisions)
        trials.append(
            {
                "threshold": t,
                "stable_decisions": stable_decisions,
                "quality_pass": len(repeats) == 3
                and stable_pool
                and stable_scores
                and stable_rankings
                and stable_decisions
                and all(m["quality_pass"] for m in measured),
                "repeats": measured,
            }
        )
    eligible = [t for t in trials if t["quality_pass"]]
    selected = (
        min(
            eligible,
            key=lambda t: (
                sum(m["near_topic"]["hits"] for m in t["repeats"]),
                sum(m["negative"]["hits"] for m in t["repeats"]),
                sum(m["returned_records"] for m in t["repeats"]),
                t["threshold"],
            ),
        )
        if eligible
        else None
    )
    return {
        "baseline": baselines,
        "reranking": reranked,
        "trials": trials,
        "quality_threshold": selected["threshold"] if selected else None,
        "rejection": selected["repeats"] if selected else None,
        "stable_pool": stable_pool,
        "stable_scores": stable_scores,
        "stable_rankings": stable_rankings,
        "unique_queries": len(queries),
        "repeats": len(repeats),
    }
