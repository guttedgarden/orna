"""P2-04 experiment paths in an owned PostgreSQL database."""

from pathlib import Path

from app.config import Settings
from app.repository import MemoryRepository
from tests.evals.database import ephemeral_eval_database, load_eval_corpus
from tests.evals.dataset import load_retrieval_split
from tests.evals.experiments.ann import run_ann
from tests.evals.experiments.lexical import run_lexical
from tests.evals.experiments.trigram import run_trigram
from tests.integration.test_eval_runner import DeterministicEvalEmbeddings

RETRIEVAL_ROOT = Path(__file__).parents[1] / "retrieval"


async def test_experiment_sql_is_isolated_and_measures_real_plans() -> None:
    dataset = load_retrieval_split(RETRIEVAL_ROOT, "dev")
    embeddings = DeterministicEvalEmbeddings()
    vectors = {
        query.case_id: await embeddings.embed_query(query.query) for query in dataset.queries
    }
    async with ephemeral_eval_database(Settings()) as db:
        await load_eval_corpus(db.pool, dataset.corpus, embeddings, db.settings)
        repository = MemoryRepository(db.pool, db.settings)
        lexical = await run_lexical(db.pool, dataset.corpus, dataset.queries, repeats=1)
        assert set(lexical["lanes"]) == {"simple", "russian", "dual"}
        assert all(size > 0 for size in lexical["index_bytes"].values())
        assert all(
            lane["diagnostic_probes_not_in_frozen_labels"] for lane in lexical["lanes"].values()
        )
        probes = {
            lane: {
                probe["name"]: probe["ranking"]
                for probe in result["diagnostic_probes_not_in_frozen_labels"]
            }
            for lane, result in lexical["lanes"].items()
        }
        assert "git-user-commit" not in probes["simple"]["ru_morphology"]
        assert "git-user-commit" in probes["russian"]["ru_morphology"]
        assert "git-user-commit" in probes["dual"]["ru_morphology"]

        trigram = await run_trigram(
            db.pool,
            dataset.corpus,
            dataset.queries,
            repository,
            vectors,
            thresholds=(0.5,),
            repeats=1,
        )
        assert trigram["extension_version"]
        assert trigram["index_bytes"] > 0
        assert "eval_identifier_trgm_gin" in trigram["forced_index_diagnostic"]["index_names"]
        assert trigram["trials"][0]["fusion_metrics"]["aggregate"]["query_count"] == 24

        ann = await run_ann(
            db.pool,
            dataset.corpus,
            dataset.queries,
            vectors,
            noise_levels=(0, 100),
            repeats=1,
        )
        assert ann["levels"][1]["counts"]["foreign_noise"] == 100
        assert (
            ann["levels"][0]["cases"][0]["visible_count"]
            == ann["levels"][1]["cases"][0]["visible_count"]
        )
        assert any(
            variant.get("actual_hnsw")
            for case in ann["levels"][1]["cases"]
            for variant in case["variants"]
        )
        assert {trial["order"] for trial in ann["order_trials"]} == {
            "ascending_noise",
            "descending_noise",
        }
