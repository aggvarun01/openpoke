import gzip
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from server.evaluation.execution_agent_retrieval import (
    RECALL_AT_5_THRESHOLD,
    _fake_embedding,
    _live_vectors,
    build_fixture,
    evaluate_benchmark,
    load_benchmark,
)


class ExecutionAgentRetrievalEvaluationTests(unittest.IsolatedAsyncioTestCase):
    def test_benchmark_has_unique_ids_and_valid_labels(self) -> None:
        benchmark = load_benchmark()
        names, profiles, cases = build_fixture(benchmark)
        agent_ids = [agent["id"] for agent in benchmark["agents"]]
        query_ids = [case["id"] for case in cases]
        known_ids = set(agent_ids)

        self.assertEqual(len(names), 30)
        self.assertEqual(len(profiles), len(names))
        self.assertEqual(len(agent_ids), len(set(agent_ids)))
        self.assertEqual(len(query_ids), len(set(query_ids)))
        self.assertEqual(len(cases), 33)
        for case in cases:
            self.assertTrue(set(case["relevant_agent_ids"]).issubset(known_ids))
            if case["scope"] == "output_dependent":
                self.assertEqual(case["expected_action"], "requires_context")
                self.assertIn(case["contextual_agent_id"], known_ids)

    async def test_offline_evaluation_is_deterministic_and_reports_baselines(self) -> None:
        results = await evaluate_benchmark()

        self.assertEqual(results["embedding_mode"], "offline_hash_proxy")
        self.assertIsNone(results["quality_threshold"])
        self.assertIsNone(results["passed"])
        self.assertEqual(results["agents"], 30)
        self.assertEqual(results["queries"], 33)
        self.assertEqual(results["output_dependent_cases_excluded"], 4)
        self.assertFalse(results["routing_decisions_measured"])
        self.assertEqual(results["hybrid"]["labeled_reuse_queries"], 25)
        self.assertEqual(
            [failure["case_id"] for failure in results["retrieval_failures"]],
            ["paraphrase-recycling", "paraphrase-household-budget"],
        )
        self.assertLessEqual(results["average_candidate_count"], 5)
        self.assertIn("recall_at_5", results["exact_name_baseline"])
        self.assertIn("recall_at_5", results["keyword_baseline"])
        self.assertGreater(
            results["average_full_roster_prompt_characters_with_profiles"],
            results["average_prompt_characters_with_retrieved_candidates"],
        )
        self.assertEqual(results, await evaluate_benchmark())
        self.assertEqual(RECALL_AT_5_THRESHOLD, 0.9)

    async def test_live_mode_applies_threshold_to_synthetic_vectors(self) -> None:
        _, profiles, cases = build_fixture()
        profile_vectors = {
            name: profile.embedding for name, profile in profiles.items()
        }
        query_vectors = {
            case["query"]: _fake_embedding(case["query"]) for case in cases
        }
        with patch(
            "server.evaluation.execution_agent_retrieval._live_vectors",
            return_value=(
                profile_vectors,
                query_vectors,
                {"cached_vectors": 63, "provider_fetches": 0},
            ),
        ):
            results = await evaluate_benchmark(live_embeddings=True)

        self.assertEqual(results["embedding_mode"], "real_embeddings")
        self.assertTrue(results["provider_calls_use_synthetic_data_only"])
        self.assertEqual(results["quality_threshold"], RECALL_AT_5_THRESHOLD)
        self.assertEqual(results["embedding_cache"]["provider_fetches"], 0)
        self.assertTrue(results["passed"])

    async def test_checked_in_real_embedding_cache_needs_no_provider(self) -> None:
        with patch(
            "server.evaluation.execution_agent_retrieval.embed_text",
            new=AsyncMock(side_effect=AssertionError("unexpected provider call")),
        ) as embed_mock:
            results = await evaluate_benchmark(live_embeddings=True)

        embed_mock.assert_not_awaited()
        self.assertEqual(results["embedding_cache"]["cached_vectors"], 63)
        self.assertEqual(results["embedding_cache"]["provider_fetches"], 0)
        self.assertEqual(results["hybrid"]["recall_at_5"], 1.0)

    async def test_live_embedding_vectors_are_cached_without_storing_text(self) -> None:
        _, profiles, cases = build_fixture()
        names = list(profiles)
        embed_mock = AsyncMock(side_effect=_fake_embedding)
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_path = Path(temp_dir) / "embeddings.json"
            with patch(
                "server.evaluation.execution_agent_retrieval.embed_text",
                new=embed_mock,
            ):
                first = await _live_vectors(
                    names, profiles, cases, cache_path=cache_path
                )
                calls_after_first_run = embed_mock.await_count
                second = await _live_vectors(
                    names, profiles, cases, cache_path=cache_path
                )

            self.assertEqual(embed_mock.await_count, calls_after_first_run)
            self.assertEqual(first[:2], second[:2])
            self.assertEqual(first[2]["provider_fetches"], calls_after_first_run)
            self.assertEqual(second[2]["provider_fetches"], 0)
            with gzip.open(cache_path, "rt", encoding="utf-8") as cache_file:
                cache_text = cache_file.read()
            self.assertNotIn(cases[0]["query"], cache_text)
            cache_payload = json.loads(cache_text)
            self.assertEqual(len(cache_payload["vectors"]), calls_after_first_run)


if __name__ == "__main__":
    unittest.main()
