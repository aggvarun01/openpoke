import unittest
from unittest.mock import patch

from server.config import Settings
from server.services.execution.embeddings import EmbeddingError
from server.services.execution.profiles import AgentProfile
from server.services.execution.retrieval import (
    MAX_AGENT_CANDIDATES,
    retrieve_agents,
)


class _FakeRoster:
    def __init__(self, names: list[str]):
        self.names = names

    def load(self) -> None:
        return None

    def get_agents(self) -> list[str]:
        return self.names


class AgentRetrievalTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.profiles: dict[str, AgentProfile] = {}
        self.names: list[str] = []
        self.settings = Settings(
            openrouter_api_key="test-key",
            execution_agent_embedding_model="test/model",
        )
        self.patches = [
            patch(
                "server.services.execution.retrieval.get_agent_roster",
                return_value=_FakeRoster(self.names),
            ),
            patch(
                "server.services.execution.retrieval.get_or_backfill_agent_profile",
                side_effect=lambda name: self.profiles.get(name),
            ),
            patch(
                "server.services.execution.retrieval.get_settings",
                return_value=self.settings,
            ),
        ]
        for item in self.patches:
            item.start()
            self.addCleanup(item.stop)

    def _profile(
        self,
        name: str,
        instruction: str,
        vector: list[float] | None = None,
        model: str | None = "test/model",
        last_active_at: str = "2026-01-01T00:00:00+00:00",
    ) -> AgentProfile:
        return AgentProfile(
            agent_name=name,
            instruction=instruction,
            last_active_at=last_active_at,
            embedding=vector,
            embedding_model=model if vector is not None else None,
            embedding_dimensions=len(vector) if vector is not None else None,
        )

    async def test_exact_name_matches_are_normalized_and_whole_phrase(self) -> None:
        self.names.extend(["Alice", "Email to Alice"])
        self.profiles["Alice"] = self._profile("Alice", "Email correspondence")
        self.profiles["Email to Alice"] = self._profile(
            "Email to Alice", "Manage Alice's email"
        )

        with patch(
            "server.services.execution.retrieval.embed_text",
            side_effect=EmbeddingError("not configured"),
        ):
            result = await retrieve_agents("  Did ALICE reply?  ")

        self.assertFalse(result.complete)
        self.assertEqual([item.agent_name for item in result.candidates], ["Alice", "Email to Alice"])
        self.assertEqual(
            [item.match_type for item in result.candidates],
            ["exact", "recent"],
        )

    async def test_semantic_results_rank_after_exact_match_and_are_limited(self) -> None:
        self.names.extend([f"Agent {index}" for index in range(7)])
        self.names.append("Exact Agent")
        for index, name in enumerate(self.names):
            vector = [1.0, float(index)]
            self.profiles[name] = self._profile(
                name,
                f"Task {index}",
                vector,
            )

        with patch(
            "server.services.execution.retrieval.embed_text",
            return_value=[1.0, 0.0],
        ):
            result = await retrieve_agents("Please route to Exact Agent")

        self.assertTrue(result.complete)
        self.assertEqual(len(result.candidates), MAX_AGENT_CANDIDATES)
        self.assertEqual(result.candidates[0].agent_name, "Exact Agent")
        self.assertEqual(result.candidates[0].match_type, "exact")
        semantic_scores = [
            item.similarity
            for item in result.candidates[1:]
            if item.match_type == "semantic"
        ]
        self.assertEqual(semantic_scores, sorted(semantic_scores, reverse=True))

    async def test_missing_vectors_mark_lookup_incomplete_and_use_recent_fallback(self) -> None:
        self.names.extend(["Recent", "Old", "Semantic"])
        self.profiles["Recent"] = self._profile(
            "Recent",
            "Recent task",
            last_active_at="2026-02-01T00:00:00+00:00",
        )
        self.profiles["Old"] = self._profile(
            "Old",
            "Old task",
            last_active_at="2025-01-01T00:00:00+00:00",
        )
        self.profiles["Semantic"] = self._profile(
            "Semantic",
            "Relevant task",
            vector=[0.8, 0.2],
        )

        with patch(
            "server.services.execution.retrieval.embed_text",
            return_value=[1.0, 0.0],
        ):
            result = await retrieve_agents("find relevant")

        self.assertFalse(result.complete)
        self.assertIn("profile_embeddings_unavailable", result.reason)
        self.assertEqual(
            [(item.agent_name, item.match_type) for item in result.candidates],
            [("Semantic", "semantic"), ("Recent", "recent"), ("Old", "recent")],
        )

    async def test_query_embedding_failure_returns_exact_and_recent_candidates(self) -> None:
        self.names.extend(["Alice", "Other"])
        self.profiles["Alice"] = self._profile(
            "Alice",
            "Email follow up",
            last_active_at="2026-01-02T00:00:00+00:00",
        )
        self.profiles["Other"] = self._profile(
            "Other",
            "Other task",
            last_active_at="2026-01-01T00:00:00+00:00",
        )

        with patch(
            "server.services.execution.retrieval.embed_text",
            side_effect=EmbeddingError("provider unavailable"),
        ):
            result = await retrieve_agents("Did Alice reply?")

        self.assertFalse(result.complete)
        self.assertIn("query_embedding_unavailable", result.reason)
        self.assertEqual(
            [(item.agent_name, item.match_type) for item in result.candidates],
            [("Alice", "exact"), ("Other", "recent")],
        )

    async def test_empty_roster_is_complete_without_embedding_request(self) -> None:
        with patch(
            "server.services.execution.retrieval.embed_text"
        ) as embed_mock:
            result = await retrieve_agents("new request")

        self.assertTrue(result.complete)
        self.assertEqual(result.candidates, [])
        embed_mock.assert_not_called()


if __name__ == "__main__":
    unittest.main()
