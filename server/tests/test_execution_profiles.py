import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from server.services.execution import log_store, profiles
from server.services.execution.log_store import ExecutionAgentLogStore
from server.services.execution.profiles import AgentProfileStore


class AgentProfileStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self._temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self._temporary_directory.cleanup)
        root = Path(self._temporary_directory.name)
        self.store = AgentProfileStore(root / "profiles.sqlite")
        self.logs = ExecutionAgentLogStore(root / "logs")

    def test_upsert_persists_bounded_normalized_instruction(self) -> None:
        instruction = "  Follow   up\nwith the customer.  "

        profile = self.store.upsert("Customer Follow-up", instruction)

        self.assertEqual(profile.instruction, "Follow up with the customer.")
        self.assertEqual(self.store.get("Customer Follow-up"), profile)

    def test_upsert_caps_instruction_length(self) -> None:
        profile = self.store.upsert("Long task", "x" * 2500)

        self.assertEqual(len(profile.instruction), 2000)

    def test_get_or_backfill_creates_only_missing_profiles(self) -> None:
        self.logs.record_request("Existing agent", "Original task")

        backfilled = self.store.get_or_backfill(
            "Existing agent", self.logs.load_latest_request("Existing agent")
        )
        self.store.upsert("Existing agent", "New task")
        existing = self.store.get_or_backfill(
            "Existing agent", self.logs.load_latest_request("Existing agent")
        )

        self.assertIsNotNone(backfilled)
        self.assertEqual(backfilled.instruction, "Original task")
        self.assertEqual(existing.instruction, "New task")

    def test_profile_helper_backfills_from_latest_recorded_request(self) -> None:
        self.logs.record_request("Existing agent", "First task")
        self.logs.record_request("Existing agent", "Latest task")

        with (
            patch.object(profiles, "get_agent_profile_store", return_value=self.store),
            patch.object(
                log_store, "get_execution_agent_logs", return_value=self.logs
            ),
        ):
            profile = profiles.get_or_backfill_agent_profile("Existing agent")

        self.assertIsNotNone(profile)
        self.assertEqual(profile.instruction, "Latest task")

    def test_get_or_backfill_without_source_returns_none(self) -> None:
        self.assertIsNone(self.store.get_or_backfill("No history", None))

    def test_clear_all_removes_profiles(self) -> None:
        self.store.upsert("Agent", "Task")

        self.store.clear_all()

        self.assertIsNone(self.store.get("Agent"))

    def test_existing_profile_table_is_upgraded_for_embeddings(self) -> None:
        legacy_path = Path(self._temporary_directory.name) / "legacy.sqlite"
        with sqlite3.connect(legacy_path) as connection:
            connection.execute(
                """
                CREATE TABLE agent_profiles (
                    agent_name TEXT PRIMARY KEY,
                    instruction TEXT NOT NULL,
                    last_active_at TEXT NOT NULL
                )
                """
            )
            connection.execute(
                "INSERT INTO agent_profiles VALUES (?, ?, ?)",
                ("Legacy agent", "Legacy task", "2026-01-01T00:00:00+00:00"),
            )

        upgraded_store = AgentProfileStore(legacy_path)
        profile = upgraded_store.get("Legacy agent")

        self.assertEqual(profile.instruction, "Legacy task")
        self.assertIsNone(profile.embedding)

    def test_embedding_is_persisted_only_for_current_instruction(self) -> None:
        self.store.upsert("Agent", "Current task")
        saved = self.store.update_embedding(
            "Agent", "Stale task", "test/model", [0.1, 0.2]
        )
        self.assertFalse(saved)

        saved = self.store.update_embedding(
            "Agent", "Current task", "test/model", [0.1, 0.2]
        )
        profile = self.store.get("Agent")

        self.assertTrue(saved)
        self.assertEqual(profile.embedding, [0.1, 0.2])
        self.assertEqual(profile.embedding_model, "test/model")
        self.assertEqual(profile.embedding_dimensions, 2)

    def test_instruction_update_invalidates_embedding(self) -> None:
        self.store.upsert("Agent", "Old task")
        self.store.update_embedding("Agent", "Old task", "test/model", [0.1])

        profile = self.store.upsert("Agent", "New task")

        self.assertIsNone(profile.embedding)
        self.assertIsNone(profile.embedding_model)
        self.assertIsNone(profile.embedding_dimensions)


if __name__ == "__main__":
    unittest.main()
