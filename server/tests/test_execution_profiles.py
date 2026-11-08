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


if __name__ == "__main__":
    unittest.main()
