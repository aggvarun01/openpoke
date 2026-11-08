import unittest
from unittest.mock import AsyncMock, patch

from server.agents.execution_agent.batch_manager import ExecutionBatchManager
from server.agents.execution_agent.runtime import ExecutionResult


class ExecutionBatchOriginTests(unittest.IsolatedAsyncioTestCase):
    async def test_dispatch_preserves_origins_for_multi_agent_batch(self) -> None:
        manager = ExecutionBatchManager()
        batch_id = await manager._register_pending_execution(
            "Email to Alice", "Check thread", "request-1"
        )
        await manager._register_pending_execution(
            "Email to Bob", "Check thread", "request-2"
        )
        result = ExecutionResult(
            agent_name="Email to Alice",
            success=True,
            response="Alice replied",
        )

        with patch.object(
            manager,
            "_dispatch_to_interaction_agent",
            new=AsyncMock(),
        ) as dispatch:
            await manager._complete_execution(batch_id, result, "Email to Alice")

            self.assertFalse(dispatch.await_count)
            await manager._complete_execution(
                batch_id,
                ExecutionResult(
                    agent_name="Email to Bob",
                    success=True,
                    response="Bob replied",
                ),
                "Email to Bob",
            )

        dispatch.assert_awaited_once()
        self.assertEqual(
            dispatch.await_args.kwargs["originating_agent_names"],
            ["Email to Alice", "Email to Bob"],
        )


if __name__ == "__main__":
    unittest.main()
