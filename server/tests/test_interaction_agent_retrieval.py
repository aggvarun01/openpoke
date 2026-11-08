import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

from server.agents.interaction_agent import agent as prompt_builder
from server.agents.interaction_agent.runtime import InteractionAgentRuntime
from server.services.execution.retrieval import (
    AgentCandidate,
    AgentRetrievalResult,
)


class InteractionAgentPromptTests(unittest.TestCase):
    def test_user_prompt_renders_candidates_and_incomplete_status(self) -> None:
        message = prompt_builder.prepare_message_with_history(
            "Find the Alice email",
            "",
            active_agents=[
                AgentCandidate(
                    agent_name='Email "Alice"',
                    instruction="Draft <then> review",
                    match_type="exact",
                )
            ],
            lookup_complete=False,
            lookup_reason="query_embedding_unavailable",
        )[0]["content"]

        self.assertIn('<active_agents status="incomplete">', message)
        self.assertIn("query_embedding_unavailable", message)
        self.assertIn('name="Email &quot;Alice&quot;"', message)
        self.assertIn("Draft &lt;then&gt; review", message)
        self.assertNotIn("<agent", message.split("<active_agents", 1)[1].split("</active_agents>", 1)[1])

    def test_agent_prompt_uses_origins_and_never_loads_roster(self) -> None:
        with patch.object(prompt_builder, "get_agent_roster", create=True) as roster:
            message = prompt_builder.prepare_message_with_history(
                "Done",
                "",
                message_type="agent",
                originating_agent_names=["Email to Alice", "Email to Bob", "Email to Alice"],
            )[0]["content"]

        self.assertIn('<active_agents source="execution_result">', message)
        self.assertEqual(message.count("<agent name="), 2)
        roster.assert_not_called()

    def test_agent_prompt_without_origin_has_no_roster(self) -> None:
        message = prompt_builder.prepare_message_with_history(
            "Important email watcher notification: New message",
            "",
            message_type="agent",
        )[0]["content"]

        self.assertIn('<active_agents source="not_provided">None</active_agents>', message)
        self.assertNotIn("<agent name=", message)


class InteractionAgentRuntimeRetrievalTests(unittest.IsolatedAsyncioTestCase):
    def _runtime(self) -> InteractionAgentRuntime:
        runtime = InteractionAgentRuntime.__new__(InteractionAgentRuntime)
        runtime.conversation_log = Mock()
        runtime._load_conversation_transcript = Mock(return_value="history")
        runtime._run_interaction_loop = AsyncMock(
            return_value=SimpleNamespace(user_messages=[], execution_agents=set())
        )
        runtime._finalize_response = Mock(return_value="ok")
        return runtime

    async def test_user_turn_injects_bounded_retrieval_and_status(self) -> None:
        runtime = self._runtime()
        retrieval = AgentRetrievalResult(
            candidates=[
                AgentCandidate("Alice", "Email thread", "exact")
            ],
            complete=False,
            reason="profile_embeddings_unavailable",
        )

        with (
            patch(
                "server.agents.interaction_agent.runtime.retrieve_agents",
                new=AsyncMock(return_value=retrieval),
            ),
            patch(
                "server.agents.interaction_agent.runtime.prepare_message_with_history",
                return_value=[],
            ) as prepare,
            patch("server.agents.interaction_agent.runtime.build_system_prompt", return_value="system"),
        ):
            result = await runtime.execute("Did Alice reply?")

        self.assertTrue(result.success)
        kwargs = prepare.call_args.kwargs
        self.assertEqual(kwargs["active_agents"], retrieval.candidates)
        self.assertFalse(kwargs["lookup_complete"])
        self.assertEqual(kwargs["lookup_reason"], "profile_embeddings_unavailable")

    async def test_agent_result_preserves_origins_without_retrieval(self) -> None:
        runtime = self._runtime()
        origins = ["Email to Alice", "Email to Bob"]

        with (
            patch(
                "server.agents.interaction_agent.runtime.retrieve_agents"
            ) as retrieve,
            patch(
                "server.agents.interaction_agent.runtime.prepare_message_with_history",
                return_value=[],
            ) as prepare,
            patch("server.agents.interaction_agent.runtime.build_system_prompt", return_value="system"),
        ):
            result = await runtime.handle_agent_message("Done", origins)

        self.assertTrue(result.success)
        self.assertEqual(prepare.call_args.kwargs["originating_agent_names"], origins)
        retrieve.assert_not_called()


if __name__ == "__main__":
    unittest.main()
