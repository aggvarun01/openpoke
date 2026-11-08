import unittest

from server.evaluation.execution_agent_retrieval import (
    RECALL_AT_5_THRESHOLD,
    build_fixture,
    evaluate_fixture,
)


class ExecutionAgentRetrievalEvaluationTests(unittest.IsolatedAsyncioTestCase):
    async def test_fixture_has_expected_scale_and_recall_threshold(self) -> None:
        names, _, cases = build_fixture()
        self.assertEqual(len(names), 50)
        self.assertEqual(len(cases), 30)

        results = await evaluate_fixture()

        self.assertGreaterEqual(results["recall_at_5"], RECALL_AT_5_THRESHOLD)
        self.assertTrue(results["passed"])
        self.assertLessEqual(results["average_candidate_count"], 5)
        self.assertLess(
            results["average_retrieved_candidate_prompt_characters"],
            results["full_roster_prompt_characters"],
        )


if __name__ == "__main__":
    unittest.main()
