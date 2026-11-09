import unittest
from unittest.mock import AsyncMock, Mock, patch

import httpx

from server.config import Settings
from server.services.execution.embeddings import (
    EMBEDDING_TIMEOUT_SECONDS,
    EmbeddingError,
    embed_text,
)


class EmbeddingClientTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.settings = Settings(
            openrouter_api_key="test-key",
            execution_agent_embedding_model="test/model",
        )
        self.settings_patch = patch(
            "server.services.execution.embeddings.get_settings",
            return_value=self.settings,
        )
        self.settings_patch.start()
        self.addCleanup(self.settings_patch.stop)

    async def test_embed_text_posts_model_and_returns_vector(self) -> None:
        response = Mock()
        response.json.return_value = {"data": [{"embedding": [0.1, 0.2]}]}
        client = AsyncMock()
        client.__aenter__.return_value = client
        client.__aexit__.return_value = False
        client.post.return_value = response

        with patch(
            "server.services.execution.embeddings.httpx.AsyncClient",
            return_value=client,
        ) as client_factory:
            vector = await embed_text("Find the email agent")

        self.assertEqual(vector, [0.1, 0.2])
        client_factory.assert_called_once_with(timeout=EMBEDDING_TIMEOUT_SECONDS)
        request = client.post.call_args
        self.assertEqual(
            request.args[0],
            "https://openrouter.ai/api/v1/embeddings",
        )
        self.assertEqual(request.kwargs["headers"]["Authorization"], "Bearer test-key")
        self.assertEqual(
            request.kwargs["json"],
            {"model": "test/model", "input": "Find the email agent"},
        )

    async def test_rejects_empty_text(self) -> None:
        with self.assertRaises(ValueError):
            await embed_text(" \n ")

    async def test_rejects_malformed_or_non_finite_vectors(self) -> None:
        for vector in ([], [float("nan")], ["bad"]):
            response = Mock()
            response.json.return_value = {"data": [{"embedding": vector}]}
            client = AsyncMock()
            client.__aenter__.return_value = client
            client.__aexit__.return_value = False
            client.post.return_value = response

            with self.subTest(vector=vector), patch(
                "server.services.execution.embeddings.httpx.AsyncClient",
                return_value=client,
            ):
                with self.assertRaises(EmbeddingError):
                    await embed_text("query")

    async def test_provider_timeout_is_explicit(self) -> None:
        client = AsyncMock()
        client.__aenter__.return_value = client
        client.__aexit__.return_value = False
        client.post.side_effect = httpx.ReadTimeout("timed out")

        with patch(
            "server.services.execution.embeddings.httpx.AsyncClient",
            return_value=client,
        ):
            with self.assertRaisesRegex(EmbeddingError, "timed out"):
                await embed_text("query")

    async def test_provider_http_error_is_explicit(self) -> None:
        request = httpx.Request("POST", "https://openrouter.ai/api/v1/embeddings")
        response = httpx.Response(429, request=request)
        client_response = Mock()
        client_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "rate limited",
            request=request,
            response=response,
        )
        client = AsyncMock()
        client.__aenter__.return_value = client
        client.__aexit__.return_value = False
        client.post.return_value = client_response

        with patch(
            "server.services.execution.embeddings.httpx.AsyncClient",
            return_value=client,
        ):
            with self.assertRaisesRegex(EmbeddingError, "HTTP 429"):
                await embed_text("query")


if __name__ == "__main__":
    unittest.main()
