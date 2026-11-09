"""OpenRouter client for Execution Agent retrieval embeddings."""

from __future__ import annotations

import math
from typing import Any

import httpx

from ...config import get_settings
from ...logging_config import logger
from ...openrouter_client.client import OpenRouterBaseURL


EMBEDDING_TIMEOUT_SECONDS = 2.0
_EMBEDDING_URL = f"{OpenRouterBaseURL}/embeddings"


class EmbeddingError(RuntimeError):
    """Raised when an embedding cannot be requested or validated."""


async def embed_text(text: str) -> list[float]:
    """Return one validated embedding vector for text using OpenRouter."""
    if not text.strip():
        raise ValueError("text must not be empty")

    settings = get_settings()
    if not settings.openrouter_api_key:
        raise EmbeddingError("OpenRouter API key is not configured")
    model = settings.execution_agent_embedding_model.strip()
    if not model:
        raise EmbeddingError("Execution Agent embedding model is not configured")

    try:
        async with httpx.AsyncClient(timeout=EMBEDDING_TIMEOUT_SECONDS) as client:
            response = await client.post(
                _EMBEDDING_URL,
                headers={
                    "Authorization": f"Bearer {settings.openrouter_api_key}",
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                },
                json={"model": model, "input": text},
            )
            response.raise_for_status()
            payload: Any = response.json()
    except httpx.TimeoutException as exc:
        logger.warning("Execution Agent embedding request timed out")
        raise EmbeddingError("Embedding request timed out") from exc
    except httpx.HTTPStatusError as exc:
        logger.warning(
            "Execution Agent embedding provider returned an error",
            extra={"status_code": exc.response.status_code},
        )
        raise EmbeddingError(
            f"Embedding provider returned HTTP {exc.response.status_code}"
        ) from exc
    except httpx.HTTPError as exc:
        logger.warning(
            "Execution Agent embedding request failed",
            extra={"error": str(exc)},
        )
        raise EmbeddingError("Embedding request failed") from exc
    except ValueError as exc:
        raise EmbeddingError("Embedding provider returned invalid JSON") from exc

    try:
        vector = payload["data"][0]["embedding"]
        if not isinstance(vector, list) or not vector:
            raise TypeError("embedding must be a non-empty list")
        result = [float(value) for value in vector]
        if not all(math.isfinite(value) for value in result):
            raise ValueError("embedding values must be finite")
        return result
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        raise EmbeddingError("Embedding provider returned an invalid vector") from exc


async def update_agent_profile_embedding(agent_name: str, instruction: str) -> bool:
    """Generate and persist a profile vector without overwriting newer profile data."""
    from .profiles import get_agent_profile_store

    try:
        settings = get_settings()
        store = get_agent_profile_store()
        profile = store.get(agent_name)
        if profile is None or profile.instruction != " ".join(
            instruction.split()
        )[:2000]:
            return False

        text = f"Agent: {agent_name}\nInstruction: {profile.instruction}"
        vector = await embed_text(text)
        return store.update_embedding(
            agent_name,
            profile.instruction,
            settings.execution_agent_embedding_model,
            vector,
        )
    except Exception as exc:
        logger.warning(
            "Execution Agent profile embedding update failed",
            extra={"agent_name": agent_name, "error": str(exc)},
        )
        return False


__all__ = [
    "EMBEDDING_TIMEOUT_SECONDS",
    "EmbeddingError",
    "embed_text",
    "update_agent_profile_embedding",
]
