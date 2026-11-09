"""Bounded hybrid retrieval for existing Execution Agents."""

from __future__ import annotations

import asyncio
import math
import re
import unicodedata
from dataclasses import dataclass
from typing import Literal, Optional

from ...config import get_settings
from ...logging_config import logger
from .embeddings import EmbeddingError, embed_text
from .profiles import AgentProfile, get_or_backfill_agent_profile
from .roster import get_agent_roster


MAX_AGENT_CANDIDATES = 5
MAX_QUERY_LENGTH = 4000


@dataclass(frozen=True)
class AgentCandidate:
    """A roster agent and the retrieval signal that selected it."""

    agent_name: str
    instruction: Optional[str]
    match_type: Literal["exact", "semantic", "recent"]
    similarity: Optional[float] = None


@dataclass(frozen=True)
class AgentRetrievalResult:
    """Bounded candidates and whether lookup can be treated as complete."""

    candidates: list[AgentCandidate]
    complete: bool
    reason: Optional[str] = None


def _normalize(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def _has_exact_name(query: str, agent_name: str) -> bool:
    normalized_name = _normalize(agent_name)
    if not normalized_name:
        return False
    phrase = re.escape(normalized_name).replace(r"\ ", r"\s+")
    return re.search(rf"(?<!\w){phrase}(?!\w)", query) is not None


def _cosine_similarity(left: list[float], right: list[float]) -> Optional[float]:
    if len(left) != len(right) or not left:
        return None
    left_norm = math.sqrt(math.fsum(value * value for value in left))
    right_norm = math.sqrt(math.fsum(value * value for value in right))
    if left_norm == 0 or right_norm == 0:
        return None
    dot_product = math.fsum(a * b for a, b in zip(left, right))
    return dot_product / (left_norm * right_norm)


def _load_roster_profiles() -> tuple[list[str], dict[str, AgentProfile]]:
    roster = get_agent_roster()
    roster.load()
    agent_names = roster.get_agents()
    profiles = {
        agent_name: profile
        for agent_name in agent_names
        if (profile := get_or_backfill_agent_profile(agent_name)) is not None
    }
    return agent_names, profiles


async def retrieve_agents(query: str) -> AgentRetrievalResult:
    """Return at most five exact or semantically relevant agents."""
    normalized_query = _normalize(query)
    if not normalized_query:
        return AgentRetrievalResult([], complete=False, reason="empty_query")

    agent_names, profiles = await asyncio.to_thread(_load_roster_profiles)
    if not agent_names:
        return AgentRetrievalResult([], complete=True)

    exact_names = sorted(
        (name for name in agent_names if _has_exact_name(normalized_query, name)),
        key=lambda name: (_normalize(name), name),
    )
    exact_candidates = [
        AgentCandidate(
            agent_name=name,
            instruction=profiles[name].instruction if name in profiles else None,
            match_type="exact",
        )
        for name in exact_names
    ]

    unavailable_reasons: set[str] = set()
    missing_profiles = set(agent_names) - profiles.keys()
    if missing_profiles:
        unavailable_reasons.add("profiles_unavailable")

    settings = get_settings()
    model = settings.execution_agent_embedding_model.strip()
    query_vector: Optional[list[float]] = None
    try:
        query_vector = await embed_text(query.strip()[:MAX_QUERY_LENGTH])
    except EmbeddingError as exc:
        logger.warning(
            "Execution Agent retrieval is degraded",
            extra={"reason": "query_embedding_unavailable", "error": str(exc)},
        )
        unavailable_reasons.add("query_embedding_unavailable")

    semantic_matches: list[tuple[float, str, AgentProfile]] = []
    profiles_without_compatible_vectors: set[str] = set()
    if query_vector is not None:
        for agent_name, profile in profiles.items():
            if (
                profile.embedding is None
                or profile.embedding_model != model
                or profile.embedding_dimensions != len(profile.embedding)
                or len(profile.embedding) != len(query_vector)
            ):
                profiles_without_compatible_vectors.add(agent_name)
                continue
            similarity = _cosine_similarity(query_vector, profile.embedding)
            if similarity is None:
                profiles_without_compatible_vectors.add(agent_name)
                continue
            semantic_matches.append((similarity, agent_name, profile))
    else:
        profiles_without_compatible_vectors.update(profiles)

    if profiles_without_compatible_vectors:
        unavailable_reasons.add("profile_embeddings_unavailable")

    semantic_matches.sort(
        key=lambda item: (-item[0], _normalize(item[1]), item[1])
    )
    exact_set = set(exact_names)
    semantic_candidates = [
        AgentCandidate(
            agent_name=agent_name,
            instruction=profile.instruction,
            match_type="semantic",
            similarity=similarity,
        )
        for similarity, agent_name, profile in semantic_matches
        if agent_name not in exact_set
    ]

    candidates = [*exact_candidates, *semantic_candidates]
    if unavailable_reasons:
        already_included = {candidate.agent_name for candidate in candidates}
        recent_profiles = sorted(
            (
                profile
                for profile in profiles.values()
                if profile.agent_name not in already_included
            ),
            key=lambda profile: _normalize(profile.agent_name),
        )
        recent_profiles.sort(key=lambda profile: profile.last_active_at, reverse=True)
        candidates.extend(
            AgentCandidate(
                agent_name=profile.agent_name,
                instruction=profile.instruction,
                match_type="recent",
            )
            for profile in recent_profiles
        )

    candidates = candidates[:MAX_AGENT_CANDIDATES]
    reason = ",".join(sorted(unavailable_reasons)) or None
    return AgentRetrievalResult(
        candidates=candidates,
        complete=not unavailable_reasons,
        reason=reason,
    )


__all__ = [
    "MAX_AGENT_CANDIDATES",
    "AgentCandidate",
    "AgentRetrievalResult",
    "retrieve_agents",
]
