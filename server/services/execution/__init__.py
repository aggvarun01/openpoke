"""Execution agent support services."""

from .embeddings import EmbeddingError, embed_text, update_agent_profile_embedding
from .log_store import ExecutionAgentLogStore, get_execution_agent_logs
from .profiles import (
    AgentProfile,
    AgentProfileStore,
    get_agent_profile_store,
    get_or_backfill_agent_profile,
)
from .roster import AgentRoster, get_agent_roster
from .retrieval import (
    MAX_AGENT_CANDIDATES,
    AgentCandidate,
    AgentRetrievalResult,
    retrieve_agents,
)

__all__ = [
    "ExecutionAgentLogStore",
    "get_execution_agent_logs",
    "EmbeddingError",
    "embed_text",
    "update_agent_profile_embedding",
    "AgentProfile",
    "AgentProfileStore",
    "get_agent_profile_store",
    "get_or_backfill_agent_profile",
    "MAX_AGENT_CANDIDATES",
    "AgentCandidate",
    "AgentRetrievalResult",
    "retrieve_agents",
    "AgentRoster",
    "get_agent_roster",
]
