"""Small deterministic evaluation for bounded Execution Agent retrieval."""

from __future__ import annotations

import asyncio
import json
from html import escape
from typing import Any
from unittest.mock import patch

from ..agents.interaction_agent.agent import prepare_message_with_history
from ..config import Settings
from ..services.execution.profiles import AgentProfile
from ..services.execution.retrieval import MAX_AGENT_CANDIDATES, retrieve_agents


MODEL = "fixture/fake-embedding-v1"
RECALL_AT_5_THRESHOLD = 0.9
_CATEGORY_TERMS = (
    ("email", ("email", "inbox", "reply", "response", "respond", "message", "correspondence", "follow-up")),
    ("reminder", ("remind", "alarm", "notification", "recurring", "nudge", "alert", "morning")),
    ("flight", ("flight", "airline", "boarding", "airfare", "ticket")),
    ("restaurant", ("restaurant", "dining", "cuisine", "menu", "reservation", "eat", "table")),
    ("budget", ("budget", "spending", "expense", "finance", "costs", "money")),
    ("meeting", ("meeting", "calendar", "invite", "availability", "attendees")),
    ("code", ("code", "review", "diff", "pull request", "repository", "changes", "improvements")),
    ("invoice", ("invoice", "vendor", "payment", "billing", "receipt", "supplier", "owed", "due")),
    ("workout", ("workout", "exercise", "training", "fitness", "gym", "strength", "conditioning", "routine")),
    ("itinerary", ("itinerary", "travel", "trip", "destination", "hotel", "away", "places", "stay")),
)
_SEMANTIC_QUERIES = (
    "Find the latest message and whether they responded.",
    "Set up something that alerts me each morning.",
    "Help me choose an airline and reserve a seat.",
    "Find somewhere good to eat and book a table.",
    "Summarize where the money went this month.",
    "Find a time everyone can attend and send an invite.",
    "Inspect the changes and suggest improvements.",
    "Check what the supplier says is owed and when it is due.",
    "Build a strength and conditioning routine.",
    "Plan places to stay and things to do while away.",
)
_UNKNOWN_QUERIES = (
    ("Sort my houseplant photos by species.", "new"),
    ("Draft a short toast for a birthday party.", "new"),
    ("Compare two note-taking systems.", "new"),
    ("Help me handle the thing we discussed.", "clarify"),
    ("Do the usual with that person.", "clarify"),
)
_MULTI_CATEGORY_PAIRS = (
    ("email", "reminder"),
    ("flight", "restaurant"),
    ("budget", "meeting"),
    ("code", "invoice"),
    ("workout", "itinerary"),
)


class _FixtureRoster:
    def __init__(self, names: list[str]):
        self._names = names

    def load(self) -> None:
        return None

    def get_agents(self) -> list[str]:
        return list(self._names)


def build_fixture() -> tuple[list[str], dict[str, AgentProfile], list[dict[str, Any]]]:
    """Build 50 synthetic agents and 30 hand-labeled routing queries."""
    names: list[str] = []
    profiles: dict[str, AgentProfile] = {}
    category_names: dict[str, list[str]] = {}

    for category, terms in _CATEGORY_TERMS:
        category_names[category] = []
        descriptor = ", ".join(terms)
        for index in range(1, 6):
            name = f"{category.title()} Task {index:02d}"
            instruction = f"Handle {descriptor} for scenario {index}."
            names.append(name)
            category_names[category].append(name)
            profiles[name] = AgentProfile(
                agent_name=name,
                instruction=instruction,
                last_active_at=f"2026-01-{index:02d}T00:00:00+00:00",
                embedding=_fake_embedding(f"Agent: {name}\nInstruction: {instruction}"),
                embedding_model=MODEL,
                embedding_dimensions=len(_CATEGORY_TERMS) + 1,
            )

    cases: list[dict[str, Any]] = []
    for (category, _), query in zip(_CATEGORY_TERMS, _SEMANTIC_QUERIES):
        names_for_category = category_names[category]
        cases.append({
            "type": "exact",
            "query": f"Use {names_for_category[0]} to handle this.",
            "expected_agent_names": [names_for_category[0]],
        })
        cases.append({
            "type": "semantic",
            "query": query,
            "expected_agent_names": names_for_category,
        })

    for first_category, second_category in _MULTI_CATEGORY_PAIRS:
        first_name = category_names[first_category][0]
        second_name = category_names[second_category][1]
        cases.append({
            "type": "multi-agent",
            "query": f"Check both {first_name} and {second_name}.",
            "expected_agent_names": [first_name, second_name],
        })

    for query, outcome in _UNKNOWN_QUERIES:
        cases.append({
            "type": outcome,
            "query": query,
            "expected_agent_names": [],
        })

    return names, profiles, cases


def _fake_embedding(text: str) -> list[float]:
    normalized_text = text.casefold()
    vector = [0.0] * (len(_CATEGORY_TERMS) + 1)
    matched = False
    for index, (_, terms) in enumerate(_CATEGORY_TERMS):
        if any(term in normalized_text for term in terms):
            vector[index] = 1.0
            matched = True
    if not matched:
        vector[-1] = 1.0
    return vector


def _baseline_roster_block(names: list[str]) -> str:
    agents = "\n".join(
        f'<agent name="{escape(name, quote=True)}" />' for name in names
    )
    return f"<active_agents>\n{agents}\n</active_agents>"


async def evaluate_fixture() -> dict[str, Any]:
    """Run the production retrieval path with deterministic fake vectors."""
    names, profiles, cases = build_fixture()
    roster = _FixtureRoster(names)

    async def fake_embed_text(text: str) -> list[float]:
        return _fake_embedding(text)

    settings = Settings(
        openrouter_api_key="offline-fixture",
        execution_agent_embedding_model=MODEL,
    )
    hits = 0
    relevant = 0
    candidate_count = 0
    candidate_prompt_characters = 0
    for case in cases:
        with (
            patch(
                "server.services.execution.retrieval.get_agent_roster",
                return_value=roster,
            ),
            patch(
                "server.services.execution.retrieval.get_or_backfill_agent_profile",
                side_effect=profiles.get,
            ),
            patch(
                "server.services.execution.retrieval.get_settings",
                return_value=settings,
            ),
            patch(
                "server.services.execution.retrieval.embed_text",
                side_effect=fake_embed_text,
            ),
        ):
            result = await retrieve_agents(case["query"])

        returned_names = {candidate.agent_name for candidate in result.candidates}
        expected_names = set(case["expected_agent_names"])
        hits += len(returned_names & expected_names)
        relevant += len(expected_names)
        candidate_count += len(result.candidates)

        message = prepare_message_with_history(
            case["query"],
            "",
            active_agents=result.candidates,
            lookup_complete=result.complete,
            lookup_reason=result.reason,
        )[0]["content"]
        active_agents = message.split("<active_agents", 1)[1].split(
            "</active_agents>", 1
        )[0]
        candidate_prompt_characters += len(active_agents)

    baseline_characters = len(_baseline_roster_block(names))
    average_candidates = candidate_count / len(cases)
    average_prompt_characters = candidate_prompt_characters / len(cases)
    recall_at_5 = hits / relevant if relevant else 1.0
    return {
        "agents": len(names),
        "queries": len(cases),
        "candidate_limit": MAX_AGENT_CANDIDATES,
        "labeled_relevant_agents": relevant,
        "retrieved_relevant_agents": hits,
        "recall_at_5": recall_at_5,
        "average_candidate_count": average_candidates,
        "average_retrieved_candidate_prompt_characters": average_prompt_characters,
        "full_roster_prompt_characters": baseline_characters,
        "retrieved_prompt_character_reduction": (
            1.0 - average_prompt_characters / baseline_characters
        ),
        "threshold": RECALL_AT_5_THRESHOLD,
        "passed": recall_at_5 >= RECALL_AT_5_THRESHOLD,
    }


def main() -> None:
    results = asyncio.run(evaluate_fixture())
    print(json.dumps(results, indent=2, sort_keys=True))
    if not results["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
