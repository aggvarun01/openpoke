"""Evaluate bounded Execution Agent retrieval on a labeled synthetic benchmark."""

from __future__ import annotations

import argparse
import asyncio
import gzip
import hashlib
import io
import json
import math
import os
import re
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Optional
from unittest.mock import patch

from ..agents.interaction_agent.agent import prepare_message_with_history
from ..config import get_settings
from ..services.execution.embeddings import embed_text
from ..services.execution.profiles import AgentProfile
from ..services.execution.retrieval import (
    MAX_AGENT_CANDIDATES,
    AgentCandidate,
    _has_exact_name,
    retrieve_agents,
)


RECALL_AT_5_THRESHOLD = 0.9
_BENCHMARK_PATH = Path(__file__).with_name(
    "execution_agent_retrieval_benchmark.json"
)
_LIVE_VECTOR_CACHE_PATH = Path(__file__).with_name(
    "execution_agent_retrieval_embedding_cache.json.gz"
)
_LIVE_VECTOR_CACHE_VERSION = 1
_FAKE_VECTOR_DIMENSIONS = 256
_STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "can", "do", "for",
    "from", "get", "have", "i", "in", "is", "it", "me", "my", "of", "on",
    "or", "please", "that", "the", "this", "to", "we", "what", "when",
    "with", "you",
}


class _FixtureRoster:
    def __init__(self, names: list[str]):
        self._names = names

    def load(self) -> None:
        return None

    def get_agents(self) -> list[str]:
        return list(self._names)


def load_benchmark() -> dict[str, Any]:
    """Load hand-authored agent records and routing labels."""
    return json.loads(_BENCHMARK_PATH.read_text(encoding="utf-8"))


def build_fixture(
    benchmark: Optional[dict[str, Any]] = None,
) -> tuple[list[str], dict[str, AgentProfile], list[dict[str, Any]]]:
    """Build retrieval inputs from the benchmark, using deterministic vectors."""
    data = benchmark or load_benchmark()
    names: list[str] = []
    profiles: dict[str, AgentProfile] = {}
    for agent in data["agents"]:
        name = agent["name"]
        instruction = agent["instruction"]
        names.append(name)
        vector = _fake_embedding(_profile_text(name, instruction))
        profiles[name] = AgentProfile(
            agent_name=name,
            instruction=instruction,
            last_active_at=agent.get("last_active_at", "2026-01-01T00:00:00+00:00"),
            embedding=vector,
            embedding_model="fixture/hash-v1",
            embedding_dimensions=len(vector),
        )
    return names, profiles, data["queries"]


def _profile_text(agent_name: str, instruction: str) -> str:
    return f"Agent: {agent_name}\nInstruction: {instruction}"


def _tokens(text: str) -> list[str]:
    return [
        token.casefold()
        for token in re.findall(r"[a-z0-9]+", text)
        if token.casefold() not in _STOP_WORDS
    ]


def _fake_embedding(text: str) -> list[float]:
    """Produce deterministic signed feature-hash vectors; not a semantic model."""
    vector = [0.0] * _FAKE_VECTOR_DIMENSIONS
    for token in _tokens(text):
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        value = int.from_bytes(digest, "big")
        index = value % _FAKE_VECTOR_DIMENSIONS
        vector[index] += 1.0 if value & (1 << 8) else -1.0
    return vector


def _cosine(left: list[float], right: list[float]) -> float:
    dot = math.fsum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(math.fsum(value * value for value in left))
    right_norm = math.sqrt(math.fsum(value * value for value in right))
    if not left_norm or not right_norm:
        return 0.0
    return dot / (left_norm * right_norm)


def _rank_exact(query: str, names: list[str]) -> list[str]:
    return sorted(
        (name for name in names if _has_exact_name(query.casefold(), name)),
        key=lambda name: (name.casefold(), name),
    )


def _rank_keyword(
    query: str, names: list[str], profiles: dict[str, AgentProfile]
) -> list[str]:
    query_terms = set(_tokens(query))
    scored: list[tuple[float, str]] = []
    for name in names:
        profile_terms = set(_tokens(f"{name} {profiles[name].instruction}"))
        union = query_terms | profile_terms
        score = len(query_terms & profile_terms) / len(union) if union else 0.0
        scored.append((score, name))
    scored.sort(key=lambda item: (-item[0], item[1].casefold(), item[1]))
    return [name for score, name in scored if score > 0]


def _render_prompt(
    query: str, candidates: list[tuple[str, Optional[str], str]]
) -> str:
    prompt_candidates = [
        AgentCandidate(agent_name=name, instruction=instruction, match_type=match_type)
        for name, instruction, match_type in candidates
    ]
    return prepare_message_with_history(
        query,
        "",
        active_agents=prompt_candidates,
        lookup_complete=True,
    )[0]["content"]


def _retrieval_metrics(
    ranked_results: list[list[str]], queries: list[dict[str, Any]]
) -> dict[str, Any]:
    scored_queries = [
        (ranked, set(case["relevant_agent_names"]))
        for ranked, case in zip(ranked_results, queries)
        if case["expected_action"] in {"reuse", "reuse_multiple"}
        and case["relevant_agent_names"]
    ]
    relevant_total = sum(len(relevant) for _, relevant in scored_queries)
    hits = sum(
        len(set(ranked[:MAX_AGENT_CANDIDATES]) & relevant)
        for ranked, relevant in scored_queries
    )
    reciprocal_ranks = []
    precisions = []
    for ranked, relevant in scored_queries:
        first_relevant_rank = next(
            (index for index, name in enumerate(ranked, 1) if name in relevant),
            None,
        )
        reciprocal_ranks.append(1 / first_relevant_rank if first_relevant_rank else 0.0)
        precisions.append(
            len(set(ranked[:MAX_AGENT_CANDIDATES]) & relevant)
            / MAX_AGENT_CANDIDATES
        )
    return {
        "labeled_reuse_queries": len(scored_queries),
        "relevant_agent_labels": relevant_total,
        "retrieved_relevant_agents_at_5": hits,
        "recall_at_5": hits / relevant_total if relevant_total else 1.0,
        "mean_reciprocal_rank": (
            sum(reciprocal_ranks) / len(reciprocal_ranks)
            if reciprocal_ranks
            else 1.0
        ),
        "mean_precision_at_5": (
            sum(precisions) / len(precisions) if precisions else 1.0
        ),
    }


async def _live_vectors(
    names: list[str],
    profiles: dict[str, AgentProfile],
    queries: list[dict[str, Any]],
    cache_path: Path = _LIVE_VECTOR_CACHE_PATH,
) -> tuple[dict[str, list[float]], dict[str, list[float]], dict[str, int]]:
    """Load cached or fetch synthetic profile/query vectors and persist the cache."""
    semaphore = asyncio.Semaphore(4)

    async def request(text: str) -> list[float]:
        async with semaphore:
            return await embed_text(text)

    profile_texts = {
        name: _profile_text(name, profiles[name].instruction) for name in names
    }
    query_texts = list(dict.fromkeys(case["query"] for case in queries))
    all_texts = list(dict.fromkeys([*profile_texts.values(), *query_texts]))
    cache: dict[str, list[float]] = {}
    if cache_path.exists():
        with gzip.open(cache_path, "rt", encoding="utf-8") as cache_file:
            payload = json.load(cache_file)
        if (
            not isinstance(payload, dict)
            or payload.get("version") != _LIVE_VECTOR_CACHE_VERSION
            or not isinstance(payload.get("vectors"), dict)
        ):
            raise ValueError(f"Invalid embedding cache format: {cache_path}")
        for key, vector in payload["vectors"].items():
            if (
                not isinstance(key, str)
                or not isinstance(vector, list)
                or not vector
                or any(not isinstance(value, (int, float)) for value in vector)
                or any(not math.isfinite(value) for value in vector)
            ):
                raise ValueError(f"Invalid vector entry in embedding cache: {cache_path}")
            cache[key] = [float(value) for value in vector]

    def cache_key(text: str) -> str:
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        return f"{get_settings().execution_agent_embedding_model.strip()}:{digest}"

    missing_texts = [text for text in all_texts if cache_key(text) not in cache]
    fetched_vectors = await asyncio.gather(*(request(text) for text in missing_texts))
    for text, vector in zip(missing_texts, fetched_vectors):
        cache[cache_key(text)] = vector

    if missing_texts:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path: Optional[Path] = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="wb",
                dir=cache_path.parent,
                prefix=f"{cache_path.name}.",
                suffix=".tmp",
                delete=False,
            ) as cache_file:
                temporary_path = Path(cache_file.name)
                with gzip.GzipFile(
                    filename="",
                    mode="wb",
                    fileobj=cache_file,
                    mtime=0,
                ) as compressed:
                    with io.TextIOWrapper(compressed, encoding="utf-8") as text_file:
                        json.dump(
                            {
                                "version": _LIVE_VECTOR_CACHE_VERSION,
                                "vectors": cache,
                            },
                            text_file,
                            separators=(",", ":"),
                        )
            os.replace(temporary_path, cache_path)
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)

    vectors_by_text = {text: cache[cache_key(text)] for text in all_texts}
    profile_vectors = {
        name: vectors_by_text[text] for name, text in profile_texts.items()
    }
    query_vectors = {text: vectors_by_text[text] for text in query_texts}
    return (
        profile_vectors,
        query_vectors,
        {
            "cached_vectors": len(all_texts) - len(missing_texts),
            "provider_fetches": len(missing_texts),
        },
    )


async def evaluate_benchmark(live_embeddings: bool = False) -> dict[str, Any]:
    """Evaluate the retrieval service and baselines on the synthetic benchmark."""
    data = load_benchmark()
    names, profiles, queries = build_fixture(data)
    roster = _FixtureRoster(names)
    agent_names_by_id = {agent["id"]: agent["name"] for agent in data["agents"]}
    scoring_queries = [
        {
            **case,
            "relevant_agent_names": [
                agent_names_by_id[agent_id]
                for agent_id in case["relevant_agent_ids"]
            ],
        }
        for case in queries
    ]

    if live_embeddings:
        profile_vectors, query_vectors, embedding_cache_stats = await _live_vectors(
            names, profiles, queries
        )
        model = get_settings().execution_agent_embedding_model.strip()
        profiles = {
            name: AgentProfile(
                agent_name=profile.agent_name,
                instruction=profile.instruction,
                last_active_at=profile.last_active_at,
                embedding=profile_vectors[name],
                embedding_model=model,
                embedding_dimensions=len(profile_vectors[name]),
            )
            for name, profile in profiles.items()
        }

        async def benchmark_embed(text: str) -> list[float]:
            return query_vectors[text]
    else:
        model = "fixture/hash-v1"
        embedding_cache_stats = {"cached_vectors": 0, "provider_fetches": 0}

        async def benchmark_embed(text: str) -> list[float]:
            return _fake_embedding(text)

    settings = get_settings().model_copy(
        update={
            "openrouter_api_key": get_settings().openrouter_api_key or "offline-fixture",
            "execution_agent_embedding_model": model,
        }
    )
    hybrid_results: list[list[str]] = []
    exact_results: list[list[str]] = []
    keyword_results: list[list[str]] = []
    failures: list[dict[str, Any]] = []
    prompt_chars = 0
    profile_roster_prompt_chars = 0
    names_only_roster_prompt_chars = 0
    candidate_counts = 0

    for case in queries:
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
                side_effect=benchmark_embed,
            ),
        ):
            result = await retrieve_agents(case["query"])

        ranked = [candidate.agent_name for candidate in result.candidates]
        hybrid_results.append(ranked)
        exact_results.append(_rank_exact(case["query"], names))
        keyword_results.append(
            _rank_keyword(case["query"], names, profiles)[:MAX_AGENT_CANDIDATES]
        )
        candidate_counts += len(ranked)
        prompt_chars += len(
            _render_prompt(
                case["query"],
                [
                    (c.agent_name, c.instruction, c.match_type)
                    for c in result.candidates
                ],
            )
        )
        profile_roster_prompt_chars += len(
            _render_prompt(
                case["query"],
                [
                    (name, profiles[name].instruction, "semantic")
                    for name in names
                ],
            )
        )
        names_only_roster_prompt_chars += len(
            _render_prompt(case["query"], [(name, None, "exact") for name in names])
        )

        expected = {
            agent_names_by_id[agent_id] for agent_id in case["relevant_agent_ids"]
        }
        if expected and not (set(ranked[:MAX_AGENT_CANDIDATES]) & expected):
            failures.append({
                "case_id": case["id"],
                "expected_agent_ids": sorted(case["relevant_agent_ids"]),
                "expected_agent_names": sorted(expected),
                "retrieved_agent_names": ranked,
            })

    hybrid_metrics = _retrieval_metrics(hybrid_results, scoring_queries)
    output = {
        "benchmark": data["metadata"]["name"],
        "benchmark_version": data["metadata"]["version"],
        "agents": len(names),
        "queries": len(queries),
        "candidate_limit": MAX_AGENT_CANDIDATES,
        "embedding_mode": "real_embeddings" if live_embeddings else "offline_hash_proxy",
        "embedding_model": model,
        "embedding_cache": embedding_cache_stats,
        "provider_calls_use_synthetic_data_only": live_embeddings,
        "action_counts": dict(Counter(case["expected_action"] for case in queries)),
        "routing_decisions_measured": False,
        "output_dependent_cases_excluded": sum(
            case["scope"] == "output_dependent" for case in queries
        ),
        "hybrid": hybrid_metrics,
        "exact_name_baseline": _retrieval_metrics(exact_results, scoring_queries),
        "keyword_baseline": _retrieval_metrics(keyword_results, scoring_queries),
        "average_candidate_count": candidate_counts / len(queries),
        "average_prompt_characters_with_retrieved_candidates": prompt_chars / len(queries),
        "average_full_roster_prompt_characters_with_profiles": (
            profile_roster_prompt_chars / len(queries)
        ),
        "average_full_roster_prompt_characters_names_only": (
            names_only_roster_prompt_chars / len(queries)
        ),
        "prompt_character_reduction_vs_full_profiles": (
            1 - prompt_chars / profile_roster_prompt_chars
            if profile_roster_prompt_chars
            else 0.0
        ),
        "prompt_character_reduction_vs_names_only_roster": (
            1 - prompt_chars / names_only_roster_prompt_chars
            if names_only_roster_prompt_chars
            else 0.0
        ),
        "retrieval_failures": failures,
    }
    if live_embeddings:
        output["quality_threshold"] = RECALL_AT_5_THRESHOLD
        output["passed"] = hybrid_metrics["recall_at_5"] >= RECALL_AT_5_THRESHOLD
    else:
        output["quality_threshold"] = None
        output["passed"] = None
        output["offline_result_note"] = (
            "Deterministic feature-hash vectors are a regression proxy, not a "
            "measure of real embedding quality."
        )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--live-embeddings",
        action="store_true",
        help=(
            "Use the configured OpenRouter embedding model on synthetic benchmark "
            "text only; this makes provider calls and may incur cost."
        ),
    )
    parser.add_argument(
        "--results-file",
        type=Path,
        help="Also write the JSON evaluation result to this path.",
    )
    args = parser.parse_args()
    results = asyncio.run(evaluate_benchmark(live_embeddings=args.live_embeddings))
    rendered = json.dumps(results, indent=2, sort_keys=True) + "\n"
    if args.results_file:
        args.results_file.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    if results["passed"] is False:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
