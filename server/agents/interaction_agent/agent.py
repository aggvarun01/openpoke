"""Interaction agent helpers for prompt construction."""

from html import escape
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from ...services.execution import AgentCandidate

_prompt_path = Path(__file__).parent / "system_prompt.md"
SYSTEM_PROMPT = _prompt_path.read_text(encoding="utf-8").strip()


# Load and return the pre-defined system prompt from markdown file
def build_system_prompt() -> str:
    """Return the static system prompt for the interaction agent."""
    return SYSTEM_PROMPT


# Build structured message with conversation history, active agents, and current turn
def prepare_message_with_history(
    latest_text: str,
    transcript: str,
    message_type: str = "user",
    active_agents: Optional[Sequence[AgentCandidate]] = None,
    lookup_complete: Optional[bool] = None,
    lookup_reason: Optional[str] = None,
    originating_agent_names: Optional[Sequence[str]] = None,
) -> List[Dict[str, str]]:
    """Compose history and a bounded retrieval/origin list with the current turn."""
    sections: List[str] = []

    sections.append(_render_conversation_history(transcript))
    if message_type == "user":
        sections.append(
            _render_active_agents(active_agents or (), lookup_complete, lookup_reason)
        )
    else:
        sections.append(_render_agent_origins(originating_agent_names))
    sections.append(_render_current_turn(latest_text, message_type))

    content = "\n\n".join(sections)
    return [{"role": "user", "content": content}]


# Format conversation transcript into XML tags for LLM context
def _render_conversation_history(transcript: str) -> str:
    history = transcript.strip()
    if not history:
        history = "None"
    return f"<conversation_history>\n{history}\n</conversation_history>"


def _render_active_agents(
    agents: Sequence[AgentCandidate],
    lookup_complete: Optional[bool],
    lookup_reason: Optional[str],
) -> str:
    """Render bounded retrieval candidates and whether lookup was complete."""
    status = "complete" if lookup_complete else "incomplete"
    rendered = [f'<active_agents status="{status}">']
    if lookup_reason:
        rendered.append(f"<lookup_status>{escape(lookup_reason)}</lookup_status>")
    for candidate in agents:
        name = escape(candidate.agent_name or "agent", quote=True)
        match_type = escape(candidate.match_type, quote=True)
        instruction = escape(candidate.instruction or "", quote=False)
        rendered.append(
            f'<agent name="{name}" match="{match_type}">'
            f"<instruction>{instruction}</instruction></agent>"
        )
    if not agents:
        rendered.append("<agents>None</agents>")
    rendered.append("</active_agents>")
    return "\n".join(rendered)


def _render_agent_origins(originating_agent_names: Optional[Sequence[str]]) -> str:
    """Render only known origins for agent-result turns; never load the roster."""
    if originating_agent_names is None:
        return '<active_agents source="not_provided">None</active_agents>'

    rendered = ['<active_agents source="execution_result">']
    seen: set[str] = set()
    for agent_name in originating_agent_names:
        if not agent_name or agent_name in seen:
            continue
        seen.add(agent_name)
        rendered.append(f'<agent name="{escape(agent_name, quote=True)}" />')
    if not seen:
        rendered.append("<agents>None</agents>")
    rendered.append("</active_agents>")
    return "\n".join(rendered)


# Wrap the current message in appropriate XML tags based on sender type
def _render_current_turn(latest_text: str, message_type: str) -> str:
    tag = "new_agent_message" if message_type == "agent" else "new_user_message"
    body = latest_text.strip()
    return f"<{tag}>\n{body}\n</{tag}>"
