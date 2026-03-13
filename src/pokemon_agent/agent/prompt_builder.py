from __future__ import annotations

import json
from dataclasses import dataclass, field

from pokemon_agent.agent.context_manager import ContextSnapshot


@dataclass(slots=True)
class PromptMetrics:
    chars: int
    approx_tokens: int
    compact: bool
    budget_tokens: int | None = None
    used_tokens: int | None = None
    section_tokens: dict[str, int] = field(default_factory=dict)
    dropped_sections: list[str] = field(default_factory=list)
    warning: str | None = None


class PromptBuilder:
    def build(self, snapshot: ContextSnapshot) -> list[dict]:
        return [
            {"role": "system", "content": snapshot.system_prompt},
            {"role": "user", "content": json.dumps(snapshot.payload, separators=(",", ":"), sort_keys=True)},
        ]

    def measure(self, messages: list[dict], snapshot: ContextSnapshot | None = None) -> PromptMetrics:
        chars = sum(len(message.get("content", "")) for message in messages)
        approx_tokens = max(1, chars // 4)
        if snapshot is None:
            compact = chars <= 2400
            warning = None if compact else "Prompt is growing large; consider pruning memory or metadata."
            return PromptMetrics(chars=chars, approx_tokens=approx_tokens, compact=compact, warning=warning)

        compact = snapshot.used_tokens <= snapshot.budget_tokens
        warning = None if compact else "Prompt remains above budget after pruning mandatory context."
        return PromptMetrics(
            chars=chars,
            approx_tokens=approx_tokens,
            compact=compact,
            budget_tokens=snapshot.budget_tokens,
            used_tokens=snapshot.used_tokens,
            section_tokens=dict(snapshot.section_tokens),
            dropped_sections=list(snapshot.dropped_sections),
            warning=warning,
        )
