from __future__ import annotations

import copy
import json
from dataclasses import asdict, dataclass, field
from typing import Any

from pokemon_agent.agent.ascii_map import build_ascii_map
from pokemon_agent.agent.progress import ProgressResult
from pokemon_agent.agent.stuck_detector import StuckState
from pokemon_agent.agent.world_map import summarize_navigation_goal
from pokemon_agent.data.walkthrough import get_current_milestone
from pokemon_agent.models.action import ActionDecision
from pokemon_agent.models.events import EventRecord
from pokemon_agent.models.memory import MemoryState
from pokemon_agent.models.planner import CandidateNextStep
from pokemon_agent.models.state import GameMode
from pokemon_agent.models.state import StructuredGameState

RESPONSE_SCHEMA = {
    "candidate_id": "<exact id from candidate_next_steps>",
    "reason": "<10 words max>",
}
PLANNER_SYSTEM_PROMPT = (
    "You control a Pokémon Red agent. Your only task: select one candidate from candidate_next_steps "
    "and return its exact id. "
    "Rules: "
    "(1) If stuck_warning is present, do not select a candidate listed in stuck_warning.failed_candidate_ids. "
    "(2) For yes/no prompts, read the dialogue and choose select_yes or select_no. "
    "(3) Otherwise prefer the candidate whose why best matches the current state and goal. "
    'Return exactly: {"candidate_id": "<id>", "reason": "<10 words max>"}. '
    "No markdown. No extra keys. candidate_id must be one of the ids shown."
)


@dataclass(slots=True)
class ActionTrace:
    turn_index: int
    action: str
    repeat: int
    reason: str
    progress_classification: str
    map_name: str
    position: dict[str, int | None]
    event_summaries: list[str]
    stuck_score: int
    used_fallback: bool
    llm_attempted: bool
    planner_source: str = "fallback"

    def to_payload(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> ActionTrace:
        return cls(
            turn_index=int(payload["turn_index"]),
            action=str(payload["action"]),
            repeat=int(payload["repeat"]),
            reason=str(payload.get("reason", "")),
            progress_classification=str(payload["progress_classification"]),
            map_name=str(payload["map_name"]),
            position={
                "x": payload.get("position", {}).get("x"),
                "y": payload.get("position", {}).get("y"),
            },
            event_summaries=[str(item) for item in payload.get("event_summaries", [])],
            stuck_score=int(payload["stuck_score"]),
            used_fallback=bool(payload["used_fallback"]),
            llm_attempted=bool(payload["llm_attempted"]),
            planner_source=str(payload.get("planner_source", "fallback")),
        )


@dataclass(slots=True)
class ContextSnapshot:
    system_prompt: str
    payload: dict[str, Any]
    budget_tokens: int
    used_tokens: int
    section_tokens: dict[str, int] = field(default_factory=dict)
    dropped_sections: list[str] = field(default_factory=list)


class ContextManager:
    def __init__(
        self,
        budget_tokens: int = 2500,
        action_window: int = 8,
        event_window: int = 6,
    ) -> None:
        self.budget_tokens = budget_tokens
        self.action_window = action_window
        self.event_window = event_window
        self.action_traces: list[ActionTrace] = []

    def build_snapshot(
        self,
        state: StructuredGameState,
        memory_state: MemoryState,
        stuck_state: StuckState | None = None,
        candidate_next_steps: list[CandidateNextStep] | None = None,
        recommended_step: CandidateNextStep | None = None,
    ) -> ContextSnapshot:
        context = self._build_context(state, memory_state, stuck_state, candidate_next_steps, recommended_step)
        system_prompt = self._build_system_prompt(state, context)
        payload = {
            "context": context,
            "response_schema": copy.deepcopy(RESPONSE_SCHEMA),
        }
        dropped_sections: list[str] = []
        self._prune_to_budget(payload, dropped_sections, system_prompt)
        section_tokens = self._measure_sections(payload, system_prompt)
        return ContextSnapshot(
            system_prompt=system_prompt,
            payload=payload,
            budget_tokens=self.budget_tokens,
            used_tokens=self._measure_total_tokens(system_prompt, payload),
            section_tokens=section_tokens,
            dropped_sections=dropped_sections,
        )

    def record_turn(
        self,
        turn_index: int,
        action: ActionDecision,
        after_state: StructuredGameState,
        progress: ProgressResult,
        events: list[EventRecord],
        stuck_state: StuckState,
        used_fallback: bool,
        llm_attempted: bool,
        planner_source: str = "fallback",
    ) -> None:
        trace = ActionTrace(
            turn_index=turn_index,
            action=action.action.value,
            repeat=action.repeat,
            reason=action.reason,
            progress_classification=progress.classification,
            map_name=after_state.map_name,
            position={"x": after_state.x, "y": after_state.y},
            event_summaries=[event.summary for event in events],
            stuck_score=stuck_state.score,
            used_fallback=used_fallback,
            llm_attempted=llm_attempted,
            planner_source=planner_source,
        )
        self.action_traces.append(trace)
        self.action_traces = self.action_traces[-self.action_window :]

    def export_state(self) -> dict[str, Any]:
        return {
            "action_traces": [trace.to_payload() for trace in self.action_traces],
        }

    def restore_state(self, payload: dict[str, Any] | None) -> None:
        traces = [] if payload is None else payload.get("action_traces", [])
        self.action_traces = [ActionTrace.from_payload(item) for item in traces][-self.action_window :]

    def _build_context(
        self,
        state: StructuredGameState,
        memory_state: MemoryState,
        stuck_state: StuckState | None,
        candidate_next_steps: list[CandidateNextStep] | None,
        recommended_step: CandidateNextStep | None,  # kept for API compat, not used
    ) -> dict[str, Any]:
        immediate_state = self._build_immediate_state(state)
        context: dict[str, Any] = {
            "immediate_state": immediate_state,
            "goal": self._build_goal(immediate_state),
        }
        context.update(self._build_mode_context(state, memory_state))
        if candidate_next_steps:
            context["candidate_next_steps"] = [self._serialize_candidate(item) for item in candidate_next_steps[:4]]
        stuck_warning = self._build_stuck_warning(stuck_state)
        if stuck_warning is not None:
            if candidate_next_steps and stuck_state is not None:
                failed_actions = set(stuck_state.recent_failed_actions[-3:])
                failed_ids = [
                    cand.id for cand in candidate_next_steps
                    if cand.action and cand.action.action.value in failed_actions
                ]
                # When oscillating/high stuck score, also flag interactable candidates so the
                # LLM avoids re-selecting them (same sign re-read pattern).
                if stuck_state.score >= 2 and (stuck_state.oscillating or stuck_state.score >= 3):
                    interactable_ids = [
                        cand.id for cand in candidate_next_steps
                        if cand.type == "MOVE_ADJACENT_TO_INTERACTABLE"
                    ]
                    failed_ids = list(dict.fromkeys(failed_ids + interactable_ids))
                if failed_ids:
                    stuck_warning["failed_candidate_ids"] = failed_ids
            context["stuck_warning"] = stuck_warning
        last_outcome = self._build_last_outcome()
        if last_outcome is not None:
            context["last_outcome"] = last_outcome
        recent_events = self._build_recent_events(memory_state)
        if recent_events:
            context["recent_events"] = recent_events
        return context

    def _build_system_prompt(self, state: StructuredGameState, context: dict[str, Any]) -> str:
        return PLANNER_SYSTEM_PROMPT

    def _build_goal(self, immediate_state: dict[str, Any]) -> str:
        milestone = immediate_state.get("current_milestone", {})
        description = str(milestone.get("description") or "")
        target_map = str(milestone.get("target_map") or "")
        if target_map and description:
            return f"Navigate to {target_map} — {description}"
        if description:
            return description
        return "Follow the walkthrough."

    def _build_immediate_state(self, state: StructuredGameState) -> dict[str, Any]:
        battle_payload = self._battle_payload(state)
        battle_kind = battle_payload.get("kind")
        inventory_names = [item.name for item in state.inventory]
        current_milestone = get_current_milestone(
            state.story_flags,
            inventory_names,
            current_map_name=state.map_name,
            badges=state.badges,
        )
        immediate: dict[str, Any] = {
            "engine_phase": state.metadata.get("engine_phase", "active"),
            "map": {"name": state.map_name, "id": state.map_id},
            "position": {"x": state.x, "y": state.y},
            "facing": state.facing,
            "mode": state.mode.value,
            "badges": list(state.badges),
            "current_milestone": {
                "id": current_milestone.id,
                "description": current_milestone.description,
                "target_map": current_milestone.target_map_name,
                "next_hint": current_milestone.route_hints[0] if current_milestone.route_hints else None,
            },
            "ui_flags": {
                "menu_open": state.menu_open,
                "text_box_open": state.text_box_open,
            },
            "battle_kind": battle_kind,
            "step": state.step,
        }
        dialogue_text = state.metadata.get("dialogue")
        if not isinstance(dialogue_text, str) or not dialogue_text.strip():
            dialogue_text = state.metadata.get("dialogue_text")
        if isinstance(dialogue_text, str) and dialogue_text.strip():
            immediate["dialogue_text"] = dialogue_text
        if state.text_box_open:
            immediate["ui_flags"]["yes_no_prompt"] = bool(state.metadata.get("yes_no_prompt"))
        return immediate

    def _build_mode_context(self, state: StructuredGameState, memory_state: MemoryState) -> dict[str, Any]:
        if state.battle_state or state.mode == GameMode.BATTLE:
            return {"battle_context": self._build_battle_context(state)}
        if state.text_box_open or state.mode == GameMode.TEXT:
            return {"dialogue_context": self._build_dialogue_context(state)}
        if self._is_overworld_mode(state) and state.navigation is not None:
            return {"overworld_context": self._build_overworld_context(state, memory_state)}
        return {}

    def _build_overworld_context(self, state: StructuredGameState, memory_state: MemoryState) -> dict[str, Any]:
        visual_map = build_ascii_map(state)
        route_info = summarize_navigation_goal(
            memory_state.long_term.world_map,
            state.map_name,
            memory_state.long_term.navigation_goal,
        )
        return {
            "visual_map": visual_map,
            "map_legend": "P=player .=walkable #=blocked ~=water @=isolated blocker D=door",
            **route_info,
        }

    def _build_dialogue_context(self, state: StructuredGameState) -> dict[str, Any]:
        dialogue_text = self._dialogue_text(state)
        return {
            "dialogue_text": dialogue_text,
            "choice_mode": "YES_NO" if self._looks_like_yes_no_prompt(dialogue_text) else "ADVANCE_OR_CHOOSE",
        }

    def _build_battle_context(self, state: StructuredGameState) -> dict[str, Any]:
        battle_state = self._battle_payload(state)
        enemy_name = (
            battle_state.get("opponent")
            or battle_state.get("enemy")
            or battle_state.get("enemy_species")
            or battle_state.get("kind")
            or "UNKNOWN"
        )
        enemy_level = self._coalesce(battle_state.get("opponent_level"), battle_state.get("enemy_level"))
        lead = state.party[0] if state.party else None
        moves = self._battle_moves(state)
        lead_payload: dict[str, Any] = {
            "name": self._coalesce(battle_state.get("player_active_species"), lead.name if lead else None, "UNKNOWN"),
            "level": self._coalesce(battle_state.get("player_active_level"), lead.level if lead else None),
            "hp": self._coalesce(battle_state.get("player_active_hp"), lead.hp if lead else None),
            "max_hp": self._coalesce(battle_state.get("player_active_max_hp"), lead.max_hp if lead else None),
            "status": lead.status if lead else None,
        }
        return {
            "enemy": {
                "kind": battle_state.get("kind"),
                "name": enemy_name,
                "level": enemy_level,
            },
            "lead_pokemon": lead_payload,
            "moves": moves[:4],
            "party_preview": [
                {
                    "name": member.name,
                    "level": member.level,
                    "hp": member.hp,
                    "max_hp": member.max_hp,
                    "status": member.status,
                }
                for member in state.party[:3]
            ],
        }

    @staticmethod
    def _dialogue_text(state: StructuredGameState) -> str:
        text = state.metadata.get("dialogue")
        if not isinstance(text, str) or not text.strip():
            text = state.metadata.get("dialogue_text")
        if isinstance(text, str) and text.strip():
            return text.strip()
        return "dialogue text unavailable"

    @staticmethod
    def _battle_moves(state: StructuredGameState) -> list[str]:
        battle_payload = ContextManager._battle_payload(state)
        battle_moves = battle_payload.get("moves")
        if not isinstance(battle_moves, list):
            battle_moves = battle_payload.get("available_moves")
        if not isinstance(battle_moves, list) and isinstance(state.metadata.get("battle_moves"), list):
            battle_moves = state.metadata.get("battle_moves")
        if not battle_moves:
            return []
        normalized: list[str] = []
        for move in battle_moves:
            if isinstance(move, dict):
                name = move.get("name")
            else:
                name = getattr(move, "name", move)
            text = str(name).strip()
            if text:
                normalized.append(text)
        return normalized

    @staticmethod
    def _looks_like_yes_no_prompt(dialogue_text: str) -> bool:
        lowered = dialogue_text.lower()
        return ("yes" in lowered and "no" in lowered) or "would you like" in lowered or "(y/n)" in lowered

    @staticmethod
    def _hp_text(hp: Any, max_hp: Any) -> str:
        if hp is None and max_hp is None:
            return "unknown HP"
        if hp is None:
            return f"?/{max_hp} HP"
        if max_hp is None:
            return f"{hp} HP"
        return f"{hp}/{max_hp} HP"

    @staticmethod
    def _is_overworld_mode(state: StructuredGameState) -> bool:
        if state.battle_state or state.text_box_open or state.menu_open:
            return False
        return state.mode == GameMode.OVERWORLD or (state.mode == GameMode.UNKNOWN and state.navigation is not None)

    @staticmethod
    def _coalesce(*values: Any) -> Any:
        for value in values:
            if value is not None:
                return value
        return None

    @staticmethod
    def _battle_payload(state: StructuredGameState) -> dict[str, Any]:
        if state.battle_state is None:
            return {}
        if isinstance(state.battle_state, dict):
            return state.battle_state
        model_dump = getattr(state.battle_state, "model_dump", None)
        if callable(model_dump):
            payload = model_dump()
            if isinstance(payload, dict):
                return payload
        return {}

    def _build_recent_events(self, memory_state: MemoryState) -> list[dict[str, Any]]:
        return [
            {
                "type": event.type.value,
                "summary": event.summary,
                "step": event.step,
            }
            for event in memory_state.recent_events[-min(self.event_window, 2) :]
        ]

    def _build_last_outcome(self) -> dict[str, Any] | None:
        if not self.action_traces:
            return None
        trace = self.action_traces[-1]
        return {
            "turn_index": trace.turn_index,
            "action": trace.action,
            "result": trace.progress_classification,
            "reason": trace.reason,
            "planner_source": trace.planner_source,
            "event_summaries": trace.event_summaries[:2],
            "stuck_score": trace.stuck_score,
        }

    def _serialize_candidate(self, candidate: CandidateNextStep) -> dict[str, Any]:
        payload = candidate.model_dump(mode="json", exclude_none=True)
        return payload

    def _build_stuck_warning(self, stuck_state: StuckState | None) -> dict[str, Any] | None:
        if stuck_state is None or stuck_state.score < 2:
            return None
        return {
            "stuck_score": stuck_state.score,
            "recent_failed_actions": stuck_state.recent_failed_actions,
            "loop_signature": stuck_state.loop_signature,
            "recovery_hint": stuck_state.recovery_hint,
        }

    def _prune_to_budget(self, payload: dict[str, Any], dropped_sections: list[str], system_prompt: str) -> None:
        context = payload["context"]
        if self._within_budget(payload, system_prompt):
            return

        # 1. Drop ASCII map first (large, only useful when stuck)
        overworld_context = context.get("overworld_context")
        if isinstance(overworld_context, dict) and "visual_map" in overworld_context:
            overworld_context.pop("visual_map", None)
            overworld_context.pop("map_legend", None)
            dropped_sections.append("overworld_ascii_map")
        if self._within_budget(payload, system_prompt):
            return

        # 2. Drop battle party preview
        battle_context = context.get("battle_context")
        if isinstance(battle_context, dict) and "party_preview" in battle_context:
            del battle_context["party_preview"]
            dropped_sections.append("battle_party_preview")
        if self._within_budget(payload, system_prompt):
            return

        # 3. Reduce then drop recent events
        recent_events = context.get("recent_events")
        if isinstance(recent_events, list) and len(recent_events) > 1:
            context["recent_events"] = recent_events[-1:]
            dropped_sections.append("recent_events_reduced")
        if self._within_budget(payload, system_prompt):
            return

        if "recent_events" in context:
            del context["recent_events"]
            dropped_sections.append("recent_events")
        if self._within_budget(payload, system_prompt):
            return

        # 4. Reduce last_outcome
        last_outcome = context.get("last_outcome")
        if isinstance(last_outcome, dict) and "event_summaries" in last_outcome:
            last_outcome["event_summaries"] = last_outcome.get("event_summaries", [])[:1]
            dropped_sections.append("last_outcome_reduced")
        if self._within_budget(payload, system_prompt):
            return

        # 5. Reduce stuck_warning (drop loop_signature, failed_candidate_ids)
        if "stuck_warning" in context:
            context["stuck_warning"].pop("loop_signature", None)
            context["stuck_warning"].pop("failed_candidate_ids", None)
            dropped_sections.append("stuck_warning_reduced")
        if self._within_budget(payload, system_prompt):
            return

        # 6. Reduce candidate list to 3
        candidates = context.get("candidate_next_steps")
        if isinstance(candidates, list) and len(candidates) > 3:
            context["candidate_next_steps"] = candidates[:3]
            dropped_sections.append("candidate_next_steps_reduced")
        if self._within_budget(payload, system_prompt):
            return

        # 7. Drop candidate why fields (near-last — most actionable data)
        candidates = context.get("candidate_next_steps")
        if isinstance(candidates, list):
            reduced_any = False
            for candidate in candidates:
                if isinstance(candidate, dict) and "why" in candidate:
                    candidate.pop("why", None)
                    reduced_any = True
            if reduced_any:
                dropped_sections.append("candidate_next_steps_reasons")
        if self._within_budget(payload, system_prompt):
            return

        # 8. Last resort: reduce to single candidate
        candidates = context.get("candidate_next_steps")
        if isinstance(candidates, list) and len(candidates) > 1:
            context["candidate_next_steps"] = candidates[:1]
            dropped_sections.append("candidate_next_steps_single")

    def _within_budget(self, payload: dict[str, Any], system_prompt: str) -> bool:
        return self._measure_total_tokens(system_prompt, payload) <= self.budget_tokens

    def _measure_sections(self, payload: dict[str, Any], system_prompt: str) -> dict[str, int]:
        context = payload.get("context", {})
        section_tokens: dict[str, int] = {
            "system_prompt": self._approx_tokens(system_prompt),
            "response_schema": self._approx_tokens(self._serialize(payload.get("response_schema", {}))),
        }
        for key, value in context.items():
            section_tokens[key] = self._approx_tokens(self._serialize(value))
        return section_tokens

    def _measure_total_tokens(self, system_prompt: str, payload: dict[str, Any]) -> int:
        chars = len(system_prompt) + len(self._serialize(payload))
        return max(1, chars // 4)

    @staticmethod
    def _approx_tokens(text: str) -> int:
        if not text:
            return 0
        return max(1, len(text) // 4)

    @staticmethod
    def _serialize(value: Any) -> str:
        return json.dumps(value, separators=(",", ":"), sort_keys=True, ensure_ascii=True)
