from __future__ import annotations

import copy
import json
from dataclasses import asdict, dataclass, field
from typing import Any

from pokemon_agent.agent.progress import ProgressResult
from pokemon_agent.agent.stuck_detector import StuckState
from pokemon_agent.data.walkthrough import get_current_milestone
from pokemon_agent.data.walkthrough import get_progress_summary
from pokemon_agent.emulator.screen_renderer import render_ascii_map
from pokemon_agent.models.action import ActionDecision
from pokemon_agent.models.events import EventRecord
from pokemon_agent.models.memory import MemoryState
from pokemon_agent.models.planner import CandidateNextStep
from pokemon_agent.models.planner import Objective
from pokemon_agent.models.state import GameMode
from pokemon_agent.models.state import StructuredGameState

RESPONSE_SCHEMA = {
    "candidate_id": "candidate_1",
    "reason": "nearest exit looks safest",
    "confidence": 0.72,
}
PLANNER_SYSTEM_PROMPT = (
    "You select the next intent for a Pokemon Red engine; you do not control inputs directly. "
    "The engine handles routing, timing, execution, validation, and recovery. "
    "Decision order: resolve forced UI, text, or battle states first, continue a still-valid objective next, "
    "then choose from candidate_next_steps. "
    "Prefer the shortest valid step most likely to trigger its expected success signal. "
    "Avoid repeating recent failed actions unless the state changed. "
    "Return exactly one JSON object with candidate_id, reason, and optional confidence. "
    "Keep reason short. No markdown or extra keys."
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
        action_window: int = 4,
        event_window: int = 4,
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
        recommended_step: CandidateNextStep | None,
    ) -> dict[str, Any]:
        immediate_state = self._build_immediate_state(state)
        walkthrough_context = self._build_walkthrough_context(immediate_state)
        context: dict[str, Any] = {
            "immediate_state": immediate_state,
            "walkthrough_context": walkthrough_context,
            "active_objective_stack": self._build_objectives(memory_state),
        }
        context.update(self._build_mode_context(state, walkthrough_context))
        if candidate_next_steps:
            context["candidate_next_steps"] = [self._serialize_candidate(item) for item in candidate_next_steps[:4]]
        if recommended_step is not None:
            context["recommended_next_step"] = self._serialize_candidate(recommended_step)
        stuck_warning = self._build_stuck_warning(stuck_state)
        if stuck_warning is not None:
            context["stuck_warning"] = stuck_warning
        last_outcome = self._build_last_outcome()
        if last_outcome is not None:
            context["last_outcome"] = last_outcome
        recent_events = self._build_recent_events(memory_state)
        if recent_events:
            context["recent_events"] = recent_events
        return context

    def _build_system_prompt(self, state: StructuredGameState, context: dict[str, Any]) -> str:
        walkthrough = context.get("walkthrough_context", {})
        milestone = str(walkthrough.get("milestone", "the active walkthrough milestone"))
        target_map = str(walkthrough.get("target_map", "the next map"))
        if state.battle_state or state.mode == GameMode.BATTLE:
            battle_context = context.get("battle_context", {})
            enemy = battle_context.get("enemy", {})
            lead = battle_context.get("lead_pokemon", {})
            moves = battle_context.get("moves", [])
            enemy_label = str(enemy.get("name") or enemy.get("kind") or "the current enemy")
            enemy_level = enemy.get("level")
            if enemy_level is not None:
                enemy_label = f"{enemy_label} lv {enemy_level}"
            lead_name = str(lead.get("name", "your lead Pokemon"))
            lead_hp = self._hp_text(lead.get("hp"), lead.get("max_hp"))
            moves_label = ", ".join(str(move) for move in moves[:4]) if isinstance(moves, list) and moves else "unavailable"
            return (
                f"{PLANNER_SYSTEM_PROMPT} Battle: Fighting {enemy_label}. "
                f"Your {lead_name} has {lead_hp}. Moves: {moves_label}. Choose wisely."
            )
        if state.text_box_open or state.mode == GameMode.TEXT:
            dialogue_context = context.get("dialogue_context", {})
            dialogue_text = str(dialogue_context.get("dialogue_text") or "dialogue text unavailable")
            return (
                f"{PLANNER_SYSTEM_PROMPT} Dialogue: NPC says: {dialogue_text!r}. "
                "Choose Yes or No if a confirmation prompt is visible; otherwise pick the safest dialogue candidate."
            )
        if self._is_overworld_mode(state):
            return (
                f"{PLANNER_SYSTEM_PROMPT} Overworld: Navigate toward {milestone}. "
                f"You are on {state.map_name}. Target: {target_map}."
            )
        return f"{PLANNER_SYSTEM_PROMPT} Stay aligned with the current walkthrough milestone: {milestone}."

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
        immediate = {
            "engine_phase": state.metadata.get("engine_phase", "active"),
            "map": {"name": state.map_name, "id": state.map_id},
            "position": {"x": state.x, "y": state.y},
            "facing": state.facing,
            "mode": state.mode.value,
            "badges": list(state.badges),
            "story_progress": get_progress_summary(
                state.story_flags,
                inventory_names,
                current_map_name=state.map_name,
                badges=state.badges,
            ),
            "current_milestone": {
                "id": current_milestone.id,
                "description": current_milestone.description,
                "target_map": current_milestone.target_map_name,
                "route_hints": current_milestone.route_hints[:1],
                "sub_steps": current_milestone.sub_steps[:1],
                "required_hms": current_milestone.required_hms,
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
        if state.navigation is not None:
            immediate["navigation_window"] = {
                "min_x": state.navigation.min_x,
                "min_y": state.navigation.min_y,
                "max_x": state.navigation.max_x,
                "max_y": state.navigation.max_y,
                "collision_hash": state.navigation.collision_hash,
            }
        return immediate

    def _build_walkthrough_context(self, immediate_state: dict[str, Any]) -> dict[str, Any]:
        milestone = immediate_state.get("current_milestone", {})
        return {
            "milestone_id": milestone.get("id"),
            "milestone": milestone.get("description"),
            "target_map": milestone.get("target_map"),
            "route_hints": list(milestone.get("route_hints", []))[:3],
        }

    def _build_mode_context(self, state: StructuredGameState, walkthrough_context: dict[str, Any]) -> dict[str, Any]:
        if state.battle_state or state.mode == GameMode.BATTLE:
            return {"battle_context": self._build_battle_context(state, walkthrough_context)}
        if state.text_box_open or state.mode == GameMode.TEXT:
            return {"dialogue_context": self._build_dialogue_context(state, walkthrough_context)}
        if self._is_overworld_mode(state) and state.navigation is not None:
            return {"overworld_context": self._build_overworld_context(state, walkthrough_context)}
        return {}

    def _build_overworld_context(self, state: StructuredGameState, walkthrough_context: dict[str, Any]) -> dict[str, Any]:
        milestone = str(walkthrough_context.get("milestone") or "the current milestone")
        target_map = str(walkthrough_context.get("target_map") or "the next map")
        visual_map = self._build_ascii_map(state)
        return {
            "navigation_prompt": f"Navigate toward {milestone}. You are on {state.map_name}. Target: {target_map}.",
            "visual_map": visual_map,
            "map_legend": "P=player .=walkable #=blocked ~=water @=isolated blocker D=door",
        }

    def _build_dialogue_context(self, state: StructuredGameState, walkthrough_context: dict[str, Any]) -> dict[str, Any]:
        dialogue_text = self._dialogue_text(state)
        milestone = str(walkthrough_context.get("milestone") or "the current milestone")
        return {
            "dialogue_prompt": f"NPC says: {dialogue_text!r}. Choose Yes or No.",
            "dialogue_text": dialogue_text,
            "choice_mode": "YES_NO" if self._looks_like_yes_no_prompt(dialogue_text) else "ADVANCE_OR_CHOOSE",
            "milestone_focus": f"Prefer the dialogue branch that best advances {milestone}.",
        }

    def _build_battle_context(self, state: StructuredGameState, walkthrough_context: dict[str, Any]) -> dict[str, Any]:
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
        milestone_target = str(walkthrough_context.get("target_map") or "the next map")
        lead_payload: dict[str, Any] = {
            "name": self._coalesce(battle_state.get("player_active_species"), lead.name if lead else None, "UNKNOWN"),
            "level": self._coalesce(battle_state.get("player_active_level"), lead.level if lead else None),
            "hp": self._coalesce(battle_state.get("player_active_hp"), lead.hp if lead else None),
            "max_hp": self._coalesce(battle_state.get("player_active_max_hp"), lead.max_hp if lead else None),
            "status": lead.status if lead else None,
        }
        return {
            "battle_prompt": (
                f"Fighting {enemy_name}"
                f"{f' lv {enemy_level}' if enemy_level is not None else ''}. "
                f"Your {lead_payload['name']} has {self._hp_text(lead_payload['hp'], lead_payload['max_hp'])}. "
                f"Moves: {', '.join(moves[:4]) if moves else 'unavailable'}. Choose wisely."
            ),
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
            "milestone_pressure": f"Winning or surviving this battle resumes progress toward {milestone_target}.",
        }

    def _build_ascii_map(self, state: StructuredGameState) -> str | None:
        if state.x is not None and state.y is not None and state.game_area is not None and state.collision_area is not None:
            return render_ascii_map(state.game_area, state.collision_area, state.x, state.y)

        navigation = state.navigation
        if navigation is None or state.x is None or state.y is None:
            return None
        walkable = {(coord.x, coord.y) for coord in navigation.walkable}
        blocked = {(coord.x, coord.y) for coord in navigation.blocked}
        lines: list[str] = []
        for y in range(navigation.min_y, navigation.max_y + 1):
            chars: list[str] = []
            for x in range(navigation.min_x, navigation.max_x + 1):
                if (x, y) == (state.x, state.y):
                    chars.append("P")
                elif (x, y) in blocked:
                    chars.append("#")
                elif (x, y) in walkable:
                    chars.append(".")
                else:
                    chars.append(" ")
            lines.append("".join(chars).rstrip())
        return "\n".join(line for line in lines if line)

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

    def _build_objectives(self, memory_state: MemoryState) -> list[dict[str, Any]]:
        objectives = memory_state.goals.active_objectives
        if not objectives:
            objectives = self._fallback_objectives(memory_state)
        return [self._serialize_objective(item) for item in objectives[:3]]

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

    def _fallback_objectives(self, memory_state: MemoryState) -> list[Objective]:
        goals = memory_state.goals
        return [
            Objective(
                id="fallback_long",
                horizon="long",
                summary=goals.long_term_goal,
                priority=30,
                success_conditions=goals.success_conditions[-2:] or ["Reach a new map or story interaction"],
            ),
            Objective(
                id="fallback_mid",
                horizon="mid",
                summary=goals.mid_term_goal,
                priority=20,
                success_conditions=["Make meaningful local progress"],
            ),
            Objective(
                id="fallback_short",
                horizon="short",
                summary=goals.short_term_goal,
                priority=10,
                success_conditions=["Cause a visible state change"],
            ),
        ]

    def _serialize_objective(self, objective: Objective) -> dict[str, Any]:
        payload = objective.model_dump(mode="json", exclude_none=True)
        payload["success_conditions"] = payload.get("success_conditions", [])[:2]
        payload["invalidation_conditions"] = payload.get("invalidation_conditions", [])[:2]
        return payload

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
            "instruction": "Avoid repeating recent no-effect actions unless the state changed.",
        }

    def _prune_to_budget(self, payload: dict[str, Any], dropped_sections: list[str], system_prompt: str) -> None:
        context = payload["context"]
        if self._within_budget(payload, system_prompt):
            return

        if "recommended_next_step" in context:
            context["recommended_next_step"].pop("why", None)
            dropped_sections.append("recommended_step_reduced")
        if self._within_budget(payload, system_prompt):
            return

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

        overworld_context = context.get("overworld_context")
        if isinstance(overworld_context, dict) and "visual_map" in overworld_context:
            overworld_context.pop("visual_map", None)
            overworld_context.pop("map_legend", None)
            dropped_sections.append("overworld_ascii_map")
        if self._within_budget(payload, system_prompt):
            return

        battle_context = context.get("battle_context")
        if isinstance(battle_context, dict) and "party_preview" in battle_context:
            del battle_context["party_preview"]
            dropped_sections.append("battle_party_preview")
        if self._within_budget(payload, system_prompt):
            return

        walkthrough_context = context.get("walkthrough_context")
        if isinstance(walkthrough_context, dict):
            walkthrough_context["route_hints"] = list(walkthrough_context.get("route_hints", []))[:1]
            dropped_sections.append("walkthrough_context_reduced")
        if self._within_budget(payload, system_prompt):
            return

        recent_events = context.get("recent_events")
        if isinstance(recent_events, list) and len(recent_events) > 2:
            context["recent_events"] = recent_events[-2:]
            dropped_sections.append("recent_events_reduced")
        if self._within_budget(payload, system_prompt):
            return

        if "recent_events" in context:
            del context["recent_events"]
            dropped_sections.append("recent_events")
        if self._within_budget(payload, system_prompt):
            return

        last_outcome = context.get("last_outcome")
        if isinstance(last_outcome, dict) and "event_summaries" in last_outcome:
            last_outcome["event_summaries"] = last_outcome.get("event_summaries", [])[:1]
            dropped_sections.append("last_outcome_reduced")
        if self._within_budget(payload, system_prompt):
            return

        if "stuck_warning" in context:
            context["stuck_warning"].pop("loop_signature", None)
            context["stuck_warning"].pop("instruction", None)
            dropped_sections.append("stuck_warning_reduced")
        if self._within_budget(payload, system_prompt):
            return

        candidates = context.get("candidate_next_steps")
        if isinstance(candidates, list) and len(candidates) > 3:
            context["candidate_next_steps"] = candidates[:3]
            dropped_sections.append("candidate_next_steps_reduced")
        if self._within_budget(payload, system_prompt):
            return

        objectives = context.get("active_objective_stack")
        if isinstance(objectives, list):
            for objective in objectives:
                objective["success_conditions"] = objective.get("success_conditions", [])[:1]
                objective["invalidation_conditions"] = objective.get("invalidation_conditions", [])[:1]
            dropped_sections.append("objective_stack_reduced")
        if self._within_budget(payload, system_prompt):
            return

        immediate_state = context.get("immediate_state", {})
        if "navigation_window" in immediate_state:
            del immediate_state["navigation_window"]
            dropped_sections.append("navigation_window")
        if self._within_budget(payload, system_prompt):
            return

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
