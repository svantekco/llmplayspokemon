from __future__ import annotations

import copy
import json
from dataclasses import asdict, dataclass, field
from typing import Any

from pokemon_agent.agent.progress import ProgressResult
from pokemon_agent.agent.stuck_detector import StuckState
from pokemon_agent.agent.world_map import summarize_navigation_goal
from pokemon_agent.data.walkthrough import get_current_milestone
from pokemon_agent.emulator.screen_renderer import build_ascii_map
from pokemon_agent.models.action import ActionDecision
from pokemon_agent.models.events import EventRecord
from pokemon_agent.models.memory import MemoryState
from pokemon_agent.models.planner import CandidateNextStep
from pokemon_agent.models.planner import CandidateRuntime
from pokemon_agent.models.planner import StrategicObjective
from pokemon_agent.models.state import GameMode
from pokemon_agent.models.state import StructuredGameState
from pokemon_agent.navigation.world_graph import load_world_graph

RESPONSE_SCHEMA = {
    "candidate_id": "<exact id from candidate_next_steps>",
}
OBJECTIVE_RESPONSE_SCHEMA = {
    "goal": "<string>",
    "target_map": "<string|null>",
    "target_landmark": "<string|null>",
    "strategy": "<string>",
    "milestone_id": "<string|null>",
    "confidence": "<float|null>",
}
PLANNER_SYSTEM_PROMPT = (
    "You control a Pokémon Red agent. Select exactly one id from candidate_next_steps. "
    "Obey mode and local_objective. Prefer the highest-priority candidate that advances the target. "
    "Never choose a blacklisted candidate. Use the dialogue only to resolve ambiguous yes/no choices. "
    'Return exactly {"candidate_id":"<id>"}. No markdown. No extra keys.'
)
OBJECTIVE_PLANNER_SYSTEM_PROMPT = (
    "You are the objective planner for a Pokémon Red agent. "
    "Return one symbolic strategic objective. "
    "Use only high-level goal text, a target map, and optionally a known landmark id. "
    "Do not return buttons, tile routes, connector ids, or exact movement sequences. "
    "Prefer plans the engine can execute deterministically from the world graph and current milestone. "
    "Return exactly one JSON object matching response_schema. No markdown."
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


def build_messages(snapshot: ContextSnapshot) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": snapshot.system_prompt},
        {"role": "user", "content": json.dumps(snapshot.payload, separators=(",", ":"), sort_keys=True)},
    ]


def measure_prompt(messages: list[dict[str, str]], snapshot: ContextSnapshot | None = None) -> dict[str, Any]:
    chars = sum(len(message.get("content", "")) for message in messages)
    approx_tokens = max(1, chars // 4)
    if snapshot is None:
        compact = chars <= 2400
        warning = None if compact else "Prompt is growing large; consider pruning memory or metadata."
        return {"chars": chars, "approx_tokens": approx_tokens, "compact": compact, "warning": warning}

    compact = snapshot.used_tokens <= snapshot.budget_tokens
    warning = None if compact else "Prompt remains above budget after pruning mandatory context."
    return {
        "chars": chars,
        "approx_tokens": approx_tokens,
        "compact": compact,
        "budget_tokens": snapshot.budget_tokens,
        "used_tokens": snapshot.used_tokens,
        "section_tokens": dict(snapshot.section_tokens),
        "dropped_sections": list(snapshot.dropped_sections),
        "warning": warning,
    }


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
        self.static_world_graph = load_world_graph()

    def build_snapshot(
        self,
        state: StructuredGameState,
        memory_state: MemoryState,
        stuck_state: StuckState | None = None,
        candidate_next_steps: list[CandidateNextStep] | None = None,
        candidate_runtime: dict[str, CandidateRuntime] | None = None,
    ) -> ContextSnapshot:
        context = self._build_context(state, memory_state, stuck_state, candidate_next_steps, candidate_runtime)
        return self._finalize_snapshot(context, PLANNER_SYSTEM_PROMPT, RESPONSE_SCHEMA)

    def build_objective_snapshot(
        self,
        state: StructuredGameState,
        memory_state: MemoryState,
        stuck_state: StuckState | None = None,
        replan_reason: str | None = None,
    ) -> ContextSnapshot:
        context = self._build_context(state, memory_state, stuck_state, candidate_next_steps=None, candidate_runtime=None)
        objective_context = dict(context)
        objective_context["planner_kind"] = "objective"
        if replan_reason:
            objective_context["replan_reason"] = replan_reason
        objective = self._serialize_objective(memory_state.long_term.objective)
        if objective is not None:
            objective_context["current_objective"] = objective
        return self._finalize_snapshot(objective_context, OBJECTIVE_PLANNER_SYSTEM_PROMPT, OBJECTIVE_RESPONSE_SCHEMA)

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

    def _finalize_snapshot(
        self,
        context: dict[str, Any],
        system_prompt: str,
        response_schema: dict[str, Any],
    ) -> ContextSnapshot:
        payload = {
            "context": context,
            "response_schema": copy.deepcopy(response_schema),
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

    def _build_context(
        self,
        state: StructuredGameState,
        memory_state: MemoryState,
        stuck_state: StuckState | None,
        candidate_next_steps: list[CandidateNextStep] | None,
        candidate_runtime: dict[str, CandidateRuntime] | None,
    ) -> dict[str, Any]:
        current_milestone = self._build_current_milestone(state)
        local_objective = self._build_local_objective(memory_state)
        context: dict[str, Any] = {
            "mode": self._planner_mode(state, memory_state),
            "current_map": self._build_current_map(state),
            "current_milestone": current_milestone,
        }
        if local_objective is not None:
            context["local_objective"] = local_objective
            next_map_hop = self._build_next_map_hop(local_objective)
            if next_map_hop is not None:
                context["next_map_hop"] = next_map_hop
        context.update(self._build_mode_context(state, memory_state))
        if candidate_next_steps:
            context["candidate_next_steps"] = [self._serialize_candidate(item) for item in candidate_next_steps[:4]]
        stuck_warning = self._build_stuck_warning(stuck_state)
        if stuck_warning is not None:
            if candidate_next_steps and stuck_state is not None:
                failed_actions = set(stuck_state.recent_failed_actions[-3:])
                goal = memory_state.long_term.navigation_goal
                failed_ids = list(goal.failed_candidate_ids) if goal is not None else []
                failed_ids = [
                    *failed_ids,
                    *[
                        cand.id for cand in candidate_next_steps
                    if candidate_runtime is not None
                    and cand.id in candidate_runtime
                    and (action := candidate_runtime[cand.id].action) is not None
                    and action.action.value in failed_actions
                    ],
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
            context["last_candidate_result"] = last_outcome
        recent_events = self._build_recent_events(memory_state)
        if recent_events:
            context["recent_events"] = recent_events
        return context

    def _build_current_map(self, state: StructuredGameState) -> dict[str, Any]:
        dialogue_text = self._get_dialogue(state)
        payload: dict[str, Any] = {
            "name": state.map_name,
            "id": state.map_id,
            "position": {"x": state.x, "y": state.y},
            "facing": state.facing,
            "step": state.step,
            "ui_flags": {
                "menu_open": state.menu_open,
                "text_box_open": state.text_box_open,
            },
        }
        if state.text_box_open:
            payload["ui_flags"]["yes_no_prompt"] = bool(state.metadata.get("yes_no_prompt"))
        if dialogue_text:
            payload["dialogue_text"] = dialogue_text
        return payload

    def _build_current_milestone(self, state: StructuredGameState) -> dict[str, Any]:
        battle_payload = self._battle_payload(state)
        inventory_names = [item.name for item in state.inventory]
        current_milestone = get_current_milestone(
            state.story_flags,
            inventory_names,
            current_map_name=state.map_name,
            badges=state.badges,
        )
        return {
            "id": current_milestone.id,
            "description": current_milestone.description,
            "target_map": current_milestone.target_map_name,
            "next_hint": current_milestone.route_hints[0] if current_milestone.route_hints else None,
            "battle_kind": battle_payload.get("kind"),
            "badges": list(state.badges),
        }

    def _build_local_objective(self, memory_state: MemoryState) -> dict[str, Any] | None:
        goal = memory_state.long_term.navigation_goal
        if goal is None:
            return None
        return {
            "kind": goal.objective_kind,
            "target_map": goal.target_map_name,
            "final_target_map": goal.final_target_map_name or goal.target_map_name,
            "target_landmark_id": goal.target_landmark_id,
            "target_landmark_type": goal.target_landmark_type,
            "current_map": goal.current_map_name,
            "engine_mode": goal.engine_mode,
            "next_map": goal.next_map_name,
            "next_hop_kind": goal.next_hop_kind,
            "next_hop_side": goal.next_hop_side,
            "target_connector_id": goal.target_connector_id,
            "failed_candidate_ids": goal.failed_candidate_ids[-4:],
        }

    def _build_next_map_hop(self, local_objective: dict[str, Any]) -> dict[str, Any] | None:
        if not local_objective.get("next_map") and not local_objective.get("next_hop_kind"):
            return None
        return {
            "next_map": local_objective.get("next_map"),
            "kind": local_objective.get("next_hop_kind"),
            "side": local_objective.get("next_hop_side"),
        }

    def _planner_mode(self, state: StructuredGameState, memory_state: MemoryState) -> str:
        if state.battle_state or state.mode == GameMode.BATTLE:
            return "battle"
        if state.menu_open or state.mode == GameMode.MENU:
            return "menu"
        if state.text_box_open or state.mode == GameMode.TEXT:
            return "text"
        goal = memory_state.long_term.navigation_goal
        if goal is None:
            return "exploration"
        return goal.engine_mode

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
        canonical_navigation = self._build_canonical_navigation(state, memory_state)
        context: dict[str, Any] = {
            "visual_map": visual_map,
            "map_legend": "P=player .=walkable #=blocked ~=water @=isolated blocker D=door N=NPC",
            "canonical_navigation": canonical_navigation,
            **route_info,
        }
        if state.npcs:
            context["npcs"] = [
                {"x": npc.tile_x, "y": npc.tile_y, "sprite": npc.sprite_index}
                for npc in state.npcs
            ]
        return context

    def _build_canonical_navigation(self, state: StructuredGameState, memory_state: MemoryState) -> dict[str, Any]:
        goal = memory_state.long_term.navigation_goal
        current_map = self.static_world_graph.get_map_by_id(state.map_id)
        if current_map is None:
            current_map = self.static_world_graph.get_map_by_name(state.map_name)
        current_map_symbol = None if current_map is None else current_map.symbol
        target_map_symbol = None
        final_target_map_symbol = None
        target_landmark_payload = None
        route_summary = None
        neighbors: list[str] = []
        nearest_pokecenter = None

        if current_map is not None:
            neighbor_symbols = [
                edge.destination_symbol
                for edge in self.static_world_graph.neighbors(current_map.symbol)
                if edge.destination_symbol is not None
            ]
            deduped_neighbors = list(dict.fromkeys(neighbor_symbols))
            neighbors = deduped_neighbors[:6]
            pokecenter = self.static_world_graph.nearest_landmark(current_map.symbol, "pokecenter")
            if pokecenter is not None:
                nearest_pokecenter = {
                    "landmark_id": pokecenter.landmark.id,
                    "map": pokecenter.landmark.map_symbol,
                }

        if goal is not None:
            target_map_symbol = self.static_world_graph.canonical_symbol(goal.target_map_name) or goal.target_map_name
            final_target_map_name = goal.final_target_map_name or goal.target_map_name
            final_target_map_symbol = self.static_world_graph.canonical_symbol(final_target_map_name) or final_target_map_name
            target_landmark = self.static_world_graph.get_landmark(goal.target_landmark_id)
            if target_landmark is not None:
                target_landmark_payload = {
                    "id": target_landmark.id,
                    "map": target_landmark.map_symbol,
                    "type": target_landmark.type,
                }
            if current_map is not None:
                route = self.static_world_graph.find_route(current_map.symbol, goal.target_map_name)
                route_summary = None if route is None else route.summary()[:8]

        return {
            "current_map": current_map_symbol or state.map_name,
            "current_map_neighbors": neighbors,
            "nearest_pokecenter": nearest_pokecenter,
            "route_summary": route_summary,
            "target_landmark": target_landmark_payload,
            "target_map": target_map_symbol,
            "final_target_map": final_target_map_symbol,
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
        enemy_level = next(
            (value for value in (battle_state.get("opponent_level"), battle_state.get("enemy_level")) if value is not None),
            None,
        )
        lead = state.party[0] if state.party else None
        moves = self._battle_moves(state)
        lead_payload: dict[str, Any] = {
            "name": next(
                (
                    value
                    for value in (battle_state.get("player_active_species"), lead.name if lead else None, "UNKNOWN")
                    if value is not None
                ),
                None,
            ),
            "level": next(
                (value for value in (battle_state.get("player_active_level"), lead.level if lead else None) if value is not None),
                None,
            ),
            "hp": next(
                (value for value in (battle_state.get("player_active_hp"), lead.hp if lead else None) if value is not None),
                None,
            ),
            "max_hp": next(
                (value for value in (battle_state.get("player_active_max_hp"), lead.max_hp if lead else None) if value is not None),
                None,
            ),
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
    def _get_dialogue(state: StructuredGameState) -> str | None:
        text = state.metadata.get("dialogue")
        if not isinstance(text, str) or not text.strip():
            text = state.metadata.get("dialogue_text")
        if isinstance(text, str) and text.strip():
            return text.strip()
        return None

    @staticmethod
    def _dialogue_text(state: StructuredGameState) -> str:
        text = ContextManager._get_dialogue(state)
        if text is not None:
            return text
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
    def _is_overworld_mode(state: StructuredGameState) -> bool:
        if state.battle_state or state.text_box_open or state.menu_open:
            return False
        return state.mode == GameMode.OVERWORLD or (state.mode == GameMode.UNKNOWN and state.navigation is not None)

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

    @staticmethod
    def _serialize_objective(plan: StrategicObjective | None) -> dict[str, Any] | None:
        if plan is None:
            return None
        return plan.model_dump(mode="json", exclude_none=True)

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

        # 4. Reduce last candidate result
        last_outcome = context.get("last_candidate_result")
        if isinstance(last_outcome, dict) and "event_summaries" in last_outcome:
            last_outcome["event_summaries"] = last_outcome.get("event_summaries", [])[:1]
            dropped_sections.append("last_candidate_result_reduced")
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
