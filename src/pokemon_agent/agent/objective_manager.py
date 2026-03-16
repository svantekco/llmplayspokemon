from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Callable

from pokemon_agent.agent.context_manager import ContextManager, build_messages, measure_prompt
from pokemon_agent.agent.controllers.protocol import NavigationTarget
from pokemon_agent.agent.llm_client import CompletionResponse, LLMUsage, OpenRouterClient
from pokemon_agent.data.walkthrough import Milestone, get_current_milestone
from pokemon_agent.models.memory import MemoryState
from pokemon_agent.models.planner import StrategicObjective
from pokemon_agent.models.state import StructuredGameState
from pokemon_agent.navigation.world_graph import Landmark, WorldGraph, map_matches


@dataclass(slots=True)
class ObjectivePlannerMetadata:
    messages: list[dict[str, str]] = field(default_factory=list)
    prompt_metrics: dict[str, Any] | None = None
    llm_usage: LLMUsage | None = None
    llm_attempted: bool = False
    llm_model: str | None = None


class ObjectiveManager:
    def __init__(
        self,
        llm_client: OpenRouterClient | None,
        context_manager: ContextManager,
        *,
        world_graph: WorldGraph,
        completion_fn: Callable[..., CompletionResponse] | None = None,
    ) -> None:
        self._llm = llm_client
        self._context_manager = context_manager
        self._world_graph = world_graph
        self._complete = completion_fn
        self._current: StrategicObjective | None = None
        self._last_milestone_id: str | None = None
        self._last_replan_turn: int = 0
        self._last_metadata = ObjectivePlannerMetadata()

    def restore_state(self, objective: StrategicObjective | None) -> None:
        self._current = objective.model_copy(deep=True) if objective is not None else None
        self._last_milestone_id = None if objective is None else objective.milestone_id
        self._last_replan_turn = 0 if objective is None else max(0, objective.generated_at_step)
        self.reset_turn_metadata()

    def reset_turn_metadata(self) -> None:
        self._last_metadata = ObjectivePlannerMetadata()

    def last_metadata(self) -> ObjectivePlannerMetadata:
        return ObjectivePlannerMetadata(
            messages=list(self._last_metadata.messages),
            prompt_metrics=None if self._last_metadata.prompt_metrics is None else dict(self._last_metadata.prompt_metrics),
            llm_usage=self._last_metadata.llm_usage,
            llm_attempted=self._last_metadata.llm_attempted,
            llm_model=self._last_metadata.llm_model,
        )

    def current_objective(self) -> StrategicObjective | None:
        return None if self._current is None else self._current.model_copy(deep=True)

    def current_milestone(self, state: StructuredGameState) -> Milestone:
        return get_current_milestone(
            state.story_flags,
            [item.name for item in state.inventory],
            current_map_name=state.map_name,
            badges=state.badges,
        )

    def replan_reason(self, state: StructuredGameState, *, stuck_score: int, turn_index: int) -> str | None:
        milestone = self.current_milestone(state)
        objective = self._current
        if objective is None:
            return "plan_completed"
        if objective.milestone_id and objective.milestone_id != milestone.id:
            return "milestone_changed"
        if self._is_map_to_unknown(state, objective):
            return "map_to_unknown"
        if self._is_plan_completed(state, milestone, objective):
            return "plan_completed"
        if stuck_score >= 8 and turn_index - self._last_replan_turn >= 10:
            return "stuck_escalation"
        return None

    def should_replan(self, state: StructuredGameState, *, stuck_score: int, turn_index: int) -> bool:
        return self.replan_reason(state, stuck_score=stuck_score, turn_index=turn_index) is not None

    def replan(
        self,
        state: StructuredGameState,
        memory_state: MemoryState,
        *,
        stuck_score: int,
        turn_index: int,
    ) -> StrategicObjective:
        milestone = self.current_milestone(state)
        reason = self.replan_reason(state, stuck_score=stuck_score, turn_index=turn_index) or "plan_completed"
        if self._llm is None or self._complete is None:
            objective = self._fallback_objective(state, milestone)
            self._set_current(
                objective,
                milestone_id=milestone.id,
                turn_index=turn_index,
                generated_at_step=state.step,
                generated_at_map=state.map_name,
            )
            return objective

        try:
            snapshot = self._context_manager.build_objective_snapshot(
                state,
                memory_state,
                replan_reason=reason,
            )
            messages = build_messages(snapshot)
            self._last_metadata.messages = list(messages)
            self._last_metadata.prompt_metrics = measure_prompt(messages, snapshot)
            self._last_metadata.llm_attempted = True
            response = self._complete(messages, purpose="objective planner")
            self._last_metadata.llm_usage = response.usage
            self._last_metadata.llm_model = response.model
            objective = self._parse_objective(response.content)
            objective = self._normalize_objective(objective, state, milestone)
        except Exception:
            objective = self._fallback_objective(state, milestone)
        self._set_current(
            objective,
            milestone_id=milestone.id,
            turn_index=turn_index,
            generated_at_step=state.step,
            generated_at_map=state.map_name,
        )
        return objective

    def mark_completed(self) -> None:
        self._current = None

    def navigation_target(self) -> NavigationTarget | None:
        objective = self._current
        if objective is None:
            return None
        target_map = objective.target_map
        landmark_id = objective.target_landmark
        if target_map is None and landmark_id is not None:
            landmark = self._world_graph.get_landmark(landmark_id)
            if landmark is not None:
                target_map = landmark.map_name
        if target_map is None and landmark_id is None:
            return None
        return NavigationTarget(
            map_name=target_map,
            landmark_id=landmark_id,
            reason=objective.goal,
        )

    def _set_current(
        self,
        objective: StrategicObjective,
        *,
        milestone_id: str,
        turn_index: int,
        generated_at_step: int,
        generated_at_map: str | None = None,
    ) -> None:
        objective.milestone_id = milestone_id
        objective.generated_at_step = generated_at_step
        objective.generated_at_map = generated_at_map
        self._current = objective
        self._last_milestone_id = milestone_id
        self._last_replan_turn = turn_index

    def _parse_objective(self, raw_text: str) -> StrategicObjective:
        payload = json.loads(self._extract_json(raw_text))
        try:
            return StrategicObjective.model_validate(payload)
        except Exception:
            legacy = StrategicObjective.from_legacy_payload(payload)
            if legacy is None:
                raise
            return legacy

    def _normalize_objective(
        self,
        objective: StrategicObjective,
        state: StructuredGameState,
        milestone: Milestone,
    ) -> StrategicObjective:
        normalized = objective.model_copy(deep=True)
        normalized.goal = normalized.goal.strip() or milestone.description
        normalized.strategy = normalized.strategy.strip() or self._fallback_strategy(state, milestone)
        if normalized.target_landmark:
            landmark = self._world_graph.get_landmark(normalized.target_landmark)
            if landmark is None:
                normalized.target_landmark = None
                normalized.target_landmark_type = None
            elif normalized.target_map is None:
                normalized.target_map = landmark.map_name
                normalized.target_landmark_type = landmark.type
            elif normalized.target_landmark_type is None:
                normalized.target_landmark_type = landmark.type
        if normalized.target_map is None:
            normalized.target_map = milestone.target_map_name
        if normalized.confidence <= 0:
            normalized.confidence = 0.8
        return normalized

    def _fallback_objective(self, state: StructuredGameState, milestone: Milestone) -> StrategicObjective:
        landmark = self._story_landmark_for_milestone(milestone)
        goal = (
            milestone.description
            if map_matches(state.map_name, milestone.target_map_name)
            else f"Move toward {milestone.target_map_name}"
        )
        return StrategicObjective(
            goal=goal,
            target_map=milestone.target_map_name,
            target_landmark=None if landmark is None else landmark.id,
            target_landmark_type=None if landmark is None else landmark.type,
            strategy=self._fallback_strategy(state, milestone),
            milestone_id=milestone.id,
            confidence=0.5,
            generated_at_step=state.step,
        )

    def _fallback_strategy(self, state: StructuredGameState, milestone: Milestone) -> str:
        if milestone.route_hints:
            return milestone.route_hints[0]
        if state.map_name and not map_matches(state.map_name, milestone.target_map_name):
            return f"Travel toward {milestone.target_map_name} and advance the current story milestone."
        return f"Use deterministic play to advance {milestone.description}"

    def _is_map_to_unknown(self, state: StructuredGameState, objective: StrategicObjective) -> bool:
        target_map = self._objective_target_map(objective)
        if not state.map_name or self._world_graph.get_map_by_name(state.map_name) is None:
            return True
        if target_map is None or map_matches(state.map_name, target_map):
            return False
        return self._world_graph.find_route(state.map_name, target_map) is None

    def _is_plan_completed(
        self,
        state: StructuredGameState,
        milestone: Milestone,
        objective: StrategicObjective,
    ) -> bool:
        landmark = self._resolve_landmark(objective)
        if landmark is not None:
            if not map_matches(state.map_name, landmark.map_name):
                return False
            if landmark.x is None or landmark.y is None or state.x is None or state.y is None:
                return not map_matches(landmark.map_name, milestone.target_map_name)
            return abs(state.x - landmark.x) + abs(state.y - landmark.y) <= 1

        target_map = self._objective_target_map(objective)
        if target_map is None or not map_matches(state.map_name, target_map):
            return False
        return not map_matches(target_map, milestone.target_map_name)

    def _objective_target_map(self, objective: StrategicObjective) -> str | None:
        if objective.target_map:
            return objective.target_map
        landmark = self._resolve_landmark(objective)
        return None if landmark is None else landmark.map_name

    def _resolve_landmark(self, objective: StrategicObjective) -> Landmark | None:
        if not objective.target_landmark:
            return None
        return self._world_graph.get_landmark(objective.target_landmark)

    def _story_landmark_for_milestone(self, milestone: Milestone) -> Landmark | None:
        target_map = self._world_graph.get_map_by_name(milestone.target_map_name)
        if target_map is None:
            return None
        story_text = " ".join([milestone.description, *milestone.route_hints, *milestone.sub_steps]).lower()
        ranked: list[tuple[int, str, Landmark]] = []
        for landmark in self._world_graph.get_landmarks_on_map(target_map.symbol):
            if landmark.type in {"route_exit", "sign"}:
                continue
            score = self._landmark_story_score(landmark, story_text)
            if score <= 0:
                continue
            ranked.append((score, landmark.id, landmark))
        if not ranked:
            return None
        ranked.sort(key=lambda item: (-item[0], item[1]))
        return ranked[0][2]

    def _landmark_story_score(self, landmark: Landmark, story_text: str) -> int:
        label = landmark.label.lower()
        score = 12 if label in story_text else 0
        landmark_type_scores = {
            "gym": ("gym", "badge", "brock", "misty", "surge", "erika", "koga", "blaine", "sabrina", "giovanni"),
            "mart": ("mart", "buy", "restock", "stock up", "parcel", "clerk"),
            "pokecenter": ("pokecenter", "poke center", "heal", "restore"),
            "cave_entrance": ("cave", "tunnel", "entrance", "moon"),
            "dungeon_entrance": ("forest", "tower", "hideout", "mansion", "safari", "road", "anne", "entrance"),
            "important_building": ("lab", "house", "dock", "museum", "fan club"),
        }
        for keyword in landmark_type_scores.get(landmark.type, ()):
            if keyword in story_text:
                score += 4
        return score

    @staticmethod
    def _extract_json(raw_text: str) -> str:
        text = raw_text.strip()
        if text.startswith("{") and text.endswith("}"):
            return text
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end < start:
            raise ValueError("Model response did not contain a JSON object")
        return text[start : end + 1]

    @staticmethod
    def _normalize_text(value: str | None) -> str:
        if not value:
            return ""
        return " ".join(re.findall(r"[a-z0-9]+", value.lower()))
