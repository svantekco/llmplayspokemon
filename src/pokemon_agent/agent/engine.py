from __future__ import annotations

import concurrent.futures
import copy
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pokemon_agent.agent.context_manager import ContextManager
from pokemon_agent.agent.battle_manager import BattleManager
from pokemon_agent.agent.executor import Executor
from pokemon_agent.agent.llm_client import CompletionResponse, LLMUsage, OpenRouterClient
from pokemon_agent.agent.menu_manager import MenuManager
from pokemon_agent.agent.memory_manager import MemoryManager
from pokemon_agent.agent.navigation import CachedRoute
from pokemon_agent.agent.navigation import advance_position
from pokemon_agent.agent.navigation import find_path
from pokemon_agent.agent.progress import ProgressDetector, ProgressResult
from pokemon_agent.agent.prompt_builder import PromptBuilder, PromptMetrics
from pokemon_agent.agent.stuck_detector import StuckDetector, StuckState
from pokemon_agent.agent.validator import ActionValidator
from pokemon_agent.agent.world_map import connectors_from_map
from pokemon_agent.agent.world_map import describe_connector
from pokemon_agent.agent.world_map import shortest_confirmed_path
from pokemon_agent.agent.world_map import world_map_stats
from pokemon_agent.models.action import ActionDecision
from pokemon_agent.models.action import ActionType
from pokemon_agent.models.events import EventRecord
from pokemon_agent.models.memory import MemoryState
from pokemon_agent.models.memory import NavigationGoal
from pokemon_agent.models.planner import CandidateNextStep
from pokemon_agent.models.planner import ExecutionPlan
from pokemon_agent.models.planner import ObjectiveHorizon
from pokemon_agent.models.planner import ObjectiveTarget
from pokemon_agent.models.planner import PlannerDecision
from pokemon_agent.models.state import GameMode
from pokemon_agent.models.state import NavigationSnapshot
from pokemon_agent.models.state import StructuredGameState

DIRECTION_DELTAS: tuple[tuple[int, int], ...] = ((0, -1), (1, 0), (0, 1), (-1, 0))
NAVIGATION_CANDIDATE_TYPES = {
    "GO_TO_MAP_EXIT",
    "EXPLORE_NEAREST_FRONTIER",
    "MOVE_ADJACENT_TO_INTERACTABLE",
    "FOLLOW_DISCOVERED_CONNECTOR",
}
YES_NO_CANDIDATE_TYPES = {"SELECT_YES", "SELECT_NO"}


@dataclass(slots=True)
class TurnResult:
    turn_index: int
    before: StructuredGameState
    action: ActionDecision
    after: StructuredGameState
    progress: ProgressResult
    stuck_state: StuckState
    events: list[EventRecord] = field(default_factory=list)
    used_fallback: bool = False
    raw_model_response: str | None = None
    prompt_messages: list[dict] = field(default_factory=list)
    prompt_metrics: PromptMetrics | None = None
    llm_usage: LLMUsage | None = None
    llm_attempted: bool = False
    llm_model: str | None = None
    planner_source: str = "fallback"
    objective_id: str | None = None
    candidate_id: str | None = None


@dataclass(slots=True)
class PlanningResult:
    action: ActionDecision
    raw_response: str | None = None
    used_fallback: bool = False
    planner_source: str = "fallback"
    messages: list[dict] = field(default_factory=list)
    prompt_metrics: PromptMetrics | None = None
    llm_usage: LLMUsage | None = None
    llm_attempted: bool = False
    llm_model: str | None = None
    objective_id: str | None = None
    candidate_id: str | None = None


@dataclass(slots=True)
class RunTelemetry:
    prompt_chars: int = 0
    approx_prompt_tokens: int = 0
    llm_prompt_tokens: int = 0
    llm_completion_tokens: int = 0
    llm_total_tokens: int = 0
    turns: int = 0
    fallback_turns: int = 0
    route_cache_turns: int = 0
    llm_calls: int = 0
    auto_selected_turns: int = 0
    execution_plan_turns: int = 0
    major_progress_turns: int = 0
    no_effect_turns: int = 0
    objective_switches: int = 0
    _last_objective_id: str | None = field(default=None, repr=False)

    def register_turn(self, turn: TurnResult) -> None:
        self.turns += 1
        if turn.planner_source == "fallback":
            self.fallback_turns += 1
        if turn.planner_source == "auto_candidate":
            self.auto_selected_turns += 1
        if turn.planner_source == "execution_plan":
            self.execution_plan_turns += 1
        if turn.planner_source in {"route_cache", "execution_plan"} and turn.action.action in {
            ActionType.MOVE_UP,
            ActionType.MOVE_DOWN,
            ActionType.MOVE_LEFT,
            ActionType.MOVE_RIGHT,
        }:
            self.route_cache_turns += 1
        if turn.llm_attempted:
            self.llm_calls += 1
        if turn.progress.classification == "major_progress":
            self.major_progress_turns += 1
        if turn.progress.classification == "no_effect":
            self.no_effect_turns += 1
        if turn.prompt_metrics:
            self.prompt_chars += turn.prompt_metrics.chars
            self.approx_prompt_tokens += turn.prompt_metrics.approx_tokens
        if turn.llm_usage:
            self.llm_prompt_tokens += turn.llm_usage.prompt_tokens or 0
            self.llm_completion_tokens += turn.llm_usage.completion_tokens or 0
            self.llm_total_tokens += turn.llm_usage.total_tokens or 0
        if turn.objective_id is not None:
            if self._last_objective_id is not None and self._last_objective_id != turn.objective_id:
                self.objective_switches += 1
            self._last_objective_id = turn.objective_id

    def to_dict(self) -> dict[str, Any]:
        turns_per_call = None if self.llm_calls == 0 else round(self.turns / self.llm_calls, 2)
        calls_per_major_progress = None if self.major_progress_turns == 0 else round(self.llm_calls / self.major_progress_turns, 2)
        prompt_tokens_per_call = None if self.llm_calls == 0 else round(self.llm_prompt_tokens / self.llm_calls, 2)
        auto_selected_candidate_ratio = 0.0 if self.turns == 0 else round(self.auto_selected_turns / self.turns, 3)
        objective_switch_rate = 0.0 if self.turns == 0 else round(self.objective_switches / self.turns, 3)
        no_effect_rate = 0.0 if self.turns == 0 else round(self.no_effect_turns / self.turns, 3)
        return {
            "prompt_chars": self.prompt_chars,
            "approx_prompt_tokens": self.approx_prompt_tokens,
            "llm_prompt_tokens": self.llm_prompt_tokens,
            "llm_completion_tokens": self.llm_completion_tokens,
            "llm_total_tokens": self.llm_total_tokens,
            "turns": self.turns,
            "fallback_turns": self.fallback_turns,
            "route_cache_turns": self.route_cache_turns,
            "llm_calls": self.llm_calls,
            "auto_selected_turns": self.auto_selected_turns,
            "execution_plan_turns": self.execution_plan_turns,
            "major_progress_turns": self.major_progress_turns,
            "no_effect_turns": self.no_effect_turns,
            "turns_per_call": turns_per_call,
            "calls_per_major_progress": calls_per_major_progress,
            "prompt_tokens_per_call": prompt_tokens_per_call,
            "auto_selected_candidate_ratio": auto_selected_candidate_ratio,
            "objective_switches": self.objective_switches,
            "objective_switch_rate": objective_switch_rate,
            "no_effect_rate": no_effect_rate,
        }


class ClosedLoopRunner:
    def __init__(
        self,
        executor: Executor,
        memory: MemoryManager,
        progress: ProgressDetector,
        stuck: StuckDetector,
        prompts: PromptBuilder,
        validator: ActionValidator,
        llm_client: OpenRouterClient | None = None,
        context_manager: ContextManager | None = None,
    ) -> None:
        self.executor = executor
        self.memory = memory
        self.progress = progress
        self.stuck = stuck
        self.prompts = prompts
        self.validator = validator
        self.llm_client = llm_client
        self.context_manager = context_manager or ContextManager()
        self.battle_manager = BattleManager()
        self.telemetry = RunTelemetry()
        self.completed_turns = 0
        self.route_cache: CachedRoute | None = None
        self.execution_plan: ExecutionPlan | None = None
        self.menu_manager = MenuManager()
        self._last_entry_direction: str | None = None  # direction we entered current map from
        self._previous_map_name: str | None = None

    _OPPOSITE_DIRECTION: dict[str, str] = {
        "north": "south",
        "south": "north",
        "east": "west",
        "west": "east",
    }

    def run_turn(self, turn_index: int) -> TurnResult:
        planning_state = self.executor.emulator.get_structured_state()
        planning = self._plan_action(planning_state)
        before = self.executor.emulator.get_structured_state()
        action = self.validator.validate(planning.action, before)
        self.executor.run(action)
        after = self.executor.emulator.get_structured_state()
        progress_result = self.progress.compare(before, after)
        stuck_state = self.stuck.update(after, action, progress_result.classification)
        # Track entry direction on map change
        if before.map_id != after.map_id or before.map_name != after.map_name:
            self._previous_map_name = before.map_name
            entered_via = self._side_for_action(action.action)
            self._last_entry_direction = self._OPPOSITE_DIRECTION.get(entered_via) if entered_via else None
            goal = self.memory.memory.long_term.navigation_goal
            if goal is not None and goal.target_map_name != after.map_name:
                goal.confirmation_required_map = after.map_name
        self._refresh_route_cache_after_turn(after, progress_result)
        self._refresh_execution_plan_after_turn(after, progress_result)
        events = self.memory.update_from_transition(before, after, action, progress_result, stuck_state)
        self.context_manager.record_turn(
            turn_index=turn_index,
            action=action,
            after_state=after,
            progress=progress_result,
            events=events,
            stuck_state=stuck_state,
            used_fallback=planning.used_fallback,
            llm_attempted=planning.llm_attempted,
            planner_source=planning.planner_source,
        )
        turn_result = TurnResult(
            turn_index=turn_index,
            before=before,
            action=action,
            after=after,
            progress=progress_result,
            stuck_state=copy.deepcopy(stuck_state),
            events=events,
            used_fallback=planning.used_fallback,
            raw_model_response=planning.raw_response,
            prompt_messages=planning.messages,
            prompt_metrics=planning.prompt_metrics,
            llm_usage=planning.llm_usage,
            llm_attempted=planning.llm_attempted,
            llm_model=planning.llm_model,
            planner_source=planning.planner_source,
            objective_id=planning.objective_id,
            candidate_id=planning.candidate_id,
        )
        self.telemetry.register_turn(turn_result)
        self.completed_turns = max(self.completed_turns, turn_index)
        return turn_result

    def run(self, turns: int) -> list[TurnResult]:
        return [self.run_turn(turn_index) for turn_index in range(1, turns + 1)]

    def _plan_action(self, state: StructuredGameState) -> PlanningResult:
        if state.is_bootstrap():
            self.route_cache = None
            self.execution_plan = None
            action = self.validator.bootstrap(state, self.stuck.state, "deterministic startup bootstrap")
            return PlanningResult(action=self.validator.validate(action, state), planner_source="bootstrap")

        cached_action = self._plan_from_cached_route(state)
        if cached_action is not None:
            plan = self.execution_plan
            source = "execution_plan" if plan is not None else "route_cache"
            return PlanningResult(
                action=cached_action,
                planner_source=source,
                objective_id=plan.objective_id if plan else None,
                candidate_id=plan.candidate_id if plan else None,
            )

        plan_action = self._plan_from_execution_plan(state)
        if plan_action is not None:
            return PlanningResult(
                action=plan_action,
                planner_source="execution_plan",
                objective_id=self.execution_plan.objective_id if self.execution_plan else None,
                candidate_id=self.execution_plan.candidate_id if self.execution_plan else None,
            )

        candidates = self._build_candidate_steps(state)
        if not candidates:
            fallback = self.validator.fallback(state, self.stuck.state, "fallback without usable candidates")
            result = PlanningResult(action=self.validator.validate(fallback, state), used_fallback=True, planner_source="fallback")
            self._clear_navigation_confirmation(state.map_name)
            return result

        recommended = candidates[0]
        if self._should_auto_select(candidates):
            compiled = self._compile_candidate(recommended, state, self._short_reason(recommended))
            if compiled is not None:
                return PlanningResult(
                    action=compiled,
                    planner_source="auto_candidate",
                    objective_id=recommended.objective_id,
                    candidate_id=recommended.id,
                )

        snapshot = self.context_manager.build_snapshot(
            state,
            self.memory.memory,
            self.stuck.state,
            candidate_next_steps=candidates,
            recommended_step=recommended,
        )
        messages = self.prompts.build(snapshot)
        prompt_metrics = self.prompts.measure(messages, snapshot)

        if self.llm_client is None:
            result = self._fallback_from_candidates(
                state,
                recommended,
                messages=messages,
                prompt_metrics=prompt_metrics,
                reason="deterministic fallback planner",
            )
            self._clear_navigation_confirmation(state.map_name)
            return result

        try:
            raw_response = self._complete_with_window_pump(messages)
            decision = self._parse_planner_decision(raw_response.content)
            chosen = self._resolve_candidate(decision, candidates) or recommended
            live_state = self.executor.emulator.get_structured_state()
            compiled = self._compile_candidate(chosen, live_state, decision.reason or self._short_reason(chosen))
            if compiled is None:
                result = self._fallback_from_candidates(
                    live_state,
                    recommended,
                    messages=messages,
                    prompt_metrics=prompt_metrics,
                    raw_response=raw_response.content,
                    llm_usage=raw_response.usage,
                    llm_model=raw_response.model,
                    llm_attempted=True,
                    reason="fallback after invalid planner choice",
                )
                self._clear_navigation_confirmation(state.map_name)
                return result
            result = PlanningResult(
                action=compiled,
                raw_response=raw_response.content,
                planner_source="llm",
                messages=messages,
                prompt_metrics=prompt_metrics,
                llm_usage=raw_response.usage,
                llm_attempted=True,
                llm_model=raw_response.model,
                objective_id=chosen.objective_id,
                candidate_id=chosen.id,
            )
            self._clear_navigation_confirmation(state.map_name)
            return result
        except Exception as exc:
            result = self._fallback_from_candidates(
                self.executor.emulator.get_structured_state(),
                recommended,
                messages=messages,
                prompt_metrics=prompt_metrics,
                raw_response=str(exc),
                llm_attempted=True,
                reason=f"fallback after llm error: {type(exc).__name__}",
            )
            self._clear_navigation_confirmation(state.map_name)
            return result

    def _fallback_from_candidates(
        self,
        state: StructuredGameState,
        recommended: CandidateNextStep,
        *,
        messages: list[dict],
        prompt_metrics: PromptMetrics,
        raw_response: str | None = None,
        llm_usage: LLMUsage | None = None,
        llm_model: str | None = None,
        llm_attempted: bool = False,
        reason: str,
    ) -> PlanningResult:
        compiled = self._compile_candidate(recommended, state, self._short_reason(recommended))
        if compiled is None:
            fallback = self.validator.fallback(state, self.stuck.state, reason)
            compiled = self.validator.validate(fallback, state)
            objective_id = None
            candidate_id = None
        else:
            objective_id = recommended.objective_id
            candidate_id = recommended.id
        return PlanningResult(
            action=compiled,
            raw_response=raw_response,
            used_fallback=True,
            planner_source="fallback",
            messages=messages,
            prompt_metrics=prompt_metrics,
            llm_usage=llm_usage,
            llm_attempted=llm_attempted,
            llm_model=llm_model,
            objective_id=objective_id,
            candidate_id=candidate_id,
        )

    def _complete_with_window_pump(self, messages: list[dict]) -> CompletionResponse:
        emulator = self.executor.emulator
        emulator.begin_planning_wait()
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(self.llm_client.complete, messages)
                while not future.done():
                    emulator.pump_planning_wait()
                return future.result()
        finally:
            emulator.end_planning_wait()

    def save_checkpoint(self, directory: str | Path) -> Path:
        checkpoint_dir = Path(directory)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        emulator_state_path = checkpoint_dir / "emulator.state"
        metadata_path = checkpoint_dir / "checkpoint.json"
        self.executor.emulator.save_state(emulator_state_path)
        payload = {
            "completed_turns": self.completed_turns,
            "memory": self.memory.memory.model_dump(),
            "stuck_state": {
                "score": self.stuck.state.score,
                "recent_signatures": list(self.stuck.state.recent_signatures),
                "recent_failed_actions": self.stuck.state.recent_failed_actions,
                "loop_signature": self.stuck.state.loop_signature,
                "recovery_hint": self.stuck.state.recovery_hint,
                "repeated_state_count": self.stuck.state.repeated_state_count,
                "steps_since_progress": self.stuck.state.steps_since_progress,
                "oscillating": self.stuck.state.oscillating,
                "recent_maps": list(self.stuck.state.recent_maps),
                "map_oscillating": self.stuck.state.map_oscillating,
            },
            "telemetry": self.telemetry.to_dict(),
            "context_state": self.context_manager.export_state(),
            "route_cache": self.route_cache.model_dump(mode="json") if self.route_cache is not None else None,
            "execution_plan": self.execution_plan.model_dump(mode="json") if self.execution_plan is not None else None,
            "last_entry_direction": self._last_entry_direction,
        }
        metadata_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return metadata_path

    def load_checkpoint(self, directory: str | Path) -> dict:
        checkpoint_dir = Path(directory)
        metadata_path = checkpoint_dir / "checkpoint.json"
        emulator_state_path = checkpoint_dir / "emulator.state"
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        self.executor.emulator.load_state(emulator_state_path)
        self.memory.memory = MemoryState.model_validate(payload["memory"])
        stuck_payload = payload["stuck_state"]
        restored = StuckState(
            score=stuck_payload["score"],
            recent_failed_actions=list(stuck_payload["recent_failed_actions"]),
            loop_signature=stuck_payload["loop_signature"],
            recovery_hint=stuck_payload["recovery_hint"],
            repeated_state_count=stuck_payload["repeated_state_count"],
            steps_since_progress=stuck_payload["steps_since_progress"],
            oscillating=stuck_payload["oscillating"],
            map_oscillating=stuck_payload.get("map_oscillating", False),
        )
        restored.recent_signatures.extend(stuck_payload["recent_signatures"])
        restored.recent_maps.extend(stuck_payload.get("recent_maps", []))
        self.stuck.restore(restored)
        telemetry = payload.get("telemetry", {})
        self.telemetry = RunTelemetry(
            prompt_chars=telemetry.get("prompt_chars", 0),
            approx_prompt_tokens=telemetry.get("approx_prompt_tokens", 0),
            llm_prompt_tokens=telemetry.get("llm_prompt_tokens", 0),
            llm_completion_tokens=telemetry.get("llm_completion_tokens", 0),
            llm_total_tokens=telemetry.get("llm_total_tokens", 0),
            turns=telemetry.get("turns", 0),
            fallback_turns=telemetry.get("fallback_turns", 0),
            route_cache_turns=telemetry.get("route_cache_turns", 0),
            llm_calls=telemetry.get("llm_calls", 0),
            auto_selected_turns=telemetry.get("auto_selected_turns", 0),
            execution_plan_turns=telemetry.get("execution_plan_turns", 0),
            major_progress_turns=telemetry.get("major_progress_turns", 0),
            no_effect_turns=telemetry.get("no_effect_turns", 0),
            objective_switches=telemetry.get("objective_switches", 0),
        )
        self.completed_turns = int(payload.get("completed_turns", 0))
        self.context_manager.restore_state(payload.get("context_state"))
        route_payload = payload.get("route_cache")
        self.route_cache = CachedRoute.model_validate(route_payload) if route_payload else None
        execution_payload = payload.get("execution_plan")
        self.execution_plan = ExecutionPlan.model_validate(execution_payload) if execution_payload else None
        self._last_entry_direction = payload.get("last_entry_direction")
        return payload

    def summary(self) -> dict[str, Any]:
        summary = self.telemetry.to_dict()
        summary.update(world_map_stats(self.memory.memory.long_term.world_map))
        return summary

    def _parse_planner_decision(self, raw_text: str) -> PlannerDecision:
        payload = self._extract_json(raw_text)
        return PlannerDecision.model_validate(json.loads(payload))

    def _extract_json(self, raw_text: str) -> str:
        text = raw_text.strip()
        if text.startswith("{") and text.endswith("}"):
            return text
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end < start:
            raise ValueError("Model response did not contain a JSON object")
        return text[start : end + 1]

    def _resolve_candidate(
        self,
        decision: PlannerDecision,
        candidates: list[CandidateNextStep],
    ) -> CandidateNextStep | None:
        for candidate in candidates:
            if decision.candidate_id and candidate.id == decision.candidate_id:
                return candidate
        for candidate in candidates:
            if decision.intent and candidate.type == decision.intent:
                return candidate
        return None

    def _build_candidate_steps(self, state: StructuredGameState) -> list[CandidateNextStep]:
        short_objective_id = self._objective_id(ObjectiveHorizon.SHORT_TERM, "short_immediate_step")
        mid_objective_id = self._objective_id(ObjectiveHorizon.MID_TERM, "mid_local_progress")
        navigation_goal = self._sync_navigation_goal(state)
        candidates: list[CandidateNextStep] = []

        if state.text_box_open:
            dialogue_text = state.metadata.get("dialogue_text")
            yes_no_prompt = bool(state.metadata.get("yes_no_prompt"))
            if yes_no_prompt:
                why = "A yes/no prompt is open and requires an explicit choice."
                if isinstance(dialogue_text, str) and dialogue_text.strip():
                    why = f"Dialogue asks for a yes/no choice: {dialogue_text.splitlines()[0][:80]}"
                return [
                    CandidateNextStep(
                        id="select_yes",
                        type="SELECT_YES",
                        why=why,
                        priority=90,
                        expected_success_signal="Prompt closes or dialogue changes after choosing yes",
                        objective_id=short_objective_id,
                        action=ActionDecision(action=ActionType.PRESS_A, repeat=1, reason="select yes"),
                        step_budget=1,
                    ),
                    CandidateNextStep(
                        id="select_no",
                        type="SELECT_NO",
                        why=why,
                        priority=90,
                        expected_success_signal="Prompt closes or dialogue changes after choosing no",
                        objective_id=short_objective_id,
                        action=ActionDecision(action=ActionType.MOVE_DOWN, repeat=1, reason="move to no"),
                        follow_up_action=ActionType.PRESS_A,
                        step_budget=2,
                    ),
                ]
            why = "Text is already open."
            if isinstance(dialogue_text, str) and dialogue_text.strip():
                why = f"Dialogue is open: {dialogue_text.splitlines()[0][:80]}"
            return [
                CandidateNextStep(
                    id="advance_text",
                    type="ADVANCE_TEXT_UNTIL_CHANGE",
                    why=why,
                    priority=100,
                    expected_success_signal="Dialogue changes or the text box closes",
                    objective_id=short_objective_id,
                    action=ActionDecision(action=ActionType.PRESS_A, repeat=1, reason="advance text"),
                    step_budget=6,
                )
            ]
        if state.menu_open:
            menu_candidates = self._dedupe_candidates(self.menu_manager.build_candidates(state, short_objective_id))
            menu_candidates.sort(key=lambda item: (-item.priority, item.id))
            return menu_candidates[:4]
        if state.battle_state:
            return self.battle_manager.build_candidates(state, short_objective_id)

        candidates.extend(self.menu_manager.build_candidates(state, short_objective_id))

        if self.stuck.state.score >= self.stuck.threshold:
            recovery_action = self.validator.fallback(state, self.stuck.state, "stuck recovery")
            candidates.append(
                CandidateNextStep(
                    id=f"recover_{recovery_action.action.value.lower()}",
                    type="RECOVER_FROM_STUCK",
                    why=self.stuck.state.recovery_hint or "Recent actions failed without progress.",
                    priority=95,
                    expected_success_signal="A different local state appears",
                    objective_id=short_objective_id,
                    action=recovery_action,
                    step_budget=1,
                )
            )

        adjacent_interaction = self._candidate_for_adjacent_interactable(state, short_objective_id)
        if adjacent_interaction is not None:
            candidates.append(adjacent_interaction)

        move_to_interaction = self._candidate_to_move_adjacent_to_interactable(state, short_objective_id)
        if move_to_interaction is not None:
            candidates.append(move_to_interaction)

        connector_route = self._build_discovered_route_candidate(state, mid_objective_id, navigation_goal)
        if connector_route is not None:
            candidates.append(connector_route)

        candidates.extend(self._build_exit_candidates(state, mid_objective_id))

        frontier = self._build_frontier_candidate(state, mid_objective_id, candidates)
        if frontier is not None:
            candidates.append(frontier)

        if not candidates:
            probe_action = self.validator.fallback(state, self.stuck.state, "probe nearby state")
            candidates.append(
                CandidateNextStep(
                    id=f"probe_{probe_action.action.value.lower()}",
                    type="SAFE_STEP",
                    why="Probe a safe nearby step while no stronger target is available.",
                    priority=25,
                    expected_success_signal="Position changes or UI opens",
                    objective_id=short_objective_id,
                    action=probe_action,
                    step_budget=1,
                )
            )

        deduped = self._dedupe_candidates(candidates)
        deduped.sort(key=lambda item: (-item.priority, item.id))
        return deduped[:4]

    def _objective_id(self, horizon: ObjectiveHorizon, default: str) -> str:
        for objective in self.memory.memory.goals.active_objectives:
            if objective.horizon == horizon:
                return objective.id
        return default

    def _candidate_for_adjacent_interactable(
        self,
        state: StructuredGameState,
        objective_id: str,
    ) -> CandidateNextStep | None:
        blocked_set = self._blocked_coordinates(state.navigation)
        if state.x is None or state.y is None or not blocked_set:
            return None
        adjacent: list[tuple[int, int, int]] = []
        for dx, dy in DIRECTION_DELTAS:
            candidate = (state.x + dx, state.y + dy)
            if candidate not in blocked_set:
                continue
            blocked_neighbors = self._blocked_neighbor_count(candidate, blocked_set)
            adjacent.append((candidate[0], candidate[1], blocked_neighbors))
        if not adjacent:
            return None
        best_x, best_y, neighbor_count = sorted(adjacent, key=lambda item: (item[2], item[0], item[1]))[0]
        if neighbor_count != 0:
            return None
        if "PRESS_A" in self.stuck.state.recent_failed_actions[-2:]:
            return None
        return CandidateNextStep(
            id=f"interact_adjacent_{best_x}_{best_y}",
            type="MOVE_ADJACENT_TO_INTERACTABLE",
            target=ObjectiveTarget(kind="interactable", map_id=state.map_id, map_name=state.map_name, x=best_x, y=best_y),
            why="An adjacent isolated blocker may be an NPC or doorway.",
            priority=82 - min(neighbor_count, 2) * 4,
            expected_success_signal="Text opens or the local state changes",
            objective_id=objective_id,
            action=ActionDecision(action=ActionType.PRESS_A, repeat=1, reason="interact nearby"),
            step_budget=1,
        )

    def _candidate_to_move_adjacent_to_interactable(
        self,
        state: StructuredGameState,
        objective_id: str,
    ) -> CandidateNextStep | None:
        if state.navigation is None or state.x is None or state.y is None:
            return None
        blocked_set = self._blocked_coordinates(state.navigation)
        walkable_set = {(coord.x, coord.y) for coord in state.navigation.walkable}
        best_choice: tuple[int, int, int, int] | None = None
        for blocked_x, blocked_y in blocked_set:
            if abs(blocked_x - state.x) + abs(blocked_y - state.y) > 4:
                continue
            if self._blocked_neighbor_count((blocked_x, blocked_y), blocked_set) != 0:
                continue
            for dx, dy in DIRECTION_DELTAS:
                target_x = blocked_x + dx
                target_y = blocked_y + dy
                if (target_x, target_y) not in walkable_set or (target_x, target_y) == (state.x, state.y):
                    continue
                route = find_path(state.navigation, state.x, state.y, target_x, target_y)
                if route is None or len(route) == 0:
                    continue
                distance = len(route)
                candidate = (distance, target_x, target_y, blocked_x * 1000 + blocked_y)
                if best_choice is None or candidate < best_choice:
                    best_choice = candidate
        if best_choice is None:
            return None
        distance, target_x, target_y, encoded = best_choice
        blocked_x = encoded // 1000
        blocked_y = encoded % 1000
        return CandidateNextStep(
            id=f"approach_interactable_{blocked_x}_{blocked_y}",
            type="MOVE_ADJACENT_TO_INTERACTABLE",
            target=ObjectiveTarget(
                kind="interactable",
                map_id=state.map_id,
                map_name=state.map_name,
                x=blocked_x,
                y=blocked_y,
                detail=f"stand at ({target_x}, {target_y})",
            ),
            why=f"A nearby interactable-looking blocker is reachable in {distance} steps.",
            priority=72 - min(distance, 6),
            expected_success_signal="Reach the adjacent tile, then open text or change state",
            objective_id=objective_id,
            target_x=target_x,
            target_y=target_y,
            follow_up_action=ActionType.PRESS_A,
            step_budget=distance + 2,
        )

    def _build_exit_candidates(
        self,
        state: StructuredGameState,
        objective_id: str,
    ) -> list[CandidateNextStep]:
        if state.navigation is None or state.x is None or state.y is None:
            return []
        side_targets: dict[str, tuple[int, int, int]] = {}
        for coordinate in state.navigation.walkable:
            side = self._boundary_side(state.navigation, coordinate.x, coordinate.y)
            if side is None or (coordinate.x, coordinate.y) == (state.x, state.y):
                continue
            route = find_path(state.navigation, state.x, state.y, coordinate.x, coordinate.y)
            if route is None or len(route) == 0:
                continue
            distance = len(route)
            current = side_targets.get(side)
            choice = (distance, coordinate.x, coordinate.y)
            if current is None or choice < current:
                side_targets[side] = choice

        candidates: list[CandidateNextStep] = []
        world_map = self.memory.memory.long_term.world_map
        known_connectors = connectors_from_map(world_map, state.map_name, confirmed_only=False)
        for side, (distance, target_x, target_y) in sorted(side_targets.items()):
            side_matches = [connector for connector in known_connectors if connector.source_side == side]
            destinations = sorted({connector.destination_map for connector in side_matches if connector.destination_map})
            if destinations:
                destination_text = ", ".join(destinations[:2])
                why = f"Exit {side} matches a discovered connector to {destination_text} ({distance} steps)."
                detail = f"{side} → {destination_text}"
            else:
                why = f"Exit {side} ({distance} steps), destination still unknown."
                detail = side

            priority = 60 - min(distance, 10)
            if side_matches and not destinations:
                priority += 8
                why = f"{why} This connector has been seen but not confirmed yet."

            if self._last_entry_direction and side == self._last_entry_direction:
                priority -= 25
                why += " (backtrack — we just came from this direction)"

            candidates.append(
                CandidateNextStep(
                    id=f"exit_{side}_{target_x}_{target_y}",
                    type="GO_TO_MAP_EXIT",
                    target=ObjectiveTarget(kind="exit", map_id=state.map_id, map_name=state.map_name, x=target_x, y=target_y, detail=detail),
                    why=why,
                    priority=priority,
                    expected_success_signal="Map changes or a new local affordance appears",
                    objective_id=objective_id,
                    target_x=target_x,
                    target_y=target_y,
                    step_budget=distance + 1,
                )
            )
        return candidates[:4]

    def _build_discovered_route_candidate(
        self,
        state: StructuredGameState,
        objective_id: str,
        goal: NavigationGoal | None,
    ) -> CandidateNextStep | None:
        if goal is None or state.navigation is None or state.x is None or state.y is None:
            return None
        route = shortest_confirmed_path(self.memory.memory.long_term.world_map, state.map_name, goal.target_map_name)
        if route is None or not route:
            return None
        connector = route[0]
        if connector.source_map != state.map_name:
            return None
        if connector.approach_x is None or connector.approach_y is None or connector.transition_action is None:
            return None
        local_route = find_path(state.navigation, state.x, state.y, connector.approach_x, connector.approach_y)
        if local_route is None:
            return None
        distance = len(local_route)
        return CandidateNextStep(
            id=f"follow_connector_{connector.id}",
            type="FOLLOW_DISCOVERED_CONNECTOR",
            target=ObjectiveTarget(
                kind="connector",
                map_id=state.map_id,
                map_name=state.map_name,
                x=connector.approach_x,
                y=connector.approach_y,
                detail=describe_connector(connector),
            ),
            why=(
                f"Follow the discovered connector toward {goal.target_map_name}: "
                f"{describe_connector(connector)}."
            ),
            priority=92 - min(distance, 10),
            expected_success_signal=f"Transition toward {goal.target_map_name}",
            objective_id=objective_id,
            target_x=connector.approach_x,
            target_y=connector.approach_y,
            follow_up_action=connector.transition_action,
            step_budget=max(1, distance + 1),
        )

    def _build_frontier_candidate(
        self,
        state: StructuredGameState,
        objective_id: str,
        existing: list[CandidateNextStep],
    ) -> CandidateNextStep | None:
        if state.navigation is None or state.x is None or state.y is None:
            return None
        reserved_targets = {
            (candidate.target_x, candidate.target_y)
            for candidate in existing
            if candidate.target_x is not None and candidate.target_y is not None
        }
        best: tuple[int, int, int] | None = None
        for coordinate in state.navigation.walkable:
            if (coordinate.x, coordinate.y) == (state.x, state.y):
                continue
            if (coordinate.x, coordinate.y) in reserved_targets:
                continue
            if self._boundary_side(state.navigation, coordinate.x, coordinate.y) is None:
                continue
            route = find_path(state.navigation, state.x, state.y, coordinate.x, coordinate.y)
            if route is None or len(route) == 0:
                continue
            distance = len(route)
            candidate = (distance, coordinate.x, coordinate.y)
            if best is None or candidate < best:
                best = candidate
        if best is None:
            return None
        distance, target_x, target_y = best
        return CandidateNextStep(
            id=f"frontier_{target_x}_{target_y}",
            type="EXPLORE_NEAREST_FRONTIER",
            target=ObjectiveTarget(kind="frontier", map_id=state.map_id, map_name=state.map_name, x=target_x, y=target_y),
            why=f"The nearest frontier tile is {distance} steps away and may unlock a better option.",
            priority=42 - min(distance, 8),
            expected_success_signal="Position changes or new ranked candidates appear",
            objective_id=objective_id,
            target_x=target_x,
            target_y=target_y,
            step_budget=distance + 1,
        )

    def _dedupe_candidates(self, candidates: list[CandidateNextStep]) -> list[CandidateNextStep]:
        seen: set[tuple[str, int | None, int | None, str | None, str | None]] = set()
        deduped: list[CandidateNextStep] = []
        for candidate in candidates:
            key = (
                candidate.type,
                candidate.target_x,
                candidate.target_y,
                candidate.action.action.value if candidate.action else None,
                candidate.target.detail if candidate.target else None,
            )
            if key in seen:
                continue
            seen.add(key)
            deduped.append(candidate)
        return deduped

    def _should_auto_select(self, candidates: list[CandidateNextStep]) -> bool:
        if not candidates:
            return False
        current_state = self.executor.emulator.get_structured_state()
        if current_state.menu_open:
            return True
        goal = self.memory.memory.long_term.navigation_goal
        if (
            goal is not None
            and goal.confirmation_required_map
            and candidates[0].type == "FOLLOW_DISCOVERED_CONNECTOR"
            and goal.confirmation_required_map == current_state.map_name
        ):
            return False
        if len(candidates) == 1:
            return True
        if any(candidate.type in YES_NO_CANDIDATE_TYPES for candidate in candidates):
            return False
        if candidates[0].type in {
            "ADVANCE_TEXT_UNTIL_CHANGE",
            "CLOSE_MENU",
            "BATTLE_DEFAULT",
            "RECOVER_FROM_STUCK",
            "OPEN_START_MENU_FOR_HM",
            "SELECT_START_MENU_OPTION_FOR_HM",
            "SELECT_HM_ITEM",
            "SELECT_PARTY_POKEMON",
            "USE_FIELD_MOVE",
        }:
            return True
        if candidates[0].type == "MOVE_ADJACENT_TO_INTERACTABLE" and candidates[0].action is not None:
            return True
        return (candidates[0].priority - candidates[1].priority) >= 15

    def _compile_candidate(
        self,
        candidate: CandidateNextStep,
        state: StructuredGameState,
        reason: str,
    ) -> ActionDecision | None:
        self.execution_plan = None
        if state.battle_state is not None:
            self.battle_manager.record_choice(candidate)
        if candidate.action is not None:
            action = candidate.action.model_copy(deep=True)
            action.reason = reason
            if candidate.type in {"ADVANCE_TEXT_UNTIL_CHANGE", "CLOSE_MENU", "BATTLE_DEFAULT"}:
                self.execution_plan = ExecutionPlan(
                    objective_id=candidate.objective_id,
                    candidate_id=candidate.id,
                    plan_type=candidate.type,
                    expected_success_signal=candidate.expected_success_signal,
                    step_budget=candidate.step_budget,
                    button_action=action.action,
                    reason=reason,
                    started_step=state.step,
                )
            elif candidate.follow_up_action is not None:
                self.execution_plan = ExecutionPlan(
                    objective_id=candidate.objective_id,
                    candidate_id=candidate.id,
                    plan_type=candidate.type,
                    expected_success_signal=candidate.expected_success_signal,
                    step_budget=1,
                    button_action=candidate.follow_up_action,
                    reason=reason,
                    started_step=state.step,
                )
            return self.validator.validate(action, state)

        if candidate.target_x is None or candidate.target_y is None:
            return None
        return self._compile_navigation_candidate(candidate, state, reason)

    def _compile_navigation_candidate(
        self,
        candidate: CandidateNextStep,
        state: StructuredGameState,
        reason: str,
    ) -> ActionDecision | None:
        if state.navigation is None or state.x is None or state.y is None:
            self.route_cache = None
            self.execution_plan = None
            return None
        route = find_path(state.navigation, state.x, state.y, candidate.target_x, candidate.target_y)
        if route is None:
            self.route_cache = None
            self.execution_plan = None
            return None
        if len(route) == 0:
            if candidate.follow_up_action is None:
                self.route_cache = None
                self.execution_plan = None
                return None
            self.execution_plan = ExecutionPlan(
                objective_id=candidate.objective_id,
                candidate_id=candidate.id,
                plan_type=candidate.type,
                target=candidate.target,
                expected_success_signal=candidate.expected_success_signal,
                step_budget=max(1, candidate.step_budget - 1),
                button_action=candidate.follow_up_action,
                target_x=candidate.target_x,
                target_y=candidate.target_y,
                follow_up_action=None,
                map_id=state.map_id,
                collision_hash=state.navigation.collision_hash,
                reason=reason,
                started_step=state.step,
            )
            return ActionDecision(action=candidate.follow_up_action, repeat=1, reason=reason)

        first_action = route[0]
        remaining = list(route[1:])
        if remaining:
            self.route_cache = CachedRoute(
                map_id=state.map_id,
                target_x=candidate.target_x,
                target_y=candidate.target_y,
                collision_hash=state.navigation.collision_hash,
                remaining_actions=remaining,
                expected_start=advance_position(state.x, state.y, first_action),
            )
        else:
            self.route_cache = None
        self.execution_plan = ExecutionPlan(
            objective_id=candidate.objective_id,
            candidate_id=candidate.id,
            plan_type=candidate.type,
            target=candidate.target,
            expected_success_signal=candidate.expected_success_signal,
            step_budget=max(candidate.step_budget, len(route)),
            button_action=None,
            target_x=candidate.target_x,
            target_y=candidate.target_y,
            follow_up_action=candidate.follow_up_action,
            map_id=state.map_id,
            collision_hash=state.navigation.collision_hash,
            reason=reason,
            started_step=state.step,
        )
        return ActionDecision(action=first_action, repeat=1, reason=reason)

    def _plan_from_execution_plan(self, state: StructuredGameState) -> ActionDecision | None:
        plan = self.execution_plan
        if plan is None:
            return None
        if plan.plan_type == "ADVANCE_TEXT_UNTIL_CHANGE":
            return self._continue_button_plan(plan, state, condition=state.text_box_open)
        if plan.plan_type in YES_NO_CANDIDATE_TYPES:
            return self._continue_button_plan(
                plan,
                state,
                condition=state.text_box_open and bool(state.metadata.get("yes_no_prompt")),
            )
        if plan.plan_type == "CLOSE_MENU":
            return self._continue_button_plan(plan, state, condition=state.menu_open)
        if plan.plan_type == "BATTLE_DEFAULT":
            return self._continue_button_plan(plan, state, condition=state.battle_state is not None)
        if plan.plan_type in NAVIGATION_CANDIDATE_TYPES and plan.follow_up_action is not None:
            if (state.x, state.y) == (plan.target_x, plan.target_y):
                follow_up_action = plan.follow_up_action
                plan.follow_up_action = None
                self.execution_plan = plan
                return ActionDecision(action=follow_up_action, repeat=1, reason=plan.reason or "Trigger connector at target")
        if plan.plan_type in NAVIGATION_CANDIDATE_TYPES and plan.target_x is not None and plan.target_y is not None:
            if (state.x, state.y) == (plan.target_x, plan.target_y) and plan.follow_up_action is None:
                self.execution_plan = None
        return None

    def _continue_button_plan(
        self,
        plan: ExecutionPlan,
        state: StructuredGameState,
        *,
        condition: bool,
    ) -> ActionDecision | None:
        if not condition or plan.button_action is None:
            self.execution_plan = None
            return None
        if plan.step_budget <= 0:
            self.execution_plan = None
            return None
        plan.step_budget -= 1
        self.execution_plan = plan if plan.step_budget > 0 else None
        return ActionDecision(action=plan.button_action, repeat=1, reason=plan.reason or "Continue current plan")

    def _plan_from_cached_route(self, state: StructuredGameState) -> ActionDecision | None:
        route = self.route_cache
        if route is None:
            return None
        if state.mode != GameMode.OVERWORLD or state.navigation is None:
            self.route_cache = None
            return None
        if route.map_id != state.map_id:
            self.route_cache = None
            return None
        if route.collision_hash != state.navigation.collision_hash:
            self.route_cache = None
            return None
        if state.x != route.expected_start.x or state.y != route.expected_start.y:
            self.route_cache = None
            return None
        if not route.remaining_actions:
            self.route_cache = None
            return None

        next_action = route.remaining_actions.pop(0)
        next_expected = advance_position(state.x or 0, state.y or 0, next_action)
        if route.remaining_actions:
            route.expected_start = next_expected
            self.route_cache = route
        else:
            self.route_cache = None
        target_text = f"({route.target_x}, {route.target_y})"
        return ActionDecision(action=next_action, repeat=1, reason=f"Continue cached route to {target_text}")

    def _refresh_route_cache_after_turn(self, state: StructuredGameState, progress: ProgressResult) -> None:
        if self.route_cache is None:
            return
        if state.mode != GameMode.OVERWORLD or state.menu_open or state.text_box_open or state.battle_state:
            self.route_cache = None
            return
        if state.navigation is None or state.navigation.collision_hash != self.route_cache.collision_hash:
            self.route_cache = None
            return
        if state.map_id != self.route_cache.map_id:
            self.route_cache = None
            return
        if progress.classification == "no_effect":
            self.route_cache = None
            return
        if state.x != self.route_cache.expected_start.x or state.y != self.route_cache.expected_start.y:
            self.route_cache = None

    def _refresh_execution_plan_after_turn(self, state: StructuredGameState, progress: ProgressResult) -> None:
        plan = self.execution_plan
        if plan is None:
            return
        if plan.plan_type in YES_NO_CANDIDATE_TYPES:
            if progress.classification == "no_effect":
                self.execution_plan = None
                return
            if not state.text_box_open or not bool(state.metadata.get("yes_no_prompt")):
                self.execution_plan = None
            return
        if plan.plan_type == "ADVANCE_TEXT_UNTIL_CHANGE":
            if not state.text_box_open:
                self.execution_plan = None
            return
        if plan.plan_type == "CLOSE_MENU":
            if not state.menu_open:
                self.execution_plan = None
            return
        if plan.plan_type == "BATTLE_DEFAULT":
            if not state.battle_state:
                self.execution_plan = None
            return
        if state.mode != GameMode.OVERWORLD or state.menu_open or state.text_box_open or state.battle_state:
            self.execution_plan = None
            return
        if plan.map_id != state.map_id:
            self.execution_plan = None
            return
        if state.navigation is not None and plan.collision_hash is not None and state.navigation.collision_hash != plan.collision_hash:
            self.execution_plan = None
            return
        if plan.target_x is not None and plan.target_y is not None and (state.x, state.y) == (plan.target_x, plan.target_y):
            if plan.follow_up_action is None:
                self.execution_plan = None
                return
        if progress.classification == "no_effect" and self.route_cache is None and plan.follow_up_action is None:
            self.execution_plan = None

    def _blocked_coordinates(self, navigation: NavigationSnapshot | None) -> set[tuple[int, int]]:
        if navigation is None:
            return set()
        return {(coord.x, coord.y) for coord in navigation.blocked}

    def _blocked_neighbor_count(self, coordinate: tuple[int, int], blocked_set: set[tuple[int, int]]) -> int:
        x, y = coordinate
        return sum((x + dx, y + dy) in blocked_set for dx, dy in DIRECTION_DELTAS)

    def _boundary_side(self, navigation: NavigationSnapshot, x: int, y: int) -> str | None:
        if y == navigation.min_y:
            return "north"
        if x == navigation.max_x:
            return "east"
        if y == navigation.max_y:
            return "south"
        if x == navigation.min_x:
            return "west"
        return None

    def _short_reason(self, candidate: CandidateNextStep) -> str:
        reason = candidate.why.strip()
        if len(reason) <= 60:
            return reason
        return f"{reason[:57].rstrip()}..."

    @staticmethod
    def _side_for_action(action: ActionType) -> str | None:
        mapping = {
            ActionType.MOVE_UP: "north",
            ActionType.MOVE_RIGHT: "east",
            ActionType.MOVE_DOWN: "south",
            ActionType.MOVE_LEFT: "west",
        }
        return mapping.get(action)

    def _sync_navigation_goal(self, state: StructuredGameState) -> NavigationGoal | None:
        target_map = None
        for objective in self.memory.memory.goals.active_objectives:
            if objective.target and objective.target.map_name and objective.horizon in {ObjectiveHorizon.MID_TERM, ObjectiveHorizon.LONG_TERM}:
                target_map = objective.target.map_name
                break

        goal = self.memory.memory.long_term.navigation_goal
        if state.is_bootstrap():
            self.memory.memory.long_term.navigation_goal = None
            return None

        if goal is not None:
            if goal.target_map_name == state.map_name:
                self.memory.memory.long_term.navigation_goal = None
                return None
            if goal.confirmation_required_map == state.map_name:
                return goal
            if target_map is None:
                return goal
            if goal.target_map_name != target_map:
                self.memory.memory.long_term.navigation_goal = None
                goal = None

        if not target_map or target_map == state.map_name:
            self.memory.memory.long_term.navigation_goal = None
            return None

        if goal is None or goal.target_map_name != target_map:
            goal = NavigationGoal(
                target_map_name=target_map,
                source="objective",
                started_step=state.step,
                last_confirmed_step=state.step,
            )
            self.memory.memory.long_term.navigation_goal = goal
        return goal

    def _clear_navigation_confirmation(self, map_name: str) -> None:
        goal = self.memory.memory.long_term.navigation_goal
        if goal is None:
            return
        if goal.confirmation_required_map == map_name:
            goal.confirmation_required_map = None
            goal.last_confirmed_step = self.executor.emulator.get_structured_state().step
