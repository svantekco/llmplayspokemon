from __future__ import annotations

from collections.abc import Callable
import concurrent.futures
import copy
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from typing import TYPE_CHECKING

from pokemon_agent.agent.controllers.battle import BattleController
from pokemon_agent.agent.controllers.cutscene import CutsceneController
from pokemon_agent.agent.controllers.dialogue import DialogueController
from pokemon_agent.agent.controllers.menu import MenuController
from pokemon_agent.agent.controllers.overworld import OverworldController
from pokemon_agent.agent.controllers.protocol import NavigationTarget
from pokemon_agent.agent.controllers.protocol import TurnContext
from pokemon_agent.agent.controllers.stubs import StubUnknownController
from pokemon_agent.agent.context_manager import ContextManager, build_messages, measure_prompt
from pokemon_agent.agent.executor import Executor
from pokemon_agent.agent.llm_client import CompletionResponse, LLMUsage, OpenRouterClient, RequestStatus
from pokemon_agent.agent.menu_manager import MenuManager
from pokemon_agent.agent.mode_dispatcher import ModeDispatcher
from pokemon_agent.agent.memory_manager import MemoryManager
from pokemon_agent.agent.navigator import Navigator
from pokemon_agent.agent.navigation import advance_position
from pokemon_agent.agent.navigation import find_path
from pokemon_agent.agent.navigation import NavigationGrid
from pokemon_agent.agent.navigation import is_real_map_edge
from pokemon_agent.agent.planning_types import PlanningResult
from pokemon_agent.agent.navigation import visible_boundary_side
from pokemon_agent.agent.progress import ProgressDetector, ProgressResult
from pokemon_agent.agent.stuck_detector import StuckDetector, StuckState
from pokemon_agent.agent.validator import ActionValidator
from pokemon_agent.agent.world_map import connectors_from_map
from pokemon_agent.agent.world_map import describe_connector
from pokemon_agent.agent.world_map import observe_state
from pokemon_agent.agent.world_map import shortest_confirmed_path
from pokemon_agent.agent.world_map import world_map_stats
from pokemon_agent.data.walkthrough import get_current_milestone
from pokemon_agent.emulator.base import EmulatorAdapter
from pokemon_agent.models.action import ActionDecision
from pokemon_agent.models.action import ActionType
from pokemon_agent.models.action import ExecutorStatus
from pokemon_agent.models.action import StepResult
from pokemon_agent.models.action import Task
from pokemon_agent.models.action import TaskKind
from pokemon_agent.models.events import EventRecord
from pokemon_agent.models.memory import ConnectorStatus
from pokemon_agent.models.memory import DiscoveredConnector
from pokemon_agent.models.memory import MemoryState
from pokemon_agent.models.memory import NavigationGoal
from pokemon_agent.models.planner import CandidateNextStep
from pokemon_agent.models.planner import CandidateRuntime
from pokemon_agent.models.planner import HumanObjectivePlan
from pokemon_agent.models.planner import InternalObjectivePlan
from pokemon_agent.models.planner import Objective
from pokemon_agent.models.planner import ObjectiveHorizon
from pokemon_agent.models.planner import ObjectivePlanEnvelope
from pokemon_agent.models.planner import ObjectivePlanStatus
from pokemon_agent.models.planner import ObjectiveTarget
from pokemon_agent.models.planner import PlannerDecision
from pokemon_agent.models.state import GameMode
from pokemon_agent.models.state import NavigationSnapshot
from pokemon_agent.models.state import StructuredGameState
from pokemon_agent.navigation.world_graph import Landmark
from pokemon_agent.navigation.world_graph import WorldGraphEdge
from pokemon_agent.navigation.world_graph import load_world_graph
from pokemon_agent.navigation.world_graph import map_matches

if TYPE_CHECKING:
    from PIL.Image import Image as PILImage

DIRECTION_DELTAS: tuple[tuple[int, int], ...] = ((0, -1), (1, 0), (0, 1), (-1, 0))
NAVIGATION_CANDIDATE_TYPES = {
    "ADVANCE_TOWARD_BOUNDARY",
    "ENTER_CONNECTOR",
    "EXPLORE_CONNECTOR",
    "EXPLORE_FOR_NPC",
    "EXPLORE_WINDOW",
}
YES_NO_CANDIDATE_TYPES = {"SELECT_YES", "SELECT_NO"}
MAX_ROUTE_CONNECTOR_HOPS = 4
STORY_INTERACTION_KEYWORDS = (
    "talk",
    "speak",
    "choose",
    "receive",
    "deliver",
    "clerk",
    "give",
    "take",
    "hand",
    "battle",
    "parcel",
    "pokedex",
)


@dataclass(frozen=True, slots=True)
class RouteHop:
    to_map: str
    direction: str


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
    prompt_metrics: dict[str, Any] | None = None
    llm_usage: LLMUsage | None = None
    llm_attempted: bool = False
    llm_model: str | None = None
    planner_source: str = "fallback"
    objective_id: str | None = None
    candidate_id: str | None = None
    candidates: list[CandidateNextStep] = field(default_factory=list)
    navigation_goal: NavigationGoal | None = None
    objective_plan: ObjectivePlanEnvelope | None = None
    executor_task: Task | None = None
    suggested_path: list[tuple[int, int]] = field(default_factory=list)
    screen_image: PILImage | None = None


@dataclass(slots=True)
class RunTelemetry:
    prompt_chars: int = 0
    approx_prompt_tokens: int = 0
    llm_prompt_tokens: int = 0
    llm_completion_tokens: int = 0
    llm_total_tokens: int = 0
    turns: int = 0
    fallback_turns: int = 0
    llm_calls: int = 0
    auto_selected_turns: int = 0
    executor_turns: int = 0
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
        if turn.planner_source == "executor":
            self.executor_turns += 1
        if turn.llm_attempted:
            self.llm_calls += 1
        if turn.progress.classification == "major_progress":
            self.major_progress_turns += 1
        if turn.progress.classification == "no_effect":
            self.no_effect_turns += 1
        if turn.prompt_metrics:
            self.prompt_chars += int(turn.prompt_metrics.get("chars", 0))
            self.approx_prompt_tokens += int(turn.prompt_metrics.get("approx_tokens", 0))
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
            "llm_calls": self.llm_calls,
            "auto_selected_turns": self.auto_selected_turns,
            "executor_turns": self.executor_turns,
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
        emulator: EmulatorAdapter,
        memory: MemoryManager,
        progress: ProgressDetector,
        stuck: StuckDetector,
        validator: ActionValidator,
        llm_client: OpenRouterClient | None = None,
        context_manager: ContextManager | None = None,
    ) -> None:
        self.emulator = emulator
        self.memory = memory
        self.progress = progress
        self.stuck = stuck
        self.validator = validator
        self.llm_client = llm_client
        self.context_manager = context_manager or ContextManager()
        self.telemetry = RunTelemetry()
        self.completed_turns = 0
        self._candidate_runtime: dict[str, CandidateRuntime] = {}
        self.menu_manager = MenuManager()
        self.static_world_graph = load_world_graph()
        self._rebuild_overworld_stack()
        # Keep the legacy executor available as a compatibility adapter for any
        # remaining task-based plans outside the overworld controller path.
        self.executor = Executor(self._connector_by_id)
        self._last_entry_direction: str | None = None
        self._previous_map_name: str | None = None
        self._objective_plan_messages: list[dict[str, str]] = []
        self._objective_plan_prompt_metrics: dict[str, Any] | None = None
        self._objective_plan_llm_usage: LLMUsage | None = None
        self._objective_plan_llm_attempted = False
        self._objective_plan_llm_model: str | None = None
        # Tracks (map_id, player_x, player_y) → turn_index of last text-box interaction.
        # Used to suppress re-interacting with the same sign/object within a cooldown window.
        self._interacted_tiles: dict[tuple[str | int, int, int], int] = {}
        self._interaction_cooldown: int = 20
        self._activity_callback: Callable[[str, str | None], None] | None = None
        self.dispatcher = self._build_mode_dispatcher()

    _OPPOSITE_DIRECTION: dict[str, str] = {
        "north": "south",
        "south": "north",
        "east": "west",
        "west": "east",
    }

    def set_activity_callback(self, callback: Callable[[str, str | None], None] | None) -> None:
        self._activity_callback = callback

    def _report_activity(self, phase: str, detail: str | None = None) -> None:
        if self._activity_callback is None:
            return
        self._activity_callback(phase, detail)

    def _build_mode_dispatcher(self) -> ModeDispatcher:
        battle_complete = None
        if self.llm_client is not None:
            battle_complete = lambda messages, purpose: self._complete_with_window_pump(messages, purpose=purpose)
        return ModeDispatcher(
            {
                GameMode.OVERWORLD: self.overworld_controller,
                GameMode.MENU: MenuController(menu_manager=self.menu_manager),
                GameMode.TEXT: DialogueController(),
                GameMode.BATTLE: BattleController(battle_complete),
                GameMode.CUTSCENE: CutsceneController(),
                GameMode.UNKNOWN: StubUnknownController(self._plan_unknown_mode),
            }
        )

    def _rebuild_overworld_stack(self) -> None:
        self.navigator = Navigator(self.static_world_graph, self.memory.memory.long_term.world_map)
        self.overworld_controller = OverworldController(
            self.navigator,
            self.memory.memory.long_term.world_map,
            static_world_graph=self.static_world_graph,
            goal_getter=lambda: self.memory.memory.long_term.navigation_goal,
            landmark_for_destination=self._landmark_for_destination_on_current_map,
            utility_action_getter=self._deterministic_overworld_utility_action,
        )

    def _build_turn_context(self, state: StructuredGameState) -> TurnContext:
        objective = self._current_turn_objective()
        previous_action: ActionDecision | None = None
        previous_progress: str | None = None
        if self.context_manager.action_traces:
            trace = self.context_manager.action_traces[-1]
            previous_progress = trace.progress_classification
            try:
                previous_action = ActionDecision(
                    action=ActionType(trace.action),
                    repeat=trace.repeat,
                    reason=trace.reason,
                )
            except ValueError:
                previous_action = None
        return TurnContext(
            objective=objective,
            navigation_target=self._build_navigation_target(objective),
            stuck_score=self.stuck.state.score,
            turn_index=self.completed_turns + 1,
            previous_action=previous_action,
            previous_progress=previous_progress,
        )

    def _current_turn_objective(self) -> Objective | None:
        objectives = self.memory.memory.goals.active_objectives
        for horizon in (ObjectiveHorizon.SHORT_TERM, ObjectiveHorizon.MID_TERM, ObjectiveHorizon.LONG_TERM):
            for objective in objectives:
                if objective.horizon == horizon:
                    return objective
        return objectives[0] if objectives else None

    def _build_navigation_target(self, objective: Objective | None) -> NavigationTarget | None:
        goal = self.memory.memory.long_term.navigation_goal
        target = objective.target if objective is not None else None
        map_name = None
        x = None
        y = None
        landmark_id = None
        reason_parts: list[str] = []
        if target is not None:
            map_name = target.map_name
            x = target.x
            y = target.y
            if target.detail:
                reason_parts.append(target.detail)
        if goal is not None:
            map_name = map_name or goal.target_map_name
            landmark_id = goal.target_landmark_id
            if goal.objective_kind:
                reason_parts.insert(0, goal.objective_kind)
        if map_name is None and x is None and y is None and landmark_id is None:
            return None
        return NavigationTarget(
            map_name=map_name,
            x=x,
            y=y,
            landmark_id=landmark_id,
            reason="; ".join(part for part in reason_parts if part),
        )

    def _deterministic_overworld_utility_action(
        self,
        state: StructuredGameState,
        context: TurnContext,
    ) -> PlanningResult | None:
        objective_id = context.objective.id if context.objective is not None else "overworld_controller"
        candidates = self.menu_manager.build_candidates(state, objective_id)
        runtime = self.menu_manager.runtime_map()
        ranked: list[tuple[tuple[int, str], ActionDecision]] = []
        for candidate in candidates:
            runtime_entry = runtime.get(candidate.id)
            if runtime_entry is None or runtime_entry.action is None:
                continue
            action = runtime_entry.action.model_copy(deep=True)
            if action.action != ActionType.PRESS_START:
                continue
            if not action.reason:
                action.reason = candidate.why
            ranked.append(((-candidate.priority, candidate.id), action))
        if not ranked:
            return None
        ranked.sort(key=lambda item: item[0])
        return PlanningResult(action=ranked[0][1], planner_source="overworld_controller")

    def run_turn(self, turn_index: int) -> TurnResult:
        self._report_activity("Planning turn", f"Turn #{turn_index}")
        self._prune_temporary_blocked_tiles(turn_index)
        before, planning = self._resolve_turn_plan()
        self._report_activity("Capturing frame", f"Turn #{turn_index}")
        screen_image = self.emulator.capture_screen_image()
        navigation_goal_snapshot = copy.deepcopy(self.memory.memory.long_term.navigation_goal)
        objective_plan_snapshot = copy.deepcopy(self.memory.memory.long_term.objective_plan)
        executor_task_snapshot = copy.deepcopy(planning.executor_task)
        active_connector_snapshot = (
            copy.deepcopy(self.overworld_controller.active_connector())
            if planning.planner_source == "overworld_controller"
            else self._connector_for_task(executor_task_snapshot)
        )
        if planning.action is None:
            raise RuntimeError("Planning did not yield an executable action")
        action = self.validator.validate(planning.action, before)
        self._report_activity("Executing action", f"{action.action.value} x{action.repeat}")
        self.emulator.set_live_path_overlay(before, planning.suggested_path)
        try:
            self.emulator.execute_action(action)
        finally:
            self.emulator.clear_live_path_overlay()
        self._report_activity("Reading new state", f"Turn #{turn_index}")
        after = self.emulator.get_structured_state()
        self._report_activity("Summarizing turn", f"{before.map_name} -> {after.map_name}")
        progress_result = self.progress.compare(before, after)
        stuck_state = self.stuck.update(after, action, progress_result.classification, progress_result)
        # Track entry direction on map change, and clear interaction cooldown on new map.
        if before.map_id != after.map_id or before.map_name != after.map_name:
            self._previous_map_name = before.map_name
            entered_via = self._side_for_action(action.action)
            self._last_entry_direction = self._OPPOSITE_DIRECTION.get(entered_via) if entered_via else None
            self._interacted_tiles.clear()
            self._clear_temporary_blocked_tiles(before)
            self._clear_temporary_blocked_tiles(after)
            self.overworld_controller.reset()
            self.executor.abort()
        if progress_result.classification == "no_effect":
            if planning.planner_source == "overworld_controller":
                self.overworld_controller.report_failure(
                    before,
                    action,
                    turn_index=turn_index,
                )
            self.executor.report_failure(before, action)
        # Record player position when a text box opens (sign/NPC interaction detected).
        if (
            not before.text_box_open
            and after.text_box_open
            and before.map_id is not None
            and before.x is not None
            and before.y is not None
        ):
            key = (before.map_id, before.x, before.y)
            self._interacted_tiles[key] = turn_index
        self._update_navigation_goal_after_turn(after, progress_result, planning)
        events = self.memory.update_from_transition(
            before,
            after,
            action,
            progress_result,
            stuck_state,
            active_connector=active_connector_snapshot,
        )
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
            candidates=copy.deepcopy(planning.candidates),
            navigation_goal=navigation_goal_snapshot,
            objective_plan=objective_plan_snapshot,
            executor_task=executor_task_snapshot,
            suggested_path=copy.deepcopy(planning.suggested_path),
            screen_image=screen_image,
        )
        self.telemetry.register_turn(turn_result)
        self.completed_turns = max(self.completed_turns, turn_index)
        self._report_activity("Turn complete", f"Turn #{turn_index} {progress_result.classification}")
        return turn_result

    def run(self, turns: int) -> list[TurnResult]:
        return [self.run_turn(turn_index) for turn_index in range(1, turns + 1)]

    def _resolve_turn_plan(self) -> tuple[StructuredGameState, PlanningResult]:
        blocked_reasons: list[str] = []
        latest_state: StructuredGameState | None = None
        for attempt_index in range(8):
            self._report_activity("Reading emulator state", f"Planning attempt {attempt_index + 1}/8")
            state = self.emulator.get_structured_state()
            latest_state = state
            if self.executor.is_active():
                task_snapshot = self.executor.current_task()
                self._report_activity("Continuing executor task", self._describe_task(task_snapshot))
                step = self.executor.step(state)
                if step.status == ExecutorStatus.STEPPING:
                    return state, PlanningResult(
                        action=step.action,
                        planner_source="executor",
                        executor_task=task_snapshot,
                        suggested_path=list(step.suggested_path),
                    )
                if step.status == ExecutorStatus.INTERRUPTED:
                    planning = self._plan_action(state)
                    planning.executor_task = task_snapshot
                    if planning.action is not None:
                        return state, planning
                    continue
                if step.status == ExecutorStatus.BLOCKED:
                    if step.blocked_reason:
                        blocked_reasons.append(step.blocked_reason)
                    self.executor.abort()
                    continue
                continue

            planning = self._plan_action(state)
            live_state = self.emulator.get_structured_state()
            latest_state = live_state
            if planning.action is not None:
                planning.suggested_path = self._suggested_path_for_action(live_state, planning.action)
                return live_state, planning
            if planning.task is None:
                fallback = self.validator.fallback(live_state, self.stuck.state, "planner produced no action")
                planning.action = self.validator.validate(fallback, live_state)
                planning.used_fallback = True
                planning.planner_source = "fallback"
                return live_state, planning

            self._report_activity("Starting executor task", self._describe_task(planning.task))
            step = self.executor.begin(planning.task, live_state, follow_up_task=planning.follow_up_task)
            planning.executor_task = planning.task.model_copy(deep=True)
            if step.status == ExecutorStatus.STEPPING:
                planning.action = step.action
                planning.suggested_path = list(step.suggested_path)
                return live_state, planning
            if step.status == ExecutorStatus.INTERRUPTED:
                interrupt_planning = self._plan_action(live_state)
                interrupt_planning.executor_task = planning.executor_task
                if interrupt_planning.action is not None:
                    return live_state, interrupt_planning
                continue
            if step.status == ExecutorStatus.BLOCKED:
                if step.blocked_reason:
                    blocked_reasons.append(step.blocked_reason)
                self.executor.abort()
                continue

        fallback_state = latest_state or self.emulator.get_structured_state()
        blocked_reason = blocked_reasons[-1] if blocked_reasons else "planner retry budget exhausted"
        fallback_reason = f"planner retry exhaustion: {blocked_reason}"
        self._report_activity("Using fallback planner", fallback_reason)
        fallback = self.validator.fallback(fallback_state, self.stuck.state, fallback_reason)
        result = PlanningResult(
            action=self.validator.validate(fallback, fallback_state),
            raw_response=fallback_reason,
            used_fallback=True,
            planner_source="fallback",
        )
        return fallback_state, self._with_objective_planner_metadata(result)

    def _plan_action(self, state: StructuredGameState) -> PlanningResult:
        self._reset_objective_planner_metadata()
        if state.is_bootstrap():
            return self._plan_bootstrap_action(state)

        self._report_activity("Refreshing objectives", state.map_name)
        observe_state(self.memory.memory.long_term.world_map, state)
        self._ensure_objective_plan(state)
        self._sync_navigation_goal(state)
        return self._with_objective_planner_metadata(self.dispatcher.dispatch(state, self._build_turn_context(state)))

    def _reset_objective_planner_metadata(self) -> None:
        self._objective_plan_messages = []
        self._objective_plan_prompt_metrics = None
        self._objective_plan_llm_usage = None
        self._objective_plan_llm_attempted = False
        self._objective_plan_llm_model = None

    def _plan_bootstrap_action(self, state: StructuredGameState) -> PlanningResult:
        self._report_activity("Handling bootstrap", state.bootstrap_phase() or "deterministic startup")
        self.executor.abort()
        self.memory.memory.long_term.objective_plan = None
        action = self.validator.bootstrap(state, self.stuck.state, "deterministic startup bootstrap")
        return PlanningResult(action=self.validator.validate(action, state), planner_source="bootstrap")

    def _plan_overworld_mode(self, state: StructuredGameState, context: TurnContext) -> PlanningResult:
        self._report_activity("Building candidates", state.map_name)
        candidates = self._build_candidate_steps(state)
        return self._plan_candidates(state, candidates)

    def _plan_unknown_mode(self, state: StructuredGameState, context: TurnContext) -> PlanningResult:
        return self._plan_overworld_mode(state, context)

    def _plan_candidates(
        self,
        state: StructuredGameState,
        candidates: list[CandidateNextStep],
    ) -> PlanningResult:
        self._report_activity("Evaluating candidates", f"{len(candidates)} option(s)")
        if not candidates:
            fallback = self.validator.fallback(state, self.stuck.state, "fallback without usable candidates")
            result = PlanningResult(action=self.validator.validate(fallback, state), used_fallback=True, planner_source="fallback")
            self._clear_navigation_confirmation(state.map_name)
            return self._with_objective_planner_metadata(result)

        recommended = candidates[0]
        force_opening_objective_llm = self._should_force_opening_objective_llm(state)
        if self._should_auto_select(candidates) and not force_opening_objective_llm:
            self._report_activity("Auto-selecting candidate", recommended.id)
            compiled = self._compile_candidate(recommended, state, self._short_reason(recommended))
            if compiled is not None:
                return self._with_objective_planner_metadata(
                    self._planning_result_from_runtime(
                        compiled,
                        planner_source="auto_candidate",
                        objective_id=recommended.objective_id,
                        candidate_id=recommended.id,
                        candidates=list(candidates),
                    )
                )

        snapshot = self.context_manager.build_snapshot(
            state,
            self.memory.memory,
            self.stuck.state,
            candidate_next_steps=candidates,
            candidate_runtime=self._candidate_runtime,
        )
        self._report_activity("Building planner prompt", f"{len(candidates)} option(s)")
        messages = build_messages(snapshot)
        prompt_metrics = measure_prompt(messages, snapshot)

        if self.llm_client is None:
            self._report_activity("Using fallback planner", "LLM disabled")
            result = self._fallback_from_candidates(
                state,
                recommended,
                candidates=candidates,
                messages=messages,
                prompt_metrics=prompt_metrics,
                reason="deterministic fallback planner",
            )
            self._clear_navigation_confirmation(state.map_name)
            return self._with_objective_planner_metadata(result)

        try:
            raw_response = self._complete_with_window_pump(messages, purpose="turn planner")
            decision = self._parse_planner_decision(raw_response.content)
            chosen = self._resolve_candidate(decision, candidates) or recommended
            live_state = self.emulator.get_structured_state()
            compiled = self._compile_candidate(chosen, live_state, decision.reason or self._short_reason(chosen))
            if compiled is None:
                result = self._fallback_from_candidates(
                    live_state,
                    recommended,
                    candidates=candidates,
                    messages=messages,
                    prompt_metrics=prompt_metrics,
                    raw_response=raw_response.content,
                    llm_usage=raw_response.usage,
                    llm_model=raw_response.model,
                    llm_attempted=True,
                    reason="fallback after invalid planner choice",
                )
                self._clear_navigation_confirmation(state.map_name)
                return self._with_objective_planner_metadata(result)
            result = self._planning_result_from_runtime(
                compiled,
                raw_response=raw_response.content,
                planner_source="llm",
                messages=messages,
                prompt_metrics=prompt_metrics,
                llm_usage=raw_response.usage,
                llm_attempted=True,
                llm_model=raw_response.model,
                objective_id=chosen.objective_id,
                candidate_id=chosen.id,
                candidates=list(candidates),
            )
            self._clear_navigation_confirmation(state.map_name)
            return self._with_objective_planner_metadata(result)
        except Exception as exc:
            result = self._fallback_from_candidates(
                self.emulator.get_structured_state(),
                recommended,
                candidates=candidates,
                messages=messages,
                prompt_metrics=prompt_metrics,
                raw_response=str(exc),
                llm_attempted=True,
                reason=f"fallback after llm error: {type(exc).__name__}",
            )
            self._clear_navigation_confirmation(state.map_name)
            return self._with_objective_planner_metadata(result)

    def _current_milestone_for_state(self, state: StructuredGameState):
        return get_current_milestone(
            state.story_flags,
            [item.name for item in state.inventory],
            current_map_name=state.map_name,
            badges=state.badges,
        )

    def _ensure_objective_plan(self, state: StructuredGameState) -> None:
        reason = self._objective_plan_replan_reason(state)
        if reason is None:
            return
        self._report_activity("Refreshing objective plan", reason)
        milestone = self._current_milestone_for_state(state)
        existing_plan = self.memory.memory.long_term.objective_plan
        if (
            existing_plan is not None
            and existing_plan.internal_plan.plan_type == "recover"
            and reason in {"plan_complete", "recover_context_changed"}
        ):
            self.memory.memory.long_term.objective_plan = self._default_objective_plan(state, milestone, reason)
            return
        if self.llm_client is None:
            self.memory.memory.long_term.objective_plan = None
            return
        try:
            snapshot = self.context_manager.build_objective_snapshot(
                state,
                self.memory.memory,
                self.stuck.state,
                replan_reason=reason,
            )
            messages = build_messages(snapshot)
            self._objective_plan_messages = messages
            self._objective_plan_prompt_metrics = measure_prompt(messages, snapshot)
            self._objective_plan_llm_attempted = True
            response = self._complete_with_window_pump(messages, purpose="objective planner")
            self._objective_plan_llm_usage = response.usage
            self._objective_plan_llm_model = response.model
            plan = self._parse_objective_plan(response.content)
            plan = self._normalize_objective_plan(plan, state, milestone, reason)
            if not self._objective_plan_is_compilable(state, plan.internal_plan):
                raise ValueError("objective plan was not compilable")
            self.memory.memory.long_term.objective_plan = plan
        except Exception:
            self.memory.memory.long_term.objective_plan = None

    def _with_objective_planner_metadata(self, result: PlanningResult) -> PlanningResult:
        if result.llm_attempted or not self._objective_plan_llm_attempted:
            return result
        result.llm_attempted = True
        result.messages = list(self._objective_plan_messages)
        result.prompt_metrics = self._objective_plan_prompt_metrics
        result.llm_usage = self._objective_plan_llm_usage
        result.llm_model = self._objective_plan_llm_model
        return result

    def _normalize_objective_plan(
        self,
        plan: ObjectivePlanEnvelope,
        state: StructuredGameState,
        milestone,
        reason: str,
    ) -> ObjectivePlanEnvelope:
        defaults = self._default_objective_plan(state, milestone, reason)
        human = plan.human_plan
        if not human.short_term_goal.strip():
            human.short_term_goal = defaults.human_plan.short_term_goal
        if not human.mid_term_goal.strip():
            human.mid_term_goal = defaults.human_plan.mid_term_goal
        if not human.long_term_goal.strip():
            human.long_term_goal = defaults.human_plan.long_term_goal
        if not human.current_strategy.strip():
            human.current_strategy = defaults.human_plan.current_strategy

        internal = plan.internal_plan
        if not internal.plan_type:
            internal.plan_type = defaults.internal_plan.plan_type
        landmark = self.static_world_graph.get_landmark(internal.target_landmark_id)
        if internal.target_map_name is None and landmark is not None:
            internal.target_map_name = landmark.map_name
        if internal.plan_type in {"go_to_map", "go_to_landmark", "interact_story_npc"} and internal.target_map_name is None:
            internal.target_map_name = defaults.internal_plan.target_map_name
        plan.status = ObjectivePlanStatus.ACTIVE
        plan.generated_at_step = state.step
        plan.valid_for_milestone_id = milestone.id
        plan.valid_for_map_name = state.map_name
        plan.replan_reason = reason
        return plan

    def _default_objective_plan(self, state: StructuredGameState, milestone, reason: str) -> ObjectivePlanEnvelope:
        if state.battle_state is not None:
            plan_type = "battle_default"
            target_map_name = state.map_name
        elif state.menu_open:
            plan_type = "resolve_menu"
            target_map_name = state.map_name
        elif state.text_box_open:
            plan_type = "advance_dialogue"
            target_map_name = state.map_name
        else:
            target_map_name = self._current_story_target(state, milestone) or milestone.target_map_name
            if self._should_story_interact_here(state, milestone, target_map_name):
                plan_type = "interact_story_npc"
            elif self._story_landmark_for_milestone(milestone) is not None and map_matches(target_map_name, milestone.target_map_name):
                plan_type = "go_to_landmark"
            else:
                plan_type = "go_to_map"
        landmark = self._story_landmark_for_milestone(milestone) if plan_type == "go_to_landmark" else None
        return ObjectivePlanEnvelope(
            human_plan=HumanObjectivePlan(
                short_term_goal=(
                    "Advance the current dialogue and watch for walkthrough-relevant cues"
                    if state.text_box_open
                    else "Resolve the current menu and return to the story route"
                    if state.menu_open
                    else f"Finish the current battle so progress toward {milestone.target_map_name} can continue"
                    if state.battle_state is not None
                    else self.memory._short_term_goal(state, milestone)
                ),
                mid_term_goal=self.memory._mid_term_goal(state, milestone),
                long_term_goal=milestone.description,
                current_strategy=self.memory._current_strategy(state, milestone),
            ),
            internal_plan=InternalObjectivePlan(
                plan_type=plan_type,
                target_map_name=target_map_name,
                target_landmark_id=None if landmark is None else landmark.id,
                target_landmark_type=None if landmark is None else landmark.type,
                target_npc_hint="story_npc" if plan_type == "interact_story_npc" else None,
                success_signal=milestone.description,
                stop_when="text_box_open" if plan_type == "interact_story_npc" else None,
                confidence=0.5,
                notes=reason,
            ),
            generated_at_step=state.step,
            valid_for_milestone_id=milestone.id,
            valid_for_map_name=state.map_name,
            replan_reason=reason,
        )

    def _objective_plan_replan_reason(self, state: StructuredGameState) -> str | None:
        plan = self.memory.memory.long_term.objective_plan
        milestone = self._current_milestone_for_state(state)
        if plan is None:
            return "missing_plan"
        if plan.status != ObjectivePlanStatus.ACTIVE:
            return "inactive_plan"
        if plan.valid_for_milestone_id and plan.valid_for_milestone_id != milestone.id:
            return "milestone_changed"
        if state.battle_state is not None and plan.internal_plan.plan_type != "battle_default":
            return "battle_mode_changed"
        if state.menu_open and plan.internal_plan.plan_type != "resolve_menu":
            return "menu_mode_changed"
        if state.text_box_open and plan.internal_plan.plan_type != "advance_dialogue":
            return "text_mode_changed"
        if self.stuck.state.score >= self.stuck.threshold and plan.internal_plan.plan_type != "recover":
            return "stuck_recovery"
        if self._objective_plan_complete(state, plan.internal_plan):
            return "plan_complete"
        if (
            plan.internal_plan.plan_type == "recover"
            and plan.valid_for_map_name
            and not map_matches(state.map_name, plan.valid_for_map_name)
        ):
            return "recover_context_changed"
        if not self._objective_plan_is_compilable(state, plan.internal_plan):
            return "plan_uncompilable"
        return None

    def _objective_plan_complete(self, state: StructuredGameState, plan: InternalObjectivePlan) -> bool:
        if plan.plan_type == "recover" and plan.target_map_name:
            return map_matches(state.map_name, plan.target_map_name)
        if plan.plan_type == "go_to_map" and plan.target_map_name:
            return map_matches(state.map_name, plan.target_map_name)
        if plan.plan_type == "go_to_landmark":
            if plan.target_landmark_id:
                landmark = self.static_world_graph.get_landmark(plan.target_landmark_id)
                if landmark is not None:
                    return map_matches(state.map_name, landmark.map_name)
            return bool(plan.target_map_name and map_matches(state.map_name, plan.target_map_name))
        if plan.plan_type == "interact_story_npc":
            return state.text_box_open
        if plan.plan_type == "advance_dialogue":
            return not state.text_box_open
        if plan.plan_type == "resolve_menu":
            return not state.menu_open
        if plan.plan_type == "battle_default":
            return state.battle_state is None
        return False

    def _objective_plan_is_compilable(self, state: StructuredGameState, plan: InternalObjectivePlan) -> bool:
        if plan.plan_type in {"advance_dialogue", "resolve_menu", "battle_default", "recover"}:
            return True
        if plan.plan_type == "interact_story_npc":
            return bool(plan.target_map_name or state.map_name)
        if plan.plan_type == "go_to_map":
            return bool(plan.target_map_name)
        if plan.plan_type == "go_to_landmark":
            if plan.target_landmark_id is not None:
                return self.static_world_graph.get_landmark(plan.target_landmark_id) is not None
            return bool(plan.target_map_name)
        return False

    def _fallback_from_candidates(
        self,
        state: StructuredGameState,
        recommended: CandidateNextStep,
        *,
        candidates: list[CandidateNextStep] | None = None,
        messages: list[dict],
        prompt_metrics: dict[str, Any],
        raw_response: str | None = None,
        llm_usage: LLMUsage | None = None,
        llm_model: str | None = None,
        llm_attempted: bool = False,
        reason: str,
    ) -> PlanningResult:
        compiled = self._compile_candidate(recommended, state, self._short_reason(recommended))
        if compiled is None:
            fallback = self.validator.fallback(state, self.stuck.state, reason)
            compiled_runtime = CandidateRuntime(action=self.validator.validate(fallback, state))
            objective_id = None
            candidate_id = None
        else:
            compiled_runtime = compiled
            objective_id = recommended.objective_id
            candidate_id = recommended.id
        return self._planning_result_from_runtime(
            compiled_runtime,
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
            candidates=list(candidates or [recommended]),
        )

    def _complete_with_window_pump(self, messages: list[dict], *, purpose: str) -> CompletionResponse:
        emulator = self.emulator
        assert self.llm_client is not None
        self._clear_llm_request_status()
        self._report_activity("Preparing LLM request", f"{purpose} ({len(messages)} message(s))")
        emulator.begin_planning_wait()
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(self.llm_client.complete, messages)
                last_status: tuple[str, str | None] | None = None
                while not future.done():
                    emulator.pump_planning_wait()
                    request_status = self._snapshot_llm_request_status()
                    status = self._format_llm_activity(request_status, purpose)
                    if status != last_status:
                        self._report_activity(*status)
                        last_status = status
                request_status = self._snapshot_llm_request_status()
                status = self._format_llm_activity(request_status, purpose)
                if status != last_status:
                    self._report_activity(*status)
                if future.exception() is None:
                    self._report_activity("Parsing LLM response", purpose)
                return future.result()
        finally:
            self._clear_llm_request_status()
            emulator.end_planning_wait()

    @staticmethod
    def _describe_task(task: Task | None) -> str | None:
        if task is None:
            return None
        if task.reason:
            return f"{task.kind.value}: {task.reason}"
        return task.kind.value

    def _clear_llm_request_status(self) -> None:
        if self.llm_client is None:
            return
        clear = getattr(self.llm_client, "clear_request_status", None)
        if callable(clear):
            clear()

    def _snapshot_llm_request_status(self) -> RequestStatus | None:
        if self.llm_client is None:
            return None
        snapshot = getattr(self.llm_client, "snapshot_request_status", None)
        if not callable(snapshot):
            return None
        return snapshot()

    @staticmethod
    def _format_llm_activity(status: RequestStatus | None, purpose: str) -> tuple[str, str | None]:
        if status is None:
            return ("Waiting for LLM response", purpose)
        if status.phase == "Preparing request":
            return ("Preparing LLM request", f"{purpose} | {status.detail}")
        if status.phase == "Waiting for rate limit":
            return ("Waiting for LLM rate limit", f"{purpose} | {status.detail}")
        if status.phase == "Sending request":
            return ("Waiting for LLM response", f"{purpose} | model {status.detail}")
        if status.phase == "Received response":
            return ("Parsing LLM response", f"{purpose} | {status.detail}")
        if status.phase == "Retrying after error":
            return ("Retrying LLM request", f"{purpose} | {status.detail}")
        if status.phase == "Request failed":
            return ("LLM request failed", f"{purpose} | {status.detail}")
        return (status.phase, status.detail)

    def save_checkpoint(self, directory: str | Path) -> Path:
        checkpoint_dir = Path(directory)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        emulator_state_path = checkpoint_dir / "emulator.state"
        metadata_path = checkpoint_dir / "checkpoint.json"
        self.emulator.save_state(emulator_state_path)
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
            "executor_state": self.executor.export_state(),
            "navigator_state": self.navigator.export_state(),
            "overworld_state": self.overworld_controller.export_state(),
            "objective_plan": (
                self.memory.memory.long_term.objective_plan.model_dump(mode="json")
                if self.memory.memory.long_term.objective_plan is not None
                else None
            ),
            "last_entry_direction": self._last_entry_direction,
        }
        metadata_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return metadata_path

    def load_checkpoint(self, directory: str | Path) -> dict:
        checkpoint_dir = Path(directory)
        metadata_path = checkpoint_dir / "checkpoint.json"
        emulator_state_path = checkpoint_dir / "emulator.state"
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        self.emulator.load_state(emulator_state_path)
        self.memory.memory = MemoryState.model_validate(payload["memory"])
        self._rebuild_overworld_stack()
        self.dispatcher = self._build_mode_dispatcher()
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
            llm_calls=telemetry.get("llm_calls", 0),
            auto_selected_turns=telemetry.get("auto_selected_turns", 0),
            executor_turns=telemetry.get("executor_turns", 0),
            major_progress_turns=telemetry.get("major_progress_turns", 0),
            no_effect_turns=telemetry.get("no_effect_turns", 0),
            objective_switches=telemetry.get("objective_switches", 0),
        )
        self.completed_turns = int(payload.get("completed_turns", 0))
        self.context_manager.restore_state(payload.get("context_state"))
        self.executor.restore_state(payload.get("executor_state"))
        self.navigator.restore_state(payload.get("navigator_state"))
        self.overworld_controller.restore_state(payload.get("overworld_state"))
        objective_payload = payload.get("objective_plan")
        self.memory.memory.long_term.objective_plan = (
            ObjectivePlanEnvelope.model_validate(objective_payload) if objective_payload else None
        )
        self._last_entry_direction = payload.get("last_entry_direction")
        return payload

    def summary(self) -> dict[str, Any]:
        summary = self.telemetry.to_dict()
        summary.update(world_map_stats(self.memory.memory.long_term.world_map))
        goals = self.memory.memory.goals
        objective_plan = self.memory.memory.long_term.objective_plan
        if objective_plan is not None:
            summary.update(
                {
                    "short_term_goal": objective_plan.human_plan.short_term_goal,
                    "mid_term_goal": objective_plan.human_plan.mid_term_goal,
                    "long_term_goal": objective_plan.human_plan.long_term_goal,
                    "current_strategy": objective_plan.human_plan.current_strategy,
                }
            )
        else:
            summary.update(
                {
                    "short_term_goal": goals.short_term_goal,
                    "mid_term_goal": goals.mid_term_goal,
                    "long_term_goal": goals.long_term_goal,
                    "current_strategy": goals.current_strategy,
                }
            )
        summary.update(self._summary_pathfinding_route())
        return summary

    def _summary_pathfinding_route(self) -> dict[str, Any]:
        goal = self.memory.memory.long_term.navigation_goal
        if goal is None:
            return {}
        current_symbol = self.static_world_graph.canonical_symbol(goal.current_map_name) or goal.current_map_name
        target_symbol = self.static_world_graph.canonical_symbol(goal.target_map_name) or goal.target_map_name
        final_target_map_name = goal.final_target_map_name or goal.target_map_name
        final_target_symbol = self.static_world_graph.canonical_symbol(final_target_map_name) or final_target_map_name
        next_symbol = self.static_world_graph.canonical_symbol(goal.next_map_name) or goal.next_map_name
        summary: dict[str, Any] = {
            "pathfinding_route": [item for item in [current_symbol] if item],
            "pathfinding_route_available": False,
            "pathfinding_target_symbol": target_symbol,
            "pathfinding_final_target_symbol": final_target_symbol,
            "pathfinding_next_symbol": next_symbol,
            "pathfinding_next_hop_kind": goal.next_hop_kind,
        }
        if goal.current_map_name is None or goal.target_map_name is None:
            return summary
        route = self.static_world_graph.find_route(goal.current_map_name, goal.target_map_name)
        if route is None:
            return summary
        summary["pathfinding_route"] = route.summary()
        summary["pathfinding_route_available"] = True
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

    def _parse_objective_plan(self, raw_text: str) -> ObjectivePlanEnvelope:
        payload = json.loads(self._extract_json(raw_text))
        return ObjectivePlanEnvelope.model_validate(payload)

    def _resolve_candidate(
        self,
        decision: PlannerDecision,
        candidates: list[CandidateNextStep],
    ) -> CandidateNextStep | None:
        for candidate in candidates:
            if candidate.id == decision.candidate_id:
                return candidate
        return None

    def _build_candidate_steps(self, state: StructuredGameState) -> list[CandidateNextStep]:
        self._candidate_runtime = {}
        short_objective_id = self._objective_id(ObjectiveHorizon.SHORT_TERM, "short_immediate_step")
        navigation_goal = self._sync_navigation_goal(state)
        objective_id = self._objective_id_for_goal(navigation_goal, short_objective_id)
        candidates: list[CandidateNextStep] = []

        candidates.extend(self.menu_manager.build_candidates(state, objective_id))
        self._candidate_runtime.update(self.menu_manager.runtime_map())

        if navigation_goal is not None and navigation_goal.engine_mode == "progression":
            candidates.extend(self._build_progression_candidates(state, objective_id, navigation_goal))
        else:
            candidates.extend(self._build_exploration_candidates(state, objective_id, navigation_goal))

        if not candidates:
            probe_action = self.validator.fallback(state, self.stuck.state, "probe nearby state")
            candidate = CandidateNextStep(
                id=f"probe_{probe_action.action.value.lower()}",
                type="SAFE_STEP",
                why="Probe a safe nearby step while no stronger target is available.",
                priority=25,
                expected_success_signal="Position changes or UI opens",
                objective_id=objective_id,
            )
            self._candidate_runtime[candidate.id] = CandidateRuntime(action=probe_action)
            candidates.append(candidate)

        deduped = self._dedupe_candidates(candidates)
        deduped.sort(key=lambda item: (-item.priority, item.id))
        return self._trim_candidate_runtime(deduped[:4])

    def _objective_id(self, horizon: ObjectiveHorizon, default: str) -> str:
        for objective in self.memory.memory.goals.active_objectives:
            if objective.horizon == horizon:
                return objective.id
        return default

    def _objective_id_for_goal(self, goal: NavigationGoal | None, fallback: str) -> str:
        if goal is None or not goal.objective_kind:
            return fallback
        return f"local_{goal.objective_kind}"

    def _build_progression_candidates(
        self,
        state: StructuredGameState,
        objective_id: str,
        goal: NavigationGoal,
    ) -> list[CandidateNextStep]:
        candidates: list[CandidateNextStep] = []
        if goal.objective_kind == "talk_to_required_npc":
            # Offer all reachable NPCs so the LLM can pick which to try.
            # After a failed interaction the cooldown filters that NPC out,
            # naturally cycling through the rest on subsequent turns.
            npc_candidates = self._build_all_npc_candidates(state, objective_id, goal, priority_boost=14)
            if npc_candidates:
                candidates.extend(npc_candidates)
            else:
                # No confirmed NPCs reachable — fall back to heuristic blockers
                interaction = self._candidate_for_adjacent_interactable(state, objective_id, priority_boost=14)
                if interaction is not None:
                    candidates.append(interaction)
                move_to_interaction = self._candidate_to_move_adjacent_to_interactable(state, objective_id, priority_boost=14)
                if move_to_interaction is not None:
                    candidates.append(move_to_interaction)
            if not candidates:
                explore = self._build_room_exploration_candidate(state, objective_id)
                if explore is not None:
                    candidates.append(explore)
            return candidates

        if goal.next_hop_kind == "boundary" and goal.next_hop_side:
            boundary = self._build_boundary_progress_candidate(state, objective_id, goal)
            if boundary is not None:
                candidates.append(boundary)
                return candidates

        connector_candidates = self._build_connector_candidates(state, objective_id, goal, prefer_story_matches=False)
        if connector_candidates:
            return connector_candidates

        exploration = self._build_window_candidate(state, objective_id, goal)
        if exploration is not None:
            candidates.append(exploration)
        return candidates

    def _build_exploration_candidates(
        self,
        state: StructuredGameState,
        objective_id: str,
        goal: NavigationGoal | None,
    ) -> list[CandidateNextStep]:
        candidates: list[CandidateNextStep] = []
        candidates.extend(self._build_connector_candidates(state, objective_id, goal, prefer_story_matches=True))

        interaction_priority = 0
        if goal is None or goal.objective_kind == "talk_to_required_npc" or map_matches(state.map_name, goal.target_map_name):
            interaction_priority = 10

        adjacent_interaction = self._candidate_for_adjacent_interactable(state, objective_id, priority_boost=interaction_priority)
        if adjacent_interaction is not None:
            candidates.append(adjacent_interaction)

        move_to_interaction = self._candidate_to_move_adjacent_to_interactable(state, objective_id, priority_boost=interaction_priority)
        if move_to_interaction is not None:
            candidates.append(move_to_interaction)

        frontier = self._build_window_candidate(state, objective_id, goal)
        if frontier is not None:
            candidates.append(frontier)
        return candidates

    def _build_boundary_progress_candidate(
        self,
        state: StructuredGameState,
        objective_id: str,
        goal: NavigationGoal,
    ) -> CandidateNextStep | None:
        if state.navigation is None or state.x is None or state.y is None or goal.next_hop_side is None:
            return None
        target = self._best_directional_target(state, goal.next_hop_side)
        if target is None:
            return None
        target_x, target_y, distance, at_real_edge = target
        move_action = self._action_for_side(goal.next_hop_side)
        if move_action is None:
            return None
        backtrack = goal.next_hop_side == self._last_entry_direction
        blacklisted = goal.next_hop_side in goal.failed_sides
        priority = 94 if at_real_edge else 82
        if backtrack:
            priority -= 18
        if blacklisted:
            priority -= 35
        candidate = CandidateNextStep(
            id=f"boundary_{goal.next_hop_side}_{target_x}_{target_y}",
            type="ADVANCE_TOWARD_BOUNDARY",
            target=ObjectiveTarget(
                kind="exit",
                map_id=state.map_id,
                map_name=state.map_name,
                x=target_x,
                y=target_y,
                detail=goal.next_hop_side,
            ),
            why=(
                f"Advance {goal.next_hop_side} toward {goal.next_map_name}."
                if at_real_edge
                else f"Move {goal.next_hop_side} to reveal the real map edge."
            ),
            priority=priority,
            expected_success_signal="The visible window expands or the map changes",
            objective_id=objective_id,
            distance=distance,
            advances_target=True,
            backtrack=backtrack,
            blacklisted=blacklisted,
        )
        self._candidate_runtime[candidate.id] = CandidateRuntime(
            task=Task(kind=TaskKind.WALK_BOUNDARY, direction=goal.next_hop_side, reason=candidate.why),
        )
        return candidate

    def _build_connector_candidates(
        self,
        state: StructuredGameState,
        objective_id: str,
        goal: NavigationGoal | None,
        *,
        prefer_story_matches: bool,
    ) -> list[CandidateNextStep]:
        if state.navigation is None or state.x is None or state.y is None:
            return []
        world_map = self.memory.memory.long_term.world_map
        story_text = self._milestone_story_text(state)
        preferred_destination = goal.next_map_name if goal is not None else self._preferred_destination_from_milestone(state)
        ranked: list[tuple[tuple[int, int, int, int, str], CandidateNextStep, CandidateRuntime]] = []
        seen_connector_keys: set[tuple[int | None, int | None, str | None, str]] = set()
        for connector in connectors_from_map(world_map, state.map_name, confirmed_only=False):
            approach = self._approach_for_connector(state, connector)
            if approach is None:
                continue
            approach_x, approach_y, follow_up_action, distance = approach
            destination_matches = bool(
                preferred_destination
                and connector.destination_map
                and map_matches(connector.destination_map, preferred_destination)
            )
            story_score = self._map_text_score(connector.destination_map or "", story_text)
            backtrack = bool(self._last_entry_direction and connector.source_side == self._last_entry_direction)
            blacklisted = bool(goal and connector.id in goal.failed_connector_ids)
            if goal and goal.next_hop_kind == "warp" and preferred_destination and connector.destination_map and not destination_matches:
                if not prefer_story_matches or story_score == 0:
                    continue
            priority = 90 if destination_matches else 82 if connector.status.value == "confirmed" else 74
            if connector.source_side is not None:
                priority += 8
            if connector.kind in {"boundary", "door"}:
                priority += 4
            elif connector.kind == "warp" and connector.destination_map is None:
                priority -= 8
            if story_score > 0:
                priority += min(10, story_score * 2)
            if connector.id == (goal.target_connector_id if goal else None):
                priority += 8
            if backtrack:
                priority -= 14
            if blacklisted:
                priority -= 35
            slug = self._slugify(connector.id)
            candidate = CandidateNextStep(
                id=f"connector_{slug}",
                type="ENTER_CONNECTOR" if connector.status.value == "confirmed" else "EXPLORE_CONNECTOR",
                target=ObjectiveTarget(
                    kind="connector",
                    map_id=state.map_id,
                    map_name=state.map_name,
                    x=approach_x,
                    y=approach_y,
                    detail=describe_connector(connector),
                ),
                why=(
                    f"Use the confirmed connector toward {connector.destination_map}."
                    if connector.status.value == "confirmed"
                    else "Test a suspected connector by walking into it."
                ),
                priority=priority,
                expected_success_signal="The map changes or a new local affordance appears",
                objective_id=objective_id,
                distance=distance,
                advances_target=destination_matches or story_score > 0,
                backtrack=backtrack,
                blacklisted=blacklisted,
            )
            runtime = CandidateRuntime(
                task=Task(
                    kind=TaskKind.ENTER_CONNECTOR,
                    connector_id=connector.id,
                    target_x=connector.source_x,
                    target_y=connector.source_y,
                    reason=candidate.why,
                ),
            )
            seen_connector_keys.add((connector.source_x, connector.source_y, connector.destination_map, connector.kind))
            ranked.append(
                (
                    (
                        1 if blacklisted else 0,
                        -priority,
                        distance,
                        0 if destination_matches else 1,
                        candidate.id,
                    ),
                    candidate,
                    runtime,
                )
            )
        ranked.extend(
            self._build_static_connector_candidates(
                state,
                objective_id,
                goal,
                preferred_destination,
                story_text,
                prefer_story_matches=prefer_story_matches,
                seen_connector_keys=seen_connector_keys,
            )
        )
        ranked.sort(key=lambda item: item[0])
        results: list[CandidateNextStep] = []
        for _sort_key, candidate, runtime in ranked[:3]:
            self._candidate_runtime[candidate.id] = runtime
            results.append(candidate)
        return results

    def _build_static_connector_candidates(
        self,
        state: StructuredGameState,
        objective_id: str,
        goal: NavigationGoal | None,
        preferred_destination: str | None,
        story_text: str,
        *,
        prefer_story_matches: bool,
        seen_connector_keys: set[tuple[int | None, int | None, str | None, str]],
    ) -> list[tuple[tuple[int, int, int, int, str], CandidateNextStep, CandidateRuntime]]:
        ranked: list[tuple[tuple[int, int, int, int, str], CandidateNextStep, CandidateRuntime]] = []
        preferred_landmark = None if goal is None else self.static_world_graph.get_landmark(goal.target_landmark_id)
        if preferred_landmark is None and goal is not None and goal.next_hop_kind == "warp" and goal.next_map_name:
            preferred_landmark = self._landmark_for_destination_on_current_map(state.map_name, goal.next_map_name)
        if preferred_landmark is not None and preferred_landmark.map_symbol == self.static_world_graph.canonical_symbol(state.map_name):
            static_entry = self._build_static_landmark_candidate(state, objective_id, preferred_landmark)
            if static_entry is not None:
                ranked.append(static_entry)
            else:
                landmark_progress = self._build_landmark_window_candidate(state, objective_id, preferred_landmark)
                if landmark_progress is not None:
                    ranked.append(landmark_progress)

        for edge in self.static_world_graph.neighbors(state.map_name):
            if edge.kind != "warp" or edge.destination_name is None or edge.destination_symbol is None:
                continue
            connector_key = (edge.x, edge.y, edge.destination_name, "warp")
            if connector_key in seen_connector_keys:
                continue
            if preferred_landmark is not None and preferred_landmark.x == edge.x and preferred_landmark.y == edge.y:
                continue
            destination_matches = bool(preferred_destination and map_matches(edge.destination_name, preferred_destination))
            story_score = self._map_text_score(edge.destination_name, story_text)
            if goal and goal.next_hop_kind == "warp" and preferred_destination and not destination_matches:
                if not prefer_story_matches or story_score == 0:
                    continue
            if not destination_matches and story_score == 0 and not prefer_story_matches and goal is not None and goal.next_hop_kind == "warp":
                continue
            static_connector = self._synthesize_static_connector(edge)
            if static_connector is None:
                continue
            approach = self._approach_for_connector(state, static_connector)
            if approach is None:
                # Approach blocked — if the player is already very close,
                # generate a low-priority nudge candidate that moves toward the
                # warp tile so the navigation window shifts to reveal a valid
                # approach on the next turn.
                if edge.x is not None and edge.y is not None and state.x is not None and state.y is not None:
                    if abs(state.x - edge.x) + abs(state.y - edge.y) <= 3:
                        nudge = self._build_nudge_toward_warp(state, edge, objective_id, destination_matches)
                        if nudge is not None:
                            ranked.append(nudge)
                continue
            approach_x, approach_y, follow_up_action, distance = approach
            priority = 84 if destination_matches else 72
            if story_score > 0:
                priority += min(10, story_score * 2)
            candidate = CandidateNextStep(
                id=f"static_warp_{self._slugify(edge.destination_symbol)}_{edge.x}_{edge.y}",
                type="ENTER_CONNECTOR",
                target=ObjectiveTarget(
                    kind="connector",
                    map_id=state.map_id,
                    map_name=state.map_name,
                    x=approach_x,
                    y=approach_y,
                    detail=f"canonical warp to {edge.destination_name}",
                ),
                why=f"Use the canonical warp to {edge.destination_name}.",
                priority=priority,
                expected_success_signal="The map changes to the expected destination",
                objective_id=objective_id,
                distance=distance,
                advances_target=destination_matches or story_score > 0,
            )
            runtime = CandidateRuntime(
                task=Task(
                    kind=TaskKind.ENTER_CONNECTOR,
                    connector_id=self._static_connector_id(edge),
                    target_x=edge.x,
                    target_y=edge.y,
                    reason=candidate.why,
                ),
            )
            ranked.append(
                (
                    (
                        0,
                        -priority,
                        distance,
                        0 if destination_matches else 1,
                        candidate.id,
                    ),
                    candidate,
                    runtime,
                )
            )
        return ranked

    def _build_static_landmark_candidate(
        self,
        state: StructuredGameState,
        objective_id: str,
        landmark: Landmark,
    ) -> tuple[tuple[int, int, int, int, str], CandidateNextStep, CandidateRuntime] | None:
        if landmark.x is None or landmark.y is None:
            return None
        approach = self._approach_for_transition_tile(state, landmark.x, landmark.y)
        if approach is None:
            return None
        approach_x, approach_y, follow_up_action, distance = approach
        candidate = CandidateNextStep(
            id=f"landmark_{landmark.id}",
            type="ENTER_CONNECTOR",
            target=ObjectiveTarget(
                kind="landmark",
                map_id=state.map_id,
                map_name=state.map_name,
                x=approach_x,
                y=approach_y,
                detail=landmark.id,
            ),
            why=f"Head for the canonical {landmark.type} landmark: {landmark.label}.",
            priority=96,
            expected_success_signal="The target building or entrance is reached",
            objective_id=objective_id,
            distance=distance,
            advances_target=True,
        )
        runtime = CandidateRuntime(
            task=Task(
                kind=TaskKind.ENTER_CONNECTOR,
                connector_id=f"landmark:{landmark.id}",
                target_x=landmark.x,
                target_y=landmark.y,
                reason=candidate.why,
            ),
        )
        return (
            (
                0,
                -candidate.priority,
                distance,
                0,
                candidate.id,
            ),
            candidate,
            runtime,
        )

    def _planning_navigation_grid(self, state: StructuredGameState) -> NavigationGrid | None:
        if state.navigation is None:
            return None
        grid = NavigationGrid(state.navigation)
        for blocked_x, blocked_y in self._temporary_blocked_tiles_for_state(state):
            grid.mark_blocked(blocked_x, blocked_y)
        return grid

    def _suggested_path_for_action(
        self,
        state: StructuredGameState,
        action: ActionDecision | None,
    ) -> list[tuple[int, int]]:
        if action is None or state.x is None or state.y is None:
            return []
        if action.action not in {
            ActionType.MOVE_UP,
            ActionType.MOVE_RIGHT,
            ActionType.MOVE_DOWN,
            ActionType.MOVE_LEFT,
        }:
            return []
        next_position = advance_position(state.x, state.y, action.action)
        return [(next_position.x, next_position.y)]

    def _planning_find_path(
        self,
        state: StructuredGameState,
        target_x: int,
        target_y: int,
        *,
        nav_grid: NavigationGrid | None = None,
    ) -> list[ActionType] | None:
        if state.x is None or state.y is None:
            return None
        nav_grid = nav_grid or self._planning_navigation_grid(state)
        if nav_grid is None:
            return None
        return nav_grid.find_path(state.x, state.y, target_x, target_y)

    def _build_landmark_window_candidate(
        self,
        state: StructuredGameState,
        objective_id: str,
        landmark: Landmark,
    ) -> tuple[tuple[int, int, int, int, str], CandidateNextStep, CandidateRuntime] | None:
        if state.navigation is None or state.x is None or state.y is None:
            return None
        if landmark.x is None or landmark.y is None:
            return None
        if state.navigation.min_x <= landmark.x <= state.navigation.max_x and state.navigation.min_y <= landmark.y <= state.navigation.max_y:
            return None

        nav_grid = self._planning_navigation_grid(state)
        if nav_grid is None:
            return None

        preferred_sides: list[str] = []
        if landmark.x < state.navigation.min_x:
            preferred_sides.append("west")
        elif landmark.x > state.navigation.max_x:
            preferred_sides.append("east")
        if landmark.y < state.navigation.min_y:
            preferred_sides.append("north")
        elif landmark.y > state.navigation.max_y:
            preferred_sides.append("south")

        best: tuple[tuple[int, int, int, int, int, str], tuple[int, int, int, str]] | None = None
        for coordinate in state.navigation.walkable:
            if (coordinate.x, coordinate.y) == (state.x, state.y):
                continue
            if not nav_grid.is_walkable(coordinate.x, coordinate.y):
                continue
            visible_side = visible_boundary_side(state.navigation, coordinate.x, coordinate.y)
            if visible_side is None:
                continue
            route = self._planning_find_path(state, coordinate.x, coordinate.y, nav_grid=nav_grid)
            if route is None or len(route) == 0:
                continue
            if preferred_sides and not any(self._moves_toward_side(state, coordinate.x, coordinate.y, side) for side in preferred_sides):
                continue
            remaining_distance = abs(landmark.x - coordinate.x) + abs(landmark.y - coordinate.y)
            rank = (
                0 if preferred_sides and visible_side in preferred_sides else 1,
                0 if is_real_map_edge(state.navigation, visible_side) else 1,
                remaining_distance,
                len(route),
                coordinate.y,
                coordinate.x,
            )
            payload = (coordinate.x, coordinate.y, len(route), visible_side)
            if best is None or rank < best[0]:
                best = (rank, payload)

        if best is None:
            return None

        target_x, target_y, distance, visible_side = best[1]
        candidate = CandidateNextStep(
            id=f"landmark_window_{landmark.id}_{target_x}_{target_y}",
            type="EXPLORE_WINDOW",
            target=ObjectiveTarget(
                kind="landmark",
                map_id=state.map_id,
                map_name=state.map_name,
                x=target_x,
                y=target_y,
                detail=landmark.id,
            ),
            why=f"Move toward the canonical {landmark.label} entrance to reveal its connector.",
            priority=88,
            expected_success_signal="The target landmark or its entrance becomes visible",
            objective_id=objective_id,
            distance=distance,
            advances_target=True,
        )
        runtime = CandidateRuntime(
            task=Task(kind=TaskKind.NAVIGATE_TO, target_x=target_x, target_y=target_y, reason=candidate.why),
        )
        return (
            (
                0,
                -candidate.priority,
                distance,
                0,
                candidate.id,
            ),
            candidate,
            runtime,
        )

    def _build_window_candidate(
        self,
        state: StructuredGameState,
        objective_id: str,
        goal: NavigationGoal | None,
    ) -> CandidateNextStep | None:
        if state.navigation is None or state.x is None or state.y is None:
            return None
        preferred_side = goal.next_hop_side if goal is not None else None
        target = self._best_directional_target(state, preferred_side, require_real_edge=False)
        if target is None:
            return None
        target_x, target_y, distance, _at_real_edge = target
        candidate = CandidateNextStep(
            id=f"window_{preferred_side or 'scan'}_{target_x}_{target_y}",
            type="EXPLORE_WINDOW",
            target=ObjectiveTarget(
                kind="frontier",
                map_id=state.map_id,
                map_name=state.map_name,
                x=target_x,
                y=target_y,
                detail=preferred_side or "nearest frontier",
            ),
            why=(
                f"Move toward the {preferred_side} edge to reveal more of the map."
                if preferred_side
                else "Move to a nearby frontier tile to reveal better options."
            ),
            priority=56 - min(distance, 8),
            expected_success_signal="New connectors or boundary tiles become visible",
            objective_id=objective_id,
            distance=distance,
            advances_target=preferred_side is not None,
        )
        self._candidate_runtime[candidate.id] = CandidateRuntime(
            task=Task(kind=TaskKind.NAVIGATE_TO, target_x=target_x, target_y=target_y, reason=candidate.why),
        )
        return candidate

    def _best_directional_target(
        self,
        state: StructuredGameState,
        side: str | None,
        *,
        require_real_edge: bool = False,
    ) -> tuple[int, int, int, bool] | None:
        assert state.navigation is not None and state.x is not None and state.y is not None
        nav_grid = self._planning_navigation_grid(state)
        if nav_grid is None:
            return None
        best: tuple[tuple[int, int, int], tuple[int, int, int, bool]] | None = None
        for coordinate in state.navigation.walkable:
            if (coordinate.x, coordinate.y) == (state.x, state.y):
                continue
            if not nav_grid.is_walkable(coordinate.x, coordinate.y):
                continue
            visible_side = visible_boundary_side(state.navigation, coordinate.x, coordinate.y)
            if visible_side is None:
                continue
            if side is not None and visible_side != side:
                if not require_real_edge:
                    if not self._moves_toward_side(state, coordinate.x, coordinate.y, side):
                        continue
                else:
                    continue
            route = self._planning_find_path(state, coordinate.x, coordinate.y, nav_grid=nav_grid)
            if route is None:
                continue
            distance = len(route)
            if distance == 0 and side is None:
                continue
            at_real_edge = bool(visible_side and is_real_map_edge(state.navigation, visible_side))
            if require_real_edge and side is not None and (visible_side != side or not at_real_edge):
                continue
            rank = self._directional_rank(side, coordinate.x, coordinate.y, distance, at_real_edge)
            payload = (coordinate.x, coordinate.y, distance, at_real_edge)
            if best is None or rank < best[0]:
                best = (rank, payload)
        return None if best is None else best[1]

    def _directional_rank(
        self,
        side: str | None,
        x: int,
        y: int,
        distance: int,
        at_real_edge: bool,
    ) -> tuple[int, int, int]:
        if side == "north":
            return (0 if at_real_edge else 1, y, distance)
        if side == "east":
            return (0 if at_real_edge else 1, -x, distance)
        if side == "south":
            return (0 if at_real_edge else 1, -y, distance)
        if side == "west":
            return (0 if at_real_edge else 1, x, distance)
        return (0, distance, x + y)

    def _moves_toward_side(self, state: StructuredGameState, x: int, y: int, side: str) -> bool:
        if side == "north":
            return y <= (state.y or 0)
        if side == "east":
            return x >= (state.x or 0)
        if side == "south":
            return y >= (state.y or 0)
        if side == "west":
            return x <= (state.x or 0)
        return True

    def _approach_for_connector(
        self,
        state: StructuredGameState,
        connector,
    ) -> tuple[int, int, ActionType, int] | None:
        if state.navigation is None or state.x is None or state.y is None:
            return None
        nav_grid = self._planning_navigation_grid(state)
        if nav_grid is None:
            return None
        if connector.approach_x is not None and connector.approach_y is not None and connector.transition_action is not None:
            route = self._planning_find_path(
                state,
                connector.approach_x,
                connector.approach_y,
                nav_grid=nav_grid,
            )
            if route is not None:
                return connector.approach_x, connector.approach_y, connector.transition_action, len(route)
        if connector.source_x is None or connector.source_y is None:
            return None
        return self._approach_for_transition_tile(state, connector.source_x, connector.source_y, nav_grid=nav_grid)

    def _approach_for_transition_tile(
        self,
        state: StructuredGameState,
        source_x: int | None,
        source_y: int | None,
        *,
        nav_grid: NavigationGrid | None = None,
    ) -> tuple[int, int, ActionType, int] | None:
        if state.navigation is None or state.x is None or state.y is None:
            return None
        if source_x is None or source_y is None:
            return None
        nav_grid = nav_grid or self._planning_navigation_grid(state)
        if nav_grid is None:
            return None
        # When the warp tile itself is walkable (e.g. a door mat or stair
        # landing), we can navigate directly onto it — stepping on it triggers
        # the warp. South-edge exits benefit from preferring a downward entry,
        # but interior staircases often sit on east/west room boundaries, so
        # keep those free to use the shortest adjacent approach instead.
        source_is_walkable = nav_grid.is_walkable(source_x, source_y)
        prefer_downward_entry = bool(source_is_walkable and source_y >= state.navigation.max_y)
        best: tuple[tuple[int, int, str], tuple[int, int, ActionType, int]] | None = None
        for move_action, dx, dy in (
            (ActionType.MOVE_UP, 0, -1),
            (ActionType.MOVE_RIGHT, 1, 0),
            (ActionType.MOVE_DOWN, 0, 1),
            (ActionType.MOVE_LEFT, -1, 0),
        ):
            approach_x = source_x - dx
            approach_y = source_y - dy
            if not nav_grid.is_walkable(approach_x, approach_y):
                continue
            route = self._planning_find_path(state, approach_x, approach_y, nav_grid=nav_grid)
            if route is None:
                continue
            distance = len(route)
            direction_penalty = 0
            if prefer_downward_entry and move_action != ActionType.MOVE_DOWN:
                direction_penalty = 100
            payload = (approach_x, approach_y, move_action, distance)
            rank = (distance + direction_penalty, direction_penalty, move_action.value)
            if best is None or rank < best[0]:
                best = (rank, payload)

        if best is not None:
            return best[1]

        # No adjacent approach tile is reachable — but if the warp tile itself
        # is walkable (e.g. a staircase landing) we can navigate directly onto
        # it.  The last move action in the route to the tile acts as the
        # follow-up, which is all that is needed to trigger the warp.
        if source_is_walkable:
            route = self._planning_find_path(state, source_x, source_y, nav_grid=nav_grid)
            if route is not None and len(route) > 0:
                return (source_x, source_y, route[-1], len(route))

        return None

    def _build_nudge_toward_warp(
        self,
        state: StructuredGameState,
        edge,
        objective_id: str,
        destination_matches: bool,
    ) -> tuple[tuple[int, int, int, int, str], CandidateNextStep, CandidateRuntime] | None:
        """Return a low-priority candidate that moves the player one step
        toward a known warp tile when the normal approach computation fails.
        This shifts the navigation window so a proper approach can be found
        on the next turn (fixes the case where the player is right next to a
        staircase but the approach tile falls outside the current screen)."""
        if state.navigation is None or state.x is None or state.y is None:
            return None
        if edge.x is None or edge.y is None:
            return None
        nav_grid = self._planning_navigation_grid(state)
        if nav_grid is None:
            return None
        # Pick the closest walkable neighbour to the warp that differs from
        # the player's current position.
        walkable = [
            (coordinate.x, coordinate.y)
            for coordinate in state.navigation.walkable
            if (coordinate.x, coordinate.y) != (state.x, state.y)
            and nav_grid.is_walkable(coordinate.x, coordinate.y)
        ]
        if not walkable:
            return None
        target_x, target_y = min(
            walkable,
            key=lambda t: (abs(t[0] - edge.x) + abs(t[1] - edge.y),
                           abs(t[0] - state.x) + abs(t[1] - state.y)),
        )
        route = self._planning_find_path(state, target_x, target_y, nav_grid=nav_grid)
        if route is None:
            return None
        priority = 68 if destination_matches else 56
        slug = self._slugify(edge.destination_symbol or f"warp_{edge.x}_{edge.y}")
        candidate = CandidateNextStep(
            id=f"nudge_warp_{slug}_{edge.x}_{edge.y}",
            type="ENTER_CONNECTOR",
            target=ObjectiveTarget(
                kind="connector",
                map_id=state.map_id,
                map_name=state.map_name,
                x=target_x,
                y=target_y,
                detail=f"nudge toward {edge.destination_name or 'warp'}",
            ),
            why=f"Move closer to the {edge.destination_name or 'warp'} connector to reveal its approach tile.",
            priority=priority,
            expected_success_signal="The approach tile for the connector becomes reachable",
            objective_id=objective_id,
            distance=len(route),
            advances_target=destination_matches,
        )
        runtime = CandidateRuntime(
            task=Task(kind=TaskKind.NAVIGATE_TO, target_x=target_x, target_y=target_y, reason=candidate.why),
        )
        return (
            (1, -priority, len(route), 0 if destination_matches else 1, candidate.id),
            candidate,
            runtime,
        )

    def _npc_tile_set(self, state: StructuredGameState) -> set[tuple[int, int]]:
        if state.navigation is None:
            return set()
        return {(npc.tile_x, npc.tile_y) for npc in state.navigation.npc_positions}

    def _candidate_for_adjacent_interactable(
        self,
        state: StructuredGameState,
        objective_id: str,
        *,
        priority_boost: int = 0,
    ) -> CandidateNextStep | None:
        blocked_set = self._blocked_coordinates(state.navigation)
        if state.x is None or state.y is None or not blocked_set:
            return None
        npc_set = self._npc_tile_set(state)
        adjacent: list[tuple[int, int, int, bool]] = []
        for dx, dy in DIRECTION_DELTAS:
            blocked_coordinate = (state.x + dx, state.y + dy)
            if blocked_coordinate not in blocked_set:
                continue
            blocked_neighbors = self._blocked_neighbor_count(blocked_coordinate, blocked_set)
            is_npc = blocked_coordinate in npc_set
            adjacent.append((blocked_coordinate[0], blocked_coordinate[1], blocked_neighbors, is_npc))
        if not adjacent:
            return None
        # Prefer confirmed NPCs (sort key 0) over isolated blockers (sort key 1)
        best_x, best_y, neighbor_count, is_npc = sorted(
            adjacent, key=lambda item: (0 if item[3] else 1, item[2], item[0], item[1])
        )[0]
        if neighbor_count != 0 and not is_npc:
            return None
        if "PRESS_A" in self.stuck.state.recent_failed_actions[-2:]:
            return None
        if state.map_id is not None and state.x is not None and state.y is not None:
            key = (state.map_id, state.x, state.y)
            last_turn = self._interacted_tiles.get(key)
            if last_turn is not None and (self.completed_turns - last_turn) < self._interaction_cooldown:
                return None
        why = "An adjacent NPC is ready to talk." if is_npc else "An adjacent isolated blocker may be an NPC or doorway."
        candidate = CandidateNextStep(
            id=f"interact_adjacent_{best_x}_{best_y}",
            type="MOVE_ADJACENT_TO_INTERACTABLE",
            target=ObjectiveTarget(kind="interactable", map_id=state.map_id, map_name=state.map_name, x=best_x, y=best_y),
            why=why,
            priority=46 + priority_boost - min(neighbor_count, 2) * 4,
            expected_success_signal="Text opens or the local state changes",
            objective_id=objective_id,
        )
        self._candidate_runtime[candidate.id] = CandidateRuntime(
            action=ActionDecision(action=ActionType.PRESS_A, repeat=1, reason="interact nearby"),
        )
        return candidate

    def _candidate_to_move_adjacent_to_interactable(
        self,
        state: StructuredGameState,
        objective_id: str,
        *,
        priority_boost: int = 0,
    ) -> CandidateNextStep | None:
        if state.navigation is None or state.x is None or state.y is None:
            return None
        nav_grid = self._planning_navigation_grid(state)
        if nav_grid is None:
            return None
        blocked_set = self._blocked_coordinates(state.navigation)
        npc_set = self._npc_tile_set(state)

        # Pass 1: scan confirmed NPC positions (no distance limit, no isolation check)
        best_choice: tuple[int, int, int, int] | None = None
        is_npc_match = False
        for npc_x, npc_y in npc_set:
            if (npc_x, npc_y) not in blocked_set:
                continue
            if abs(npc_x - state.x) + abs(npc_y - state.y) <= 1:
                continue  # already adjacent, handled by _candidate_for_adjacent_interactable
            for dx, dy in DIRECTION_DELTAS:
                target_x = npc_x + dx
                target_y = npc_y + dy
                if not nav_grid.is_walkable(target_x, target_y) or (target_x, target_y) == (state.x, state.y):
                    continue
                if state.map_id is not None:
                    approach_key = (state.map_id, target_x, target_y)
                    last_turn = self._interacted_tiles.get(approach_key)
                    if last_turn is not None and (self.completed_turns - last_turn) < self._interaction_cooldown:
                        continue
                route = self._planning_find_path(state, target_x, target_y, nav_grid=nav_grid)
                if route is None or len(route) == 0:
                    continue
                distance = len(route)
                choice = (distance, target_x, target_y, npc_x * 1000 + npc_y)
                if best_choice is None or choice < best_choice:
                    best_choice = choice
                    is_npc_match = True

        # Pass 2: fallback to isolated blocked tiles within distance 4
        if best_choice is None:
            for blocked_x, blocked_y in blocked_set:
                if abs(blocked_x - state.x) + abs(blocked_y - state.y) > 4:
                    continue
                if self._blocked_neighbor_count((blocked_x, blocked_y), blocked_set) != 0:
                    continue
                for dx, dy in DIRECTION_DELTAS:
                    target_x = blocked_x + dx
                    target_y = blocked_y + dy
                    if not nav_grid.is_walkable(target_x, target_y) or (target_x, target_y) == (state.x, state.y):
                        continue
                    if state.map_id is not None:
                        approach_key = (state.map_id, target_x, target_y)
                        last_turn = self._interacted_tiles.get(approach_key)
                        if last_turn is not None and (self.completed_turns - last_turn) < self._interaction_cooldown:
                            continue
                    route = self._planning_find_path(state, target_x, target_y, nav_grid=nav_grid)
                    if route is None or len(route) == 0:
                        continue
                    distance = len(route)
                    choice = (distance, target_x, target_y, blocked_x * 1000 + blocked_y)
                    if best_choice is None or choice < best_choice:
                        best_choice = choice

        if best_choice is None:
            return None
        distance, target_x, target_y, encoded = best_choice
        blocked_x = encoded // 1000
        blocked_y = encoded % 1000
        why = (
            f"A confirmed NPC is reachable in {distance} steps."
            if is_npc_match
            else f"A nearby interactable-looking blocker is reachable in {distance} steps."
        )
        candidate = CandidateNextStep(
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
            why=why,
            priority=34 + priority_boost - min(distance, 6),
            expected_success_signal="Reach the adjacent tile, then open text or change state",
            objective_id=objective_id,
            distance=distance,
        )
        self._candidate_runtime[candidate.id] = CandidateRuntime(
            task=Task(kind=TaskKind.NAVIGATE_ADJACENT, target_x=blocked_x, target_y=blocked_y, reason=candidate.why),
            follow_up_task=Task(kind=TaskKind.INTERACT, reason=candidate.why),
        )
        return candidate

    def _build_all_npc_candidates(
        self,
        state: StructuredGameState,
        objective_id: str,
        goal: NavigationGoal,
        *,
        priority_boost: int = 0,
    ) -> list[CandidateNextStep]:
        """Build a candidate for every reachable NPC in the room.

        Returns up to 3 NPC candidates sorted by distance so the LLM can
        choose which NPC to approach.  When an interaction doesn't advance
        the story the cooldown on ``_interacted_tiles`` naturally filters
        that NPC out on the next cycle, surfacing the remaining ones.
        """
        if state.navigation is None or state.x is None or state.y is None:
            return []
        nav_grid = self._planning_navigation_grid(state)
        if nav_grid is None:
            return []
        npc_set = self._npc_tile_set(state)
        blocked_set = self._blocked_coordinates(state.navigation)

        # Collect the best approach for each distinct NPC tile.
        npc_approaches: list[tuple[int, int, int, int, int]] = []  # (distance, adj_x, adj_y, npc_x, npc_y)
        for npc_x, npc_y in npc_set:
            if (npc_x, npc_y) not in blocked_set:
                continue
            best_for_npc: tuple[int, int, int] | None = None
            for dx, dy in DIRECTION_DELTAS:
                adj_x, adj_y = npc_x + dx, npc_y + dy
                if not nav_grid.is_walkable(adj_x, adj_y):
                    continue
                # Respect interaction cooldown
                if state.map_id is not None:
                    approach_key = (state.map_id, adj_x, adj_y)
                    last_turn = self._interacted_tiles.get(approach_key)
                    if last_turn is not None and (self.completed_turns - last_turn) < self._interaction_cooldown:
                        continue
                route = self._planning_find_path(state, adj_x, adj_y, nav_grid=nav_grid)
                if route is None:
                    continue
                distance = len(route)
                if best_for_npc is None or distance < best_for_npc[0]:
                    best_for_npc = (distance, adj_x, adj_y)
            if best_for_npc is not None:
                npc_approaches.append((best_for_npc[0], best_for_npc[1], best_for_npc[2], npc_x, npc_y))

        npc_approaches.sort()  # closest first
        candidates: list[CandidateNextStep] = []
        for rank, (distance, adj_x, adj_y, npc_x, npc_y) in enumerate(npc_approaches[:3]):
            # Highest priority for closest, decreasing for farther NPCs
            priority = 48 + priority_boost - rank * 4 - min(distance, 6)
            is_adjacent = distance == 0
            if is_adjacent:
                candidate = CandidateNextStep(
                    id=f"talk_npc_{npc_x}_{npc_y}",
                    type="MOVE_ADJACENT_TO_INTERACTABLE",
                    target=ObjectiveTarget(kind="npc", map_id=state.map_id, map_name=state.map_name, x=npc_x, y=npc_y),
                    why=f"Talk to NPC at ({npc_x}, {npc_y}) — already adjacent.",
                    priority=priority + 10,
                    expected_success_signal="Text opens with story dialogue",
                    objective_id=objective_id,
                    distance=0,
                )
                self._candidate_runtime[candidate.id] = CandidateRuntime(
                    action=ActionDecision(action=ActionType.PRESS_A, repeat=1, reason=f"talk to NPC at ({npc_x}, {npc_y})"),
                )
            else:
                candidate = CandidateNextStep(
                    id=f"talk_npc_{npc_x}_{npc_y}",
                    type="EXPLORE_FOR_NPC",
                    target=ObjectiveTarget(
                        kind="npc",
                        map_id=state.map_id,
                        map_name=state.map_name,
                        x=npc_x,
                        y=npc_y,
                        detail=f"walk to ({adj_x}, {adj_y}) then talk",
                    ),
                    why=f"Walk to NPC at ({npc_x}, {npc_y}), {distance} steps away, and talk.",
                    priority=priority,
                    expected_success_signal="Text opens with story dialogue",
                    objective_id=objective_id,
                    distance=distance,
                )
                self._candidate_runtime[candidate.id] = CandidateRuntime(
                    task=Task(kind=TaskKind.NAVIGATE_ADJACENT, target_x=npc_x, target_y=npc_y, reason=candidate.why),
                    follow_up_task=Task(kind=TaskKind.INTERACT, reason=f"talk to NPC at ({npc_x}, {npc_y})"),
                )
            candidates.append(candidate)
        return candidates

    def _build_room_exploration_candidate(
        self,
        state: StructuredGameState,
        objective_id: str,
    ) -> CandidateNextStep | None:
        """Walk to the farthest reachable tile to explore the room."""
        if state.navigation is None or state.x is None or state.y is None:
            return None
        nav_grid = self._planning_navigation_grid(state)
        if nav_grid is None:
            return None
        best_frontier: tuple[int, int, int] | None = None
        for coordinate in state.navigation.walkable:
            if (coordinate.x, coordinate.y) == (state.x, state.y):
                continue
            if not nav_grid.is_walkable(coordinate.x, coordinate.y):
                continue
            route = self._planning_find_path(state, coordinate.x, coordinate.y, nav_grid=nav_grid)
            if route is None or len(route) == 0:
                continue
            distance = len(route)
            if best_frontier is None or distance > best_frontier[0]:
                best_frontier = (distance, coordinate.x, coordinate.y)

        if best_frontier is None:
            return None
        distance, target_x, target_y = best_frontier
        candidate = CandidateNextStep(
            id=f"npc_search_{target_x}_{target_y}",
            type="EXPLORE_FOR_NPC",
            target=ObjectiveTarget(
                kind="npc_search",
                map_id=state.map_id,
                map_name=state.map_name,
                x=target_x,
                y=target_y,
                detail="room frontier exploration",
            ),
            why="No NPC directly reachable; exploring the room to find an approach path.",
            priority=38,
            expected_success_signal="New tiles become accessible or an NPC becomes reachable",
            objective_id=objective_id,
            distance=distance,
        )
        self._candidate_runtime[candidate.id] = CandidateRuntime(
            task=Task(kind=TaskKind.NAVIGATE_TO, target_x=target_x, target_y=target_y, reason=candidate.why),
        )
        return candidate

    def _dedupe_candidates(self, candidates: list[CandidateNextStep]) -> list[CandidateNextStep]:
        seen: set[tuple[str, int | None, int | None, str | None, str | None]] = set()
        deduped: list[CandidateNextStep] = []
        for candidate in candidates:
            runtime = self._candidate_runtime_for(candidate)
            task = runtime.task
            key = (
                candidate.type,
                task.target_x if task is not None else None,
                task.target_y if task is not None else None,
                runtime.action.action.value if runtime.action else None,
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
        current_state = self.emulator.get_structured_state()
        if current_state.menu_open:
            return True
        if candidates[0].blacklisted:
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
            "ADVANCE_TOWARD_BOUNDARY",
            "ENTER_CONNECTOR",
        }:
            return True
        if candidates[0].type == "MOVE_ADJACENT_TO_INTERACTABLE" and self._candidate_runtime_for(candidates[0]).action is not None:
            return True
        if candidates[0].advances_target and not candidates[1].advances_target:
            return True
        return (candidates[0].priority - candidates[1].priority) >= 12

    def _should_force_opening_objective_llm(self, state: StructuredGameState) -> bool:
        if self.llm_client is None:
            return False
        if state.is_bootstrap():
            return False
        if state.mode != GameMode.OVERWORLD or state.menu_open or state.text_box_open or state.battle_state is not None:
            return False
        milestone = get_current_milestone(
            state.story_flags,
            [item.name for item in state.inventory],
            current_map_name=state.map_name,
            badges=state.badges,
        )
        if milestone.id != "get_starter":
            return False
        if not map_matches(state.map_name, "Red's House 2F"):
            return False
        if self._just_exited_bootstrap():
            return True
        return self.telemetry.turns == 0 and not self._has_seen_known_location("Red's House 2F")

    def _just_exited_bootstrap(self) -> bool:
        if not self.context_manager.action_traces:
            return False
        return self.context_manager.action_traces[-1].planner_source == "bootstrap"

    def _has_seen_known_location(self, target_map_name: str) -> bool:
        return any(
            map_matches(known_map_name, target_map_name)
            for known_map_name in self.memory.memory.long_term.known_locations.values()
        )

    def _compile_candidate(
        self,
        candidate: CandidateNextStep,
        state: StructuredGameState,
        reason: str,
    ) -> CandidateRuntime | None:
        runtime = self._candidate_runtime_for(candidate)
        if runtime.action is not None:
            action = runtime.action.model_copy(deep=True)
            action.reason = reason
            return CandidateRuntime(action=self.validator.validate(action, state))
        if runtime.task is not None:
            task = runtime.task.model_copy(deep=True)
            task.reason = reason
            follow_up_task = runtime.follow_up_task.model_copy(deep=True) if runtime.follow_up_task is not None else None
            if follow_up_task is not None and not follow_up_task.reason:
                follow_up_task.reason = reason
            return CandidateRuntime(task=task, follow_up_task=follow_up_task)
        compiled_task = self._compile_task_for_candidate(candidate, state, reason)
        if compiled_task is None:
            return None
        return CandidateRuntime(task=compiled_task[0], follow_up_task=compiled_task[1])

    def _compile_task_for_candidate(
        self,
        candidate: CandidateNextStep,
        state: StructuredGameState,
        reason: str,
    ) -> tuple[Task, Task | None] | None:
        target = candidate.target
        target_x = target.x if target is not None else None
        target_y = target.y if target is not None else None
        if candidate.type == "ADVANCE_TOWARD_BOUNDARY":
            direction = target.detail if target is not None else None
            if direction is None:
                return None
            return Task(kind=TaskKind.WALK_BOUNDARY, direction=direction, reason=reason), None
        if candidate.type in {"ENTER_CONNECTOR", "EXPLORE_CONNECTOR"}:
            source_x, source_y = self._connector_source_for_candidate(candidate)
            connector_id = self._connector_id_for_candidate(candidate) or candidate.id
            task = Task(
                kind=TaskKind.ENTER_CONNECTOR,
                connector_id=connector_id,
                target_x=source_x,
                target_y=source_y,
                reason=reason,
            )
            return task, None
        if candidate.type == "MOVE_ADJACENT_TO_INTERACTABLE":
            if target_x is None or target_y is None:
                return None
            return (
                Task(kind=TaskKind.NAVIGATE_ADJACENT, target_x=target_x, target_y=target_y, reason=reason),
                Task(kind=TaskKind.INTERACT, reason=reason),
            )
        if candidate.type == "EXPLORE_WINDOW":
            if target_x is None or target_y is None:
                return None
            return Task(kind=TaskKind.NAVIGATE_TO, target_x=target_x, target_y=target_y, reason=reason), None
        return None

    def _planning_result_from_runtime(
        self,
        runtime: CandidateRuntime,
        *,
        raw_response: str | None = None,
        used_fallback: bool = False,
        planner_source: str,
        messages: list[dict] | None = None,
        prompt_metrics: dict[str, Any] | None = None,
        llm_usage: LLMUsage | None = None,
        llm_attempted: bool = False,
        llm_model: str | None = None,
        objective_id: str | None = None,
        candidate_id: str | None = None,
        candidates: list[CandidateNextStep] | None = None,
    ) -> PlanningResult:
        return PlanningResult(
            action=runtime.action,
            task=runtime.task,
            follow_up_task=runtime.follow_up_task,
            raw_response=raw_response,
            used_fallback=used_fallback,
            planner_source=planner_source,
            messages=list(messages or []),
            prompt_metrics=prompt_metrics,
            llm_usage=llm_usage,
            llm_attempted=llm_attempted,
            llm_model=llm_model,
            objective_id=objective_id,
            candidate_id=candidate_id,
            candidates=list(candidates or []),
        )

    def _connector_by_id(self, connector_id: str) -> object | None:
        connector = self.memory.memory.long_term.world_map.connectors.get(connector_id)
        if connector is not None:
            return connector
        return self._static_connector_by_id(connector_id)

    def _static_connector_by_id(self, connector_id: str | None) -> DiscoveredConnector | None:
        parsed = self._parse_static_connector_id(connector_id)
        if parsed is None:
            return None
        source_symbol, destination_symbol, source_x, source_y = parsed
        edge = self._find_static_warp_edge(source_symbol, destination_symbol, source_x, source_y)
        if edge is None:
            return None
        return self._synthesize_static_connector(edge)

    def _parse_static_connector_id(self, connector_id: str | None) -> tuple[str | None, str | None, int, int] | None:
        if not connector_id or not connector_id.startswith("static:"):
            return None
        parts = connector_id.split(":")
        if len(parts) == 4:
            _prefix, destination_symbol, source_x, source_y = parts
            source_symbol = None
        elif len(parts) == 5:
            _prefix, source_symbol, destination_symbol, source_x, source_y = parts
        else:
            return None
        try:
            return source_symbol, destination_symbol or None, int(source_x), int(source_y)
        except ValueError:
            return None

    def _find_static_warp_edge(
        self,
        source_symbol: str | None,
        destination_symbol: str | None,
        source_x: int,
        source_y: int,
    ) -> WorldGraphEdge | None:
        if source_symbol:
            edge = self.static_world_graph.get_warp_at(source_symbol, source_x, source_y)
            if edge is not None and (destination_symbol is None or edge.destination_symbol == destination_symbol):
                return edge
        for graph_map in self.static_world_graph.maps():
            edge = self.static_world_graph.get_warp_at(graph_map.symbol, source_x, source_y)
            if edge is None:
                continue
            if destination_symbol is not None and edge.destination_symbol != destination_symbol:
                continue
            return edge
        return None

    def _synthesize_static_connector(self, edge: WorldGraphEdge) -> DiscoveredConnector | None:
        if edge.x is None or edge.y is None:
            return None
        boundary_side = self._static_warp_boundary_side(edge)
        activation_mode = "push" if boundary_side is not None else "step_on"
        transition_action = self._action_for_side(boundary_side) if boundary_side is not None else None
        return DiscoveredConnector(
            id=f"{edge.source_name}::tile::{edge.x}:{edge.y}",
            source_map=edge.source_name,
            source_x=edge.x,
            source_y=edge.y,
            kind="warp",
            activation_mode=activation_mode,
            status=ConnectorStatus.CONFIRMED,
            approach_x=edge.x if activation_mode == "push" else None,
            approach_y=edge.y if activation_mode == "push" else None,
            transition_action=transition_action,
            destination_map=edge.destination_name,
        )

    def _static_warp_boundary_side(self, edge: WorldGraphEdge) -> str | None:
        if edge.x is None or edge.y is None:
            return None
        # Only preserve boundary-push semantics for "return to LAST_MAP"-style
        # exits. Explicit warps can legitimately sit on a room edge (stairs,
        # ladders), and treating them as boundary pushes makes the executor aim
        # for the unreachable warp tile instead of its approach tile.
        if not self._static_warp_uses_boundary_push(edge):
            return None
        graph_map = self.static_world_graph.get_map_by_id(edge.source_symbol)
        if graph_map is None:
            return None
        max_x = max(0, graph_map.width * 2 - 1)
        max_y = max(0, graph_map.height * 2 - 1)
        if edge.y <= 0:
            return "north"
        if edge.x >= max_x:
            return "east"
        if edge.y >= max_y:
            return "south"
        if edge.x <= 0:
            return "west"
        return None

    def _static_warp_uses_boundary_push(self, edge: WorldGraphEdge) -> bool:
        return edge.original_destination_symbol == "LAST_MAP" or edge.resolution_method == "last_map_proximity"

    def _static_connector_id(self, edge: WorldGraphEdge) -> str:
        return f"static:{edge.destination_symbol}:{edge.x}:{edge.y}"

    def _connector_for_task(self, task: Task | None) -> DiscoveredConnector | None:
        if task is None or task.kind != TaskKind.ENTER_CONNECTOR or task.connector_id is None:
            return None
        connector = self._connector_by_id(task.connector_id)
        return connector if isinstance(connector, DiscoveredConnector) else None

    def _connector_id_for_candidate(self, candidate: CandidateNextStep) -> str | None:
        runtime = self._candidate_runtime_for(candidate)
        if runtime.task is not None:
            return runtime.task.connector_id
        if candidate.id.startswith("connector_"):
            slug = candidate.id.removeprefix("connector_")
            for connector_id in self.memory.memory.long_term.world_map.connectors:
                if self._slugify(connector_id) == slug:
                    return connector_id
        return None

    def _connector_source_for_candidate(self, candidate: CandidateNextStep) -> tuple[int | None, int | None]:
        connector_id = self._connector_id_for_candidate(candidate)
        if connector_id is not None:
            connector = self._connector_by_id(connector_id)
            if connector is not None:
                return connector.source_x, connector.source_y
        if candidate.target is None:
            return None, None
        return candidate.target.x, candidate.target.y

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
        runtime = self._candidate_runtime_for(candidate)
        assert runtime.target_x is not None and runtime.target_y is not None
        route = self._find_path_with_temporary_blockers(state, runtime.target_x, runtime.target_y)
        if route is None:
            self.route_cache = None
            self.execution_plan = None
            return None
        if len(route) == 0:
            if runtime.follow_up_action is None:
                self.route_cache = None
                self.execution_plan = None
                return None
            self.execution_plan = ExecutionPlan(
                objective_id=candidate.objective_id,
                candidate_id=candidate.id,
                plan_type=candidate.type,
                target=candidate.target,
                expected_success_signal=candidate.expected_success_signal,
                step_budget=max(1, runtime.step_budget - 1),
                button_action=runtime.follow_up_action,
                target_x=runtime.target_x,
                target_y=runtime.target_y,
                follow_up_action=None,
                map_id=state.map_id,
                collision_hash=state.navigation.collision_hash,
                reason=reason,
                started_step=state.step,
            )
            return ActionDecision(action=runtime.follow_up_action, repeat=1, reason=reason)

        first_action = route[0]
        remaining = list(route[1:])
        if remaining:
            self.route_cache = CachedRoute(
                map_id=state.map_id,
                target_x=runtime.target_x,
                target_y=runtime.target_y,
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
            step_budget=max(runtime.step_budget, len(route)),
            button_action=None,
            target_x=runtime.target_x,
            target_y=runtime.target_y,
            follow_up_action=runtime.follow_up_action,
            map_id=state.map_id,
            collision_hash=state.navigation.collision_hash,
            reason=reason,
            started_step=state.step,
        )
        return ActionDecision(action=first_action, repeat=1, reason=reason)

    def _candidate_runtime_for(self, candidate: CandidateNextStep) -> CandidateRuntime:
        return self._candidate_runtime.get(candidate.id, CandidateRuntime())

    def _trim_candidate_runtime(self, candidates: list[CandidateNextStep]) -> list[CandidateNextStep]:
        keep_ids = {candidate.id for candidate in candidates}
        self._candidate_runtime = {
            candidate_id: runtime for candidate_id, runtime in self._candidate_runtime.items() if candidate_id in keep_ids
        }
        return candidates

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
        if plan.plan_type in NAVIGATION_CANDIDATE_TYPES and plan.target_x is not None and plan.target_y is not None:
            if (state.x, state.y) == (plan.target_x, plan.target_y):
                if plan.follow_up_action is not None:
                    follow_up_action = plan.follow_up_action
                    plan.follow_up_action = None
                    self.execution_plan = plan
                    return ActionDecision(action=follow_up_action, repeat=1, reason=plan.reason or "Trigger connector at target")
                self.execution_plan = None
                return None
            # Not at target yet – recompute path and continue navigating.
            if (
                state.mode == GameMode.OVERWORLD
                and state.navigation is not None
                and state.x is not None
                and state.y is not None
                and not state.text_box_open
                and not state.menu_open
                and state.battle_state is None
                and plan.map_id == state.map_id
                and plan.step_budget > 0
            ):
                route = self._find_path_with_temporary_blockers(state, plan.target_x, plan.target_y)
                if route:
                    plan.step_budget -= 1
                    first_action = route[0]
                    remaining = list(route[1:])
                    if remaining:
                        self.route_cache = CachedRoute(
                            map_id=state.map_id,
                            target_x=plan.target_x,
                            target_y=plan.target_y,
                            collision_hash=state.navigation.collision_hash,
                            remaining_actions=remaining,
                            expected_start=advance_position(state.x, state.y, first_action),
                        )
                    return ActionDecision(action=first_action, repeat=1, reason=plan.reason or "Continue navigation to target")
            # Path not found or budget exhausted – abandon plan.
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
        if state.x != route.expected_start.x or state.y != route.expected_start.y:
            self.route_cache = None
            return None
        if not route.remaining_actions:
            self.route_cache = None
            return None

        next_action = route.remaining_actions.pop(0)
        walkable = {(coord.x, coord.y) for coord in state.navigation.walkable}
        next_expected = advance_position(state.x or 0, state.y or 0, next_action)
        if (next_expected.x, next_expected.y) not in walkable:
            self.route_cache = None
            return None
        if route.remaining_actions:
            route.expected_start = next_expected
            self.route_cache = route
        else:
            self.route_cache = None
        target_text = f"({route.target_x}, {route.target_y})"
        return ActionDecision(action=next_action, repeat=1, reason=f"Continue cached route to {target_text}")

    def _drain_cached_route_steps(self) -> int:
        """Execute all remaining cached-route steps in a tight loop.

        Called immediately after the first step of a cached route (or any
        navigation action that set up a route cache).  Returns the number
        of extra emulator actions executed.  Stops early if the game state
        becomes unsafe (map change, text box, battle, etc.).
        """
        extra_steps = 0
        while self.route_cache is not None and self.route_cache.remaining_actions:
            state = self.emulator.get_structured_state()
            if state.mode != GameMode.OVERWORLD:
                self.route_cache = None
                break
            if state.text_box_open or state.menu_open or state.battle_state is not None:
                self.route_cache = None
                break
            route = self.route_cache
            if state.map_id != route.map_id:
                self.route_cache = None
                break
            if state.x != route.expected_start.x or state.y != route.expected_start.y:
                self.route_cache = None
                break
            next_action = route.remaining_actions.pop(0)
            if state.navigation is not None:
                walkable = {(c.x, c.y) for c in state.navigation.walkable}
                next_pos = advance_position(state.x or 0, state.y or 0, next_action)
                if (next_pos.x, next_pos.y) not in walkable:
                    self.route_cache = None
                    break
            step = ActionDecision(action=next_action, repeat=1, reason="drain cached route")
            self.emulator.execute_action(self.validator.validate(step, state))
            extra_steps += 1
            if route.remaining_actions:
                route.expected_start = advance_position(state.x or 0, state.y or 0, next_action)
            else:
                self.route_cache = None

        # If we reached the target and the execution plan has a follow-up
        # action (e.g. walk into a door / stairs), execute it immediately.
        plan = self.execution_plan
        if (
            plan is not None
            and plan.follow_up_action is not None
            and plan.plan_type in NAVIGATION_CANDIDATE_TYPES
            and plan.target_x is not None
            and plan.target_y is not None
        ):
            state = self.emulator.get_structured_state()
            if (
                state.mode == GameMode.OVERWORLD
                and (state.x, state.y) == (plan.target_x, plan.target_y)
                and not state.text_box_open
                and not state.menu_open
                and state.battle_state is None
            ):
                follow_up = ActionDecision(
                    action=plan.follow_up_action,
                    repeat=1,
                    reason=plan.reason or "follow-up at target",
                )
                self.emulator.execute_action(self.validator.validate(follow_up, state))
                plan.follow_up_action = None
                self.execution_plan = plan
                extra_steps += 1

        return extra_steps

    def _find_path_with_temporary_blockers(
        self,
        state: StructuredGameState,
        target_x: int,
        target_y: int,
    ) -> list[ActionType] | None:
        navigation = state.navigation
        if navigation is None or state.x is None or state.y is None:
            return None
        blocked_tiles = self._temporary_blocked_tiles_for_state(state)
        if not blocked_tiles:
            return find_path(navigation, state.x, state.y, target_x, target_y)

        exempt = {(state.x, state.y), (target_x, target_y)}
        walkable = [
            coordinate
            for coordinate in navigation.walkable
            if (coordinate.x, coordinate.y) not in blocked_tiles or (coordinate.x, coordinate.y) in exempt
        ]
        adjusted_navigation = navigation.model_copy(update={"walkable": walkable})
        return find_path(adjusted_navigation, state.x, state.y, target_x, target_y)

    def _temporary_blocked_tiles_for_state(self, state: StructuredGameState) -> set[tuple[int, int]]:
        return self.navigator.blocked_tiles_for_state(state)

    def _record_failed_move_blocker(
        self,
        before: StructuredGameState,
        after: StructuredGameState,
        action: ActionDecision,
        progress: ProgressResult,
        turn_index: int,
    ) -> None:
        if progress.classification != "no_effect":
            return
        if before.mode != GameMode.OVERWORLD or before.x is None or before.y is None:
            return
        if action.action not in {ActionType.MOVE_UP, ActionType.MOVE_RIGHT, ActionType.MOVE_DOWN, ActionType.MOVE_LEFT}:
            return
        if before.map_id != after.map_id or before.map_name != after.map_name:
            return
        if after.x != before.x or after.y != before.y:
            return
        blocked_coordinate = advance_position(before.x, before.y, action.action)
        self.navigator.mark_blocked(
            before,
            blocked_coordinate.x,
            blocked_coordinate.y,
            turn_index=turn_index,
        )

    def _prune_temporary_blocked_tiles(self, turn_index: int) -> None:
        self.navigator.prune_blocked(turn_index)

    def _clear_temporary_blocked_tiles(self, state: StructuredGameState) -> None:
        map_key = self._state_map_key(state)
        self.navigator.clear_map(map_key)

    def _state_map_key(self, state: StructuredGameState) -> str | int | None:
        return state.map_id if state.map_id is not None else (state.map_name or None)

    def _refresh_route_cache_after_turn(self, state: StructuredGameState, progress: ProgressResult) -> None:
        if self.route_cache is None:
            return
        if state.mode != GameMode.OVERWORLD or state.menu_open or state.text_box_open or state.battle_state:
            self.route_cache = None
            return
        if state.navigation is None:
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
        # For navigation plans the re-path logic in _plan_from_execution_plan
        # recomputes using the current collision data, so a hash change is not
        # fatal.  Only invalidate non-navigation plans on collision changes.
        if plan.plan_type not in NAVIGATION_CANDIDATE_TYPES:
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
        return visible_boundary_side(navigation, x, y)

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

    def _capped_route_target(self, current_map_name: str | None, final_target_map_name: str | None) -> str | None:
        if not current_map_name or not final_target_map_name:
            return final_target_map_name
        route = self.static_world_graph.find_route(current_map_name, final_target_map_name)
        if route is None or len(route.edges) <= MAX_ROUTE_CONNECTOR_HOPS:
            return final_target_map_name
        return route.maps[MAX_ROUTE_CONNECTOR_HOPS].name

    def _graph_exits_from(self, map_name: str | None) -> list[RouteHop]:
        if not map_name:
            return []
        graph_map = self.static_world_graph.get_map_by_name(map_name)
        if graph_map is None:
            return []
        exits: list[RouteHop] = []
        for edge in self.static_world_graph.neighbors(graph_map.symbol):
            direction = edge.direction or edge.kind
            if edge.destination_name is None or direction is None:
                continue
            exits.append(RouteHop(to_map=edge.destination_name, direction=direction))
        return exits

    def _next_hop_toward(self, from_map: str | None, to_map: str | None) -> RouteHop | None:
        if not from_map or not to_map:
            return None
        route = self.static_world_graph.find_route(from_map, to_map)
        if route is None or not route.edges:
            return None
        edge = route.edges[0]
        if edge.destination_name is None:
            return None
        return RouteHop(
            to_map=edge.destination_name,
            direction=edge.direction or edge.kind,
        )

    def _requires_current_map_story_interaction(self, state: StructuredGameState, milestone) -> bool:
        if not state.map_name:
            return False
        connections = self._graph_exits_from(state.map_name)
        if not connections or any(connection.direction != "warp" for connection in connections):
            return False
        canonical_name = self.static_world_graph.canonical_name(state.map_name) or state.map_name
        normalized_map = " ".join(re.findall(r"[a-z0-9]+", canonical_name.lower()))
        if not normalized_map:
            return False
        for line in (milestone.description, *milestone.route_hints, *milestone.sub_steps):
            normalized_line = " ".join(re.findall(r"[a-z0-9]+", line.lower()))
            if normalized_map not in normalized_line:
                continue
            if any(keyword in normalized_line for keyword in STORY_INTERACTION_KEYWORDS):
                return True
        return False

    def _sync_navigation_goal(self, state: StructuredGameState) -> NavigationGoal | None:
        if state.is_bootstrap():
            self.memory.memory.long_term.navigation_goal = None
            return None
        milestone = get_current_milestone(
            state.story_flags,
            [item.name for item in state.inventory],
            current_map_name=state.map_name,
            badges=state.badges,
        )
        objective_plan = self.memory.memory.long_term.objective_plan
        if objective_plan is not None:
            compiled_goal = self._sync_navigation_goal_from_objective_plan(state, milestone, objective_plan.internal_plan)
            if compiled_goal is not None:
                return compiled_goal
        goal = self.memory.memory.long_term.navigation_goal
        final_target_map_name = milestone.target_map_name
        logical_target_map_name = self._current_story_target(state, milestone)
        target_map_name = final_target_map_name if logical_target_map_name is None else logical_target_map_name
        if logical_target_map_name is not None and not map_matches(state.map_name, logical_target_map_name):
            target_map_name = self._capped_route_target(state.map_name, logical_target_map_name) or logical_target_map_name
        target_landmark = (
            self._story_landmark_for_milestone(milestone)
            if map_matches(target_map_name, milestone.target_map_name)
            else None
        )

        hop = None if map_matches(state.map_name, target_map_name) else self._next_hop_toward(state.map_name, target_map_name)
        objective_kind = "explore_for_matching_connector"
        engine_mode = "exploration"
        next_map_name = None
        next_hop_kind = None
        next_hop_side = None
        if hop is not None:
            next_map_name = hop.to_map
            next_hop_kind = "warp" if hop.direction == "warp" else "boundary"
            next_hop_side = None if hop.direction == "warp" else hop.direction
            engine_mode = "progression"
            objective_kind = "reach_boundary_side" if next_hop_kind == "boundary" else "explore_for_matching_connector"
            if next_hop_kind == "warp" and next_map_name is not None:
                target_landmark = self._landmark_for_destination_on_current_map(state.map_name, next_map_name) or target_landmark
        elif target_landmark is not None and map_matches(state.map_name, target_map_name):
            objective_kind = "reach_landmark"
            engine_mode = "progression"
        elif self._should_story_interact_here(state, milestone, target_map_name):
            objective_kind = "talk_to_required_npc"
            engine_mode = "progression"

        if (
            goal is None
            or goal.source != "objective"
            or goal.target_map_name != target_map_name
            or goal.final_target_map_name != final_target_map_name
        ):
            goal = NavigationGoal(
                target_map_name=target_map_name,
                final_target_map_name=final_target_map_name,
                target_landmark_id=None if target_landmark is None else target_landmark.id,
                target_landmark_type=None if target_landmark is None else target_landmark.type,
                source="objective",
                started_step=state.step,
            )
            self.memory.memory.long_term.navigation_goal = goal
        goal.current_map_name = state.map_name
        goal.target_map_name = target_map_name
        goal.final_target_map_name = final_target_map_name
        goal.target_landmark_id = None if target_landmark is None else target_landmark.id
        goal.target_landmark_type = None if target_landmark is None else target_landmark.type
        goal.objective_kind = objective_kind
        goal.engine_mode = engine_mode
        goal.next_map_name = next_map_name
        goal.next_hop_kind = next_hop_kind
        goal.next_hop_side = next_hop_side
        goal.last_confirmed_step = state.step
        if goal.next_hop_kind != "warp":
            goal.target_connector_id = None
        if goal.target_connector_id and goal.next_map_name is not None:
            connector = self.memory.memory.long_term.world_map.connectors.get(goal.target_connector_id)
            if connector is None or (connector.destination_map and not map_matches(connector.destination_map, goal.next_map_name)):
                goal.target_connector_id = None
        if goal.target_connector_id is None and goal.next_hop_kind == "warp":
            for connector in connectors_from_map(self.memory.memory.long_term.world_map, state.map_name, confirmed_only=True):
                if connector.destination_map and goal.next_map_name and map_matches(connector.destination_map, goal.next_map_name):
                    goal.target_connector_id = connector.id
                    goal.objective_kind = "enter_selected_connector"
                    break
        if goal.target_connector_id is not None:
            goal.objective_kind = "enter_selected_connector"
        return goal

    def _sync_navigation_goal_from_objective_plan(
        self,
        state: StructuredGameState,
        milestone,
        plan: InternalObjectivePlan,
    ) -> NavigationGoal | None:
        if not self._objective_plan_is_compilable(state, plan):
            return None
        if plan.plan_type not in {"go_to_map", "go_to_landmark", "interact_story_npc"}:
            self.memory.memory.long_term.navigation_goal = None
            return None

        final_target_landmark = self.static_world_graph.get_landmark(plan.target_landmark_id)
        final_target_map_name = plan.target_map_name or (
            final_target_landmark.map_name if final_target_landmark is not None else milestone.target_map_name
        )
        # If a go_to_* plan has already reached its local/intermediate map, do
        # not let that stale target mask the next story hop for another turn.
        if (
            plan.plan_type in {"go_to_map", "go_to_landmark"}
            and final_target_map_name
            and map_matches(state.map_name, final_target_map_name)
            and not map_matches(state.map_name, milestone.target_map_name)
        ):
            self.memory.memory.long_term.navigation_goal = None
            return None
        target_map_name = final_target_map_name
        if plan.plan_type in {"go_to_map", "go_to_landmark"} and not map_matches(state.map_name, final_target_map_name):
            target_map_name = self._capped_route_target(state.map_name, final_target_map_name) or final_target_map_name
        target_landmark = final_target_landmark if map_matches(target_map_name, final_target_map_name) else None
        goal = self.memory.memory.long_term.navigation_goal

        hop = None if map_matches(state.map_name, target_map_name) else self._next_hop_toward(state.map_name, target_map_name)
        objective_kind = "explore_for_matching_connector"
        engine_mode = "progression"
        next_map_name = None
        next_hop_kind = None
        next_hop_side = None

        if plan.plan_type == "interact_story_npc":
            objective_kind = "talk_to_required_npc"
        elif hop is not None:
            next_map_name = hop.to_map
            next_hop_kind = "warp" if hop.direction == "warp" else "boundary"
            next_hop_side = None if hop.direction == "warp" else hop.direction
            objective_kind = "reach_boundary_side" if next_hop_kind == "boundary" else "explore_for_matching_connector"
            if next_hop_kind == "warp" and next_map_name is not None:
                target_landmark = self._landmark_for_destination_on_current_map(state.map_name, next_map_name) or target_landmark
        elif plan.plan_type == "go_to_landmark" and target_landmark is not None and map_matches(state.map_name, target_map_name):
            objective_kind = "reach_landmark"
        elif map_matches(state.map_name, target_map_name) and self._should_story_interact_here(state, milestone, target_map_name):
            objective_kind = "talk_to_required_npc"
        else:
            engine_mode = "exploration"

        if (
            goal is None
            or goal.source != "objective_plan"
            or goal.target_map_name != target_map_name
            or goal.final_target_map_name != final_target_map_name
        ):
            goal = NavigationGoal(
                target_map_name=target_map_name,
                final_target_map_name=final_target_map_name,
                source="objective_plan",
                started_step=state.step,
            )
            self.memory.memory.long_term.navigation_goal = goal

        goal.current_map_name = state.map_name
        goal.target_map_name = target_map_name
        goal.final_target_map_name = final_target_map_name
        goal.target_landmark_id = None if target_landmark is None else target_landmark.id
        goal.target_landmark_type = None if target_landmark is None else target_landmark.type
        goal.objective_kind = objective_kind
        goal.engine_mode = engine_mode
        goal.next_map_name = next_map_name
        goal.next_hop_kind = next_hop_kind
        goal.next_hop_side = next_hop_side
        goal.last_confirmed_step = state.step
        if goal.next_hop_kind != "warp":
            goal.target_connector_id = None

        if goal.target_connector_id and goal.next_map_name is not None:
            connector = self.memory.memory.long_term.world_map.connectors.get(goal.target_connector_id)
            if connector is None or (connector.destination_map and not map_matches(connector.destination_map, goal.next_map_name)):
                goal.target_connector_id = None
        if goal.target_connector_id is None and goal.next_hop_kind == "warp":
            for connector in connectors_from_map(self.memory.memory.long_term.world_map, state.map_name, confirmed_only=True):
                if connector.destination_map and goal.next_map_name and map_matches(connector.destination_map, goal.next_map_name):
                    goal.target_connector_id = connector.id
                    goal.objective_kind = "enter_selected_connector"
                    break
        if goal.target_connector_id is not None:
            goal.objective_kind = "enter_selected_connector"
        return goal

    def _clear_navigation_confirmation(self, map_name: str) -> None:
        del map_name
        return None

    def _update_navigation_goal_after_turn(
        self,
        state: StructuredGameState,
        progress: ProgressResult,
        planning: PlanningResult,
    ) -> None:
        goal = self.memory.memory.long_term.navigation_goal
        if goal is None:
            return
        if progress.classification == "major_progress" and "map_id" in progress.changed_fields:
            goal.failed_candidate_ids.clear()
            goal.failed_connector_ids.clear()
            goal.failed_sides.clear()
            goal.target_connector_id = None
            goal.last_confirmed_step = state.step
            return
        if planning.candidate_id is not None:
            goal.last_candidate_id = planning.candidate_id
        if progress.classification == "no_effect" and planning.candidate_id is not None:
            goal.failed_candidate_ids = [item for item in [*goal.failed_candidate_ids, planning.candidate_id] if item][-4:]
            if goal.target_connector_id is not None:
                goal.failed_connector_ids = [item for item in [*goal.failed_connector_ids, goal.target_connector_id] if item][-4:]
            if goal.next_hop_side is not None:
                goal.failed_sides = [item for item in [*goal.failed_sides, goal.next_hop_side] if item][-3:]
        elif progress.classification in {"movement_success", "interaction_success", "major_progress"}:
            if planning.candidate_id in goal.failed_candidate_ids:
                goal.failed_candidate_ids = [item for item in goal.failed_candidate_ids if item != planning.candidate_id]
            if goal.target_connector_id is not None and goal.target_connector_id in goal.failed_connector_ids:
                goal.failed_connector_ids = [item for item in goal.failed_connector_ids if item != goal.target_connector_id]
            if goal.next_hop_side is not None and goal.next_hop_side in goal.failed_sides:
                goal.failed_sides = [item for item in goal.failed_sides if item != goal.next_hop_side]

    def _current_story_target(self, state: StructuredGameState, milestone) -> str | None:
        if map_matches(state.map_name, milestone.target_map_name):
            return None
        if self._requires_current_map_story_interaction(state, milestone):
            return state.map_name
        return milestone.target_map_name

    def _preferred_destination_from_milestone(self, state: StructuredGameState) -> str | None:
        milestone = get_current_milestone(
            state.story_flags,
            [item.name for item in state.inventory],
            current_map_name=state.map_name,
            badges=state.badges,
        )
        story_text = self._milestone_story_text(state)
        best: tuple[tuple[int, int, int, int, str], str] | None = None
        for connection in self._graph_exits_from(state.map_name):
            if connection.to_map == state.map_name:
                continue
            route = self.static_world_graph.find_route(connection.to_map, milestone.target_map_name)
            route_length = 999 if route is None else len(route.edges)
            score = self._map_text_score(connection.to_map, story_text)
            backtrack = bool(self._previous_map_name and map_matches(connection.to_map, self._previous_map_name))
            rank = (
                0 if map_matches(connection.to_map, milestone.target_map_name) else 1,
                route_length,
                0 if score > 0 else 1,
                1 if backtrack else 0,
                connection.to_map,
            )
            if best is None or rank < best[0]:
                best = (rank, connection.to_map)
        return None if best is None else best[1]

    def _story_landmark_for_milestone(self, milestone) -> Landmark | None:
        target_map = self.static_world_graph.get_map_by_name(milestone.target_map_name)
        if target_map is None:
            return None
        story_text = " ".join([milestone.description, *milestone.route_hints, *milestone.sub_steps]).lower()
        ranked: list[tuple[int, str, Landmark]] = []
        for landmark in self.static_world_graph.get_landmarks_on_map(target_map.symbol):
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

    def _landmark_for_destination_on_current_map(self, current_map_name: str | None, destination_map_name: str | None) -> Landmark | None:
        if not current_map_name or not destination_map_name:
            return None
        ranked: list[tuple[tuple[int, str], Landmark]] = []
        for landmark in self.static_world_graph.get_landmarks_on_map(current_map_name):
            if landmark.x is None or landmark.y is None:
                continue
            if landmark.destination_name is None or not map_matches(landmark.destination_name, destination_map_name):
                continue
            rank = (0 if landmark.type == "important_building" else 1, landmark.id)
            ranked.append((rank, landmark))
        ranked.sort(key=lambda item: item[0])
        return None if not ranked else ranked[0][1]

    def _landmark_story_score(self, landmark: Landmark, story_text: str) -> int:
        label = landmark.label.lower()
        score = 0
        if label in story_text:
            score += 12
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

    def _milestone_story_text(self, state: StructuredGameState) -> str:
        milestone = get_current_milestone(
            state.story_flags,
            [item.name for item in state.inventory],
            current_map_name=state.map_name,
            badges=state.badges,
        )
        return " ".join([milestone.description, *milestone.route_hints, *milestone.sub_steps]).lower()

    def _map_text_score(self, map_name: str, story_text: str) -> int:
        if not map_name or not story_text:
            return 0
        if map_name.lower() in story_text:
            return 5
        tokens = [token for token in re.findall(r"[a-z0-9]+", map_name.lower()) if token and token not in {"city", "town", "route", "road", "gym", "house", "gate", "lab", "forest", "cave", "center", "pokecenter", "mart"}]
        return sum(1 for token in tokens if token in story_text)

    def _has_nearby_interactable(self, state: StructuredGameState) -> bool:
        if state.navigation is None or state.x is None or state.y is None:
            return False
        nav_grid = self._planning_navigation_grid(state)
        if nav_grid is None:
            return False
        # Check confirmed NPC positions first
        npc_set = self._npc_tile_set(state)
        blocked_set = self._blocked_coordinates(state.navigation)
        for npc_x, npc_y in npc_set:
            for dx, dy in DIRECTION_DELTAS:
                if nav_grid.is_walkable(npc_x + dx, npc_y + dy):
                    return True
        # Fallback: isolated blocked tile heuristic
        for dx, dy in DIRECTION_DELTAS:
            blocked_coordinate = (state.x + dx, state.y + dy)
            if blocked_coordinate in blocked_set and self._blocked_neighbor_count(blocked_coordinate, blocked_set) == 0:
                return True
        for blocked_x, blocked_y in blocked_set:
            if abs(blocked_x - state.x) + abs(blocked_y - state.y) > 4:
                continue
            if self._blocked_neighbor_count((blocked_x, blocked_y), blocked_set) != 0:
                continue
            for dx, dy in DIRECTION_DELTAS:
                if nav_grid.is_walkable(blocked_x + dx, blocked_y + dy):
                    return True
        return False

    def _should_story_interact_here(
        self,
        state: StructuredGameState,
        milestone,
        target_map: str | None,
    ) -> bool:
        if not self._has_nearby_interactable(state):
            return False
        if target_map and map_matches(state.map_name, target_map):
            return True
        if map_matches(state.map_name, milestone.target_map_name):
            return True
        return self._requires_current_map_story_interaction(state, milestone)

    def _is_story_relevant_interior(self, state: StructuredGameState) -> bool:
        milestone = get_current_milestone(
            state.story_flags,
            [item.name for item in state.inventory],
            current_map_name=state.map_name,
            badges=state.badges,
        )
        return self._requires_current_map_story_interaction(state, milestone)

    @staticmethod
    def _action_for_side(side: str | None) -> ActionType | None:
        if side == "north":
            return ActionType.MOVE_UP
        if side == "east":
            return ActionType.MOVE_RIGHT
        if side == "south":
            return ActionType.MOVE_DOWN
        if side == "west":
            return ActionType.MOVE_LEFT
        return None

    @staticmethod
    def _slugify(value: str) -> str:
        slug = "_".join(re.findall(r"[a-z0-9]+", value.lower()))
        return slug or "candidate"
