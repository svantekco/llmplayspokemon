from __future__ import annotations

from collections.abc import Callable

from pokemon_agent.agent.controllers.protocol import TurnContext
from pokemon_agent.agent.navigator import Navigator
from pokemon_agent.agent.navigation import advance_position
from pokemon_agent.agent.planning_types import PlanningResult
from pokemon_agent.agent.stuck_detector import StuckState
from pokemon_agent.models.action import ActionDecision
from pokemon_agent.models.action import ActionType
from pokemon_agent.models.state import GameMode
from pokemon_agent.models.state import NPCSprite
from pokemon_agent.models.state import StructuredGameState

RecoveryEscalation = Callable[[StructuredGameState, TurnContext], PlanningResult | None]
StuckStateGetter = Callable[[], StuckState]
ConnectorStateGetter = Callable[[], object | None]
_MOVE_ACTIONS: tuple[ActionType, ...] = (
    ActionType.MOVE_UP,
    ActionType.MOVE_RIGHT,
    ActionType.MOVE_DOWN,
    ActionType.MOVE_LEFT,
)
_OPPOSITE_ACTIONS = {
    ActionType.MOVE_UP: ActionType.MOVE_DOWN,
    ActionType.MOVE_RIGHT: ActionType.MOVE_LEFT,
    ActionType.MOVE_DOWN: ActionType.MOVE_UP,
    ActionType.MOVE_LEFT: ActionType.MOVE_RIGHT,
}


class RecoveryController:
    def __init__(
        self,
        navigator: Navigator,
        *,
        stuck_state_getter: StuckStateGetter,
        threshold: int = 4,
        escalation_callback: RecoveryEscalation | None = None,
        active_connector_getter: ConnectorStateGetter | None = None,
    ) -> None:
        self._navigator = navigator
        self._stuck_state_getter = stuck_state_getter
        self._threshold = max(1, int(threshold))
        self._escalation_callback = escalation_callback
        self._active_connector_getter = active_connector_getter
        self._consecutive_failures = 0
        self._random_walk_index = 0
        self._active_map_key: str | int | None = None

    def step(self, state: StructuredGameState, context: TurnContext) -> PlanningResult | None:
        stuck_state = self._stuck_state_getter()
        failure_count = max(self._consecutive_failures, stuck_state.steps_since_progress)
        if not self._should_recover(state, stuck_state, failure_count):
            self.reset()
            return None

        map_key = self._navigator.state_map_key(state)
        if map_key != self._active_map_key:
            self._random_walk_index = 0
            self._active_map_key = map_key

        severity = max(self._threshold, failure_count)
        if severity >= self._threshold + 6:
            result = self._escalate(state, context)
        elif severity >= self._threshold + 4:
            result = self._backtrack_or_random_walk(state, context)
        elif severity >= self._threshold + 2:
            result = self._interact_nearby(state)
        else:
            result = self._try_alternate_path(state, context)

        if result is None or result.action is None:
            return None
        result.planner_source = "recovery_controller"
        return result

    def reset(self) -> None:
        self._consecutive_failures = 0
        self._random_walk_index = 0
        self._active_map_key = None

    def record_outcome(self, progress_classification: str) -> None:
        if progress_classification in {"no_effect", "regression"}:
            self._consecutive_failures += 1
            return
        self.reset()

    def export_state(self) -> dict[str, object] | None:
        if self._consecutive_failures == 0 and self._random_walk_index == 0 and self._active_map_key is None:
            return None
        return {
            "consecutive_failures": self._consecutive_failures,
            "random_walk_index": self._random_walk_index,
            "active_map_key": self._active_map_key,
        }

    def restore_state(self, payload: dict[str, object] | None) -> None:
        self.reset()
        if not payload:
            return
        consecutive_failures = payload.get("consecutive_failures")
        if isinstance(consecutive_failures, int):
            self._consecutive_failures = max(0, consecutive_failures)
        random_walk_index = payload.get("random_walk_index")
        if isinstance(random_walk_index, int):
            self._random_walk_index = max(0, random_walk_index)
        active_map_key = payload.get("active_map_key")
        if isinstance(active_map_key, (str, int)):
            self._active_map_key = active_map_key

    def _should_recover(self, state: StructuredGameState, stuck_state: StuckState, failure_count: int) -> bool:
        if state.is_bootstrap():
            return False
        if state.battle_state or state.menu_open or state.text_box_open:
            return False
        if state.mode not in {GameMode.OVERWORLD, GameMode.UNKNOWN}:
            return False
        if self._active_connector_getter is not None and self._active_connector_getter() is not None:
            return False
        return stuck_state.score >= self._threshold and failure_count > 0

    def _try_alternate_path(
        self,
        state: StructuredGameState,
        context: TurnContext,
    ) -> PlanningResult | None:
        target = context.navigation_target
        if (
            target is not None
            and target.x is not None
            and target.y is not None
            and (target.map_name is None or target.map_name == state.map_name)
            and state.x is not None
            and state.y is not None
        ):
            ranked: list[tuple[tuple[int, int, str], ActionType]] = []
            for action, next_x, next_y in self._adjacent_moves(state, context):
                distance = abs(target.x - next_x) + abs(target.y - next_y)
                ranked.append(((distance, self._same_as_previous_penalty(action, context), action.value), action))
            if ranked:
                ranked.sort(key=lambda item: item[0])
                return self._move_action(ranked[0][1], "recovery: try an alternate path toward the current target")

        moves = self._adjacent_moves(state, context)
        if not moves:
            return None
        preferred = sorted(
            moves,
            key=lambda item: (self._same_as_previous_penalty(item[0], context), item[0].value),
        )[0][0]
        return self._move_action(preferred, "recovery: break the local movement loop")

    def _interact_nearby(self, state: StructuredGameState) -> PlanningResult:
        if state.x is None or state.y is None:
            return self._press_a("recovery: probe a nearby interaction")
        npc = self._nearest_npc(state)
        if npc is None:
            return self._press_a("recovery: probe a nearby interaction")
        distance = abs(state.x - npc.tile_x) + abs(state.y - npc.tile_y)
        if distance == 1:
            face_action = self._action_toward(state.x, state.y, npc.tile_x, npc.tile_y)
            expected_facing = self._facing_name(face_action)
            if face_action is not None and state.facing != expected_facing:
                return self._move_action(face_action, "recovery: face the nearby NPC")
            return self._press_a("recovery: interact with the nearby NPC")
        adjacent = self._navigator.best_adjacent_tile(npc.tile_x, npc.tile_y)
        if adjacent is None:
            return self._press_a("recovery: probe a nearby interaction")
        route = self._navigator.find_local_path(state, adjacent[0], adjacent[1])
        if route:
            return self._move_action(route[0], "recovery: move next to a nearby NPC")
        return self._press_a("recovery: probe a nearby interaction")

    def _backtrack_or_random_walk(
        self,
        state: StructuredGameState,
        context: TurnContext,
    ) -> PlanningResult:
        previous_action = None if context.previous_action is None else context.previous_action.action
        reverse = _OPPOSITE_ACTIONS.get(previous_action)
        available = self._adjacent_moves(state, context)
        if reverse is not None:
            for action, _next_x, _next_y in available:
                if action == reverse:
                    return self._move_action(action, "recovery: backtrack out of the stuck pocket")

        if not available:
            return self._press_a("recovery: fall back to a local interaction")

        ordered_actions = self._rotate_actions()
        for action in ordered_actions:
            for candidate, _next_x, _next_y in available:
                if candidate == action:
                    self._random_walk_index += 1
                    return self._move_action(candidate, "recovery: random walk to reveal a new route")
        return self._press_a("recovery: fall back to a local interaction")

    def _escalate(
        self,
        state: StructuredGameState,
        context: TurnContext,
    ) -> PlanningResult:
        if self._escalation_callback is not None:
            result = self._escalation_callback(state, context)
            if result is not None and result.action is not None:
                if not result.action.reason:
                    result.action.reason = "recovery: reroute after escalating the objective"
                return result
        fallback = self._press_a("recovery: fallback interaction while escalation is unavailable")
        fallback.used_fallback = True
        fallback.raw_response = "recovery escalation unavailable"
        return fallback

    def _adjacent_moves(
        self,
        state: StructuredGameState,
        context: TurnContext,
    ) -> list[tuple[ActionType, int, int]]:
        if state.navigation is None or state.x is None or state.y is None:
            return []
        walkable = {(coordinate.x, coordinate.y) for coordinate in state.navigation.walkable}
        blocked = self._navigator.blocked_tiles_for_state(state)
        moves: list[tuple[ActionType, int, int]] = []
        for action in self._preferred_actions(context):
            coordinate = advance_position(state.x, state.y, action)
            position = (coordinate.x, coordinate.y)
            if position not in walkable or position in blocked:
                continue
            moves.append((action, coordinate.x, coordinate.y))
        return moves

    def _preferred_actions(self, context: TurnContext) -> list[ActionType]:
        previous = None if context.previous_action is None else context.previous_action.action
        if previous not in _MOVE_ACTIONS:
            return list(_MOVE_ACTIONS)
        if previous in {ActionType.MOVE_UP, ActionType.MOVE_DOWN}:
            return [ActionType.MOVE_RIGHT, ActionType.MOVE_LEFT, previous, _OPPOSITE_ACTIONS[previous]]
        return [ActionType.MOVE_UP, ActionType.MOVE_DOWN, previous, _OPPOSITE_ACTIONS[previous]]

    def _rotate_actions(self) -> list[ActionType]:
        offset = self._random_walk_index % len(_MOVE_ACTIONS)
        return [*_MOVE_ACTIONS[offset:], *_MOVE_ACTIONS[:offset]]

    @staticmethod
    def _nearest_npc(state: StructuredGameState) -> NPCSprite | None:
        if not state.npcs or state.x is None or state.y is None:
            return None
        return min(
            state.npcs,
            key=lambda npc: (abs(state.x - npc.tile_x) + abs(state.y - npc.tile_y), npc.tile_y, npc.tile_x),
        )

    @staticmethod
    def _same_as_previous_penalty(action: ActionType, context: TurnContext) -> int:
        previous = None if context.previous_action is None else context.previous_action.action
        if previous == action:
            return 2
        if _OPPOSITE_ACTIONS.get(previous) == action:
            return 1
        return 0

    @staticmethod
    def _action_toward(player_x: int, player_y: int, target_x: int, target_y: int) -> ActionType | None:
        if target_x == player_x and target_y == player_y - 1:
            return ActionType.MOVE_UP
        if target_x == player_x + 1 and target_y == player_y:
            return ActionType.MOVE_RIGHT
        if target_x == player_x and target_y == player_y + 1:
            return ActionType.MOVE_DOWN
        if target_x == player_x - 1 and target_y == player_y:
            return ActionType.MOVE_LEFT
        return None

    @staticmethod
    def _facing_name(action: ActionType | None) -> str | None:
        if action == ActionType.MOVE_UP:
            return "UP"
        if action == ActionType.MOVE_RIGHT:
            return "RIGHT"
        if action == ActionType.MOVE_DOWN:
            return "DOWN"
        if action == ActionType.MOVE_LEFT:
            return "LEFT"
        return None

    @staticmethod
    def _move_action(action: ActionType, reason: str) -> PlanningResult:
        return PlanningResult(action=ActionDecision(action=action, repeat=1, reason=reason))

    @staticmethod
    def _press_a(reason: str) -> PlanningResult:
        return PlanningResult(action=ActionDecision(action=ActionType.PRESS_A, repeat=1, reason=reason))
