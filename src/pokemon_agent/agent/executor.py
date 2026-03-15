from __future__ import annotations

from typing import Callable

from pokemon_agent.agent.navigation import NavigationGrid
from pokemon_agent.agent.navigation import facing_action_for_target
from pokemon_agent.agent.navigation import facing_name_for_action
from pokemon_agent.agent.navigation import is_real_map_edge
from pokemon_agent.agent.navigation import visible_boundary_side
from pokemon_agent.models.action import ActionDecision
from pokemon_agent.models.action import ActionType
from pokemon_agent.models.action import ExecutorStatus
from pokemon_agent.models.action import StepResult
from pokemon_agent.models.action import Task
from pokemon_agent.models.action import TaskKind
from pokemon_agent.models.memory import DiscoveredConnector
from pokemon_agent.models.state import GameMode
from pokemon_agent.models.state import StructuredGameState


class Executor:
    def __init__(
        self,
        connector_lookup: Callable[[str], DiscoveredConnector | None] | None = None,
        *,
        max_retries: int = 3,
    ) -> None:
        self._connector_lookup = connector_lookup
        self._max_retries = max(1, int(max_retries))
        self._task: Task | None = None
        self._follow_up_task: Task | None = None
        self._retries = 0
        self._started_map_key: str | int | None = None
        self._nav_grid = NavigationGrid()
        self._pending_blocked_reason: str | None = None
        self._last_action_mode: str | None = None

    def is_active(self) -> bool:
        return self._task is not None or self._pending_blocked_reason is not None

    def current_task(self) -> Task | None:
        if self._task is None:
            return None
        return self._task.model_copy(deep=True)

    def blocked_tiles(self) -> set[tuple[int, int]]:
        return set(self._nav_grid._blocked)

    def map_key(self) -> str | int | None:
        return self._started_map_key

    def begin(
        self,
        task: Task,
        state: StructuredGameState,
        *,
        follow_up_task: Task | None = None,
    ) -> StepResult:
        self._task = task.model_copy(deep=True)
        self._follow_up_task = follow_up_task.model_copy(deep=True) if follow_up_task is not None else None
        self._retries = 0
        self._started_map_key = self._state_map_key(state)
        self._pending_blocked_reason = None
        self._last_action_mode = None
        self._nav_grid = NavigationGrid(state.navigation)
        return self.step(state)

    def step(self, state: StructuredGameState) -> StepResult:
        self._nav_grid.refresh(state.navigation)
        self._last_action_mode = None

        if self._pending_blocked_reason is not None:
            blocked_reason = self._pending_blocked_reason
            self.abort()
            return StepResult(status=ExecutorStatus.BLOCKED, blocked_reason=blocked_reason)

        if self._task is None:
            return StepResult(status=ExecutorStatus.DONE)

        current_map_key = self._state_map_key(state)
        if self._started_map_key is not None and current_map_key is not None and current_map_key != self._started_map_key:
            self.abort()
            return StepResult(status=ExecutorStatus.DONE)

        if self._is_interrupted(state):
            return StepResult(status=ExecutorStatus.INTERRUPTED)

        while self._task is not None:
            result = self._step_task(state, self._task)
            if result.status == ExecutorStatus.DONE and result.action is None:
                if self._follow_up_task is not None:
                    self._task = self._follow_up_task
                    self._follow_up_task = None
                    continue
                self.abort()
            return result

        return StepResult(status=ExecutorStatus.DONE)

    def report_move_failed(self, state: StructuredGameState, action: ActionDecision) -> None:
        if self._task is None:
            return
        if self._last_action_mode != "move":
            return
        if state.x is None or state.y is None:
            return
        blocked_coordinate = self._advance_position(state.x, state.y, action.action)
        if blocked_coordinate is None:
            return
        self._nav_grid.mark_blocked(*blocked_coordinate)
        self._retries += 1
        if self._retries >= self._max_retries:
            self._pending_blocked_reason = f"Movement to {blocked_coordinate} failed repeatedly"

    def abort(self) -> None:
        self._task = None
        self._follow_up_task = None
        self._retries = 0
        self._started_map_key = None
        self._pending_blocked_reason = None
        self._last_action_mode = None
        self._nav_grid.clear_blocked()

    def export_state(self) -> dict[str, object] | None:
        if not self.is_active():
            return None
        return {
            "task": self._task.model_dump(mode="json") if self._task is not None else None,
            "follow_up_task": self._follow_up_task.model_dump(mode="json") if self._follow_up_task is not None else None,
            "retries": self._retries,
            "started_map_key": self._started_map_key,
            "blocked_tiles": sorted([list(item) for item in self._nav_grid._blocked]),
            "pending_blocked_reason": self._pending_blocked_reason,
        }

    def restore_state(self, payload: dict[str, object] | None) -> None:
        self.abort()
        if not payload:
            return
        task_payload = payload.get("task")
        follow_up_payload = payload.get("follow_up_task")
        self._task = Task.model_validate(task_payload) if task_payload else None
        self._follow_up_task = Task.model_validate(follow_up_payload) if follow_up_payload else None
        self._retries = int(payload.get("retries", 0))
        self._started_map_key = payload.get("started_map_key")
        self._pending_blocked_reason = payload.get("pending_blocked_reason")
        for item in payload.get("blocked_tiles", []):
            if isinstance(item, (list, tuple)) and len(item) == 2:
                self._nav_grid.mark_blocked(int(item[0]), int(item[1]))

    def _step_task(self, state: StructuredGameState, task: Task) -> StepResult:
        if task.kind == TaskKind.NAVIGATE_TO:
            assert task.target_x is not None and task.target_y is not None
            return self._step_navigate_to(state, task, task.target_x, task.target_y)
        if task.kind == TaskKind.NAVIGATE_ADJACENT:
            assert task.target_x is not None and task.target_y is not None
            return self._step_navigate_adjacent(state, task)
        if task.kind == TaskKind.INTERACT:
            self._task = None
            return self._emit_action(ActionType.PRESS_A, task.reason or "interact")
        if task.kind == TaskKind.PRESS_BUTTON:
            assert task.button is not None
            return self._step_press_button(task)
        if task.kind == TaskKind.ENTER_CONNECTOR:
            return self._step_enter_connector(state, task)
        if task.kind == TaskKind.WALK_BOUNDARY:
            return self._step_walk_boundary(state, task)
        self._task = None
        return StepResult(status=ExecutorStatus.BLOCKED, blocked_reason=f"Unsupported task: {task.kind.value}")

    def _step_press_button(self, task: Task) -> StepResult:
        assert task.button is not None
        action = self._emit_action(task.button, task.reason or "press button")
        task.repeat -= 1
        if task.repeat <= 0:
            self._task = None
        return action

    def _step_navigate_to(
        self,
        state: StructuredGameState,
        task: Task,
        target_x: int,
        target_y: int,
    ) -> StepResult:
        if state.x is None or state.y is None:
            self._task = None
            return StepResult(status=ExecutorStatus.BLOCKED, blocked_reason="Missing player coordinates")
        if (state.x, state.y) == (target_x, target_y):
            self._task = None
            return StepResult(status=ExecutorStatus.DONE)
        route = self._nav_grid.find_path(state.x, state.y, target_x, target_y)
        if not route:
            self._task = None
            return StepResult(
                status=ExecutorStatus.BLOCKED,
                blocked_reason=f"No path to ({target_x}, {target_y})",
            )
        return self._emit_action(
            route[0],
            task.reason or f"navigate to ({target_x}, {target_y})",
            suggested_path=self._coordinates_for_route(state.x, state.y, route),
        )

    def _step_navigate_adjacent(self, state: StructuredGameState, task: Task) -> StepResult:
        assert task.target_x is not None and task.target_y is not None
        if state.x is None or state.y is None:
            self._task = None
            return StepResult(status=ExecutorStatus.BLOCKED, blocked_reason="Missing player coordinates")
        if abs(state.x - task.target_x) + abs(state.y - task.target_y) == 1:
            adjacent = (state.x, state.y)
        else:
            adjacent = self._nav_grid.best_adjacent_tile(task.target_x, task.target_y)
        if adjacent is None:
            self._task = None
            return StepResult(status=ExecutorStatus.BLOCKED, blocked_reason="No walkable adjacent tile")
        if (state.x, state.y) != adjacent:
            return self._step_navigate_to(state, task, adjacent[0], adjacent[1])
        face_action = facing_action_for_target(state.x, state.y, task.target_x, task.target_y)
        expected_facing = facing_name_for_action(face_action)
        if face_action is not None and state.facing != expected_facing:
            return self._emit_action(face_action, task.reason or "face adjacent target", mode="face")
        self._task = None
        return StepResult(status=ExecutorStatus.DONE)

    def _step_enter_connector(self, state: StructuredGameState, task: Task) -> StepResult:
        connector = self._lookup_connector(task.connector_id)
        approach = self._approach_for_connector(state, task, connector)
        if approach is None:
            self._task = None
            return StepResult(status=ExecutorStatus.BLOCKED, blocked_reason="No connector approach found")

        approach_x, approach_y, transition_action = approach
        if state.x is None or state.y is None:
            self._task = None
            return StepResult(status=ExecutorStatus.BLOCKED, blocked_reason="Missing player coordinates")
        if (state.x, state.y) == (approach_x, approach_y):
            blocked_coordinate = self._advance_position(state.x, state.y, transition_action)
            suggested_path = [] if blocked_coordinate is None else [blocked_coordinate]
            return self._emit_action(
                transition_action,
                task.reason or "enter connector",
                suggested_path=suggested_path,
            )
        return self._step_navigate_to(state, task, approach_x, approach_y)

    def _step_walk_boundary(self, state: StructuredGameState, task: Task) -> StepResult:
        assert task.direction is not None
        if state.navigation is None or state.x is None or state.y is None:
            self._task = None
            return StepResult(status=ExecutorStatus.BLOCKED, blocked_reason="Missing navigation snapshot")
        target = self._best_boundary_target(state, task.direction)
        if target is None:
            self._task = None
            return StepResult(
                status=ExecutorStatus.BLOCKED,
                blocked_reason=f"No boundary route for {task.direction}",
            )
        target_x, target_y = target
        if (state.x, state.y) == (target_x, target_y):
            action = self._action_for_direction(task.direction)
            if action is None:
                self._task = None
                return StepResult(
                    status=ExecutorStatus.BLOCKED,
                    blocked_reason=f"Unsupported boundary direction: {task.direction}",
                )
            blocked_coordinate = self._advance_position(state.x, state.y, action)
            suggested_path = [] if blocked_coordinate is None else [blocked_coordinate]
            return self._emit_action(
                action,
                task.reason or f"walk {task.direction}",
                suggested_path=suggested_path,
            )
        return self._step_navigate_to(state, task, target_x, target_y)

    def _best_boundary_target(
        self,
        state: StructuredGameState,
        direction: str,
    ) -> tuple[int, int] | None:
        navigation = self._nav_grid.snapshot
        if navigation is None or state.x is None or state.y is None:
            return None
        best: tuple[tuple[int, int, int], tuple[int, int]] | None = None
        for coordinate in navigation.walkable:
            if (coordinate.x, coordinate.y) == (state.x, state.y):
                continue
            visible_side = visible_boundary_side(navigation, coordinate.x, coordinate.y)
            if visible_side is None:
                continue
            if visible_side != direction and not self._moves_toward_side(state.x, state.y, coordinate.x, coordinate.y, direction):
                continue
            route = self._nav_grid.find_path(state.x, state.y, coordinate.x, coordinate.y)
            if route is None:
                continue
            rank = self._directional_rank(
                direction,
                coordinate.x,
                coordinate.y,
                len(route),
                is_real_map_edge(navigation, visible_side),
            )
            if best is None or rank < best[0]:
                best = (rank, (coordinate.x, coordinate.y))
        return None if best is None else best[1]

    def _approach_for_connector(
        self,
        state: StructuredGameState,
        task: Task,
        connector: DiscoveredConnector | None,
    ) -> tuple[int, int, ActionType] | None:
        if state.x is None or state.y is None:
            return None
        if (
            connector is not None
            and connector.approach_x is not None
            and connector.approach_y is not None
            and connector.transition_action is not None
        ):
            route = self._nav_grid.find_path(state.x, state.y, connector.approach_x, connector.approach_y)
            if route is not None:
                return connector.approach_x, connector.approach_y, connector.transition_action

        source_x = connector.source_x if connector is not None else task.target_x
        source_y = connector.source_y if connector is not None else task.target_y
        if source_x is None or source_y is None:
            return None
        return self._approach_for_transition_tile(state, source_x, source_y)

    def _approach_for_transition_tile(
        self,
        state: StructuredGameState,
        source_x: int,
        source_y: int,
    ) -> tuple[int, int, ActionType] | None:
        best: tuple[tuple[int, int, str], tuple[int, int, ActionType]] | None = None
        source_is_walkable = self._nav_grid.is_walkable(source_x, source_y)
        for action, dx, dy in (
            (ActionType.MOVE_UP, 0, -1),
            (ActionType.MOVE_RIGHT, 1, 0),
            (ActionType.MOVE_DOWN, 0, 1),
            (ActionType.MOVE_LEFT, -1, 0),
        ):
            approach_x = source_x - dx
            approach_y = source_y - dy
            if not self._nav_grid.is_walkable(approach_x, approach_y):
                continue
            route = self._nav_grid.find_path(state.x, state.y, approach_x, approach_y)
            if route is None:
                continue
            direction_penalty = 0
            if source_is_walkable and action != ActionType.MOVE_DOWN:
                direction_penalty = 100
            rank = (len(route) + direction_penalty, direction_penalty, action.value)
            if best is None or rank < best[0]:
                best = (rank, (approach_x, approach_y, action))

        if best is not None:
            return best[1]

        if source_is_walkable:
            route = self._nav_grid.find_path(state.x, state.y, source_x, source_y)
            if route:
                return source_x, source_y, route[-1]
        return None

    def _emit_action(
        self,
        action: ActionType,
        reason: str,
        *,
        mode: str | None = None,
        suggested_path: list[tuple[int, int]] | None = None,
    ) -> StepResult:
        self._last_action_mode = mode or self._action_mode_for(action)
        return StepResult(
            status=ExecutorStatus.STEPPING,
            action=ActionDecision(action=action, repeat=1, reason=reason),
            suggested_path=list(suggested_path or []),
        )

    def _coordinates_for_route(
        self,
        start_x: int,
        start_y: int,
        route: list[ActionType],
    ) -> list[tuple[int, int]]:
        coordinates: list[tuple[int, int]] = []
        current_x = start_x
        current_y = start_y
        for action in route:
            next_position = self._advance_position(current_x, current_y, action)
            if next_position is None:
                break
            coordinates.append(next_position)
            current_x, current_y = next_position
        return coordinates

    def _lookup_connector(self, connector_id: str | None) -> DiscoveredConnector | None:
        if connector_id is None or self._connector_lookup is None:
            return None
        return self._connector_lookup(connector_id)

    def _action_for_direction(self, direction: str) -> ActionType | None:
        mapping = {
            "north": ActionType.MOVE_UP,
            "east": ActionType.MOVE_RIGHT,
            "south": ActionType.MOVE_DOWN,
            "west": ActionType.MOVE_LEFT,
        }
        return mapping.get(direction)

    def _directional_rank(
        self,
        direction: str,
        x: int,
        y: int,
        distance: int,
        at_real_edge: bool,
    ) -> tuple[int, int, int]:
        if direction == "north":
            return (0 if at_real_edge else 1, y, distance)
        if direction == "east":
            return (0 if at_real_edge else 1, -x, distance)
        if direction == "south":
            return (0 if at_real_edge else 1, -y, distance)
        if direction == "west":
            return (0 if at_real_edge else 1, x, distance)
        return (0, distance, x + y)

    def _moves_toward_side(
        self,
        current_x: int,
        current_y: int,
        target_x: int,
        target_y: int,
        direction: str,
    ) -> bool:
        if direction == "north":
            return target_y <= current_y
        if direction == "east":
            return target_x >= current_x
        if direction == "south":
            return target_y >= current_y
        if direction == "west":
            return target_x <= current_x
        return True

    def _advance_position(self, x: int, y: int, action: ActionType) -> tuple[int, int] | None:
        deltas = {
            ActionType.MOVE_UP: (0, -1),
            ActionType.MOVE_RIGHT: (1, 0),
            ActionType.MOVE_DOWN: (0, 1),
            ActionType.MOVE_LEFT: (-1, 0),
        }
        delta = deltas.get(action)
        if delta is None:
            return None
        return x + delta[0], y + delta[1]

    def _action_mode_for(self, action: ActionType) -> str:
        if action in {
            ActionType.MOVE_UP,
            ActionType.MOVE_RIGHT,
            ActionType.MOVE_DOWN,
            ActionType.MOVE_LEFT,
        }:
            return "move"
        return "button"

    def _is_interrupted(self, state: StructuredGameState) -> bool:
        return (
            state.mode != GameMode.OVERWORLD
            or state.menu_open
            or state.text_box_open
            or state.battle_state is not None
        )

    def _state_map_key(self, state: StructuredGameState) -> str | int | None:
        return state.map_id if state.map_id is not None else (state.map_name or None)
