from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from pokemon_agent.agent.controllers.protocol import NavigationTarget
from pokemon_agent.agent.controllers.protocol import TurnContext
from pokemon_agent.agent.navigator import Navigator
from pokemon_agent.agent.planning_types import PlanningResult
from pokemon_agent.agent.navigation import advance_position
from pokemon_agent.models.action import ActionDecision
from pokemon_agent.models.action import ActionType
from pokemon_agent.models.memory import DiscoveredConnector
from pokemon_agent.models.memory import NavigationGoal
from pokemon_agent.models.memory import WorldMapMemory
from pokemon_agent.models.state import NPCSprite
from pokemon_agent.models.state import StructuredGameState
from pokemon_agent.navigation.world_graph import Landmark
from pokemon_agent.navigation.world_graph import WorldGraph
from pokemon_agent.navigation.world_graph import map_matches


@dataclass(slots=True)
class ConnectorExecutionState:
    connector_id: str | None
    activation_mode: str
    source_x: int | None
    source_y: int | None
    approach_x: int | None
    approach_y: int | None
    transition_action: ActionType | None
    failed_transition_attempts: int = 0
    tried_interact_fallback: bool = False


class OverworldController:
    def __init__(
        self,
        navigator: Navigator,
        world_map: WorldMapMemory,
        *,
        static_world_graph: WorldGraph,
        goal_getter: Callable[[], NavigationGoal | None] | None = None,
        landmark_for_destination: Callable[[str | None, str | None], Landmark | None] | None = None,
        utility_action_getter: Callable[[StructuredGameState, TurnContext], PlanningResult | None] | None = None,
    ) -> None:
        self._navigator = navigator
        self._world_map = world_map
        self._static_world_graph = static_world_graph
        self._goal_getter = goal_getter
        self._landmark_for_destination = landmark_for_destination
        self._utility_action_getter = utility_action_getter
        self._last_action_mode: str | None = None
        self._connector_state: ConnectorExecutionState | None = None
        self._active_connector: DiscoveredConnector | None = None
        self._active_map_key: str | int | None = None

    def step(self, state: StructuredGameState, context: TurnContext) -> PlanningResult:
        map_key = self._navigator.state_map_key(state)
        if map_key != self._active_map_key:
            self._connector_state = None
            self._active_map_key = map_key
        self._navigator.update(state, turn_index=context.turn_index)
        if state.navigation is None or state.x is None or state.y is None:
            return self._emit_action(ActionType.PRESS_A, "advance while overworld state settles", mode="interact")

        if self._utility_action_getter is not None:
            utility_action = self._utility_action_getter(state, context)
            if utility_action is not None and utility_action.action is not None:
                self._last_action_mode = "utility"
                return utility_action

        goal = self._goal_getter() if self._goal_getter is not None else None
        local_target = self._local_target(state, context.navigation_target)
        target_map = self._target_map(state, context.navigation_target, goal)

        if (
            local_target is not None
            and (target_map is None or map_matches(state.map_name, target_map))
            and (state.x, state.y) != local_target
        ):
            return self._walk_to(
                state,
                local_target[0],
                local_target[1],
                context.navigation_target.reason or "walk to local objective",
            )

        if goal is not None:
            connector = self._goal_connector(state, goal)
            if connector is not None:
                return self._enter_connector(
                    state,
                    connector,
                    goal.objective_kind or "enter target connector",
                )

        if target_map is not None and not map_matches(state.map_name, target_map):
            route = self._navigator.next_route_step(
                state.map_name,
                target_map,
                preferred_connector_id=None if goal is None else goal.target_connector_id,
            )
            if route is not None and route.connector is not None:
                return self._enter_connector(
                    state,
                    route.connector,
                    f"travel toward {route.destination_map or target_map}",
                )
            if route is not None and route.edge is not None:
                static_connector = self._synthesize_static_connector(route.edge)
                if static_connector is not None:
                    return self._enter_connector(
                        state,
                        static_connector,
                        f"travel toward {route.destination_map or target_map}",
                    )
            if route is not None and route.kind == "connection" and route.direction is not None:
                return self._walk_boundary(
                    state,
                    route.direction,
                    f"follow the {route.direction} boundary toward {route.destination_map or target_map}",
                )
            if route is not None and route.kind == "warp":
                landmark = self._route_landmark(state.map_name, route.destination_map)
                if landmark is not None and landmark.x is not None and landmark.y is not None:
                    return self._act_on_landmark(
                        state,
                        landmark,
                        f"enter {landmark.label}",
                    )
                if route.edge is not None and route.edge.x is not None and route.edge.y is not None:
                    return self._enter_tile(
                        state,
                        route.edge.x,
                        route.edge.y,
                        f"enter warp toward {route.destination_map or target_map}",
                    )
                if route.direction is not None:
                    return self._walk_boundary(
                        state,
                        route.direction,
                        f"explore toward {route.direction} exit for {target_map}",
                    )
            return self._explore(
                state,
                preferred_direction=None if route is None else route.direction,
                reason=f"explore toward {target_map}",
            )

        landmark = self._goal_landmark(state, goal)
        if landmark is not None:
            return self._act_on_landmark(
                state,
                landmark,
                f"head for {landmark.label}",
            )

        if goal is not None and goal.objective_kind == "talk_to_required_npc":
            npc_action = self._talk_to_visible_npc(state)
            if npc_action is not None:
                return npc_action

        return self._explore(
            state,
            preferred_direction=None if goal is None else goal.next_hop_side,
            reason="explore overworld",
        )

    def reset(self) -> None:
        self._last_action_mode = None
        self._connector_state = None
        self._active_connector = None
        self._active_map_key = None

    def active_connector(self) -> DiscoveredConnector | None:
        return self._active_connector

    def report_failure(
        self,
        state: StructuredGameState,
        action: ActionDecision,
        *,
        turn_index: int,
    ) -> bool:
        if self._report_connector_failure(state, action):
            return False
        if self._last_action_mode != "move":
            return False
        if state.x is None or state.y is None:
            return False
        coordinate = self._advance_position(state.x, state.y, action.action)
        if coordinate is None:
            return False
        self._navigator.mark_blocked(
            state,
            coordinate[0],
            coordinate[1],
            turn_index=turn_index,
        )
        return True

    def export_state(self) -> dict[str, object] | None:
        if self._connector_state is None and self._active_map_key is None:
            return None
        transition_action = None
        if self._connector_state is not None and self._connector_state.transition_action is not None:
            transition_action = self._connector_state.transition_action.value
        return {
            "active_map_key": self._active_map_key,
            "last_action_mode": self._last_action_mode,
            "connector_state": None
            if self._connector_state is None
            else {
                "connector_id": self._connector_state.connector_id,
                "activation_mode": self._connector_state.activation_mode,
                "source_x": self._connector_state.source_x,
                "source_y": self._connector_state.source_y,
                "approach_x": self._connector_state.approach_x,
                "approach_y": self._connector_state.approach_y,
                "transition_action": transition_action,
                "failed_transition_attempts": self._connector_state.failed_transition_attempts,
                "tried_interact_fallback": self._connector_state.tried_interact_fallback,
            },
        }

    def restore_state(self, payload: dict[str, object] | None) -> None:
        self.reset()
        if not payload:
            return
        self._active_map_key = payload.get("active_map_key")
        last_action_mode = payload.get("last_action_mode")
        self._last_action_mode = last_action_mode if isinstance(last_action_mode, str) else None
        connector_payload = payload.get("connector_state")
        if not isinstance(connector_payload, dict):
            return
        transition_action = None
        transition_value = connector_payload.get("transition_action")
        if isinstance(transition_value, str):
            try:
                transition_action = ActionType(transition_value)
            except ValueError:
                transition_action = None
        self._connector_state = ConnectorExecutionState(
            connector_id=connector_payload.get("connector_id")
            if isinstance(connector_payload.get("connector_id"), str)
            else None,
            activation_mode=str(connector_payload.get("activation_mode") or "step_on"),
            source_x=connector_payload.get("source_x")
            if isinstance(connector_payload.get("source_x"), int)
            else None,
            source_y=connector_payload.get("source_y")
            if isinstance(connector_payload.get("source_y"), int)
            else None,
            approach_x=connector_payload.get("approach_x")
            if isinstance(connector_payload.get("approach_x"), int)
            else None,
            approach_y=connector_payload.get("approach_y")
            if isinstance(connector_payload.get("approach_y"), int)
            else None,
            transition_action=transition_action,
            failed_transition_attempts=int(connector_payload.get("failed_transition_attempts", 0)),
            tried_interact_fallback=bool(connector_payload.get("tried_interact_fallback", False)),
        )

    def _local_target(
        self,
        state: StructuredGameState,
        target: NavigationTarget | None,
    ) -> tuple[int, int] | None:
        if target is None or target.x is None or target.y is None:
            return None
        if target.map_name and not map_matches(state.map_name, target.map_name):
            return None
        return (target.x, target.y)

    def _target_map(
        self,
        state: StructuredGameState,
        target: NavigationTarget | None,
        goal: NavigationGoal | None,
    ) -> str | None:
        if target is not None and target.map_name and not map_matches(state.map_name, target.map_name):
            return target.map_name
        if goal is not None:
            return goal.target_map_name
        return None

    def _goal_connector(
        self,
        state: StructuredGameState,
        goal: NavigationGoal,
    ) -> DiscoveredConnector | None:
        if not goal.target_connector_id:
            return None
        connector = self._world_map.connectors.get(goal.target_connector_id)
        if connector is None:
            return None
        if not map_matches(connector.source_map, state.map_name):
            return None
        return connector

    def _goal_landmark(
        self,
        state: StructuredGameState,
        goal: NavigationGoal | None,
    ) -> Landmark | None:
        if goal is None or goal.target_landmark_id is None:
            return None
        landmark = self._static_world_graph.get_landmark(goal.target_landmark_id)
        if landmark is None or not map_matches(landmark.map_name, state.map_name):
            return None
        return landmark

    def _route_landmark(
        self,
        current_map_name: str | None,
        destination_map_name: str | None,
    ) -> Landmark | None:
        if self._landmark_for_destination is None:
            return None
        return self._landmark_for_destination(current_map_name, destination_map_name)

    def _synthesize_static_connector(self, edge) -> DiscoveredConnector | None:
        if edge.x is None or edge.y is None:
            return None
        boundary_side = self._static_warp_boundary_side(edge)
        activation_mode = "push" if boundary_side is not None else "step_on"
        transition_action = self._action_for_direction(boundary_side) if boundary_side is not None else None
        return DiscoveredConnector(
            id=f"{edge.source_name}::tile::{edge.x}:{edge.y}",
            source_map=edge.source_name,
            source_x=edge.x,
            source_y=edge.y,
            kind="warp",
            activation_mode=activation_mode,
            approach_x=edge.x if activation_mode == "push" else None,
            approach_y=edge.y if activation_mode == "push" else None,
            transition_action=transition_action,
            destination_map=edge.destination_name,
        )

    def _static_warp_boundary_side(self, edge) -> str | None:
        if edge.x is None or edge.y is None:
            return None
        if not self._static_warp_uses_boundary_push(edge):
            return None
        graph_map = self._static_world_graph.get_map_by_id(edge.source_symbol)
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

    @staticmethod
    def _static_warp_uses_boundary_push(edge) -> bool:
        return edge.original_destination_symbol == "LAST_MAP" or edge.resolution_method == "last_map_proximity"

    def _walk_to(
        self,
        state: StructuredGameState,
        target_x: int,
        target_y: int,
        reason: str,
    ) -> PlanningResult:
        route = self._navigator.find_local_path(state, target_x, target_y)
        if route is None:
            return self._explore(state, preferred_direction=None, reason=f"reroute after blocked path to ({target_x}, {target_y})")
        if not route:
            return self._emit_action(ActionType.PRESS_A, reason or "confirm arrival", mode="interact")
        return self._emit_action(
            route[0],
            reason or f"walk to ({target_x}, {target_y})",
            mode="move",
            suggested_path=self._navigator.coordinates_for_route(state, route),
        )

    def _walk_adjacent_and_interact(
        self,
        state: StructuredGameState,
        target_x: int,
        target_y: int,
        reason: str,
    ) -> PlanningResult:
        assert state.x is not None and state.y is not None
        if abs(state.x - target_x) + abs(state.y - target_y) == 1:
            face_action = self._facing_action_for_target(state.x, state.y, target_x, target_y)
            expected_facing = self._facing_name_for_action(face_action)
            if face_action is not None and state.facing != expected_facing:
                return self._emit_action(face_action, reason or "face target", mode="face")
            return self._emit_action(ActionType.PRESS_A, reason or "interact", mode="interact")
        adjacent = self._navigator.best_adjacent_tile(target_x, target_y)
        if adjacent is None:
            return self._explore(state, preferred_direction=None, reason=reason or "search for adjacent tile")
        return self._walk_to(state, adjacent[0], adjacent[1], reason)

    def _enter_tile(
        self,
        state: StructuredGameState,
        source_x: int,
        source_y: int,
        reason: str,
    ) -> PlanningResult:
        approach = self._navigator.approach_for_tile(state, source_x, source_y)
        if approach is None:
            return self._explore(state, preferred_direction=None, reason=reason)
        if state.x == approach.x and state.y == approach.y:
            suggested_path = []
            next_position = self._advance_position(state.x, state.y, approach.transition_action)
            if next_position is not None:
                suggested_path = [next_position]
            return self._emit_action(
                approach.transition_action,
                reason,
                mode="move",
                suggested_path=suggested_path,
            )
        return self._walk_to(state, approach.x, approach.y, reason)

    def _enter_connector(
        self,
        state: StructuredGameState,
        connector: DiscoveredConnector,
        reason: str,
    ) -> PlanningResult:
        connector_state = self._ensure_connector_state(connector)
        activation_mode = self._connector_activation_mode(connector)
        if (
            activation_mode == "push"
            and connector_state.source_x is not None
            and connector_state.source_y is not None
            and connector_state.transition_action is not None
        ):
            return self._step_push_connector(state, connector_state, reason)
        if activation_mode == "interact":
            return self._step_interact_connector(state, connector_state, reason)
        approach = self._navigator.approach_for_connector(state, connector)
        if approach is None:
            return self._explore(state, preferred_direction=connector.source_side, reason=reason)
        if state.x == approach.x and state.y == approach.y:
            suggested_path = []
            next_position = self._advance_position(state.x, state.y, approach.transition_action)
            if next_position is not None:
                suggested_path = [next_position]
            return self._emit_action(
                approach.transition_action,
                reason,
                mode="move",
                suggested_path=suggested_path,
            )
        return self._walk_to(state, approach.x, approach.y, reason)

    def _step_push_connector(
        self,
        state: StructuredGameState,
        connector_state: ConnectorExecutionState,
        reason: str,
    ) -> PlanningResult:
        assert connector_state.source_x is not None
        assert connector_state.source_y is not None
        assert connector_state.transition_action is not None
        anchor_x, anchor_y = self._push_anchor(connector_state)
        if state.x is None or state.y is None:
            return self._emit_action(ActionType.PRESS_A, reason or "advance push connector", mode="interact")
        if (state.x, state.y) != (anchor_x, anchor_y):
            return self._walk_to(state, anchor_x, anchor_y, reason)
        if connector_state.failed_transition_attempts >= 2:
            expected_facing = self._facing_name_for_action(connector_state.transition_action)
            if state.facing != expected_facing:
                return self._emit_action(connector_state.transition_action, reason or "face connector", mode="face")
            if not connector_state.tried_interact_fallback:
                return self._emit_action(ActionType.PRESS_A, reason or "interact with connector", mode="connector_interact")
            self._connector_state = None
            return self._explore(state, preferred_direction=None, reason=f"{reason} fallback exhausted")
        suggested_path = []
        next_position = self._advance_position(state.x, state.y, connector_state.transition_action)
        if next_position is not None:
            suggested_path = [next_position]
        return self._emit_action(
            connector_state.transition_action,
            reason or "enter connector",
            mode="connector_transition",
            suggested_path=suggested_path,
        )

    def _step_interact_connector(
        self,
        state: StructuredGameState,
        connector_state: ConnectorExecutionState,
        reason: str,
    ) -> PlanningResult:
        if state.x is None or state.y is None:
            return self._emit_action(ActionType.PRESS_A, reason or "interact with connector", mode="connector_interact")
        approach_x = connector_state.approach_x
        approach_y = connector_state.approach_y
        if approach_x is not None and approach_y is not None and (state.x, state.y) != (approach_x, approach_y):
            return self._walk_to(state, approach_x, approach_y, reason)
        source_x = connector_state.source_x
        source_y = connector_state.source_y
        if source_x is not None and source_y is not None and abs(state.x - source_x) + abs(state.y - source_y) == 1:
            face_action = self._facing_action_for_target(state.x, state.y, source_x, source_y)
            expected_facing = self._facing_name_for_action(face_action)
            if face_action is not None and state.facing != expected_facing:
                return self._emit_action(face_action, reason or "face connector", mode="face")
        return self._emit_action(ActionType.PRESS_A, reason or "enter connector", mode="connector_interact")

    def _walk_boundary(
        self,
        state: StructuredGameState,
        direction: str,
        reason: str,
    ) -> PlanningResult:
        target = self._navigator.best_boundary_target(state, direction)
        action = self._action_for_direction(direction)
        if target is None or action is None:
            return self._explore(state, preferred_direction=None, reason=reason)
        if state.x == target[0] and state.y == target[1]:
            suggested_path = []
            next_position = self._advance_position(target[0], target[1], action)
            if next_position is not None:
                suggested_path = [next_position]
            return self._emit_action(action, reason, mode="move", suggested_path=suggested_path)
        return self._walk_to(state, target[0], target[1], reason)

    def _act_on_landmark(
        self,
        state: StructuredGameState,
        landmark: Landmark,
        reason: str,
    ) -> PlanningResult:
        if landmark.x is None or landmark.y is None:
            return self._explore(state, preferred_direction=None, reason=reason)
        if landmark.type == "sign":
            return self._walk_adjacent_and_interact(state, landmark.x, landmark.y, reason)
        if landmark.destination_name is not None or landmark.type in {
            "important_building",
            "cave_entrance",
            "dungeon_entrance",
            "gym",
            "mart",
            "pokecenter",
            "route_exit",
        }:
            return self._enter_tile(state, landmark.x, landmark.y, reason)
        return self._walk_to(state, landmark.x, landmark.y, reason)

    def _talk_to_visible_npc(self, state: StructuredGameState) -> PlanningResult | None:
        if not state.npcs or state.x is None or state.y is None:
            return None
        best: tuple[tuple[int, int, int], NPCSprite] | None = None
        for npc in state.npcs:
            rank = (
                abs(state.x - npc.tile_x) + abs(state.y - npc.tile_y),
                npc.tile_y,
                npc.tile_x,
            )
            if best is None or rank < best[0]:
                best = (rank, npc)
        if best is None:
            return None
        npc = best[1]
        return self._walk_adjacent_and_interact(
            state,
            npc.tile_x,
            npc.tile_y,
            "talk to nearby NPC",
        )

    def _explore(
        self,
        state: StructuredGameState,
        *,
        preferred_direction: str | None,
        reason: str,
    ) -> PlanningResult:
        directions: list[str] = []
        if preferred_direction:
            directions.append(preferred_direction)
        for direction in ("north", "east", "south", "west"):
            if direction not in directions:
                directions.append(direction)
        for direction in directions:
            action = self._action_for_direction(direction)
            if action is None:
                continue
            target = self._navigator.best_boundary_target(state, direction)
            if target is None:
                continue
            if state.x == target[0] and state.y == target[1]:
                suggested_path = []
                next_position = self._advance_position(target[0], target[1], action)
                if next_position is not None:
                    suggested_path = [next_position]
                return self._emit_action(
                    action,
                    reason or f"explore toward {direction}",
                    mode="move",
                    suggested_path=suggested_path,
                )
            return self._walk_to(state, target[0], target[1], reason or f"explore toward {direction}")
        return self._emit_action(ActionType.PRESS_A, reason or "safe overworld fallback", mode="interact")

    def _emit_action(
        self,
        action: ActionType,
        reason: str,
        *,
        mode: str,
        suggested_path: list[tuple[int, int]] | None = None,
    ) -> PlanningResult:
        self._last_action_mode = mode
        return PlanningResult(
            action=ActionDecision(action=action, repeat=1, reason=reason),
            planner_source="overworld_controller",
            suggested_path=list(suggested_path or []),
        )

    def _ensure_connector_state(self, connector: DiscoveredConnector) -> ConnectorExecutionState:
        activation_mode = self._connector_activation_mode(connector)
        source_x = connector.source_x
        source_y = connector.source_y
        approach_x = connector.approach_x
        approach_y = connector.approach_y
        transition_action = connector.transition_action
        if (
            self._connector_state is not None
            and self._connector_state.connector_id == connector.id
            and self._connector_state.activation_mode == activation_mode
            and self._connector_state.source_x == source_x
            and self._connector_state.source_y == source_y
            and self._connector_state.transition_action == transition_action
        ):
            if self._connector_state.approach_x is None:
                self._connector_state.approach_x = approach_x
            if self._connector_state.approach_y is None:
                self._connector_state.approach_y = approach_y
            self._active_connector = connector
            return self._connector_state
        self._connector_state = ConnectorExecutionState(
            connector_id=connector.id,
            activation_mode=activation_mode,
            source_x=source_x,
            source_y=source_y,
            approach_x=approach_x,
            approach_y=approach_y,
            transition_action=transition_action,
        )
        self._active_connector = connector
        return self._connector_state

    @staticmethod
    def _connector_activation_mode(connector: DiscoveredConnector) -> str:
        if connector.activation_mode in {"step_on", "push", "interact"}:
            return connector.activation_mode
        if connector.transition_action == ActionType.PRESS_A:
            return "interact"
        return "step_on"

    def _report_connector_failure(
        self,
        state: StructuredGameState,
        action: ActionDecision,
    ) -> bool:
        if self._connector_state is None or state.x is None or state.y is None:
            return False
        if self._connector_state.activation_mode == "push":
            anchor_x, anchor_y = self._push_anchor(self._connector_state)
            if (
                self._last_action_mode == "connector_transition"
                and action.action == self._connector_state.transition_action
                and (state.x, state.y) == (anchor_x, anchor_y)
            ):
                self._connector_state.failed_transition_attempts += 1
                return True
            if self._last_action_mode == "face" and action.action == self._connector_state.transition_action:
                return True
            if self._last_action_mode == "connector_interact" and action.action == ActionType.PRESS_A:
                self._connector_state.tried_interact_fallback = True
                return True
            return False
        if (
            self._connector_state.activation_mode == "interact"
            and self._last_action_mode == "connector_interact"
            and action.action == ActionType.PRESS_A
        ):
            self._connector_state = None
            self._active_connector = None
            return True
        return False

    @staticmethod
    def _advance_position(x: int, y: int, action: ActionType) -> tuple[int, int] | None:
        if action not in {
            ActionType.MOVE_UP,
            ActionType.MOVE_RIGHT,
            ActionType.MOVE_DOWN,
            ActionType.MOVE_LEFT,
        }:
            return None
        coordinate = advance_position(x, y, action)
        return (coordinate.x, coordinate.y)

    @staticmethod
    def _action_for_direction(direction: str | None) -> ActionType | None:
        mapping = {
            "north": ActionType.MOVE_UP,
            "east": ActionType.MOVE_RIGHT,
            "south": ActionType.MOVE_DOWN,
            "west": ActionType.MOVE_LEFT,
        }
        return mapping.get(direction or "")

    @staticmethod
    def _facing_action_for_target(
        player_x: int,
        player_y: int,
        target_x: int,
        target_y: int,
    ) -> ActionType | None:
        dx = target_x - player_x
        dy = target_y - player_y
        if (dx, dy) == (0, -1):
            return ActionType.MOVE_UP
        if (dx, dy) == (1, 0):
            return ActionType.MOVE_RIGHT
        if (dx, dy) == (0, 1):
            return ActionType.MOVE_DOWN
        if (dx, dy) == (-1, 0):
            return ActionType.MOVE_LEFT
        return None

    @staticmethod
    def _facing_name_for_action(action: ActionType | None) -> str | None:
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
    def _push_anchor(connector_state: ConnectorExecutionState) -> tuple[int, int]:
        if (
            connector_state.approach_x is not None
            and connector_state.approach_y is not None
            and (connector_state.approach_x, connector_state.approach_y)
            != (connector_state.source_x, connector_state.source_y)
        ):
            return connector_state.approach_x, connector_state.approach_y
        assert connector_state.source_x is not None
        assert connector_state.source_y is not None
        return connector_state.source_x, connector_state.source_y
