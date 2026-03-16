from pokemon_agent.agent.controllers.protocol import NavigationTarget
from pokemon_agent.agent.controllers.protocol import TurnContext
from pokemon_agent.agent.controllers.recovery import RecoveryController
from pokemon_agent.agent.planning_types import PlanningResult
from pokemon_agent.agent.stuck_detector import StuckState
from pokemon_agent.models.action import ActionDecision
from pokemon_agent.models.action import ActionType
from pokemon_agent.models.state import GameMode
from pokemon_agent.models.state import NPCSprite
from pokemon_agent.models.state import NavigationSnapshot
from pokemon_agent.models.state import StructuredGameState
from pokemon_agent.models.state import WorldCoordinate


class _NavigatorStub:
    def __init__(self) -> None:
        self.blocked: set[tuple[int, int]] = set()

    def state_map_key(self, state: StructuredGameState) -> str | int | None:
        return state.map_id if state.map_id is not None else state.map_name

    def blocked_tiles_for_state(self, state: StructuredGameState) -> set[tuple[int, int]]:
        del state
        return set(self.blocked)

    def best_adjacent_tile(self, target_x: int, target_y: int) -> tuple[int, int] | None:
        return (target_x, target_y + 1)

    def find_local_path(
        self,
        state: StructuredGameState,
        target_x: int,
        target_y: int,
    ) -> list[ActionType] | None:
        if state.x == target_x and state.y == target_y:
            return []
        if state.x == target_x and target_y == (state.y or 0) - 1:
            return [ActionType.MOVE_UP]
        if state.y == target_y and target_x == (state.x or 0) + 1:
            return [ActionType.MOVE_RIGHT]
        if state.y == target_y and target_x == (state.x or 0) - 1:
            return [ActionType.MOVE_LEFT]
        if state.x == target_x and target_y == (state.y or 0) + 1:
            return [ActionType.MOVE_DOWN]
        return None


def _state(
    *,
    x: int = 5,
    y: int = 5,
    facing: str = "UP",
    npcs: list[NPCSprite] | None = None,
) -> StructuredGameState:
    walkable = [
        WorldCoordinate(x=5, y=5),
        WorldCoordinate(x=5, y=4),
        WorldCoordinate(x=6, y=5),
        WorldCoordinate(x=5, y=6),
        WorldCoordinate(x=4, y=5),
    ]
    return StructuredGameState(
        map_name="Mock Town",
        map_id="mock_town",
        x=x,
        y=y,
        facing=facing,
        mode=GameMode.OVERWORLD,
        navigation=NavigationSnapshot(
            min_x=0,
            min_y=0,
            max_x=9,
            max_y=9,
            player=WorldCoordinate(x=x, y=y),
            walkable=walkable,
            blocked=[],
            collision_hash="mock-town",
        ),
        npcs=list(npcs or []),
    )


def test_recovery_controller_tries_alternate_path_at_first_tier() -> None:
    stuck = StuckState(score=4, steps_since_progress=4)
    controller = RecoveryController(_NavigatorStub(), stuck_state_getter=lambda: stuck, threshold=4)

    result = controller.step(
        _state(),
        TurnContext(
            turn_index=5,
            stuck_score=4,
            previous_action=ActionDecision(action=ActionType.MOVE_UP, repeat=1, reason="blocked"),
            navigation_target=NavigationTarget(map_name="Mock Town", x=7, y=5, reason="reach target"),
        ),
    )

    assert result is not None
    assert result.planner_source == "recovery_controller"
    assert result.action is not None
    assert result.action.action == ActionType.MOVE_RIGHT


def test_recovery_controller_interacts_nearby_at_second_tier() -> None:
    stuck = StuckState(score=6, steps_since_progress=6)
    controller = RecoveryController(_NavigatorStub(), stuck_state_getter=lambda: stuck, threshold=4)

    result = controller.step(
        _state(npcs=[NPCSprite(sprite_index=1, tile_x=5, tile_y=4)], facing="UP"),
        TurnContext(turn_index=6, stuck_score=6),
    )

    assert result is not None
    assert result.action is not None
    assert result.action.action == ActionType.PRESS_A


def test_recovery_controller_backtracks_before_random_walk() -> None:
    stuck = StuckState(score=8, steps_since_progress=8)
    controller = RecoveryController(_NavigatorStub(), stuck_state_getter=lambda: stuck, threshold=4)

    result = controller.step(
        _state(),
        TurnContext(
            turn_index=8,
            stuck_score=8,
            previous_action=ActionDecision(action=ActionType.MOVE_RIGHT, repeat=1, reason="blocked"),
        ),
    )

    assert result is not None
    assert result.action is not None
    assert result.action.action == ActionType.MOVE_LEFT


def test_recovery_controller_uses_escalation_callback_at_fourth_tier() -> None:
    stuck = StuckState(score=10, steps_since_progress=10)
    calls: list[int] = []

    def escalate(state: StructuredGameState, context: TurnContext) -> PlanningResult:
        del state, context
        calls.append(1)
        return PlanningResult(
            action=ActionDecision(action=ActionType.PRESS_B, repeat=1, reason="reroute objective"),
            planner_source="overworld_controller",
        )

    controller = RecoveryController(
        _NavigatorStub(),
        stuck_state_getter=lambda: stuck,
        threshold=4,
        escalation_callback=escalate,
    )

    result = controller.step(_state(), TurnContext(turn_index=10, stuck_score=10))

    assert result is not None
    assert result.planner_source == "recovery_controller"
    assert result.action is not None
    assert result.action.action == ActionType.PRESS_B
    assert len(calls) == 1
