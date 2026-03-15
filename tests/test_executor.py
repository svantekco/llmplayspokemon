from pokemon_agent.agent.executor import Executor
from pokemon_agent.agent.navigation import build_navigation_snapshot_from_tiles
from pokemon_agent.models.action import ActionType
from pokemon_agent.models.action import ExecutorStatus
from pokemon_agent.models.action import Task
from pokemon_agent.models.action import TaskKind
from pokemon_agent.models.memory import DiscoveredConnector
from pokemon_agent.models.state import GameMode
from pokemon_agent.models.state import StructuredGameState


def _state(
    *,
    x: int,
    y: int,
    width: int = 5,
    height: int = 5,
    blocked_tiles: list[tuple[int, int]] | None = None,
    facing: str = "DOWN",
    mode: GameMode = GameMode.OVERWORLD,
    text_box_open: bool = False,
) -> StructuredGameState:
    navigation = build_navigation_snapshot_from_tiles(
        width=width,
        height=height,
        player_x=x,
        player_y=y,
        blocked_tiles=blocked_tiles or [],
        collision_hash="mock-grid",
    )
    return StructuredGameState(
        map_name="Mock Town",
        map_id="mock_town",
        x=x,
        y=y,
        facing=facing,
        mode=mode,
        text_box_open=text_box_open,
        navigation=navigation,
        step=1,
    )


def test_executor_completes_navigation_task_over_multiple_steps() -> None:
    executor = Executor()
    task = Task(kind=TaskKind.NAVIGATE_TO, target_x=2, target_y=2, reason="walk to tile")

    first = executor.begin(task, _state(x=2, y=4))
    assert first.status == ExecutorStatus.STEPPING
    assert first.action is not None
    assert first.action.action == ActionType.MOVE_UP

    done = executor.step(_state(x=2, y=2))
    assert done.status == ExecutorStatus.DONE
    assert executor.is_active() is False


def test_executor_reports_blocked_after_repeated_failed_move_reports() -> None:
    executor = Executor(max_retries=3)
    state = _state(x=1, y=2, width=3, height=3)
    task = Task(kind=TaskKind.NAVIGATE_TO, target_x=1, target_y=0, reason="climb corridor")

    first = executor.begin(task, state)
    assert first.action is not None
    assert first.action.action == ActionType.MOVE_UP

    executor.report_move_failed(state, first.action)
    second = executor.step(state)
    assert second.action is not None
    assert second.action.action in {ActionType.MOVE_LEFT, ActionType.MOVE_RIGHT}

    executor.report_move_failed(state, second.action)
    third = executor.step(state)
    assert third.action is not None
    assert third.action.action in {ActionType.MOVE_LEFT, ActionType.MOVE_RIGHT}

    executor.report_move_failed(state, third.action)
    blocked = executor.step(state)
    assert blocked.status == ExecutorStatus.BLOCKED
    assert blocked.blocked_reason is not None


def test_executor_interrupts_and_resumes_navigation_task() -> None:
    executor = Executor()
    task = Task(kind=TaskKind.NAVIGATE_TO, target_x=2, target_y=0, reason="walk north")

    first = executor.begin(task, _state(x=2, y=2))
    assert first.status == ExecutorStatus.STEPPING

    interrupted = executor.step(_state(x=2, y=1, mode=GameMode.TEXT, text_box_open=True))
    assert interrupted.status == ExecutorStatus.INTERRUPTED
    assert executor.is_active() is True

    resumed = executor.step(_state(x=2, y=1))
    assert resumed.status == ExecutorStatus.STEPPING
    assert resumed.action is not None
    assert resumed.action.action == ActionType.MOVE_UP


def test_executor_navigate_adjacent_applies_facing_then_follow_up_interact() -> None:
    executor = Executor()
    task = Task(kind=TaskKind.NAVIGATE_ADJACENT, target_x=2, target_y=1, reason="talk to blocker")
    follow_up = Task(kind=TaskKind.INTERACT, reason="talk to blocker")

    state = _state(x=1, y=1, blocked_tiles=[(2, 1)], facing="DOWN")
    first = executor.begin(task, state, follow_up_task=follow_up)
    assert first.status == ExecutorStatus.STEPPING
    assert first.action is not None
    assert first.action.action == ActionType.MOVE_RIGHT

    second = executor.step(_state(x=1, y=1, blocked_tiles=[(2, 1)], facing="RIGHT"))
    assert second.status == ExecutorStatus.STEPPING
    assert second.action is not None
    assert second.action.action == ActionType.PRESS_A


def test_executor_recomputes_connector_approach_after_failed_move() -> None:
    connector = DiscoveredConnector(
        id="stairs",
        source_map="Red's House 2F",
        source_x=6,
        source_y=1,
        kind="warp",
        approach_x=6,
        approach_y=2,
        transition_action=ActionType.MOVE_UP,
    )
    executor = Executor(lambda connector_id: connector if connector_id == "stairs" else None)
    state = _state(x=3, y=6, width=8, height=8)
    task = Task(
        kind=TaskKind.ENTER_CONNECTOR,
        connector_id="stairs",
        target_x=6,
        target_y=1,
        reason="walk to stairs",
    )

    first = executor.begin(task, state)
    assert first.status == ExecutorStatus.STEPPING
    assert first.action is not None
    assert first.action.action == ActionType.MOVE_UP

    executor.report_move_failed(state, first.action)
    second = executor.step(state)
    assert second.status == ExecutorStatus.STEPPING
    assert second.action is not None
    assert second.action.action == ActionType.MOVE_RIGHT


def test_executor_steps_off_walkable_connector_tile_to_reapproach() -> None:
    connector = DiscoveredConnector(
        id="stairs",
        source_map="Red's House 2F",
        source_x=6,
        source_y=1,
        kind="warp",
        approach_x=6,
        approach_y=2,
        transition_action=ActionType.MOVE_UP,
    )
    executor = Executor(lambda connector_id: connector if connector_id == "stairs" else None)
    task = Task(
        kind=TaskKind.ENTER_CONNECTOR,
        connector_id="stairs",
        target_x=6,
        target_y=1,
        reason="walk to stairs",
    )

    result = executor.begin(task, _state(x=6, y=1, width=8, height=8))

    assert result.status == ExecutorStatus.STEPPING
    assert result.action is not None
    assert result.action.action == ActionType.MOVE_DOWN
    assert result.suggested_path == [(6, 2)]
