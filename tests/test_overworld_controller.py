from pokemon_agent.agent.controllers.overworld import OverworldController
from pokemon_agent.agent.controllers.protocol import NavigationTarget
from pokemon_agent.agent.controllers.protocol import TurnContext
from pokemon_agent.agent.navigator import Navigator
from pokemon_agent.agent.navigation import build_navigation_snapshot_from_tiles
from pokemon_agent.models.action import ActionType
from pokemon_agent.models.memory import ConnectorStatus
from pokemon_agent.models.memory import DiscoveredConnector
from pokemon_agent.models.memory import DiscoveredMap
from pokemon_agent.models.memory import NavigationGoal
from pokemon_agent.models.memory import WorldMapMemory
from pokemon_agent.models.state import GameMode
from pokemon_agent.models.state import StructuredGameState
from pokemon_agent.navigation.world_graph import load_world_graph


def _state(
    *,
    map_name: str = "Mock Town",
    map_id: str | int = "mock_town",
    x: int,
    y: int,
    width: int = 5,
    height: int = 5,
    blocked_tiles: list[tuple[int, int]] | None = None,
    facing: str = "DOWN",
) -> StructuredGameState:
    return StructuredGameState(
        map_name=map_name,
        map_id=map_id,
        x=x,
        y=y,
        facing=facing,
        mode=GameMode.OVERWORLD,
        navigation=build_navigation_snapshot_from_tiles(
            width=width,
            height=height,
            player_x=x,
            player_y=y,
            blocked_tiles=blocked_tiles or [],
            collision_hash="overworld-grid",
        ),
    )


def _controller(
    world_map: WorldMapMemory | None = None,
    *,
    goal_getter=None,
) -> OverworldController:
    graph = load_world_graph()
    memory = world_map or WorldMapMemory()
    navigator = Navigator(graph, memory)
    return OverworldController(
        navigator,
        memory,
        static_world_graph=graph,
        goal_getter=goal_getter,
        landmark_for_destination=lambda _current, _destination: None,
    )


def test_overworld_controller_walks_to_local_target() -> None:
    controller = _controller()
    state = _state(x=2, y=4, width=5, height=5)

    planning = controller.step(
        state,
        TurnContext(
            turn_index=1,
            navigation_target=NavigationTarget(map_name="Mock Town", x=2, y=0, reason="reach north exit"),
        ),
    )

    assert planning.planner_source == "overworld_controller"
    assert planning.action is not None
    assert planning.action.action == ActionType.MOVE_UP
    assert planning.suggested_path[:2] == [(2, 3), (2, 2)]


def test_overworld_controller_enters_confirmed_connector_toward_target_map() -> None:
    world_map = WorldMapMemory(
        maps={
            "Red's House 2F": DiscoveredMap(map_name="Red's House 2F", connectors=["stairs"]),
            "Red's House 1F": DiscoveredMap(map_name="Red's House 1F"),
        },
        connectors={
            "stairs": DiscoveredConnector(
                id="stairs",
                source_map="Red's House 2F",
                source_x=7,
                source_y=1,
                kind="warp",
                status=ConnectorStatus.CONFIRMED,
                approach_x=6,
                approach_y=1,
                transition_action=ActionType.MOVE_RIGHT,
                destination_map="Red's House 1F",
                destination_x=7,
                destination_y=1,
            )
        },
    )
    goal = NavigationGoal(target_map_name="Red's House 1F", target_connector_id="stairs")
    controller = _controller(world_map, goal_getter=lambda: goal)
    state = _state(
        map_name="Red's House 2F",
        map_id="reds_house_2f",
        x=6,
        y=1,
        width=8,
        height=8,
        blocked_tiles=[(7, 1)],
        facing="RIGHT",
    )

    planning = controller.step(
        state,
        TurnContext(
            turn_index=1,
            navigation_target=NavigationTarget(map_name="Red's House 1F", reason="descend the stairs"),
        ),
    )

    assert planning.action is not None
    assert planning.action.action == ActionType.MOVE_RIGHT
    assert planning.suggested_path == [(7, 1)]


def test_overworld_controller_marks_failed_move_and_repaths() -> None:
    controller = _controller()
    state = _state(x=1, y=2, width=3, height=3)
    context = TurnContext(
        turn_index=1,
        navigation_target=NavigationTarget(map_name="Mock Town", x=1, y=0, reason="walk north"),
    )

    first = controller.step(state, context)
    assert first.action is not None
    assert first.action.action == ActionType.MOVE_UP

    controller.report_failure(state, first.action, turn_index=1)
    second = controller.step(state, context)

    assert second.action is not None
    assert second.action.action in {ActionType.MOVE_LEFT, ActionType.MOVE_RIGHT}
