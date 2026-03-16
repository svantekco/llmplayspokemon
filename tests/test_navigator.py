from pokemon_agent.agent.navigator import Navigator
from pokemon_agent.agent.navigation import build_navigation_snapshot_from_tiles
from pokemon_agent.models.action import ActionType
from pokemon_agent.models.memory import ConnectorStatus
from pokemon_agent.models.memory import DiscoveredConnector
from pokemon_agent.models.memory import DiscoveredMap
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
) -> StructuredGameState:
    return StructuredGameState(
        map_name=map_name,
        map_id=map_id,
        x=x,
        y=y,
        mode=GameMode.OVERWORLD,
        navigation=build_navigation_snapshot_from_tiles(
            width=width,
            height=height,
            player_x=x,
            player_y=y,
            blocked_tiles=blocked_tiles or [],
            collision_hash="navigator-grid",
        ),
    )


def test_navigator_avoids_and_expires_temporary_blockers() -> None:
    navigator = Navigator(load_world_graph(), WorldMapMemory(), blocked_ttl=2)
    state = _state(x=2, y=4, width=5, height=5)

    navigator.update(state, turn_index=1)
    assert navigator.find_local_path(state, 2, 0) == [
        ActionType.MOVE_UP,
        ActionType.MOVE_UP,
        ActionType.MOVE_UP,
        ActionType.MOVE_UP,
    ]

    navigator.mark_blocked(state, 2, 3, turn_index=1)
    navigator.update(state, turn_index=1)
    assert navigator.find_local_path(state, 2, 0) == [
        ActionType.MOVE_RIGHT,
        ActionType.MOVE_UP,
        ActionType.MOVE_UP,
        ActionType.MOVE_UP,
        ActionType.MOVE_UP,
        ActionType.MOVE_LEFT,
    ]

    navigator.prune_blocked(4)
    navigator.update(state, turn_index=4)
    assert navigator.find_local_path(state, 2, 0) == [
        ActionType.MOVE_UP,
        ActionType.MOVE_UP,
        ActionType.MOVE_UP,
        ActionType.MOVE_UP,
    ]


def test_navigator_returns_confirmed_route_step_between_maps() -> None:
    world_map = WorldMapMemory(
        maps={
            "Mock Town": DiscoveredMap(map_name="Mock Town", connectors=["Mock Town::side::east"]),
            "Route 1": DiscoveredMap(map_name="Route 1", connectors=["Route 1::side::east"]),
            "Route 2": DiscoveredMap(map_name="Route 2"),
        },
        connectors={
            "Mock Town::side::east": DiscoveredConnector(
                id="Mock Town::side::east",
                source_map="Mock Town",
                source_side="east",
                kind="boundary",
                status=ConnectorStatus.CONFIRMED,
                approach_x=8,
                approach_y=5,
                transition_action=ActionType.MOVE_RIGHT,
                destination_map="Route 1",
                destination_x=1,
                destination_y=5,
            ),
            "Route 1::side::east": DiscoveredConnector(
                id="Route 1::side::east",
                source_map="Route 1",
                source_side="east",
                kind="boundary",
                status=ConnectorStatus.CONFIRMED,
                approach_x=1,
                approach_y=5,
                transition_action=ActionType.MOVE_RIGHT,
                destination_map="Route 2",
                destination_x=1,
                destination_y=5,
            ),
        },
    )
    navigator = Navigator(load_world_graph(), world_map)

    route = navigator.next_route_step("Mock Town", "Route 2")

    assert route is not None
    assert route.connector is not None
    assert route.connector.id == "Mock Town::side::east"
    assert route.destination_map == "Route 1"
    assert route.direction == "east"
