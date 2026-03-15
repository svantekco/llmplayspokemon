from pokemon_agent.agent.world_map import confirm_transition
from pokemon_agent.agent.world_map import observe_state
from pokemon_agent.models.action import ActionDecision
from pokemon_agent.models.action import ActionType
from pokemon_agent.models.memory import ConnectorStatus
from pokemon_agent.models.memory import DiscoveredConnector
from pokemon_agent.models.memory import DiscoveredMap
from pokemon_agent.models.memory import WorldMapMemory
from pokemon_agent.models.state import GameMode
from pokemon_agent.models.state import NavigationSnapshot
from pokemon_agent.models.state import StructuredGameState
from pokemon_agent.models.state import WorldCoordinate


def test_observe_state_latest_tile_observation_wins_when_walkable_and_blocked_conflict() -> None:
    world_map = WorldMapMemory()
    observe_state(
        world_map,
        StructuredGameState(
            map_name="Mock Room",
            map_id="mock_room",
            mode=GameMode.OVERWORLD,
            step=1,
            navigation=NavigationSnapshot(
                min_x=0,
                min_y=0,
                max_x=2,
                max_y=2,
                player=WorldCoordinate(x=1, y=1),
                walkable=[WorldCoordinate(x=0, y=0), WorldCoordinate(x=1, y=1)],
                blocked=[WorldCoordinate(x=2, y=2)],
            ),
        ),
    )

    observe_state(
        world_map,
        StructuredGameState(
            map_name="Mock Room",
            map_id="mock_room",
            mode=GameMode.OVERWORLD,
            step=2,
            navigation=NavigationSnapshot(
                min_x=0,
                min_y=0,
                max_x=2,
                max_y=2,
                player=WorldCoordinate(x=1, y=1),
                walkable=[WorldCoordinate(x=2, y=2), WorldCoordinate(x=1, y=1)],
                blocked=[WorldCoordinate(x=0, y=0)],
            ),
        ),
    )

    discovered_map = world_map.maps["Mock Room"]
    walkable = {(coordinate.x, coordinate.y) for coordinate in discovered_map.walkable}
    blocked = {(coordinate.x, coordinate.y) for coordinate in discovered_map.blocked}
    assert walkable == {(1, 1), (2, 2)}
    assert blocked == {(0, 0)}
    assert not walkable & blocked


def test_observe_state_prunes_bogus_indoor_connectors_and_keeps_canonical_warp() -> None:
    world_map = WorldMapMemory(
        maps={
            "Red's House 2F": DiscoveredMap(
                map_name="Red's House 2F",
                map_id="REDS_HOUSE_2F",
                connectors=[
                    "Red's House 2F::side::north",
                    "Red's House 2F::side::west",
                    "Red's House 2F::side::south",
                    "Red's House 2F::side::east",
                ],
            )
        },
        connectors={
            "Red's House 2F::side::north": DiscoveredConnector(
                id="Red's House 2F::side::north",
                source_map="Red's House 2F",
                source_side="north",
                source_x=0,
                source_y=0,
                kind="door",
            ),
            "Red's House 2F::side::west": DiscoveredConnector(
                id="Red's House 2F::side::west",
                source_map="Red's House 2F",
                source_side="west",
                source_x=0,
                source_y=1,
                kind="boundary",
            ),
            "Red's House 2F::side::south": DiscoveredConnector(
                id="Red's House 2F::side::south",
                source_map="Red's House 2F",
                source_side="south",
                source_x=0,
                source_y=7,
                kind="door",
            ),
            "Red's House 2F::side::east": DiscoveredConnector(
                id="Red's House 2F::side::east",
                source_map="Red's House 2F",
                source_side="east",
                source_x=7,
                source_y=6,
                kind="door",
            ),
        },
    )

    observe_state(
        world_map,
        StructuredGameState(
            map_name="Red's House 2F",
            map_id="REDS_HOUSE_2F",
            mode=GameMode.OVERWORLD,
            step=99,
            navigation=NavigationSnapshot(
                min_x=0,
                min_y=0,
                max_x=7,
                max_y=7,
                player=WorldCoordinate(x=6, y=2),
                walkable=[
                    WorldCoordinate(x=6, y=1),
                    WorldCoordinate(x=6, y=2),
                    WorldCoordinate(x=6, y=6),
                    WorldCoordinate(x=0, y=1),
                ],
                blocked=[
                    WorldCoordinate(x=7, y=1),
                    WorldCoordinate(x=0, y=0),
                ],
            ),
        ),
    )

    discovered_map = world_map.maps["Red's House 2F"]
    assert discovered_map.connectors == ["Red's House 2F::tile::7:1"]
    assert set(world_map.connectors.keys()) == {"Red's House 2F::tile::7:1"}
    connector = world_map.connectors["Red's House 2F::tile::7:1"]
    assert connector.kind == "warp"
    assert connector.status == ConnectorStatus.SUSPECTED
    assert connector.source_x == 7
    assert connector.source_y == 1


def test_confirm_transition_uses_tile_connector_for_canonical_edge_warp() -> None:
    world_map = WorldMapMemory()
    connector = confirm_transition(
        world_map,
        previous=StructuredGameState(
            map_name="Red's House 2F",
            map_id="REDS_HOUSE_2F",
            x=6,
            y=1,
            mode=GameMode.OVERWORLD,
            step=10,
        ),
        action=ActionDecision(action=ActionType.MOVE_RIGHT, reason="Step onto the stairs"),
        current=StructuredGameState(
            map_name="Red's House 1F",
            map_id="REDS_HOUSE_1F",
            x=7,
            y=1,
            mode=GameMode.OVERWORLD,
            step=11,
        ),
    )

    assert connector is not None
    assert connector.id == "Red's House 2F::tile::7:1"
    assert connector.kind == "warp"
    assert connector.status == ConnectorStatus.CONFIRMED
    assert connector.destination_map == "Red's House 1F"


def test_confirm_transition_uses_active_connector_for_push_doors() -> None:
    world_map = WorldMapMemory()
    active_connector = DiscoveredConnector(
        id="static:PALLET_TOWN:3:7",
        source_map="Red's House 1F",
        source_x=3,
        source_y=7,
        kind="warp",
        activation_mode="push",
        approach_x=3,
        approach_y=7,
        transition_action=ActionType.MOVE_DOWN,
        destination_map="Pallet Town",
    )

    connector = confirm_transition(
        world_map,
        previous=StructuredGameState(
            map_name="Red's House 1F",
            map_id="REDS_HOUSE_1F",
            x=3,
            y=7,
            mode=GameMode.OVERWORLD,
            step=20,
        ),
        action=ActionDecision(action=ActionType.MOVE_DOWN, reason="Push through the door"),
        current=StructuredGameState(
            map_name="Pallet Town",
            map_id="PALLET_TOWN",
            x=5,
            y=5,
            mode=GameMode.OVERWORLD,
            step=21,
        ),
        active_connector=active_connector,
    )

    assert connector is not None
    assert connector.id == "Red's House 1F::tile::3:7"
    assert connector.source_x == 3
    assert connector.source_y == 7
    assert connector.activation_mode == "push"
    assert connector.transition_action == ActionType.MOVE_DOWN
    assert connector.destination_map == "Pallet Town"
