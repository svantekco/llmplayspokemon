from pokemon_agent.agent.context_manager import ContextManager
from pokemon_agent.agent.engine import ClosedLoopRunner
from pokemon_agent.agent.memory_manager import MemoryManager
from pokemon_agent.agent.progress import ProgressDetector
from pokemon_agent.agent.stuck_detector import StuckDetector
from pokemon_agent.agent.validator import ActionValidator
from pokemon_agent.models.planner import HumanObjectivePlan, InternalObjectivePlan, ObjectivePlanEnvelope
from pokemon_agent.models.memory import MemoryState, NavigationGoal
from pokemon_agent.models.state import GameMode, NavigationSnapshot, StructuredGameState, WorldCoordinate
from pokemon_agent.navigation.world_graph import load_world_graph


class _StubEmulator:
    def get_structured_state(self) -> StructuredGameState:
        return StructuredGameState()

    def execute_action(self, action) -> None:  # pragma: no cover - unused in these tests
        del action


def test_world_graph_routes_early_game_city_path() -> None:
    route = load_world_graph().find_route("Pallet Town", "Viridian City")

    assert route is not None
    assert route.summary() == ["PALLET_TOWN", "ROUTE_1", "VIRIDIAN_CITY"]


def test_world_graph_resolves_viridian_city_pokecenter_warp() -> None:
    warp = load_world_graph().get_warp_at("Viridian City", 23, 25)

    assert warp is not None
    assert warp.destination_symbol == "VIRIDIAN_POKECENTER"
    assert warp.kind == "warp"


def test_world_graph_classifies_viridian_city_landmarks() -> None:
    landmarks = load_world_graph().get_landmarks_on_map("Viridian City")
    landmark_types = {landmark.type for landmark in landmarks}
    landmark_ids = {landmark.id for landmark in landmarks}

    assert {"pokecenter", "mart", "gym", "route_exit", "sign"}.issubset(landmark_types)
    assert "viridian_city_pokecenter" in landmark_ids
    assert "viridian_city_mart" in landmark_ids
    assert "viridian_city_gym" in landmark_ids


def test_world_graph_finds_nearest_pokecenter_from_route_1() -> None:
    result = load_world_graph().nearest_landmark("Route 1", "pokecenter")

    assert result is not None
    assert result.landmark.id == "viridian_city_pokecenter"
    assert result.route.summary() == ["ROUTE_1", "VIRIDIAN_CITY"]


def test_engine_navigation_goal_targets_pewter_gym_landmark() -> None:
    runner = ClosedLoopRunner(
        emulator=_StubEmulator(),
        memory=MemoryManager(),
        progress=ProgressDetector(),
        stuck=StuckDetector(),
        validator=ActionValidator(),
    )
    state = StructuredGameState(
        map_name="Pewter City",
        map_id=0x02,
        mode=GameMode.OVERWORLD,
        story_flags=["got_starter", "oak_received_parcel", "got_pokedex"],
        step=24,
    )

    goal = runner._sync_navigation_goal(state)

    assert goal is not None
    assert goal.target_landmark_id == "pewter_city_gym"
    assert goal.target_landmark_type == "gym"
    assert goal.objective_kind == "reach_landmark"


def test_engine_compiles_objective_plan_landmark_target() -> None:
    runner = ClosedLoopRunner(
        emulator=_StubEmulator(),
        memory=MemoryManager(),
        progress=ProgressDetector(),
        stuck=StuckDetector(),
        validator=ActionValidator(),
    )
    runner.memory.memory.long_term.objective_plan = ObjectivePlanEnvelope(
        human_plan=HumanObjectivePlan(
            short_term_goal="Head to the gym.",
            mid_term_goal="Reach Pewter Gym.",
            long_term_goal="Defeat Brock in Pewter City Gym and earn the Boulder Badge.",
            current_strategy="Use the symbolic landmark target.",
        ),
        internal_plan=InternalObjectivePlan(
            plan_type="go_to_landmark",
            target_map_name="Pewter City",
            target_landmark_id="pewter_city_gym",
            target_landmark_type="gym",
            success_signal="Reach the gym",
            confidence=0.95,
        ),
        valid_for_milestone_id="gym1_brock",
        valid_for_map_name="Pewter City",
    )
    state = StructuredGameState(
        map_name="Pewter City",
        map_id=0x02,
        mode=GameMode.OVERWORLD,
        story_flags=["got_starter", "oak_received_parcel", "got_pokedex"],
        step=24,
    )

    goal = runner._sync_navigation_goal(state)

    assert goal is not None
    assert goal.source == "objective_plan"
    assert goal.target_landmark_id == "pewter_city_gym"
    assert goal.objective_kind == "reach_landmark"


def test_context_manager_adds_canonical_navigation_grounding() -> None:
    memory_state = MemoryState()
    memory_state.long_term.navigation_goal = NavigationGoal(
        target_map_name="Viridian City",
        target_landmark_id="viridian_city_pokecenter",
        target_landmark_type="pokecenter",
        current_map_name="Route 1",
        engine_mode="progression",
    )
    state = StructuredGameState(
        map_name="Route 1",
        map_id=0x0C,
        x=5,
        y=5,
        mode=GameMode.OVERWORLD,
        navigation=NavigationSnapshot(
            min_x=0,
            min_y=0,
            max_x=9,
            max_y=9,
            player=WorldCoordinate(x=5, y=5),
            walkable=[WorldCoordinate(x=5, y=5), WorldCoordinate(x=5, y=4)],
            blocked=[WorldCoordinate(x=6, y=5)],
            collision_hash="route-1",
        ),
    )

    snapshot = ContextManager().build_snapshot(state, memory_state)
    canonical = snapshot.payload["context"]["overworld_context"]["canonical_navigation"]

    assert canonical["current_map"] == "ROUTE_1"
    assert canonical["target_map"] == "VIRIDIAN_CITY"
    assert canonical["target_landmark"]["id"] == "viridian_city_pokecenter"
    assert canonical["nearest_pokecenter"]["landmark_id"] == "viridian_city_pokecenter"
    assert canonical["route_summary"] == ["ROUTE_1", "VIRIDIAN_CITY"]
