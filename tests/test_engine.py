import json
import time
from pathlib import Path

import numpy as np
import pytest

from pokemon_agent.agent.engine import ClosedLoopRunner
from pokemon_agent.agent.memory_manager import MemoryManager
from pokemon_agent.agent.progress import ProgressDetector
from pokemon_agent.agent.stuck_detector import StuckDetector
from pokemon_agent.agent.validator import ActionValidator
import pokemon_agent.agent.menu_manager as menu_manager_module
from pokemon_agent.data.walkthrough import Milestone
from pokemon_agent.emulator.mock import MockEmulatorAdapter
from pokemon_agent.models.action import ActionType
from pokemon_agent.models.memory import ConnectorStatus, DiscoveredConnector, DiscoveredMap
from pokemon_agent.models.planner import Objective, ObjectiveHorizon, ObjectiveTarget
from pokemon_agent.models.state import InventoryItem
from pokemon_agent.models.state import GameMode


class _PlanningAwareMock(MockEmulatorAdapter):
    def __init__(self) -> None:
        super().__init__()
        self.planning_pump_count = 0

    def pump_planning_wait(self) -> None:
        self.planning_pump_count += 1


class _PlanningAdvanceMock(MockEmulatorAdapter):
    def __init__(self) -> None:
        super().__init__()
        self.state.metadata["dialogue"] = None
        self._planning_advanced = False

    def pump_planning_wait(self) -> None:
        if self._planning_advanced:
            return
        self.state.x = 6
        self.state.step += 1
        self._planning_advanced = True


class _NoNpcMock(MockEmulatorAdapter):
    def __init__(self) -> None:
        super().__init__()
        self.maps["Mock Town"]["npc"] = None
        self._sync_navigation()


class _ScreenAreaMock(_NoNpcMock):
    def get_structured_state(self):
        state = super().get_structured_state()
        game_area = np.zeros((18, 20), dtype=np.uint32)
        collision_area = np.zeros((18, 20), dtype=np.uint32)
        game_area[4:6, 4:6] = 99
        collision_area[4:6, 4:6] = 1
        collision_area[8, 0] = 1
        state.game_area = game_area.tolist()
        state.collision_area = collision_area.tolist()
        return state


class _DynamicObstacleMock(_NoNpcMock):
    def __init__(self) -> None:
        super().__init__()
        self._obstacle_inserted = False

    def execute_action(self, action):
        super().execute_action(action)
        if self._obstacle_inserted:
            return
        if action.action in {ActionType.MOVE_UP, ActionType.MOVE_DOWN, ActionType.MOVE_LEFT, ActionType.MOVE_RIGHT}:
            current_map = self.maps[self.state.map_name]
            current_map["blocked"].add((self.state.x + 1, self.state.y))
            self._obstacle_inserted = True
            self._sync_navigation()


class _MultiRouteMock(MockEmulatorAdapter):
    def __init__(self) -> None:
        super().__init__()
        self.maps = {
            "Mock Town": {
                "size": (10, 10),
                "blocked": set(),
                "warp": {(9, 5): ("Route 1", 1, 5)},
                "npc": None,
            },
            "Route 1": {
                "size": (4, 8),
                "blocked": set(),
                "warp": {(0, 5): ("Mock Town", 8, 5), (2, 5): ("Route 2", 1, 5)},
                "npc": None,
            },
            "Route 2": {
                "size": (4, 8),
                "blocked": set(),
                "warp": {(0, 5): ("Route 1", 1, 5)},
                "npc": None,
            },
        }
        self.state.map_name = "Mock Town"
        self.state.map_id = "mock_town"
        self.state.x = 7
        self.state.y = 5
        self._sync_navigation()


class _BootstrapMock(MockEmulatorAdapter):
    def __init__(self) -> None:
        super().__init__()
        self.state.map_name = "Title Screen"
        self.state.map_id = None
        self.state.x = None
        self.state.y = None
        self.state.facing = None
        self.state.mode = GameMode.CUTSCENE
        self.state.metadata = {"engine_phase": "bootstrap", "bootstrap_phase": "title_screen"}


_ENCODE = {" ": 0x7F, "'": 0xE0, "!": 0xE7, "?": 0xE6}
_ENCODE.update({letter: 0x80 + index for index, letter in enumerate("ABCDEFGHIJKLMNOPQRSTUVWXYZ")})
_ENCODE.update({letter: 0xA0 + index for index, letter in enumerate("abcdefghijklmnopqrstuvwxyz")})
_CUT_PROGRESS_FLAGS = [
    "got_starter",
    "oak_received_parcel",
    "got_pokedex",
    "beat_brock",
    "beat_misty",
    "got_ss_ticket",
    "got_hm01_cut",
]


def _menu_grid(labels: list[str], *, top_x: int = 12, top_y: int = 2) -> list[list[int]]:
    grid = np.full((18, 20), 383, dtype=np.uint32)
    for index, label in enumerate(labels):
        row = top_y + (index * 2)
        for offset, char in enumerate(label[: max(0, 20 - (top_x + 1))], start=top_x + 1):
            grid[row, offset] = _ENCODE.get(char, 0x7F)
    return grid.tolist()


class _VermilionNoNpcMock(_NoNpcMock):
    def __init__(self) -> None:
        super().__init__()
        self.maps["Vermilion City"] = self.maps.pop("Mock Town")
        self.state.map_name = "Vermilion City"
        self.state.map_id = "vermilion_city"
        self.state.story_flags = list(_CUT_PROGRESS_FLAGS)
        self.state.inventory = [InventoryItem(name="HM Cut", count=1)]
        self._sync_navigation()


class _StartMenuMock(_VermilionNoNpcMock):
    def __init__(self) -> None:
        super().__init__()
        self.state.menu_open = True
        self.state.mode = GameMode.MENU
        self.state.navigation = None
        self.state.game_area = _menu_grid(["POKEDEX", "POKEMON", "ITEM", "SAVE", "EXIT"])
        self.state.metadata["ram_context"] = {
            "ui": {
                "window_y": 0,
                "top_menu_item_x": 12,
                "top_menu_item_y": 2,
                "current_menu_item": 0,
                "max_menu_item": 4,
            }
        }

    def execute_action(self, action):
        self.state.metadata["last_button"] = action.action.value
        self.advance_frames(1)


class _YesNoPromptMock(MockEmulatorAdapter):
    def __init__(self) -> None:
        super().__init__()
        self.maps["Oak's Lab"] = {"size": (6, 6), "blocked": set(), "warp": {}, "npc": None}
        self.state.map_name = "Oak's Lab"
        self.state.map_id = 0x28
        self.state.mode = GameMode.TEXT
        self.state.text_box_open = True
        self.state.menu_open = False
        self.state.navigation = None
        self.state.metadata = {
            "dialogue_text": "Take the item?\nYES\nNO",
            "yes_no_prompt": True,
            "cursor": "YES",
        }

    def execute_action(self, action):
        self.state.step += 1
        if not self.state.text_box_open:
            return
        if action.action == ActionType.MOVE_DOWN:
            self.state.metadata["cursor"] = "NO"
            self.state.metadata["dialogue_text"] = "Take the item?\nYES\n>NO"
            return
        if action.action == ActionType.PRESS_A:
            self.state.metadata["selected"] = self.state.metadata.get("cursor", "YES")
            self.state.text_box_open = False
            self.state.mode = GameMode.OVERWORLD
            self.state.metadata["yes_no_prompt"] = False
            return


class _PlainDialogueMock(MockEmulatorAdapter):
    def __init__(self) -> None:
        super().__init__()
        self.maps["Oak's Lab"] = {"size": (6, 6), "blocked": set(), "warp": {}, "npc": None}
        self.state.map_name = "Oak's Lab"
        self.state.map_id = 0x28
        self.state.mode = GameMode.TEXT
        self.state.text_box_open = True
        self.state.menu_open = False
        self.state.navigation = None
        self.state.metadata = {
            "dialogue_text": "Oak: Take this.",
            "yes_no_prompt": False,
        }

    def execute_action(self, action):
        self.state.step += 1
        if self.state.text_box_open and action.action == ActionType.PRESS_A:
            self.state.text_box_open = False
            self.state.mode = GameMode.OVERWORLD


class _StuckOverworldNoNavigationMock(MockEmulatorAdapter):
    def __init__(self) -> None:
        super().__init__()
        self.maps["Mock Town"]["npc"] = None
        self.state.metadata["dialogue"] = None

    def get_structured_state(self):
        state = self.state.model_copy(deep=True)
        state.navigation = None
        return state


class _SlowCandidateLLM:
    def __init__(self) -> None:
        self.calls = 0

    def complete(self, messages):
        time.sleep(0.05)
        self.calls += 1
        from pokemon_agent.agent.llm_client import CompletionResponse

        payload = json.loads(messages[-1]["content"])
        candidate_id = payload["context"]["candidate_next_steps"][0]["id"]
        return CompletionResponse(content=json.dumps({"candidate_id": candidate_id, "reason": "test"}), model="fake")


class _ChooseCandidateLLM:
    def __init__(self, index: int = 0) -> None:
        self.index = index
        self.calls = 0

    def complete(self, messages):
        self.calls += 1
        from pokemon_agent.agent.llm_client import CompletionResponse

        payload = json.loads(messages[-1]["content"])
        candidates = payload["context"]["candidate_next_steps"]
        candidate_id = candidates[min(self.index, len(candidates) - 1)]["id"]
        return CompletionResponse(content=json.dumps({"candidate_id": candidate_id, "reason": "pick candidate"}), model="fake")


class _BadLLM:
    def __init__(self) -> None:
        self.calls = 0

    def complete(self, messages):
        self.calls += 1
        from pokemon_agent.agent.llm_client import CompletionResponse

        return CompletionResponse(content="not json", model="fake")


def _build_runner(emulator, llm_client=None):
    return ClosedLoopRunner(
        emulator=emulator,
        memory=MemoryManager(),
        progress=ProgressDetector(),
        stuck=StuckDetector(),
        validator=ActionValidator(max_repeat=4),
        llm_client=llm_client,
    )


def _setup_overworld_walk(emulator: MockEmulatorAdapter) -> None:
    emulator.maps["Mock Town"]["npc"] = None
    emulator._sync_navigation()


def _setup_menu_recovery(emulator: MockEmulatorAdapter) -> None:
    emulator.state.menu_open = True
    emulator.state.mode = GameMode.MENU


def _setup_dialogue_recovery(emulator: MockEmulatorAdapter) -> None:
    emulator.state.text_box_open = True
    emulator.state.mode = GameMode.TEXT
    emulator.state.metadata["dialogue"] = "Testing dialogue"


def _setup_battle_recovery(emulator: MockEmulatorAdapter) -> None:
    emulator.state.battle_state = {"kind": "WILD", "opponent": "RATTATA", "enemy_species": "RATTATA"}
    emulator.state.mode = GameMode.BATTLE


def _seed_discovered_route(runner: ClosedLoopRunner) -> None:
    runner.memory.memory.goals.active_objectives = [
        Objective(
            id="long_route2",
            horizon=ObjectiveHorizon.LONG_TERM,
            target=ObjectiveTarget(kind="map", map_name="Route 2"),
        ),
        Objective(
            id="mid_route2",
            horizon=ObjectiveHorizon.MID_TERM,
            target=ObjectiveTarget(kind="map", map_name="Route 2"),
        ),
        Objective(
            id="short_here",
            horizon=ObjectiveHorizon.SHORT_TERM,
            target=ObjectiveTarget(kind="map", map_name="Mock Town", x=7, y=5),
        ),
    ]
    world_map = runner.memory.memory.long_term.world_map
    world_map.maps["Mock Town"] = DiscoveredMap(map_name="Mock Town", map_id="mock_town", connectors=["Mock Town::side::east"])
    world_map.maps["Route 1"] = DiscoveredMap(map_name="Route 1", map_id="route_1", connectors=["Route 1::side::east"])
    world_map.maps["Route 2"] = DiscoveredMap(map_name="Route 2", map_id="route_2", connectors=[])
    world_map.connectors["Mock Town::side::east"] = DiscoveredConnector(
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
    )
    world_map.connectors["Route 1::side::east"] = DiscoveredConnector(
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
    )


def test_closed_loop_runner_executes_one_mock_turn():
    emulator = MockEmulatorAdapter()
    runner = _build_runner(emulator)

    result = runner.run_turn(1)

    assert result.llm_attempted is False
    assert result.planner_source == "auto_candidate"
    assert result.action.action == ActionType.PRESS_A
    assert result.progress.classification == "interaction_success"
    assert result.prompt_metrics is None


def test_runner_checkpoint_round_trip(tmp_path: Path):
    emulator = MockEmulatorAdapter()
    runner = _build_runner(emulator)
    runner.run_turn(1)
    runner.save_checkpoint(tmp_path)

    restored_runner = _build_runner(MockEmulatorAdapter())
    payload = restored_runner.load_checkpoint(tmp_path)

    assert payload["completed_turns"] == 1
    assert "context_state" in payload
    assert "execution_plan" in payload
    restored_state = restored_runner.emulator.get_structured_state()
    assert restored_state.text_box_open is True
    assert restored_runner.completed_turns == 1
    assert len(restored_runner.context_manager.action_traces) == 1


@pytest.mark.parametrize(
    ("setup", "turns", "expected_action", "expected_classification"),
    [
        (_setup_overworld_walk, 4, "MOVE_RIGHT", "movement_success"),
        (_setup_menu_recovery, 2, "PRESS_B", "interaction_success"),
        (_setup_dialogue_recovery, 2, "PRESS_A", "interaction_success"),
        (_setup_battle_recovery, 2, "PRESS_A", "major_progress"),
    ],
)
def test_mock_recovery_scenarios(setup, turns, expected_action, expected_classification):
    emulator = MockEmulatorAdapter()
    setup(emulator)
    runner = _build_runner(emulator)

    results = runner.run(turns)

    assert results
    assert results[0].action.action.value == expected_action
    assert expected_classification in [turn.progress.classification for turn in results]


def test_runner_pumps_emulator_while_waiting_for_llm():
    emulator = _PlanningAwareMock()
    emulator.maps["Mock Town"]["npc"] = None
    emulator._sync_navigation()
    llm = _SlowCandidateLLM()
    runner = _build_runner(emulator, llm_client=llm)

    result = runner.run_turn(1)

    assert result.llm_attempted is True
    assert emulator.planning_pump_count > 0
    assert llm.calls == 1


def test_runner_passes_current_ascii_map_in_llm_payload():
    class _CaptureMapLLM:
        def __init__(self) -> None:
            self.visual_map = None

        def complete(self, messages):
            from pokemon_agent.agent.llm_client import CompletionResponse

            payload = json.loads(messages[-1]["content"])
            self.visual_map = payload["context"]["overworld_context"]["visual_map"]
            candidate_id = payload["context"]["candidate_next_steps"][0]["id"]
            return CompletionResponse(content=json.dumps({"candidate_id": candidate_id, "reason": "use captured map"}), model="fake")

    emulator = _ScreenAreaMock()
    llm = _CaptureMapLLM()
    runner = _build_runner(emulator, llm_client=llm)

    result = runner.run_turn(1)

    assert result.llm_attempted is True
    assert llm.visual_map is not None
    assert "P" in llm.visual_map
    assert "~" in llm.visual_map


def test_runner_skips_llm_during_bootstrap():
    emulator = _BootstrapMock()
    runner = _build_runner(emulator, llm_client=_SlowCandidateLLM())

    result = runner.run_turn(1)

    assert result.llm_attempted is False
    assert result.action.action.value == "PRESS_START"
    assert result.prompt_messages == []
    assert result.prompt_metrics is None


def test_runner_uses_live_state_after_planning_wait():
    emulator = _PlanningAdvanceMock()
    emulator.maps["Mock Town"]["npc"] = None
    emulator._sync_navigation()
    llm = _SlowCandidateLLM()
    runner = _build_runner(emulator, llm_client=llm)

    result = runner.run_turn(1)

    assert result.before.x == 6
    assert result.after.x == 7
    assert result.progress.classification == "movement_success"


def test_runner_uses_execution_plan_for_followup_dialogue_without_llm():
    emulator = MockEmulatorAdapter()
    llm = _ChooseCandidateLLM()
    runner = _build_runner(emulator, llm_client=llm)

    first = runner.run_turn(1)
    second = runner.run_turn(2)
    third = runner.run_turn(3)

    assert llm.calls == 0
    assert first.planner_source == "auto_candidate"
    assert second.planner_source == "auto_candidate"
    assert third.planner_source == "execution_plan"
    assert third.action.action == ActionType.PRESS_A


def test_runner_uses_execution_plan_for_cached_route_without_repeat_llm_calls():
    emulator = _NoNpcMock()
    llm = _ChooseCandidateLLM(index=0)
    runner = _build_runner(emulator, llm_client=llm)

    first = runner.run_turn(1)
    second = runner.run_turn(2)

    assert llm.calls == 1
    assert first.planner_source == "llm"
    assert second.planner_source == "execution_plan"
    assert second.llm_attempted is False
    assert runner.summary()["llm_calls"] == 1


def test_runner_calls_llm_only_when_multiple_candidates_exist():
    emulator = _NoNpcMock()
    llm = _ChooseCandidateLLM(index=1)
    runner = _build_runner(emulator, llm_client=llm)

    result = runner.run_turn(1)

    assert llm.calls == 1
    assert result.llm_attempted is True
    assert result.planner_source == "llm"
    assert result.prompt_metrics is not None


def test_runner_auto_selects_when_one_candidate_dominates():
    emulator = MockEmulatorAdapter()
    llm = _ChooseCandidateLLM(index=1)
    runner = _build_runner(emulator, llm_client=llm)

    result = runner.run_turn(1)

    assert llm.calls == 0
    assert result.planner_source == "auto_candidate"
    assert result.action.action == ActionType.PRESS_A


def test_runner_opens_menu_proactively_for_required_hm(monkeypatch) -> None:
    emulator = _VermilionNoNpcMock()
    monkeypatch.setattr(
        menu_manager_module,
        "get_current_milestone",
        lambda *args, **kwargs: Milestone(
            id="gym3_surge",
            description="Teach Cut and enter the Vermilion Gym.",
            completion_flag=None,
            completion_item=None,
            prerequisite_flags=["got_hm01_cut"],
            prerequisite_items=[],
            target_map_name="Vermilion City",
            route_hints=["Open the menu and prepare Cut."],
            sub_steps=["Teach Cut if needed.", "Use Cut near the gym tree."],
            required_hms=["Cut"],
            next_milestone_id=None,
        ),
    )
    runner = _build_runner(emulator)

    result = runner.run_turn(1)

    assert result.planner_source == "auto_candidate"
    assert result.action.action == ActionType.PRESS_START


def test_runner_uses_menu_manager_when_start_menu_is_open() -> None:
    emulator = _StartMenuMock()
    runner = _build_runner(emulator)

    result = runner.run_turn(1)

    assert result.planner_source == "auto_candidate"
    assert result.action.action == ActionType.MOVE_DOWN


def test_runner_prefers_local_recovery_over_press_start_when_stuck_in_overworld() -> None:
    emulator = _StuckOverworldNoNavigationMock()
    runner = _build_runner(emulator)
    runner.stuck.state.score = 44
    runner.stuck.state.recent_failed_actions = ["PRESS_A", "PRESS_A", "PRESS_A", "PRESS_A", "PRESS_B"]
    runner.stuck.state.recovery_hint = "Try a local recovery action before reopening the menu."

    result = runner.run_turn(1)

    assert result.planner_source == "auto_candidate"
    assert result.action.action in {
        ActionType.MOVE_UP,
        ActionType.MOVE_RIGHT,
        ActionType.MOVE_DOWN,
        ActionType.MOVE_LEFT,
    }


def test_runner_uses_llm_for_yes_no_text_prompt():
    emulator = _YesNoPromptMock()
    llm = _ChooseCandidateLLM(index=1)
    runner = _build_runner(emulator, llm_client=llm)

    first = runner.run_turn(1)
    second = runner.run_turn(2)

    assert llm.calls == 1
    assert first.llm_attempted is True
    assert first.planner_source == "llm"
    assert first.action.action == ActionType.MOVE_DOWN
    assert second.planner_source == "execution_plan"
    assert second.action.action == ActionType.PRESS_A
    assert emulator.state.metadata["selected"] == "NO"


def test_runner_skips_llm_for_plain_dialogue_text():
    emulator = _PlainDialogueMock()
    llm = _ChooseCandidateLLM(index=1)
    runner = _build_runner(emulator, llm_client=llm)

    result = runner.run_turn(1)

    assert llm.calls == 0
    assert result.llm_attempted is False
    assert result.planner_source == "auto_candidate"
    assert result.action.action == ActionType.PRESS_A


def test_runner_falls_back_when_llm_response_is_invalid():
    emulator = _NoNpcMock()
    llm = _BadLLM()
    runner = _build_runner(emulator, llm_client=llm)

    result = runner.run_turn(1)

    assert result.used_fallback is True
    assert result.planner_source == "fallback"
    assert result.llm_attempted is True


def test_runner_restores_cached_route_from_checkpoint(tmp_path: Path):
    emulator = _NoNpcMock()
    llm = _ChooseCandidateLLM(index=0)
    runner = _build_runner(emulator, llm_client=llm)

    first = runner.run_turn(1)
    assert runner.route_cache is not None
    runner.save_checkpoint(tmp_path)

    restored_llm = _ChooseCandidateLLM(index=0)
    restored_runner = _build_runner(_NoNpcMock(), llm_client=restored_llm)
    payload = restored_runner.load_checkpoint(tmp_path)
    second = restored_runner.run_turn(2)

    assert first.planner_source == "llm"
    assert payload["execution_plan"] is not None
    assert second.planner_source == "execution_plan"
    assert restored_llm.calls == 0


def test_runner_replans_when_route_is_invalidated():
    emulator = _DynamicObstacleMock()
    llm = _ChooseCandidateLLM(index=0)
    runner = _build_runner(emulator, llm_client=llm)

    first = runner.run_turn(1)
    second = runner.run_turn(2)

    assert first.planner_source == "llm"
    assert llm.calls == 1
    assert second.planner_source != "execution_plan"
    assert runner.route_cache is None


def test_runner_rechecks_navigation_goal_after_map_change():
    emulator = _MultiRouteMock()
    llm = _ChooseCandidateLLM(index=0)
    runner = _build_runner(emulator, llm_client=llm)
    _seed_discovered_route(runner)

    first = runner.run_turn(1)
    second = runner.run_turn(2)
    third = runner.run_turn(3)

    assert first.planner_source in {"llm", "auto_candidate"}
    assert second.planner_source == "execution_plan"
    assert second.after.map_name == "Route 1"
    assert third.before.map_name == "Route 1"
    assert third.llm_attempted is True
    assert third.planner_source == "llm"
    assert third.action.action == ActionType.MOVE_RIGHT


def test_runner_checkpoint_preserves_world_map_and_navigation_goal(tmp_path: Path):
    emulator = _MultiRouteMock()
    runner = _build_runner(emulator)
    _seed_discovered_route(runner)
    runner.memory.memory.long_term.navigation_goal = runner._sync_navigation_goal(emulator.get_structured_state())

    runner.save_checkpoint(tmp_path)
    restored_runner = _build_runner(_MultiRouteMock())
    restored_runner.load_checkpoint(tmp_path)

    restored_goal = restored_runner.memory.memory.long_term.navigation_goal
    restored_world_map = restored_runner.memory.memory.long_term.world_map
    assert restored_goal is not None
    assert restored_goal.target_map_name == "Route 2"
    assert "Mock Town::side::east" in restored_world_map.connectors
    assert restored_world_map.connectors["Mock Town::side::east"].destination_map == "Route 1"
