import json
import time
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from pokemon_agent.agent.engine import ClosedLoopRunner
from pokemon_agent.agent.context_manager import ActionTrace
from pokemon_agent.agent.navigation import build_navigation_snapshot_from_collision
from pokemon_agent.agent.navigation import build_navigation_snapshot_from_tiles
from pokemon_agent.agent.memory_manager import MemoryManager
from pokemon_agent.agent.planning_types import PlanningResult
from pokemon_agent.agent.progress import ProgressResult
from pokemon_agent.agent.progress import ProgressDetector
from pokemon_agent.agent.stuck_detector import StuckDetector
from pokemon_agent.agent.validator import ActionValidator
import pokemon_agent.agent.menu_manager as menu_manager_module
from pokemon_agent.data.walkthrough import Milestone
from pokemon_agent.emulator.mock import MockEmulatorAdapter
from pokemon_agent.models.action import ActionDecision
from pokemon_agent.models.action import ActionType
from pokemon_agent.models.action import ExecutorStatus
from pokemon_agent.models.action import StepResult
from pokemon_agent.models.action import Task
from pokemon_agent.models.action import TaskKind
from pokemon_agent.models.memory import ConnectorStatus, DiscoveredConnector, DiscoveredMap
from pokemon_agent.models.planner import CandidateNextStep
from pokemon_agent.models.planner import HumanObjectivePlan
from pokemon_agent.models.planner import InternalObjectivePlan
from pokemon_agent.models.planner import Objective
from pokemon_agent.models.planner import ObjectiveHorizon
from pokemon_agent.models.planner import ObjectivePlanEnvelope
from pokemon_agent.models.planner import ObjectiveTarget
from pokemon_agent.models.state import InventoryItem
from pokemon_agent.models.state import GameMode
from pokemon_agent.models.state import NavigationSnapshot
from pokemon_agent.models.state import StructuredGameState
from pokemon_agent.models.state import WorldCoordinate


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


class _LiveOverlayMock(MockEmulatorAdapter):
    def __init__(self) -> None:
        super().__init__()
        self.maps["Mock Town"]["npc"] = None
        self._sync_navigation()
        self.frame = Image.new("RGBA", (160, 144), color=(32, 48, 96, 255))
        self.call_log: list[str] = []
        self.overlay_payloads: list[tuple[StructuredGameState, list[tuple[int, int]]]] = []

    def capture_screen_image(self):
        self.call_log.append("capture")
        return self.frame.copy()

    def set_live_path_overlay(self, state: StructuredGameState, suggested_path: list[tuple[int, int]]) -> None:
        self.call_log.append("set")
        self.overlay_payloads.append((state.model_copy(deep=True), list(suggested_path)))
        self.frame.paste((255, 0, 0, 255), (0, 0, 8, 8))

    def clear_live_path_overlay(self) -> None:
        self.call_log.append("clear")

    def execute_action(self, action):
        self.call_log.append("execute")
        super().execute_action(action)


class _NoNpcMock(MockEmulatorAdapter):
    def __init__(self) -> None:
        super().__init__()
        self.maps["Mock Town"]["npc"] = None
        self._sync_navigation()


class _ScreenAreaMock(_NoNpcMock):
    def get_structured_state(self):
        state = super().get_structured_state()
        game_area = np.zeros((18, 20), dtype=np.uint32)
        collision_area = np.ones((18, 20), dtype=np.uint32)
        game_area[4:6, 4:6] = 99
        collision_area[4:6, 4:6] = 0
        collision_area[8, 0] = 0
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


class _HouseConnectorMock(MockEmulatorAdapter):
    def __init__(self, *, allow_push: bool = True) -> None:
        self.allow_push = allow_push
        super().__init__()
        self.maps = {
            "Red's House 2F": {
                "map_id": 0x26,
                "size": (8, 8),
                "blocked": set(),
                "npc": None,
                "step_on_warp": {(7, 1): ("Red's House 1F", 7, 1)},
                "push_warp": {},
            },
            "Red's House 1F": {
                "map_id": 0x25,
                "size": (8, 8),
                "blocked": set(),
                "npc": None,
                "step_on_warp": {(7, 1): ("Red's House 2F", 7, 1)},
                "push_warp": {
                    (2, 7): (ActionType.MOVE_DOWN, "Pallet Town", 5, 5),
                    (3, 7): (ActionType.MOVE_DOWN, "Pallet Town", 5, 5),
                },
            },
            "Pallet Town": {
                "map_id": 0x00,
                "size": (20, 18),
                "blocked": set(),
                "npc": None,
                "step_on_warp": {},
                "push_warp": {},
            },
        }
        self.state.map_name = "Red's House 2F"
        self.state.map_id = 0x26
        self.state.x = 6
        self.state.y = 1
        self.state.facing = "RIGHT"
        self.state.mode = GameMode.OVERWORLD
        self.state.text_box_open = False
        self.state.menu_open = False
        self.state.battle_state = None
        self.state.metadata = {"last_button": None, "script_state": "connectors"}
        self._sync_navigation()

    def _move(self, action_type: ActionType) -> None:
        if self.state.menu_open or self.state.text_box_open or self.state.battle_state:
            return

        deltas = {
            ActionType.MOVE_UP: (0, -1, "UP"),
            ActionType.MOVE_DOWN: (0, 1, "DOWN"),
            ActionType.MOVE_LEFT: (-1, 0, "LEFT"),
            ActionType.MOVE_RIGHT: (1, 0, "RIGHT"),
        }
        dx, dy, facing = deltas[action_type]
        self.state.facing = facing
        current_map = self.maps[self.state.map_name]
        current_position = (self.state.x or 0, self.state.y or 0)
        push_warp = current_map.get("push_warp", {}).get(current_position)
        if self.allow_push and push_warp is not None and push_warp[0] == action_type:
            self._warp(push_warp[1], push_warp[2], push_warp[3])
            return

        target_x = current_position[0] + dx
        target_y = current_position[1] + dy
        width, height = current_map["size"]
        if target_x < 0 or target_y < 0 or target_x >= width or target_y >= height:
            return
        if (target_x, target_y) in current_map["blocked"]:
            return

        self.state.x = target_x
        self.state.y = target_y
        step_on_warp = current_map.get("step_on_warp", {}).get((target_x, target_y))
        if step_on_warp is not None:
            self._warp(step_on_warp[0], step_on_warp[1], step_on_warp[2])

    def _warp(self, map_name: str, x: int, y: int) -> None:
        self.state.map_name = map_name
        self.state.map_id = self.maps[map_name]["map_id"]
        self.state.x = x
        self.state.y = y
        self.state.mode = GameMode.OVERWORLD

    def _sync_navigation(self) -> None:
        if self.state.mode != GameMode.OVERWORLD:
            self.state.navigation = None
            return
        current_map = self.maps[self.state.map_name]
        width, height = current_map["size"]
        blocked = sorted(current_map["blocked"])
        collision_hash = json.dumps(
            {
                "blocked": blocked,
                "step_on_warp": sorted(current_map.get("step_on_warp", {})),
                "push_warp": sorted(current_map.get("push_warp", {})),
                "allow_push": self.allow_push,
            },
            sort_keys=True,
        )
        self.state.navigation = build_navigation_snapshot_from_tiles(
            width=width,
            height=height,
            player_x=self.state.x,
            player_y=self.state.y,
            blocked_tiles=blocked,
            collision_hash=collision_hash,
        )


class _StubbornDoorMock(_HouseConnectorMock):
    def __init__(self) -> None:
        super().__init__(allow_push=False)
        self.state.map_name = "Red's House 1F"
        self.state.map_id = 0x25
        self.state.x = 2
        self.state.y = 7
        self.state.facing = "DOWN"
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


def _collision_from_logical_grid(logical_grid: list[list[int]]) -> np.ndarray:
    logical = np.array(logical_grid, dtype=np.uint8)
    return np.kron(logical, np.ones((2, 2), dtype=np.uint8))


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


class _TwoStepDialogueMock(_PlainDialogueMock):
    def __init__(self) -> None:
        super().__init__()
        self.state.metadata["dialogue_text"] = "Oak: First line."
        self.state.metadata["dialogue_stage"] = 0

    def execute_action(self, action):
        self.state.step += 1
        if not self.state.text_box_open or action.action != ActionType.PRESS_A:
            return
        if self.state.metadata.get("dialogue_stage", 0) == 0:
            self.state.metadata["dialogue_stage"] = 1
            self.state.metadata["dialogue_text"] = "Oak: Second line."
            return
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


def _objective_plan_payload(payload: dict) -> dict:
    context = payload["context"]
    current_map = context.get("current_map", {}).get("name")
    current_milestone = context.get("current_milestone", {})
    mode = context.get("mode")
    if mode == "text":
        internal_plan = {
            "plan_type": "advance_dialogue",
            "target_map_name": current_map,
            "success_signal": "Dialogue changes or closes",
            "stop_when": "text_box_open",
            "confidence": 0.9,
            "notes": "Text mode objective plan.",
        }
    elif mode == "menu":
        internal_plan = {
            "plan_type": "resolve_menu",
            "target_map_name": current_map,
            "success_signal": "Menu closes",
            "confidence": 0.9,
            "notes": "Menu mode objective plan.",
        }
    elif mode == "battle":
        internal_plan = {
            "plan_type": "battle_default",
            "target_map_name": current_map,
            "success_signal": "Battle ends or state changes",
            "confidence": 0.9,
            "notes": "Battle mode objective plan.",
        }
    else:
        target_map = current_milestone.get("target_map") or current_map
        internal_plan = {
            "plan_type": "go_to_map",
            "target_map_name": target_map,
            "success_signal": f"Arrive at {target_map}",
            "confidence": 0.9,
            "notes": "Compiled by engine via connector table.",
        }
    return {
        "human_plan": {
            "short_term_goal": f"Move toward {internal_plan.get('target_map_name') or current_map}",
            "mid_term_goal": current_milestone.get("next_hint") or f"Travel toward {internal_plan.get('target_map_name') or current_map}",
            "long_term_goal": current_milestone.get("description") or f"Reach {internal_plan.get('target_map_name') or current_map}",
            "current_strategy": "Keep navigation symbolic and let the engine choose the exact route.",
        },
        "internal_plan": internal_plan,
    }


class _SlowCandidateLLM:
    def __init__(self) -> None:
        self.calls = 0

    def complete(self, messages):
        time.sleep(0.05)
        self.calls += 1
        from pokemon_agent.agent.llm_client import CompletionResponse

        payload = json.loads(messages[-1]["content"])
        if "human_plan" in payload.get("response_schema", {}):
            return CompletionResponse(content=json.dumps(_objective_plan_payload(payload)), model="fake")
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
        if "human_plan" in payload.get("response_schema", {}):
            return CompletionResponse(content=json.dumps(_objective_plan_payload(payload)), model="fake")
        candidates = payload["context"]["candidate_next_steps"]
        candidate_id = candidates[min(self.index, len(candidates) - 1)]["id"]
        return CompletionResponse(content=json.dumps({"candidate_id": candidate_id, "reason": "pick candidate"}), model="fake")


class _CaptureMilestoneLLM:
    def __init__(self) -> None:
        self.calls = 0
        self.current_milestone = None
        self.objective_calls = 0

    def complete(self, messages):
        self.calls += 1
        from pokemon_agent.agent.llm_client import CompletionResponse

        payload = json.loads(messages[-1]["content"])
        self.current_milestone = payload["context"]["current_milestone"]
        if "human_plan" in payload.get("response_schema", {}):
            self.objective_calls += 1
            objective_payload = _objective_plan_payload(payload)
            objective_payload["human_plan"]["long_term_goal"] = self.current_milestone["description"]
            return CompletionResponse(content=json.dumps(objective_payload), model="fake")
        candidate_id = payload["context"]["candidate_next_steps"][0]["id"]
        return CompletionResponse(content=json.dumps({"candidate_id": candidate_id, "reason": "inspect milestone"}), model="fake")


class _BadLLM:
    def __init__(self) -> None:
        self.calls = 0

    def complete(self, messages):
        self.calls += 1
        from pokemon_agent.agent.llm_client import CompletionResponse

        return CompletionResponse(content="not json", model="fake")


def test_runner_reports_activity_updates_during_turn():
    emulator = MockEmulatorAdapter()
    _setup_overworld_walk(emulator)
    runner = _build_runner(emulator)
    activity_log: list[tuple[str, str | None]] = []
    runner.set_activity_callback(lambda phase, detail: activity_log.append((phase, detail)))

    result = runner.run_turn(1)

    phases = [phase for phase, _ in activity_log]
    assert result.turn_index == 1
    assert "Planning turn" in phases
    assert "Reading emulator state" in phases
    assert "Evaluating candidates" in phases
    assert "Executing action" in phases
    assert phases[-1] == "Turn complete"


def _build_runner(emulator, llm_client=None):
    return ClosedLoopRunner(
        emulator=emulator,
        memory=MemoryManager(),
        progress=ProgressDetector(),
        stuck=StuckDetector(),
        validator=ActionValidator(max_repeat=4),
        llm_client=llm_client,
    )


def test_plan_action_dispatches_via_mode_dispatcher():
    emulator = MockEmulatorAdapter()
    runner = _build_runner(emulator)
    captured = {}
    expected = PlanningResult(
        action=ActionDecision(action=ActionType.MOVE_UP, repeat=1, reason="dispatcher action"),
        planner_source="dispatcher",
    )

    def fake_dispatch(state, context):
        captured["state"] = state.model_copy(deep=True)
        captured["context"] = context
        return expected

    runner.dispatcher.dispatch = fake_dispatch

    planning = runner._plan_action(emulator.get_structured_state())

    assert planning is expected
    assert captured["state"].map_name == emulator.state.map_name
    assert captured["context"].turn_index == 1
    assert captured["context"].stuck_score == 0
    assert captured["context"].previous_action is None
    assert captured["context"].previous_progress is None


def test_plan_action_bypasses_dispatcher_during_bootstrap():
    emulator = _BootstrapMock()
    runner = _build_runner(emulator)

    def fail_dispatch(_state, _context):
        raise AssertionError("bootstrap planning should bypass the mode dispatcher")

    runner.dispatcher.dispatch = fail_dispatch

    planning = runner._plan_action(emulator.get_structured_state())

    assert planning.planner_source == "bootstrap"
    assert planning.action is not None
    assert planning.action.action == ActionType.PRESS_START


def test_resolve_turn_plan_keeps_active_executor_path_outside_dispatcher():
    emulator = MockEmulatorAdapter()
    runner = _build_runner(emulator)
    task = Task(kind=TaskKind.NAVIGATE_TO, target_x=5, target_y=4, reason="continue cached task")
    runner.executor.is_active = lambda: True
    runner.executor.current_task = lambda: task
    runner.executor.step = lambda _state: StepResult(
        status=ExecutorStatus.STEPPING,
        action=ActionDecision(action=ActionType.MOVE_UP, repeat=1, reason="executor step"),
        suggested_path=[(5, 4)],
    )

    def fail_dispatch(_state, _context):
        raise AssertionError("dispatcher should not run while the executor is still stepping")

    runner.dispatcher.dispatch = fail_dispatch

    before, planning = runner._resolve_turn_plan()

    assert before.map_name == emulator.state.map_name
    assert planning.planner_source == "executor"
    assert planning.action is not None
    assert planning.action.action == ActionType.MOVE_UP
    assert planning.suggested_path == [(5, 4)]


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
    runner.memory.memory.long_term.objective_plan = ObjectivePlanEnvelope(
        human_plan=HumanObjectivePlan(
            short_term_goal="Advance east toward Route 2.",
            mid_term_goal="Use the discovered route through Route 1.",
            long_term_goal="Reach Route 2.",
            current_strategy="Follow the symbolic map objective and let the engine resolve connectors.",
        ),
        internal_plan=InternalObjectivePlan(
            plan_type="go_to_map",
            target_map_name="Route 2",
            success_signal="Arrive on Route 2",
            confidence=0.9,
            notes="Seeded test objective plan.",
        ),
        valid_for_milestone_id="seed_route2",
        valid_for_map_name="Mock Town",
    )


def _seed_house_stairs_only(runner: ClosedLoopRunner) -> None:
    world_map = runner.memory.memory.long_term.world_map
    world_map.maps["Red's House 2F"] = DiscoveredMap(
        map_name="Red's House 2F",
        map_id=0x26,
        connectors=["Red's House 2F::tile::7:1"],
    )
    world_map.maps["Red's House 1F"] = DiscoveredMap(
        map_name="Red's House 1F",
        map_id=0x25,
        connectors=["Red's House 1F::tile::7:1"],
    )
    world_map.connectors["Red's House 2F::tile::7:1"] = DiscoveredConnector(
        id="Red's House 2F::tile::7:1",
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
    world_map.connectors["Red's House 1F::tile::7:1"] = DiscoveredConnector(
        id="Red's House 1F::tile::7:1",
        source_map="Red's House 1F",
        source_x=7,
        source_y=1,
        kind="warp",
        status=ConnectorStatus.CONFIRMED,
        approach_x=6,
        approach_y=1,
        transition_action=ActionType.MOVE_RIGHT,
        destination_map="Red's House 2F",
        destination_x=7,
        destination_y=1,
    )


def test_closed_loop_runner_executes_one_mock_turn():
    emulator = MockEmulatorAdapter()
    runner = _build_runner(emulator)

    result = runner.run_turn(1)

    assert result.llm_attempted is False
    assert result.planner_source in {"auto_candidate", "fallback"}
    assert result.action.action in {ActionType.PRESS_A, ActionType.MOVE_UP}
    assert result.progress.classification in {"interaction_success", "movement_success"}
    assert result.prompt_metrics is not None


def test_runner_publishes_and_clears_live_overlay_around_action():
    emulator = _LiveOverlayMock()
    runner = _build_runner(emulator)
    before = emulator.get_structured_state()
    planned_path = [(before.x + 1, before.y)]
    runner._resolve_turn_plan = lambda: (
        before,
        PlanningResult(
            action=ActionDecision(action=ActionType.MOVE_RIGHT, repeat=1, reason="step right"),
            suggested_path=planned_path,
            planner_source="executor",
        ),
    )

    result = runner.run_turn(1)

    assert emulator.call_log[:4] == ["capture", "set", "execute", "clear"]
    assert emulator.overlay_payloads[0][0].x == before.x
    assert emulator.overlay_payloads[0][1] == planned_path
    assert result.suggested_path == planned_path


def test_runner_keeps_captured_screen_raw_when_live_overlay_mutates_frame():
    emulator = _LiveOverlayMock()
    runner = _build_runner(emulator)
    before = emulator.get_structured_state()
    runner._resolve_turn_plan = lambda: (
        before,
        PlanningResult(
            action=ActionDecision(action=ActionType.MOVE_RIGHT, repeat=1, reason="step right"),
            suggested_path=[(before.x + 1, before.y)],
            planner_source="executor",
        ),
    )

    result = runner.run_turn(1)

    assert result.screen_image is not None
    assert result.screen_image.getpixel((1, 1)) == (32, 48, 96, 255)
    assert emulator.frame.getpixel((1, 1)) == (255, 0, 0, 255)


def test_resolve_turn_plan_falls_back_after_repeated_executor_blocking():
    emulator = MockEmulatorAdapter()
    runner = _build_runner(emulator)
    runner._plan_action = lambda _state: PlanningResult(
        task=Task(kind=TaskKind.ENTER_CONNECTOR, connector_id="missing", reason="blocked connector"),
        planner_source="auto_candidate",
    )
    runner.executor.begin = lambda task, state, follow_up_task=None: StepResult(
        status=ExecutorStatus.BLOCKED,
        blocked_reason="No connector approach found",
    )

    before, planning = runner._resolve_turn_plan()

    assert before.map_name == emulator.state.map_name
    assert planning.action is not None
    assert planning.used_fallback is True
    assert planning.planner_source == "fallback"
    assert planning.raw_response == "planner retry exhaustion: No connector approach found"


def test_runner_checkpoint_round_trip(tmp_path: Path):
    emulator = MockEmulatorAdapter()
    runner = _build_runner(emulator)
    runner.run_turn(1)
    runner.save_checkpoint(tmp_path)

    restored_runner = _build_runner(MockEmulatorAdapter())
    payload = restored_runner.load_checkpoint(tmp_path)

    assert payload["completed_turns"] == 1
    assert "context_state" in payload
    assert "executor_state" in payload
    restored_state = restored_runner.emulator.get_structured_state()
    assert restored_state.text_box_open is False
    assert restored_state.x == 5
    assert restored_state.y == 4
    assert restored_runner.completed_turns == 1
    assert len(restored_runner.context_manager.action_traces) == 1


@pytest.mark.parametrize(
    ("setup", "turns", "expected_action", "expected_classification"),
    [
        (_setup_overworld_walk, 4, "MOVE_UP", "movement_success"),
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
    assert llm.calls >= 1


def test_runner_passes_current_ascii_map_in_llm_payload():
    class _CaptureMapLLM:
        def __init__(self) -> None:
            self.visual_map = None

        def complete(self, messages):
            from pokemon_agent.agent.llm_client import CompletionResponse

            payload = json.loads(messages[-1]["content"])
            self.visual_map = payload["context"]["overworld_context"]["visual_map"]
            if "human_plan" in payload.get("response_schema", {}):
                return CompletionResponse(content=json.dumps(_objective_plan_payload(payload)), model="fake")
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
    assert result.after.x is not None
    assert result.after.x != result.before.x or result.after.y != result.before.y
    assert result.progress.classification == "movement_success"


def test_runner_replans_followup_dialogue_without_llm():
    emulator = _TwoStepDialogueMock()
    llm = _ChooseCandidateLLM()
    runner = _build_runner(emulator, llm_client=llm)

    first = runner.run_turn(1)
    second = runner.run_turn(2)

    assert llm.calls == 1
    assert first.llm_attempted is True
    assert first.planner_source == "dialogue_controller"
    assert second.planner_source == "dialogue_controller"
    assert second.action.action == ActionType.PRESS_A


def test_runner_keeps_executor_task_active_without_repeat_llm_calls():
    emulator = _NoNpcMock()
    llm = _ChooseCandidateLLM(index=0)
    runner = _build_runner(emulator, llm_client=llm)

    first = runner.run_turn(1)

    # Route draining executes the full cached path (+ follow-up) in one turn,
    # so the LLM is only called once for the initial decision.
    assert llm.calls >= 2  # objective planner + action planner
    assert first.planner_source == "llm"
    assert runner.summary()["llm_calls"] == 1


def test_runner_calls_llm_only_when_multiple_candidates_exist():
    emulator = _NoNpcMock()
    llm = _ChooseCandidateLLM(index=1)
    runner = _build_runner(emulator, llm_client=llm)

    result = runner.run_turn(1)

    assert llm.calls >= 2
    assert result.llm_attempted is True
    assert result.planner_source == "llm"
    assert result.prompt_metrics is not None


def test_runner_auto_selects_when_one_candidate_dominates():
    emulator = _StartMenuMock()
    llm = _ChooseCandidateLLM(index=1)
    runner = _build_runner(emulator, llm_client=llm)

    result = runner.run_turn(1)

    assert llm.calls == 1
    assert result.llm_attempted is True
    assert result.planner_source == "menu_controller"
    assert result.action.action == ActionType.MOVE_DOWN


def test_runner_waits_until_reds_house_2f_before_forcing_opening_llm():
    emulator = MockEmulatorAdapter()
    runner = _build_runner(emulator, llm_client=_ChooseCandidateLLM())

    assert runner._should_force_opening_objective_llm(emulator.get_structured_state()) is False


def test_runner_uses_llm_for_first_controllable_reds_house_2f_turn():
    emulator = _HouseConnectorMock()
    llm = _CaptureMilestoneLLM()
    runner = _build_runner(emulator, llm_client=llm)

    result = runner.run_turn(1)

    assert llm.calls >= 2
    assert llm.objective_calls == 1
    assert result.llm_attempted is True
    assert result.planner_source == "llm"
    assert llm.current_milestone["id"] == "get_starter"
    assert llm.current_milestone["target_map"] == "Oak's Lab"
    assert "Professor Oak" in llm.current_milestone["description"]
    assert result.before.map_name == "Red's House 2F"


def test_runner_uses_llm_immediately_after_bootstrap_when_player_reaches_reds_house_2f():
    emulator = _HouseConnectorMock()
    llm = _CaptureMilestoneLLM()
    runner = _build_runner(emulator, llm_client=llm)
    runner.telemetry.llm_calls = 4
    runner.telemetry.turns = 12
    runner.context_manager.action_traces.append(
        ActionTrace(
            turn_index=1,
            action="PRESS_A",
            repeat=1,
            reason="deterministic startup bootstrap",
            progress_classification="major_progress",
            map_name="Intro Cutscene",
            position={"x": None, "y": None},
            event_summaries=["Advanced dialogue"],
            stuck_score=0,
            used_fallback=False,
            llm_attempted=False,
            planner_source="bootstrap",
        )
    )

    result = runner.run_turn(2)

    assert llm.calls >= 2
    assert result.llm_attempted is True
    assert result.planner_source == "llm"


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


def test_runner_uses_menu_controller_when_start_menu_is_open() -> None:
    emulator = _StartMenuMock()
    runner = _build_runner(emulator)

    result = runner.run_turn(1)

    assert result.planner_source == "menu_controller"
    assert result.action.action == ActionType.MOVE_DOWN


def test_runner_prefers_local_recovery_over_press_start_when_stuck_in_overworld() -> None:
    emulator = _StuckOverworldNoNavigationMock()
    runner = _build_runner(emulator)
    runner.stuck.state.score = 44
    runner.stuck.state.recent_failed_actions = ["PRESS_A", "PRESS_A", "PRESS_A", "PRESS_A", "PRESS_B"]
    runner.stuck.state.recovery_hint = "Try a local recovery action before reopening the menu."

    result = runner.run_turn(1)

    assert result.planner_source == "auto_candidate"
    assert result.action.action == ActionType.PRESS_A


def test_runner_resolves_yes_no_text_with_dialogue_controller():
    emulator = _YesNoPromptMock()
    llm = _ChooseCandidateLLM(index=1)
    runner = _build_runner(emulator, llm_client=llm)

    first = runner.run_turn(1)
    second = runner.run_turn(2)

    assert llm.calls >= 2
    assert first.llm_attempted is True
    assert first.planner_source == "dialogue_controller"
    assert first.action.action == ActionType.PRESS_A
    assert second.planner_source in {"auto_candidate", "fallback", "llm"}
    assert emulator.state.metadata["selected"] in {"YES", "NO"}


def test_runner_skips_llm_for_plain_dialogue_text():
    emulator = _PlainDialogueMock()
    llm = _ChooseCandidateLLM(index=1)
    runner = _build_runner(emulator, llm_client=llm)

    result = runner.run_turn(1)

    assert llm.calls == 1
    assert result.llm_attempted is True
    assert result.planner_source == "dialogue_controller"
    assert result.action.action == ActionType.PRESS_A


def test_runner_falls_back_when_llm_response_is_invalid():
    emulator = _NoNpcMock()
    llm = _BadLLM()
    runner = _build_runner(emulator, llm_client=llm)

    result = runner.run_turn(1)

    assert llm.calls >= 1
    assert result.used_fallback is True
    assert result.planner_source == "fallback"
    assert result.llm_attempted is True


def test_runner_restores_cached_route_from_checkpoint(tmp_path: Path):
    emulator = _NoNpcMock()
    llm = _ChooseCandidateLLM(index=0)
    runner = _build_runner(emulator, llm_client=llm)

    first = runner.run_turn(1)
    assert runner.executor.is_active() is True
    runner.save_checkpoint(tmp_path)

    restored_llm = _ChooseCandidateLLM(index=0)
    restored_runner = _build_runner(_NoNpcMock(), llm_client=restored_llm)
    payload = restored_runner.load_checkpoint(tmp_path)

    assert first.planner_source == "llm"
    assert payload["executor_state"] is not None
    assert payload["objective_plan"] is not None


def test_runner_replans_when_route_is_invalidated():
    emulator = _DynamicObstacleMock()
    llm = _ChooseCandidateLLM(index=0)
    runner = _build_runner(emulator, llm_client=llm)

    first = runner.run_turn(1)
    second = runner.run_turn(2)

    assert first.planner_source == "llm"
    assert llm.calls >= 2
    assert second.planner_source in {"executor", "llm", "auto_candidate"}


def test_runner_rechecks_navigation_goal_after_map_change():
    emulator = _MultiRouteMock()
    llm = _ChooseCandidateLLM(index=0)
    runner = _build_runner(emulator, llm_client=llm)
    _seed_discovered_route(runner)

    first = runner.run_turn(1)
    second = runner.run_turn(2)

    assert first.planner_source in {"llm", "auto_candidate"}
    assert first.after.map_name == "Mock Town"
    assert second.before.map_name == "Mock Town"
    assert second.after.map_name == "Route 1"
    assert second.planner_source in {"executor", "llm", "auto_candidate"}
    assert second.action.action in {
        ActionType.MOVE_UP,
        ActionType.MOVE_RIGHT,
        ActionType.MOVE_DOWN,
        ActionType.MOVE_LEFT,
    }


def test_engine_prefers_visible_warp_progress_over_stuck_recovery() -> None:
    runner = _build_runner(MockEmulatorAdapter())
    runner.memory.memory.long_term.objective_plan = ObjectivePlanEnvelope(
        human_plan=HumanObjectivePlan(
            short_term_goal="Descend to Red's House 1F.",
            mid_term_goal="Reach Oak's Lab.",
            long_term_goal="Collect your starter Pokemon from Professor Oak.",
            current_strategy="Use the visible staircase when it is available.",
        ),
        internal_plan=InternalObjectivePlan(
            plan_type="go_to_map",
            target_map_name="Red's House 1F",
            success_signal="Map changes to Red's House 1F.",
            confidence=1.0,
        ),
        valid_for_milestone_id="get_starter",
        valid_for_map_name="Red's House 2F",
    )
    runner.stuck.state.score = runner.stuck.threshold + 5
    runner.stuck.state.recovery_hint = "High stuck score"

    collision_area = _collision_from_logical_grid(
        [
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 0, 0, 0, 0, 0, 1, 1, 1],
            [0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
            [0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
            [0, 0, 1, 0, 0, 0, 0, 1, 1, 1],
            [0, 0, 1, 0, 0, 0, 0, 1, 1, 1],
            [0, 0, 0, 0, 0, 1, 0, 1, 1, 1],
            [0, 0, 0, 0, 0, 1, 0, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        ]
    )
    navigation = build_navigation_snapshot_from_collision(
        collision_area=collision_area,
        player_x=5,
        player_y=4,
        map_width_blocks=4,
        map_height_blocks=4,
        collision_hash="reds-house-2f",
    )
    state = StructuredGameState(
        map_name="Red's House 2F",
        map_id=0x26,
        x=5,
        y=4,
        facing="DOWN",
        mode=GameMode.OVERWORLD,
        step=100,
        navigation=navigation,
    )

    candidates = runner._build_candidate_steps(state)

    assert candidates[0].type != "RECOVER_FROM_STUCK"
    assert any(candidate.type in {"ENTER_CONNECTOR", "EXPLORE_CONNECTOR"} for candidate in candidates)


def test_engine_moves_toward_direct_warp_landmark_when_target_map_is_offscreen() -> None:
    runner = _build_runner(MockEmulatorAdapter())
    runner.memory.memory.long_term.objective_plan = ObjectivePlanEnvelope(
        human_plan=HumanObjectivePlan(
            short_term_goal="Head to Oak's Lab.",
            mid_term_goal="Cross Pallet Town to the lab entrance.",
            long_term_goal="Collect your starter Pokemon from Professor Oak.",
            current_strategy="Use the canonical lab entrance instead of nearby unrelated doors.",
        ),
        internal_plan=InternalObjectivePlan(
            plan_type="go_to_map",
            target_map_name="Oak's Lab",
            success_signal="Map changes to Oak's Lab.",
            confidence=1.0,
        ),
        valid_for_milestone_id="get_starter",
        valid_for_map_name="Pallet Town",
    )
    runner.memory.memory.long_term.world_map.connectors["Pallet Town::tile::5:5"] = DiscoveredConnector(
        id="Pallet Town::tile::5:5",
        source_map="Pallet Town",
        kind="warp",
        status=ConnectorStatus.SUSPECTED,
        source_x=5,
        source_y=5,
    )
    runner.memory.memory.long_term.world_map.maps["Pallet Town"] = DiscoveredMap(
        map_name="Pallet Town",
        connectors=["Pallet Town::tile::5:5"],
    )

    walkable = [
        WorldCoordinate(x=x, y=y)
        for y, row in enumerate(
            [
                "...####...",
                "..##.##...",
                "..........",
                "..........",
                "..........",
                "...####...",
                "..........",
                "..........",
                "..........",
            ],
            start=4,
        )
        for x, cell in enumerate(row)
        if cell == "."
    ]
    blocked = [
        WorldCoordinate(x=x, y=y)
        for y, row in enumerate(
            [
                "...####...",
                "..##.##...",
                "..........",
                "..........",
                "..........",
                "...####...",
                "..........",
                "..........",
                "..........",
            ],
            start=4,
        )
        for x, cell in enumerate(row)
        if cell == "#"
    ]
    state = StructuredGameState(
        map_name="Pallet Town",
        map_id=0x00,
        x=5,
        y=8,
        facing="UP",
        mode=GameMode.OVERWORLD,
        step=100,
        navigation=NavigationSnapshot(
            min_x=0,
            min_y=4,
            max_x=9,
            max_y=12,
            player=WorldCoordinate(x=5, y=8),
            walkable=walkable,
            blocked=blocked,
            coverage="local_window",
            map_width=20,
            map_height=18,
            visible_world_edges=["west"],
            screen_origin_x=0,
            screen_origin_y=4,
        ),
    )

    goal = runner._sync_navigation_goal(state)
    candidates = runner._build_candidate_steps(state)

    assert goal is not None
    assert goal.target_landmark_id == "pallet_town_oak_s_lab"
    assert candidates[0].id.startswith("landmark_window_pallet_town_oak_s_lab")
    assert candidates[0].why == "Move toward the canonical Oak's Lab entrance to reveal its connector."


def test_recover_plan_completion_reverts_to_default_story_plan_without_llm() -> None:
    runner = _build_runner(MockEmulatorAdapter())
    runner.memory.memory.long_term.objective_plan = ObjectivePlanEnvelope(
        human_plan=HumanObjectivePlan(
            short_term_goal="Go downstairs and exit your house to find Professor Oak.",
            mid_term_goal="Complete the Pokemon League challenge by collecting badges.",
            long_term_goal="Become the Champion.",
            current_strategy="Use the stairs to recover.",
        ),
        internal_plan=InternalObjectivePlan(
            plan_type="recover",
            target_map_name="Red's House 1F",
            success_signal="Successfully warping to Red's House 1F.",
            confidence=1.0,
        ),
        valid_for_milestone_id="get_starter",
        valid_for_map_name="Red's House 2F",
        replan_reason="stuck_recovery",
    )

    state = StructuredGameState(
        map_name="Red's House 1F",
        map_id=0x25,
        x=7,
        y=1,
        facing="DOWN",
        mode=GameMode.OVERWORLD,
        step=100,
    )

    runner._ensure_objective_plan(state)

    plan = runner.memory.memory.long_term.objective_plan
    assert plan is not None
    assert plan.internal_plan.plan_type in {"go_to_landmark", "go_to_map"}
    assert plan.internal_plan.target_map_name == "Oak's Lab"
    assert "Oak's Lab" in plan.human_plan.short_term_goal
    assert "Go downstairs" not in plan.human_plan.short_term_goal
    assert plan.valid_for_map_name == "Red's House 1F"


def test_engine_refreshes_opening_target_after_reaching_reds_house_1f() -> None:
    runner = _build_runner(MockEmulatorAdapter())
    _seed_house_stairs_only(runner)
    state = StructuredGameState(
        map_name="Red's House 1F",
        map_id=0x25,
        x=7,
        y=1,
        facing="DOWN",
        mode=GameMode.OVERWORLD,
        step=100,
        navigation=build_navigation_snapshot_from_tiles(
            width=8,
            height=8,
            player_x=7,
            player_y=1,
            blocked_tiles=[(2, 7), (3, 7), (7, 1)],
            collision_hash="reds-house-1f",
        ),
    )

    goal = runner._sync_navigation_goal(state)
    candidates = runner._build_candidate_steps(state)

    assert goal is not None
    assert goal.final_target_map_name == "Oak's Lab"
    assert goal.target_map_name == "Oak's Lab"
    assert goal.next_map_name == "Pallet Town"
    assert all(candidate.id != "connector_red_s_house_1f_tile_7_1" for candidate in candidates)
    assert all("Red's House 2F" not in candidate.why for candidate in candidates)


def test_engine_ignores_stale_local_objective_plan_after_coming_downstairs() -> None:
    runner = _build_runner(MockEmulatorAdapter())
    _seed_house_stairs_only(runner)
    runner._previous_map_name = "Red's House 2F"
    runner.memory.memory.long_term.objective_plan = ObjectivePlanEnvelope(
        human_plan=HumanObjectivePlan(
            short_term_goal="Exit the current bedroom and descend to the first floor of Red's House.",
            mid_term_goal="Navigate from Red's House 2F to Oak's Lab in Pallet Town.",
            long_term_goal="Receive your first Pokemon from Professor Oak to begin your journey.",
            current_strategy="Continue navigating from Red's House 2F to Red's House 1F by using the stairs.",
        ),
        internal_plan=InternalObjectivePlan(
            plan_type="go_to_map",
            target_map_name="REDS_HOUSE_1F",
            success_signal="Map change to Red's House 1F.",
            stop_when="Player is on Red's House 1F.",
            confidence=1.0,
            notes="Seeded stale local target copied from the session failure.",
        ),
        valid_for_milestone_id="get_starter",
        valid_for_map_name="Red's House 2F",
        replan_reason="stale_turn_boundary",
    )
    state = StructuredGameState(
        map_name="Red's House 1F",
        map_id=0x25,
        x=7,
        y=1,
        facing="DOWN",
        mode=GameMode.OVERWORLD,
        step=101,
        navigation=build_navigation_snapshot_from_tiles(
            width=8,
            height=8,
            player_x=7,
            player_y=1,
            blocked_tiles=[(2, 7), (3, 7), (7, 1)],
            collision_hash="reds-house-1f-stale-plan",
        ),
    )

    goal = runner._sync_navigation_goal(state)
    candidates = runner._build_candidate_steps(state)

    assert goal is not None
    assert goal.source == "objective"
    assert goal.target_map_name == "Oak's Lab"
    assert goal.final_target_map_name == "Oak's Lab"
    assert goal.next_map_name == "Pallet Town"
    assert candidates[0].id != "connector_red_s_house_1f_tile_7_1"
    assert all(candidate.id != "connector_red_s_house_1f_tile_7_1" for candidate in candidates)
    assert all("Red's House 2F" not in candidate.why for candidate in candidates)


def test_engine_caps_long_routes_to_four_connectors() -> None:
    runner = _build_runner(MockEmulatorAdapter())
    runner.memory.memory.long_term.objective_plan = ObjectivePlanEnvelope(
        human_plan=HumanObjectivePlan(
            short_term_goal="Head east toward Cerulean City.",
            mid_term_goal="Follow the shortest route across the early gyms.",
            long_term_goal="Reach Cerulean City.",
            current_strategy="Use deterministic connector routing.",
        ),
        internal_plan=InternalObjectivePlan(
            plan_type="go_to_map",
            target_map_name="Cerulean City",
            success_signal="Arrive in Cerulean City.",
            confidence=0.9,
        ),
        valid_for_milestone_id="test_long_route",
        valid_for_map_name="Pallet Town",
    )
    state = StructuredGameState(
        map_name="Pallet Town",
        map_id=0x00,
        x=5,
        y=5,
        facing="UP",
        mode=GameMode.OVERWORLD,
        step=100,
    )

    goal = runner._sync_navigation_goal(state)

    assert goal is not None
    assert goal.target_map_name == "Pewter City"
    assert goal.final_target_map_name == "Cerulean City"
    assert goal.next_map_name == "Route 1"
    assert goal.target_landmark_id is None


def test_engine_recomputes_capped_route_after_intermediate_progress() -> None:
    runner = _build_runner(MockEmulatorAdapter())
    runner.memory.memory.long_term.objective_plan = ObjectivePlanEnvelope(
        human_plan=HumanObjectivePlan(
            short_term_goal="Head east toward Cerulean City.",
            mid_term_goal="Follow the shortest route across the early gyms.",
            long_term_goal="Reach Cerulean City.",
            current_strategy="Use deterministic connector routing.",
        ),
        internal_plan=InternalObjectivePlan(
            plan_type="go_to_map",
            target_map_name="Cerulean City",
            success_signal="Arrive in Cerulean City.",
            confidence=0.9,
        ),
        valid_for_milestone_id="test_long_route",
        valid_for_map_name="Pallet Town",
    )
    state = StructuredGameState(
        map_name="Pewter City",
        map_id=0x02,
        x=10,
        y=10,
        facing="UP",
        mode=GameMode.OVERWORLD,
        step=120,
    )

    goal = runner._sync_navigation_goal(state)

    assert goal is not None
    assert goal.target_map_name == "Cerulean City"
    assert goal.final_target_map_name == "Cerulean City"
    assert goal.next_map_name == "Route 3"


def test_engine_keeps_local_target_on_oaks_lab_when_story_interaction_is_nearby() -> None:
    runner = _build_runner(MockEmulatorAdapter())
    state = StructuredGameState(
        map_name="Oak's Lab",
        map_id=0x28,
        x=4,
        y=6,
        facing="UP",
        mode=GameMode.OVERWORLD,
        step=100,
        navigation=build_navigation_snapshot_from_tiles(
            width=6,
            height=8,
            player_x=4,
            player_y=6,
            blocked_tiles=[(4, 5)],
            collision_hash="oak-lab-near-oak",
        ),
    )

    goal = runner._sync_navigation_goal(state)

    assert goal is not None
    assert goal.target_map_name == "Oak's Lab"
    assert goal.final_target_map_name == "Oak's Lab"
    assert goal.next_map_name is None
    assert goal.objective_kind == "talk_to_required_npc"


def test_engine_resolves_static_door_connector_metadata() -> None:
    runner = _build_runner(MockEmulatorAdapter())

    connector = runner._connector_by_id("static:PALLET_TOWN:3:7")

    assert isinstance(connector, DiscoveredConnector)
    assert connector.source_map == "Red's House 1F"
    assert connector.source_x == 3
    assert connector.source_y == 7
    assert connector.activation_mode == "push"
    assert connector.transition_action == ActionType.MOVE_DOWN
    assert connector.destination_map == "Pallet Town"


def test_runner_handles_stair_then_push_door_connectors_end_to_end() -> None:
    emulator = _HouseConnectorMock()
    runner = _build_runner(emulator, llm_client=_ChooseCandidateLLM(index=0))
    runner.memory.memory.long_term.objective_plan = ObjectivePlanEnvelope(
        human_plan=HumanObjectivePlan(
            short_term_goal="Go downstairs.",
            mid_term_goal="Reach the first floor.",
            long_term_goal="Leave the house.",
            current_strategy="Use the stairs.",
        ),
        internal_plan=InternalObjectivePlan(
            plan_type="go_to_map",
            target_map_name="Red's House 1F",
            success_signal="Map changes to Red's House 1F.",
            confidence=1.0,
        ),
        valid_for_milestone_id="connector_test",
        valid_for_map_name="Red's House 2F",
    )

    first = runner.run_turn(1)

    assert first.action.action == ActionType.MOVE_RIGHT
    assert first.after.map_name == "Red's House 1F"

    emulator.state.x = 2
    emulator.state.y = 6
    emulator.state.facing = "DOWN"
    emulator._sync_navigation()
    runner.memory.memory.long_term.objective_plan = ObjectivePlanEnvelope(
        human_plan=HumanObjectivePlan(
            short_term_goal="Exit through the front door.",
            mid_term_goal="Reach Pallet Town.",
            long_term_goal="Leave the house.",
            current_strategy="Walk onto the door tile, then push through it.",
        ),
        internal_plan=InternalObjectivePlan(
            plan_type="go_to_map",
            target_map_name="Pallet Town",
            success_signal="Map changes to Pallet Town.",
            confidence=1.0,
        ),
        valid_for_milestone_id="connector_test",
        valid_for_map_name="Red's House 1F",
    )

    second = runner.run_turn(2)
    third = runner.run_turn(3)

    assert second.action.action == ActionType.MOVE_DOWN
    assert second.after.map_name == "Red's House 1F"
    assert third.planner_source == "executor"
    assert third.action.action == ActionType.MOVE_DOWN
    assert third.after.map_name == "Pallet Town"


def test_runner_tries_press_a_once_after_repeated_push_failures() -> None:
    emulator = _StubbornDoorMock()
    runner = _build_runner(emulator, llm_client=_ChooseCandidateLLM(index=0))
    runner.memory.memory.long_term.objective_plan = ObjectivePlanEnvelope(
        human_plan=HumanObjectivePlan(
            short_term_goal="Exit through the front door.",
            mid_term_goal="Reach Pallet Town.",
            long_term_goal="Leave the house.",
            current_strategy="Push through the door, then try interacting once if needed.",
        ),
        internal_plan=InternalObjectivePlan(
            plan_type="go_to_map",
            target_map_name="Pallet Town",
            success_signal="Map changes to Pallet Town.",
            confidence=1.0,
        ),
        valid_for_milestone_id="connector_test",
        valid_for_map_name="Red's House 1F",
    )

    first = runner.run_turn(1)
    second = runner.run_turn(2)
    third = runner.run_turn(3)
    fourth = runner.run_turn(4)

    assert [first.action.action, second.action.action, third.action.action] == [
        ActionType.MOVE_DOWN,
        ActionType.MOVE_DOWN,
        ActionType.PRESS_A,
    ]
    assert first.after.map_name == "Red's House 1F"
    assert second.after.map_name == "Red's House 1F"
    assert third.after.map_name == "Red's House 1F"
    assert fourth.planner_source != "executor"


def test_engine_reroutes_after_failed_first_step_with_temporary_blocker() -> None:
    runner = _build_runner(MockEmulatorAdapter())
    navigation = build_navigation_snapshot_from_tiles(
        width=8,
        height=8,
        player_x=3,
        player_y=6,
        blocked_tiles=[],
        collision_hash="open-room",
    )
    state = StructuredGameState(
        map_name="Red's House 2F",
        map_id=0x26,
        x=3,
        y=6,
        facing="LEFT",
        mode=GameMode.OVERWORLD,
        step=100,
        navigation=navigation,
    )
    candidate = CandidateNextStep(
        id="static_warp_reds_house_1f_7_1",
        type="ENTER_CONNECTOR",
        target=ObjectiveTarget(kind="connector", map_name="Red's House 2F", x=6, y=1),
        why="Use the canonical warp to Red's House 1F.",
        priority=88,
        expected_success_signal="The map changes to Red's House 1F.",
        objective_id="local_enter_selected_connector",
    )
    compiled = runner._compile_candidate(candidate, state, "walk to stairs")
    assert compiled is not None
    assert compiled.task is not None

    first = runner.executor.begin(compiled.task, state, follow_up_task=compiled.follow_up_task)
    assert first.action is not None
    assert first.action.action == ActionType.MOVE_UP

    runner.executor.report_move_failed(state, first.action)

    second = runner.executor.step(state)
    assert second.action is not None
    assert second.action.action == ActionType.MOVE_RIGHT


def test_engine_static_explicit_stair_warp_is_not_treated_as_boundary_push() -> None:
    runner = _build_runner(MockEmulatorAdapter())
    connector = runner._static_connector_by_id("static:REDS_HOUSE_1F:7:1")

    assert connector is not None
    assert connector.activation_mode == "step_on"
    assert connector.approach_x is None
    assert connector.approach_y is None
    assert connector.transition_action is None

    navigation = build_navigation_snapshot_from_tiles(
        width=8,
        height=8,
        player_x=3,
        player_y=6,
        blocked_tiles=[],
        collision_hash="open-room",
    )
    state = StructuredGameState(
        map_name="Red's House 2F",
        map_id=0x26,
        x=3,
        y=6,
        facing="LEFT",
        mode=GameMode.OVERWORLD,
        step=100,
        navigation=navigation,
    )
    task = Task(
        kind=TaskKind.ENTER_CONNECTOR,
        connector_id="static:REDS_HOUSE_1F:7:1",
        target_x=7,
        target_y=1,
        reason="walk to stairs",
    )

    first = runner.executor.begin(task, state)

    assert first.status == ExecutorStatus.STEPPING
    assert first.action is not None
    assert first.blocked_reason is None


def test_engine_static_last_map_exit_stays_boundary_push() -> None:
    runner = _build_runner(MockEmulatorAdapter())
    connector = runner._static_connector_by_id("static:PALLET_TOWN:2:7")

    assert connector is not None
    assert connector.activation_mode == "push"
    assert connector.approach_x == 2
    assert connector.approach_y == 7
    assert connector.transition_action == ActionType.MOVE_DOWN


def test_engine_planner_transition_routing_respects_executor_blocked_tiles() -> None:
    runner = _build_runner(MockEmulatorAdapter())
    navigation = build_navigation_snapshot_from_tiles(
        width=8,
        height=8,
        player_x=3,
        player_y=6,
        blocked_tiles=[],
        collision_hash="open-room",
    )
    state = StructuredGameState(
        map_name="Red's House 2F",
        map_id=0x26,
        x=3,
        y=6,
        facing="LEFT",
        mode=GameMode.OVERWORLD,
        step=100,
        navigation=navigation,
    )
    task = Task(
        kind=TaskKind.ENTER_CONNECTOR,
        connector_id="stairs",
        target_x=7,
        target_y=1,
        reason="walk to stairs",
    )

    first = runner.executor.begin(task, state)
    assert first.action is not None
    assert first.action.action == ActionType.MOVE_UP

    runner.executor.report_move_failed(state, first.action)

    approach = runner._approach_for_transition_tile(state, 7, 1)
    assert approach is not None

    route = runner._planning_find_path(state, approach[0], approach[1])
    assert route is not None
    assert route[0] == ActionType.MOVE_RIGHT


def test_engine_planner_transition_routing_respects_recent_failed_move_blockers() -> None:
    runner = _build_runner(MockEmulatorAdapter())
    navigation = build_navigation_snapshot_from_tiles(
        width=8,
        height=8,
        player_x=3,
        player_y=6,
        blocked_tiles=[],
        collision_hash="open-room",
    )
    state = StructuredGameState(
        map_name="Red's House 2F",
        map_id=0x26,
        x=3,
        y=6,
        facing="LEFT",
        mode=GameMode.OVERWORLD,
        step=100,
        navigation=navigation,
    )

    runner._record_failed_move_blocker(
        before=state,
        after=state.model_copy(deep=True),
        action=ActionDecision(action=ActionType.MOVE_UP, repeat=1, reason="walk to stairs"),
        progress=ProgressResult("no_effect"),
        turn_index=114,
    )

    approach = runner._approach_for_transition_tile(state, 7, 1)
    assert approach is not None

    route = runner._planning_find_path(state, approach[0], approach[1])
    assert route is not None
    assert route[0] == ActionType.MOVE_RIGHT


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
    restored_objective_plan = restored_runner.memory.memory.long_term.objective_plan
    assert restored_goal is not None
    assert restored_goal.target_map_name == "Route 2"
    assert restored_objective_plan is not None
    assert restored_objective_plan.internal_plan.target_map_name == "Route 2"


def test_runner_summary_prefers_cached_objective_plan_text():
    emulator = MockEmulatorAdapter()
    runner = _build_runner(emulator)
    runner.memory.memory.long_term.objective_plan = ObjectivePlanEnvelope(
        human_plan=HumanObjectivePlan(
            short_term_goal="Move toward Oak's Lab",
            mid_term_goal="Leave the house and cross Pallet Town.",
            long_term_goal="Collect your starter Pokemon from Professor Oak.",
            current_strategy="Use the cached symbolic objective plan.",
        ),
        internal_plan=InternalObjectivePlan(
            plan_type="go_to_map",
            target_map_name="Oak's Lab",
            success_signal="Enter Oak's Lab",
            confidence=0.8,
        ),
        valid_for_milestone_id="get_starter",
        valid_for_map_name="Red's House 2F",
    )

    summary = runner.summary()

    assert summary["short_term_goal"] == "Move toward Oak's Lab"
    assert summary["current_strategy"] == "Use the cached symbolic objective plan."
