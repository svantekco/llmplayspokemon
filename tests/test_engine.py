import json
import time
from pathlib import Path

from pokemon_agent.agent.engine import ClosedLoopRunner
from pokemon_agent.agent.executor import Executor
from pokemon_agent.agent.memory_manager import MemoryManager
from pokemon_agent.agent.progress import ProgressDetector
from pokemon_agent.agent.prompt_builder import PromptBuilder
from pokemon_agent.agent.stuck_detector import StuckDetector
from pokemon_agent.agent.validator import ActionValidator
from pokemon_agent.agent.evaluator import ScenarioEvaluator
from pokemon_agent.emulator.mock import MockEmulatorAdapter
from pokemon_agent.models.action import ActionType
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
        executor=Executor(emulator),
        memory=MemoryManager(),
        progress=ProgressDetector(),
        stuck=StuckDetector(),
        prompts=PromptBuilder(),
        validator=ActionValidator(max_repeat=4),
        llm_client=llm_client,
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
    restored_state = restored_runner.executor.emulator.get_structured_state()
    assert restored_state.text_box_open is True
    assert restored_runner.completed_turns == 1
    assert len(restored_runner.context_manager.action_traces) == 1


def test_scenario_evaluator_runs_mock_scenarios():
    results = ScenarioEvaluator().run()
    assert results
    assert all(result.passed for result in results)


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
