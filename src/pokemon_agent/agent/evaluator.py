from __future__ import annotations

from dataclasses import dataclass, field
from pokemon_agent.agent.engine import ClosedLoopRunner
from pokemon_agent.agent.executor import Executor
from pokemon_agent.agent.memory_manager import MemoryManager
from pokemon_agent.agent.progress import ProgressDetector
from pokemon_agent.agent.prompt_builder import PromptBuilder
from pokemon_agent.agent.stuck_detector import StuckDetector
from pokemon_agent.agent.validator import ActionValidator
from pokemon_agent.emulator.mock import MockEmulatorAdapter
from pokemon_agent.models.state import GameMode


@dataclass(slots=True)
class EvalScenario:
    name: str
    turns: int
    setup: callable


@dataclass(slots=True)
class EvalResult:
    name: str
    turns: int
    fallback_turns: int
    llm_calls: int
    auto_selected_turns: int
    no_effect_turns: int
    major_progress: int
    interaction_success: int
    movement_success: int
    max_stuck: int
    passed: bool
    notes: list[str] = field(default_factory=list)


class ScenarioEvaluator:
    def __init__(self) -> None:
        self.scenarios = [
            EvalScenario("overworld_walk", 4, self._setup_overworld_walk),
            EvalScenario("menu_recovery", 2, self._setup_menu_recovery),
            EvalScenario("dialogue_recovery", 2, self._setup_dialogue_recovery),
            EvalScenario("battle_recovery", 2, self._setup_battle_recovery),
        ]

    def run(self) -> list[EvalResult]:
        results: list[EvalResult] = []
        for scenario in self.scenarios:
            emulator = MockEmulatorAdapter()
            scenario.setup(emulator)
            runner = ClosedLoopRunner(
                executor=Executor(emulator),
                memory=MemoryManager(),
                progress=ProgressDetector(),
                stuck=StuckDetector(),
                prompts=PromptBuilder(),
                validator=ActionValidator(max_repeat=4),
                llm_client=None,
            )
            turns = runner.run(scenario.turns)
            classifications = [turn.progress.classification for turn in turns]
            first_action = turns[0].action.action.value
            passed, notes = self._judge_scenario(scenario.name, first_action, classifications)
            results.append(
                EvalResult(
                    name=scenario.name,
                    turns=len(turns),
                    fallback_turns=sum(1 for turn in turns if turn.used_fallback),
                    llm_calls=runner.summary().get("llm_calls", 0),
                    auto_selected_turns=runner.summary().get("auto_selected_turns", 0),
                    no_effect_turns=runner.summary().get("no_effect_turns", 0),
                    major_progress=sum(1 for turn in turns if turn.progress.classification == "major_progress"),
                    interaction_success=sum(1 for turn in turns if turn.progress.classification == "interaction_success"),
                    movement_success=sum(1 for turn in turns if turn.progress.classification == "movement_success"),
                    max_stuck=max(turn.stuck_state.score for turn in turns),
                    passed=passed,
                    notes=notes,
                )
            )
        return results

    def _judge_scenario(self, name: str, first_action: str, classifications: list[str]) -> tuple[bool, list[str]]:
        notes: list[str] = []
        passed = True
        if name == "overworld_walk":
            passed = "movement_success" in classifications
            notes.append(f"first_action={first_action}")
        elif name == "menu_recovery":
            passed = first_action == "PRESS_B" and "interaction_success" in classifications
            notes.append(f"first_action={first_action}")
        elif name == "dialogue_recovery":
            passed = first_action == "PRESS_A"
            notes.append(f"first_action={first_action}")
        elif name == "battle_recovery":
            passed = first_action == "PRESS_A" and "major_progress" in classifications
            notes.append(f"first_action={first_action}")
        return passed, notes

    def _setup_overworld_walk(self, emulator: MockEmulatorAdapter) -> None:
        emulator.maps["Mock Town"]["npc"] = None
        emulator._sync_navigation()

    def _setup_menu_recovery(self, emulator: MockEmulatorAdapter) -> None:
        emulator.state.menu_open = True
        emulator.state.mode = GameMode.MENU

    def _setup_dialogue_recovery(self, emulator: MockEmulatorAdapter) -> None:
        emulator.state.text_box_open = True
        emulator.state.mode = GameMode.TEXT
        emulator.state.metadata["dialogue"] = "Testing dialogue"

    def _setup_battle_recovery(self, emulator: MockEmulatorAdapter) -> None:
        emulator.state.battle_state = {"kind": "WILD", "opponent": "RATTATA"}
        emulator.state.mode = GameMode.BATTLE
