from __future__ import annotations

from pokemon_agent.emulator.base import EmulatorAdapter
from pokemon_agent.models.action import ActionDecision


class Executor:
    def __init__(self, emulator: EmulatorAdapter) -> None:
        self.emulator = emulator

    def run(self, action: ActionDecision) -> None:
        self.emulator.execute_action(action)
