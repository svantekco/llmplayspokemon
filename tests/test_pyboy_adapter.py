from __future__ import annotations

from pokemon_agent.emulator.pyboy_adapter import PyBoyAdapter


class _FakePyBoy:
    def __init__(self) -> None:
        self.calls: list[tuple[str, object, object]] = []

    def tick(self, count=1, render=True, sound=True):
        self.calls.append(("tick", render, sound))
        return True


def test_pyboy_adapter_pumps_planning_wait_with_tick():
    adapter = PyBoyAdapter.__new__(PyBoyAdapter)
    adapter.pyboy = _FakePyBoy()
    adapter.step = 10
    adapter.config = type("Config", (), {"pyboy_window": "SDL2"})()

    adapter.pump_planning_wait()

    assert adapter.pyboy.calls == [("tick", True, False)]
    assert adapter.step == 11
