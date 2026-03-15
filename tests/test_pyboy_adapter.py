from __future__ import annotations

import numpy as np
from PIL import Image

from pokemon_agent.emulator.pyboy_adapter import PyBoyAdapter
from pokemon_agent.models.state import GameMode
from pokemon_agent.models.state import NavigationSnapshot
from pokemon_agent.models.state import StructuredGameState
from pokemon_agent.models.state import WorldCoordinate
from pokemon_agent.ui.debug_overlay import WorldPathOverlayPainter


class _FakeScreen:
    def __init__(self) -> None:
        self.ndarray = np.full((144, 160, 4), [10, 20, 30, 255], dtype=np.uint8)
        self.image = Image.fromarray(self.ndarray.copy(), "RGBA")


class _FakeWindowSDL2:
    def __init__(self, screen: _FakeScreen) -> None:
        self.screen = screen
        self.frames: list[np.ndarray] = []

    def post_tick(self) -> None:
        self.frames.append(self.screen.ndarray.copy())


class _FakePluginManager:
    def __init__(self, screen: _FakeScreen) -> None:
        self.window_sdl2 = _FakeWindowSDL2(screen)


class _FakePyBoy:
    def __init__(self) -> None:
        self.screen = _FakeScreen()
        self._plugin_manager = _FakePluginManager(self.screen)
        self.calls: list[tuple[str, object, object]] = []

    def tick(self, count=1, render=True, sound=True):
        self.calls.append(("tick", render, sound))
        return True


def _overlay_state() -> StructuredGameState:
    return StructuredGameState(
        map_name="Mock Town",
        map_id="mock_town",
        x=1,
        y=1,
        facing="RIGHT",
        mode=GameMode.OVERWORLD,
        navigation=NavigationSnapshot(
            min_x=0,
            min_y=0,
            max_x=9,
            max_y=8,
            player=WorldCoordinate(x=1, y=1),
            walkable=[
                WorldCoordinate(x=1, y=1),
                WorldCoordinate(x=2, y=1),
                WorldCoordinate(x=3, y=1),
            ],
            blocked=[],
            collision_hash="mock",
            screen_origin_x=0,
            screen_origin_y=0,
        ),
    )


def _build_adapter(*, window: str = "SDL2", live_path_overlay_enabled: bool = False) -> PyBoyAdapter:
    adapter = PyBoyAdapter.__new__(PyBoyAdapter)
    adapter.pyboy = _FakePyBoy()
    adapter.step = 10
    adapter.config = type("Config", (), {"pyboy_window": window})()
    adapter.live_path_overlay_enabled = live_path_overlay_enabled
    adapter._live_overlay_state = None
    adapter._live_suggested_path = []
    adapter._path_overlay_painter = WorldPathOverlayPainter()
    return adapter


def test_pyboy_adapter_pumps_planning_wait_with_tick():
    adapter = _build_adapter()

    adapter.pump_planning_wait()

    assert adapter.pyboy.calls == [("tick", True, False)]
    assert adapter.step == 11


def test_pyboy_adapter_presents_live_path_overlay_and_restores_buffer():
    adapter = _build_adapter(live_path_overlay_enabled=True)
    state = _overlay_state()
    original = adapter.pyboy.screen.ndarray.copy()

    adapter.set_live_path_overlay(state, [(2, 1), (3, 1)])
    adapter.advance_frames(1)

    assert adapter.pyboy._plugin_manager.window_sdl2.frames
    presented = adapter.pyboy._plugin_manager.window_sdl2.frames[0]
    assert adapter.pyboy.screen.ndarray.tolist() == original.tolist()
    assert presented[24, 40].tolist() != original[24, 40].tolist()


def test_pyboy_adapter_live_path_overlay_is_noop_for_non_sdl_windows():
    adapter = _build_adapter(window="null", live_path_overlay_enabled=False)
    state = _overlay_state()

    adapter.set_live_path_overlay(state, [(2, 1), (3, 1)])
    adapter.advance_frames(1)

    assert adapter.pyboy._plugin_manager.window_sdl2.frames == []
