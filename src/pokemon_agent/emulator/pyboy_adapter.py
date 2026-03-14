from __future__ import annotations

from pathlib import Path
import time
from pokemon_agent.config import AppConfig
from pokemon_agent.emulator.base import EmulatorAdapter
from pokemon_agent.emulator.pokemon_red_ram_map import rom_profile_metadata
from pokemon_agent.emulator.pokemon_red_ram_map import verify_pokemon_red_rom_file
from pokemon_agent.emulator.state_extractor import PokemonRedStateExtractor
from pokemon_agent.models.action import ActionDecision
from pokemon_agent.models.action import ActionType
from pokemon_agent.models.state import StructuredGameState, GameMode

try:
    from pyboy import PyBoy
    from pyboy.utils import WindowEvent
except Exception:  # pragma: no cover
    PyBoy = None
    WindowEvent = None


class PyBoyAdapter(EmulatorAdapter):
    ACTION_TO_BUTTON = {
        ActionType.MOVE_UP: "UP",
        ActionType.MOVE_DOWN: "DOWN",
        ActionType.MOVE_LEFT: "LEFT",
        ActionType.MOVE_RIGHT: "RIGHT",
        ActionType.PRESS_A: "A",
        ActionType.PRESS_B: "B",
        ActionType.PRESS_START: "START",
    }

    def __init__(self, rom_path: str, config: AppConfig | None = None) -> None:
        if PyBoy is None:
            raise RuntimeError("PyBoy is not installed or failed to import.")
        if not Path(rom_path).exists():
            raise FileNotFoundError(f"ROM not found: {rom_path}")
        self.config = config or AppConfig()
        self.rom_path = rom_path
        self.rom_verification = verify_pokemon_red_rom_file(rom_path)
        self.button_map = self._build_button_map()
        self.pyboy = PyBoy(rom_path, window=self.config.pyboy_window)
        self.step = 0
        self.extractor = PokemonRedStateExtractor(self.pyboy, rom_profile=rom_profile_metadata(self.rom_verification))
        self.advance_frames(self.config.pyboy_boot_frames)

    def get_raw_state(self):
        state = self.get_structured_state()
        return {
            "rom_path": self.rom_path,
            "step": self.step,
            "state": state.model_dump(),
        }

    def get_structured_state(self) -> StructuredGameState:
        try:
            return self.extractor.extract(self.step)
        except Exception as exc:
            return StructuredGameState(
                map_name="UNKNOWN",
                mode=GameMode.UNKNOWN,
                step=self.step,
                metadata={"note": f"State extraction failed: {exc}"},
            )

    def press_button(self, button: str) -> None:
        if WindowEvent is None or button not in self.button_map:
            raise ValueError(f"Unsupported button: {button}")
        press_event, release_event = self.button_map[button]
        self.pyboy.send_input(press_event)
        self.advance_frames(self.config.pyboy_press_frames)
        self.pyboy.send_input(release_event)
        self.advance_frames(2)

    def advance_frames(self, n: int) -> None:
        for _ in range(n):
            self.pyboy.tick()
            self.step += 1

    def execute_action(self, action: ActionDecision) -> None:
        button = self.ACTION_TO_BUTTON[action.action]
        for _ in range(action.repeat):
            self.press_button(button)
            self.advance_frames(self.config.pyboy_post_action_frames)

    def close(self) -> None:
        self.pyboy.stop()

    def _build_button_map(self) -> dict[str, tuple]:
        if WindowEvent is None:
            raise RuntimeError("PyBoy WindowEvent helpers are unavailable.")
        return {
            "UP": (WindowEvent.PRESS_ARROW_UP, WindowEvent.RELEASE_ARROW_UP),
            "DOWN": (WindowEvent.PRESS_ARROW_DOWN, WindowEvent.RELEASE_ARROW_DOWN),
            "LEFT": (WindowEvent.PRESS_ARROW_LEFT, WindowEvent.RELEASE_ARROW_LEFT),
            "RIGHT": (WindowEvent.PRESS_ARROW_RIGHT, WindowEvent.RELEASE_ARROW_RIGHT),
            "A": (WindowEvent.PRESS_BUTTON_A, WindowEvent.RELEASE_BUTTON_A),
            "B": (WindowEvent.PRESS_BUTTON_B, WindowEvent.RELEASE_BUTTON_B),
            "START": (WindowEvent.PRESS_BUTTON_START, WindowEvent.RELEASE_BUTTON_START),
        }

    def save_state(self, path: str | Path) -> None:
        target = Path(path)
        with target.open("wb") as handle:
            self.pyboy.save_state(handle)
        target.with_suffix(".json").write_text(f'{{"step": {self.step}}}', encoding="utf-8")

    def load_state(self, path: str | Path) -> None:
        target = Path(path)
        with target.open("rb") as handle:
            self.pyboy.load_state(handle)
        step_path = target.with_suffix(".json")
        if step_path.exists():
            import json

            payload = json.loads(step_path.read_text(encoding="utf-8"))
            self.step = int(payload.get("step", self.step))

    def begin_planning_wait(self) -> None:
        return None

    def pump_planning_wait(self) -> None:
        try:
            render = self.config.pyboy_window != "null"
            self.pyboy.tick(1, render=render, sound=False)
            self.step += 1
            if not render:
                time.sleep(0.01)
        except Exception:
            time.sleep(0.01)

    def end_planning_wait(self) -> None:
        return None

    def capture_screen_image(self):
        image = self.pyboy.screen.image
        return image.copy() if image is not None else None
