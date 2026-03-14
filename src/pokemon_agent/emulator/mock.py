from __future__ import annotations

import json
from pathlib import Path
from pokemon_agent.agent.navigation import build_navigation_snapshot_from_tiles
from pokemon_agent.emulator.base import EmulatorAdapter
from pokemon_agent.models.action import ActionDecision, ActionType
from pokemon_agent.models.state import BattleContext
from pokemon_agent.models.state import InventoryItem
from pokemon_agent.models.state import PartyMember
from pokemon_agent.models.state import StructuredGameState, GameMode


class MockEmulatorAdapter(EmulatorAdapter):
    def __init__(self) -> None:
        self.maps = {
            "Mock Town": {
                "size": (10, 10),
                "blocked": {(4, 5), (4, 6), (2, 2)},
                "warp": {(9, 5): ("Route 1", 1, 5)},
                "npc": (6, 5),
            },
            "Route 1": {
                "size": (12, 8),
                "blocked": {(3, 3), (3, 4), (8, 6)},
                "warp": {(0, 5): ("Mock Town", 8, 5)},
                "npc": None,
            },
        }
        self.received_parcel = False
        self.dialogue_stage = 0
        self.state = StructuredGameState(
            map_name="Mock Town",
            map_id="mock_town",
            x=5,
            y=5,
            facing="DOWN",
            mode=GameMode.OVERWORLD,
            party=[PartyMember(name="Charmander", hp=20, max_hp=20)],
            inventory=[],
            metadata={"last_button": None, "script_state": "intro"},
        )
        self._sync_navigation()

    def get_raw_state(self):
        return self.state.model_dump()

    def get_structured_state(self) -> StructuredGameState:
        self._sync_navigation()
        return self.state.model_copy(deep=True)

    def press_button(self, button: str) -> None:
        self.state.metadata["last_button"] = button

    def advance_frames(self, n: int) -> None:
        self.state.step += n

    def execute_action(self, action: ActionDecision) -> None:
        for _ in range(action.repeat):
            self.state.metadata["last_button"] = action.action.value
            if action.action in {
                ActionType.MOVE_UP,
                ActionType.MOVE_DOWN,
                ActionType.MOVE_LEFT,
                ActionType.MOVE_RIGHT,
            }:
                self._move(action.action)
            elif action.action == ActionType.PRESS_A:
                self._press_a()
            elif action.action == ActionType.PRESS_B:
                self._press_b()
            elif action.action == ActionType.PRESS_START:
                self._press_start()
            self.advance_frames(1)
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
        target_x = (self.state.x or 0) + dx
        target_y = (self.state.y or 0) + dy
        width, height = current_map["size"]
        if target_x < 0 or target_y < 0 or target_x >= width or target_y >= height:
            return
        if (target_x, target_y) in current_map["blocked"]:
            return

        self.state.x = target_x
        self.state.y = target_y
        warp = current_map["warp"].get((target_x, target_y))
        if warp:
            map_name, new_x, new_y = warp
            self.state.map_name = map_name
            self.state.map_id = map_name.lower().replace(" ", "_")
            self.state.x = new_x
            self.state.y = new_y
            self.state.mode = GameMode.OVERWORLD

    def _press_a(self) -> None:
        if self.state.battle_state:
            self.state.battle_state = None
            self.state.mode = GameMode.OVERWORLD
            self.state.metadata["last_battle_result"] = "resolved"
            return

        if self.state.menu_open:
            self.state.menu_open = False
            self.state.mode = GameMode.OVERWORLD
            return

        if self.state.text_box_open:
            if self.dialogue_stage == 0:
                self.dialogue_stage = 1
                self.state.metadata["dialogue"] = "Received Parcel!"
            else:
                self.state.text_box_open = False
                self.state.mode = GameMode.OVERWORLD
                self.state.metadata["dialogue"] = None
                if not self.received_parcel:
                    self.received_parcel = True
                    self.state.inventory.append(InventoryItem(name="Parcel", count=1))
                    self.state.metadata["script_state"] = "parcel_received"
            return

        npc = self.maps[self.state.map_name]["npc"]
        if npc and abs((self.state.x or 0) - npc[0]) + abs((self.state.y or 0) - npc[1]) == 1:
            self.state.text_box_open = True
            self.state.mode = GameMode.TEXT
            self.dialogue_stage = 0
            self.state.metadata["dialogue"] = "Aide: Take this parcel."
            return

        if self.state.map_name == "Route 1" and (self.state.x, self.state.y) == (5, 2) and not self.state.battle_state:
            self.state.battle_state = BattleContext(kind="WILD", opponent="PIDGEY", enemy_species="PIDGEY")
            self.state.mode = GameMode.BATTLE

    def _press_b(self) -> None:
        if self.state.text_box_open:
            self.state.text_box_open = False
        elif self.state.menu_open:
            self.state.menu_open = False
        elif self.state.battle_state:
            self.state.battle_state = None
            self.state.metadata["last_battle_result"] = "escaped"
        self.state.mode = GameMode.OVERWORLD if not self.state.battle_state else GameMode.BATTLE

    def _press_start(self) -> None:
        if self.state.text_box_open or self.state.battle_state:
            return
        self.state.menu_open = not self.state.menu_open
        self.state.mode = GameMode.MENU if self.state.menu_open else GameMode.OVERWORLD

    def _sync_navigation(self) -> None:
        if self.state.mode != GameMode.OVERWORLD:
            self.state.navigation = None
            return
        current_map = self.maps[self.state.map_name]
        width, height = current_map["size"]
        blocked = set(current_map["blocked"])
        npc = current_map.get("npc")
        if npc is not None:
            blocked.add(npc)
        blocked.update(current_map.get("warp", {}).keys())
        self.state.navigation = build_navigation_snapshot_from_tiles(
            width=width,
            height=height,
            player_x=self.state.x,
            player_y=self.state.y,
            blocked_tiles=blocked,
            collision_hash=self._collision_hash(current_map),
        )

    def _collision_hash(self, current_map: dict) -> str:
        blocked = sorted(tuple(item) for item in current_map["blocked"])
        npc = current_map.get("npc")
        warp = sorted(tuple(item) for item in current_map.get("warp", {}).keys())
        return json.dumps({"blocked": blocked, "npc": npc, "warp": warp}, sort_keys=True)

    def save_state(self, path: str | Path) -> None:
        target = Path(path)
        payload = {
            "state": self.state.model_dump(),
            "received_parcel": self.received_parcel,
            "dialogue_stage": self.dialogue_stage,
        }
        target.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def load_state(self, path: str | Path) -> None:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        self.state = StructuredGameState.model_validate(payload["state"])
        self.received_parcel = bool(payload["received_parcel"])
        self.dialogue_stage = int(payload["dialogue_stage"])
