from __future__ import annotations

from hashlib import sha1
from pokemon_agent.agent.navigation import build_navigation_snapshot_from_collision
from pokemon_agent.data.map_names import POKEMON_RED_MAP_NAMES
from pokemon_agent.emulator.pokemon_red_ram_map import build_ram_context
from pokemon_agent.emulator.text_reader import decode_screen_text
from pokemon_agent.emulator.text_reader import detect_yes_no_prompt
from pokemon_agent.models.state import BattleContext
from pokemon_agent.models.state import InventoryItem
from pokemon_agent.models.state import GameMode
from pokemon_agent.models.state import PartyMember
from pokemon_agent.models.state import StructuredGameState

KNOWN_MAP_NAMES = POKEMON_RED_MAP_NAMES


class PokemonRedStateExtractor:
    def __init__(self, pyboy, rom_profile: dict | None = None) -> None:
        self.pyboy = pyboy
        self.rom_profile = rom_profile or {}

    def extract(self, step: int) -> StructuredGameState:
        ram_context = build_ram_context(self.pyboy.memory)
        player = ram_context["player"]
        ui = ram_context["ui"]
        battle = ram_context["battle"]
        party = self._build_party(ram_context)
        inventory = self._build_inventory(ram_context)
        story_flags = list(ram_context.get("story", {}).get("active_flags", []))
        badges = list(player.get("badges", {}).get("names", []))
        map_id = player["map_id"]
        x = player["x"]
        y = player["y"]
        battle_flag = battle["flag"]
        moving_direction_raw = player["moving_direction_raw"]
        direction_raw = player["direction_raw"]
        joy_ignore = ui["joy_ignore"]
        window_y = ui["window_y"]

        window_visible = window_y < 0x90
        text_box_open = window_visible and joy_ignore != 0 and battle_flag == 0
        menu_open = window_visible and not text_box_open and battle_flag == 0
        mode = self._infer_mode(battle_flag, menu_open, text_box_open)
        game_area = self.pyboy.game_wrapper.game_area()
        collision_area = self.pyboy.game_wrapper.game_area_collision()
        dialogue_text = decode_screen_text(game_area) if text_box_open else None
        yes_no_prompt = detect_yes_no_prompt(game_area) if text_box_open else False
        tile_hash = self._hash_grid(game_area)
        collision_hash = self._hash_grid(collision_area)
        map_context = ram_context.get("map", {})
        navigation = None
        if mode == GameMode.OVERWORLD:
            navigation = build_navigation_snapshot_from_collision(
                collision_area=collision_area,
                player_x=x,
                player_y=y,
                map_width_blocks=map_context.get("width") if isinstance(map_context, dict) else None,
                map_height_blocks=map_context.get("height") if isinstance(map_context, dict) else None,
                collision_hash=collision_hash,
            )
        bootstrap_phase = self._detect_bootstrap_phase(
            step=step,
            map_id=map_id,
            x=x,
            y=y,
            battle_flag=battle_flag,
            game_area=game_area,
            ram_context=ram_context,
        )

        if bootstrap_phase is not None:
            bootstrap_text_box_open = window_visible and bootstrap_phase == "intro_cutscene"
            bootstrap_dialogue_text = decode_screen_text(game_area) if bootstrap_text_box_open else None
            bootstrap_yes_no_prompt = detect_yes_no_prompt(game_area) if bootstrap_text_box_open else False
            return StructuredGameState(
                map_name=self._bootstrap_label(bootstrap_phase),
                map_id=None,
                x=None,
                y=None,
                facing=None,
                mode=GameMode.CUTSCENE,
                menu_open=bootstrap_phase == "title_menu",
                text_box_open=bootstrap_text_box_open,
                battle_state=None,
                navigation=None,
                party=party,
                inventory=inventory,
                story_flags=story_flags,
                badges=badges,
                step=step,
                game_area=self._serialize_grid(game_area),
                collision_area=self._serialize_grid(collision_area),
                metadata={
                    "tile_hash": tile_hash,
                    "collision_hash": collision_hash,
                    "battle_flag": battle_flag,
                    "window_y": window_y,
                    "joy_ignore": joy_ignore,
                    "direction_raw": direction_raw,
                    "moving_direction_raw": moving_direction_raw,
                    "raw_map_id": map_id,
                    "raw_x": x,
                    "raw_y": y,
                    "dialogue_text": bootstrap_dialogue_text,
                    "yes_no_prompt": bootstrap_yes_no_prompt,
                    "story_flags": story_flags,
                    "badges": badges,
                    "ram_profile": self.rom_profile,
                    "ram_context": ram_context,
                    "engine_phase": "bootstrap",
                    "bootstrap_phase": bootstrap_phase,
                    "confirmed_fields": ["story_flags", "badges"],
                    "heuristic_fields": [
                        "bootstrap_phase",
                        "map_name",
                        "menu_open",
                        "text_box_open",
                        "mode",
                    ],
                },
            )

        return StructuredGameState(
            map_name=KNOWN_MAP_NAMES.get(map_id, f"MAP_{map_id:02X}"),
            map_id=map_id,
            x=x,
            y=y,
            facing=self._decode_direction(direction_raw, moving_direction_raw),
            mode=mode,
            menu_open=menu_open,
            text_box_open=text_box_open,
            battle_state=self._battle_state(battle),
            navigation=navigation,
            party=party,
            inventory=inventory,
            story_flags=story_flags,
            badges=badges,
            step=step,
            game_area=self._serialize_grid(game_area),
            collision_area=self._serialize_grid(collision_area),
            metadata={
                "tile_hash": tile_hash,
                "collision_hash": collision_hash,
                "battle_flag": battle_flag,
                "window_y": window_y,
                "joy_ignore": joy_ignore,
                "direction_raw": direction_raw,
                "moving_direction_raw": moving_direction_raw,
                "dialogue_text": dialogue_text,
                "yes_no_prompt": yes_no_prompt,
                "story_flags": story_flags,
                "badges": badges,
                "ram_profile": self.rom_profile,
                "ram_context": ram_context,
                "engine_phase": "active",
                "confirmed_fields": ["map_id", "x", "y", "battle_state", "party", "inventory", "story_flags", "badges"],
                "heuristic_fields": ["map_name", "facing", "menu_open", "text_box_open", "mode"],
            },
        )

    def _battle_state(self, battle_payload: dict[str, object]) -> BattleContext | None:
        battle_flag = int(battle_payload.get("flag", 0))
        if battle_flag == 0:
            return None
        context = battle_payload.get("context")
        if isinstance(context, dict):
            return BattleContext.model_validate(context)
        kind = str(battle_payload.get("kind") or f"UNKNOWN_{battle_flag}")
        return BattleContext(kind=kind)

    def _build_party(self, ram_context: dict[str, object]) -> list[PartyMember]:
        return [PartyMember.model_validate(member) for member in ram_context.get("party", [])]

    def _build_inventory(self, ram_context: dict[str, object]) -> list[InventoryItem]:
        return [InventoryItem.model_validate(item) for item in ram_context.get("inventory", [])]

    def _infer_mode(self, battle_flag: int, menu_open: bool, text_box_open: bool) -> GameMode:
        if battle_flag:
            return GameMode.BATTLE
        if text_box_open:
            return GameMode.TEXT
        if menu_open:
            return GameMode.MENU
        return GameMode.OVERWORLD

    def _decode_direction(self, direction_raw: int, moving_direction_raw: int) -> str | None:
        direct_lookup = {
            0x00: "DOWN",
            0x04: "UP",
            0x08: "LEFT",
            0x0C: "RIGHT",
        }
        if direction_raw in direct_lookup:
            return direct_lookup[direction_raw]

        fallback_lookup = {
            0x00: "DOWN",
            0x01: "UP",
            0x02: "LEFT",
            0x03: "RIGHT",
        }
        if moving_direction_raw in fallback_lookup:
            return fallback_lookup[moving_direction_raw]
        if direction_raw in fallback_lookup:
            return fallback_lookup[direction_raw]
        return f"RAW_{direction_raw:02X}"

    def _detect_bootstrap_phase(
        self,
        step: int,
        map_id: int,
        x: int,
        y: int,
        battle_flag: int,
        game_area,
        ram_context: dict[str, object],
    ) -> str | None:
        player = ram_context["player"]
        map_context = ram_context["map"]

        if battle_flag != 0:
            return None

        if self._is_title_menu(game_area):
            return "title_menu"
        if self._is_title_screen(game_area):
            return "title_screen"
        if map_id == 0 and x == 0 and y == 0:
            if step < 1600:
                return "boot_sequence"
            return "intro_cutscene"
        if player["party_count"] == 0 and map_context["width"] == 0 and map_context["height"] == 0:
            return "intro_cutscene"
        if player["party_count"] == 0 and self._is_empty_background(game_area):
            return "intro_cutscene"
        return None

    def _is_title_screen(self, game_area) -> bool:
        return (
            int(game_area[1][2]) == 128
            and int(game_area[1][17]) == 143
            and int(game_area[7][2]) == 305
            and int(game_area[7][17]) == 320
            and int(game_area[17][2]) == 321
            and int(game_area[17][17]) == 334
        )

    def _is_title_menu(self, game_area) -> bool:
        return (
            int(game_area[0][0]) == 377
            and int(game_area[0][14]) == 379
            and int(game_area[5][0]) == 381
            and int(game_area[5][14]) == 382
            and int(game_area[2][1]) == 237
        )

    def _bootstrap_label(self, bootstrap_phase: str) -> str:
        labels = {
            "boot_sequence": "Boot Sequence",
            "title_screen": "Title Screen",
            "title_menu": "Title Menu",
            "intro_cutscene": "Intro Cutscene",
        }
        return labels.get(bootstrap_phase, "Bootstrap")

    def _is_empty_background(self, game_area) -> bool:
        return bool((game_area == 383).all())

    def _hash_grid(self, grid) -> str:
        return sha1(grid.tobytes()).hexdigest()[:12]

    def _serialize_grid(self, grid) -> list[list[int]]:
        rows = int(getattr(grid, "shape", [0, 0])[0] if grid is not None else 0)
        cols = int(getattr(grid, "shape", [0, 0])[1] if grid is not None else 0)
        return [[int(grid[row][col]) for col in range(cols)] for row in range(rows)]
