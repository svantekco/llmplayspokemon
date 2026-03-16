#!/usr/bin/env python3
from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import re
import subprocess
import sys
from tempfile import TemporaryDirectory
from collections import defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pokemon_agent.data.map_names import POKEMON_RED_MAP_NAMES

LAST_MAP_SYMBOL = "LAST_MAP"
UNUSED_SYMBOL_PREFIX = "UNUSED_MAP"
MULTIPLAYER_SYMBOLS = {"TRADE_CENTER", "COLOSSEUM"}
COPY_SUFFIXES = ("_COPY",)
SIGN_TEXT_TOKENS = ("SIGN", "TIPS", "NOTICE", "POSTER", "BINOCULARS")
GENERIC_BUILDING_TOKENS = {"HOUSE", "ROOM", "COPY"}
IMPORTANT_BUILDING_TOKENS = {
    "LAB",
    "MUSEUM",
    "FAN",
    "DOJO",
    "DOCK",
    "MANSION",
    "HIDEOUT",
    "TOWER",
    "SAFARI",
    "SILPH",
    "BIKE",
    "DAYCARE",
    "FAN",
    "NAME",
    "BILL",
    "WARDEN",
    "FUJI",
    "PSYCHIC",
}
DUNGEON_TOKENS = {
    "FOREST",
    "TOWER",
    "MANSION",
    "HIDEOUT",
    "VICTORY",
    "SAFARI",
    "SILPH",
    "SEAFOAM",
    "POWER",
}
CAVE_TOKENS = {"CAVE", "CAVERN", "TUNNEL", "DIGLETTS", "MOON", "ROCK"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pret-dir",
        type=Path,
        help="Path to an existing pret/pokered checkout. If omitted, the script clones a temporary checkout and removes it afterwards.",
    )
    parser.add_argument(
        "--world-graph-out",
        type=Path,
        default=PROJECT_ROOT / "src/pokemon_agent/generated/world_graph.json",
        help="Output path for the generated world graph JSON.",
    )
    parser.add_argument(
        "--landmarks-out",
        type=Path,
        default=PROJECT_ROOT / "src/pokemon_agent/generated/landmarks.json",
        help="Output path for the generated landmarks JSON.",
    )
    parser.add_argument(
        "--map-objects-out",
        type=Path,
        default=PROJECT_ROOT / "src/pokemon_agent/generated/map_objects.json",
        help="Output path for the generated map objects JSON.",
    )
    parser.add_argument(
        "--type-chart-out",
        type=Path,
        default=PROJECT_ROOT / "src/pokemon_agent/generated/type_chart.json",
        help="Output path for the generated type chart JSON.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    with _pret_checkout(args.pret_dir) as pret_dir:
        name_labels = _parse_name_labels(pret_dir / "data/maps/names.asm")
        indoor_group_anchors = _parse_indoor_group_anchors(pret_dir / "data/maps/town_map_entries.asm", name_labels)
        maps = _parse_map_constants(pret_dir / "constants/map_constants.asm")
        header_data = _parse_headers(pret_dir / "data/maps/headers")
        object_data = _parse_objects(pret_dir / "data/maps/objects")

        for symbol, payload in maps.items():
            map_id = payload["id"]
            payload["name"] = POKEMON_RED_MAP_NAMES.get(map_id, _humanize_symbol(symbol))
            payload["slug"] = _slug(payload["name"])
            payload["anchor_name"] = indoor_group_anchors.get(payload.get("group"))
            payload["file"] = None
            payload["tileset"] = None
            payload["connections"] = []
            payload["warps"] = []
            payload["bg_events"] = []
            payload["object_events"] = []
            payload["is_unused"] = symbol.startswith(UNUSED_SYMBOL_PREFIX)
            payload["is_multiplayer"] = symbol in MULTIPLAYER_SYMBOLS
            payload["is_copy"] = symbol.endswith(COPY_SUFFIXES) or "COPY" in symbol
            payload["routing_enabled"] = not payload["is_unused"] and not payload["is_multiplayer"]

        for symbol, payload in header_data.items():
            if symbol not in maps:
                continue
            maps[symbol]["file"] = payload["file"]
            maps[symbol]["tileset"] = payload["tileset"]
            maps[symbol]["connections"] = payload["connections"]

        for symbol, payload in object_data.items():
            if symbol not in maps:
                continue
            maps[symbol]["warps"] = payload["warps"]
            maps[symbol]["bg_events"] = payload["bg_events"]
            maps[symbol]["object_events"] = payload["object_events"]

        _resolve_last_map_warps(maps)

        world_graph = _build_world_graph(maps, pret_dir)
        landmarks = _build_landmarks(maps, world_graph)
        generated_at = str(world_graph["meta"]["generated_at"])
        pret_commit = str(world_graph["meta"]["pret_commit"])
        tileset_ids, tileset_order = _parse_sequential_constants(pret_dir / "constants/tileset_constants.asm")
        sprite_ids, _ = _parse_sequential_constants(pret_dir / "constants/sprite_constants.asm")
        tileset_blocksets = _parse_tileset_blocksets(pret_dir / "gfx/tilesets.asm")
        warp_tile_ids = _parse_warp_tile_ids(
            pret_dir / "data/tilesets/warp_tile_ids.asm",
            tileset_order=tileset_order,
        )
        door_tile_ids = _parse_door_tile_ids(pret_dir / "data/tilesets/door_tile_ids.asm")
        special_warps = _parse_special_warps(pret_dir / "data/maps/special_warps.asm", maps)
        map_objects = _build_map_objects(
            maps,
            pret_dir,
            generated_at=generated_at,
            pret_commit=pret_commit,
            sprite_ids=sprite_ids,
            tileset_ids=tileset_ids,
            tileset_blocksets=tileset_blocksets,
            warp_tile_ids=warp_tile_ids,
            door_tile_ids=door_tile_ids,
            special_warps=special_warps,
        )
        type_chart = _build_type_chart(
            pret_dir / "data/types/type_matchups.asm",
            pret_dir / "constants/type_constants.asm",
            generated_at=generated_at,
            pret_commit=pret_commit,
        )

        for output in (
            args.world_graph_out,
            args.landmarks_out,
            args.map_objects_out,
            args.type_chart_out,
        ):
            output.parent.mkdir(parents=True, exist_ok=True)

        args.world_graph_out.write_text(json.dumps(world_graph, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        args.landmarks_out.write_text(json.dumps(landmarks, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        args.map_objects_out.write_text(json.dumps(map_objects, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        args.type_chart_out.write_text(json.dumps(type_chart, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"Wrote {args.world_graph_out}")
        print(f"Wrote {args.landmarks_out}")
        print(f"Wrote {args.map_objects_out}")
        print(f"Wrote {args.type_chart_out}")
    return 0


@contextmanager
def _pret_checkout(provided_dir: Path | None) -> Path:
    if provided_dir is not None:
        pret_dir = provided_dir.resolve()
        if not pret_dir.exists():
            raise SystemExit(f"PRET checkout not found: {pret_dir}")
        yield pret_dir
        return

    with TemporaryDirectory(prefix="pret-pokered-") as temp_dir:
        clone_dir = Path(temp_dir) / "pokered"
        subprocess.run(
            ["git", "clone", "--depth", "1", "https://github.com/pret/pokered", str(clone_dir)],
            check=True,
        )
        yield clone_dir


def _parse_map_constants(path: Path) -> dict[str, dict[str, Any]]:
    maps: dict[str, dict[str, Any]] = {}
    current_group: str | None = None
    for line in path.read_text(encoding="utf-8").splitlines():
        match = re.match(r"\s*map_const\s+(\w+),\s*(\d+),\s*(\d+)\s*;\s*\$(..)", line)
        if match:
            symbol, width, height, hex_id = match.groups()
            maps[symbol] = {
                "symbol": symbol,
                "id": int(hex_id, 16),
                "width": int(width),
                "height": int(height),
                "group": current_group,
            }
            continue
        group_match = re.match(r"\s*end_indoor_group\s+(\w+)", line)
        if group_match:
            current_group = group_match.group(1)
    return maps


def _parse_headers(directory: Path) -> dict[str, dict[str, Any]]:
    headers: dict[str, dict[str, Any]] = {}
    for path in sorted(directory.glob("*.asm")):
        symbol: str | None = None
        file_name: str | None = None
        tileset: str | None = None
        connections: list[dict[str, Any]] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            header_match = re.match(r"\s*map_header\s+(\w+),\s*(\w+),\s*(\w+),\s*(.+)", line)
            if header_match:
                file_name, symbol, tileset, _flags = header_match.groups()
                continue
            conn_match = re.match(r"\s*connection\s+(north|south|east|west),\s*(\w+),\s*(\w+),\s*(-?\d+)", line)
            if conn_match:
                direction, target_file, target_symbol, offset = conn_match.groups()
                connections.append(
                    {
                        "direction": direction,
                        "target_file": target_file,
                        "target_symbol": target_symbol,
                        "offset": int(offset),
                    }
                )
        if symbol is not None:
            headers[symbol] = {"file": file_name, "tileset": tileset, "connections": connections}
    return headers


def _parse_objects(directory: Path) -> dict[str, dict[str, Any]]:
    objects: dict[str, dict[str, Any]] = {}
    for path in sorted(directory.glob("*.asm")):
        warps: list[dict[str, Any]] = []
        bg_events: list[dict[str, Any]] = []
        object_events: list[dict[str, Any]] = []
        map_symbol: str | None = None
        section: str | None = None
        for line in path.read_text(encoding="utf-8").splitlines():
            end_match = re.match(r"\s*def_warps_to\s+(\w+)", line)
            if end_match:
                map_symbol = end_match.group(1)
                break
            if re.match(r"\s*def_warp_events", line):
                section = "warps"
                continue
            if re.match(r"\s*def_bg_events", line):
                section = "bg_events"
                continue
            if re.match(r"\s*def_object_events", line):
                section = "object_events"
                continue
            if section == "warps":
                if re.match(r"\s*warp_event\s+", line):
                    parts = [part.strip() for part in re.sub(r"^\s*warp_event\s+", "", line).split(",")]
                    if len(parts) < 4:
                        continue
                    x_token, y_token, destination_symbol, destination_warp_id_token = parts[:4]
                    x = _parse_int_token(x_token)
                    y = _parse_int_token(y_token)
                    destination_warp_id = _parse_int_token(destination_warp_id_token)
                    if x is None or y is None or destination_warp_id is None:
                        continue
                    warps.append(
                        {
                            "index": len(warps) + 1,
                            "x": x,
                            "y": y,
                            "destination_symbol": destination_symbol,
                            "destination_warp_id": destination_warp_id,
                        }
                    )
            elif section == "bg_events":
                if re.match(r"\s*bg_event\s+", line):
                    parts = [part.strip() for part in re.sub(r"^\s*bg_event\s+", "", line).split(",")]
                    if len(parts) < 3:
                        continue
                    x = _parse_int_token(parts[0])
                    y = _parse_int_token(parts[1])
                    if x is None or y is None:
                        continue
                    bg_events.append({"index": len(bg_events) + 1, "x": x, "y": y, "text_id": parts[2]})
            elif section == "object_events":
                if re.match(r"\s*object_event\s+", line):
                    parts = [part.strip() for part in re.sub(r"^\s*object_event\s+", "", line).split(",")]
                    if len(parts) < 6:
                        continue
                    x = _parse_int_token(parts[0])
                    y = _parse_int_token(parts[1])
                    if x is None or y is None:
                        continue
                    sprite, movement, facing, text_id = parts[2:6]
                    arg7 = parts[6] if len(parts) > 6 else None
                    arg8 = _parse_int_token(parts[7]) if len(parts) > 7 else None
                    object_events.append(
                        {
                            "index": len(object_events) + 1,
                            "x": x,
                            "y": y,
                            "sprite": sprite,
                            "movement": movement,
                            "facing": facing,
                            "text_id": text_id,
                            "arg7": arg7,
                            "arg8": arg8,
                        }
                    )
        if map_symbol is not None:
            objects[map_symbol] = {
                "warps": warps,
                "bg_events": bg_events,
                "object_events": object_events,
            }
    return objects


def _parse_sequential_constants(path: Path) -> tuple[dict[str, int], list[str]]:
    values: dict[str, int] = {}
    ordered: list[str] = []
    current = 0
    active = False
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = _strip_comment(raw_line)
        if not line:
            continue
        if line.startswith("const_def"):
            current = 0
            active = True
            continue
        if not active:
            continue
        next_match = re.match(r"const_next\s+([$\w-]+)", line)
        if next_match:
            parsed = _parse_int_token(next_match.group(1))
            if parsed is not None:
                current = parsed
            continue
        const_match = re.match(r"const\s+(\w+)", line)
        if const_match:
            symbol = const_match.group(1)
            values[symbol] = current
            ordered.append(symbol)
            current += 1
    return values, ordered


def _parse_tileset_blocksets(path: Path) -> dict[str, str]:
    mapping: dict[str, str] = {}
    pending_symbols: list[str] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = _strip_comment(raw_line)
        if not line:
            continue
        label_match = re.match(r"(\w+)_Block::(?:\s+INCBIN\s+\"([^\"]+)\")?", line)
        if label_match:
            label, incbin_path = label_match.groups()
            pending_symbols.append(_camel_to_symbol(label))
            if incbin_path is not None:
                for symbol in pending_symbols:
                    mapping[symbol] = incbin_path
                pending_symbols = []
            continue
        incbin_match = re.match(r'INCBIN\s+"([^"]+)"', line)
        if incbin_match and pending_symbols:
            for symbol in pending_symbols:
                mapping[symbol] = incbin_match.group(1)
            pending_symbols = []
    return mapping


def _parse_warp_tile_ids(path: Path, *, tileset_order: list[str]) -> dict[str, list[int]]:
    pointer_labels: list[str] = []
    collecting_pointers = False
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = _strip_comment(raw_line)
        if not line:
            continue
        if line == "WarpTileIDPointers:":
            collecting_pointers = True
            continue
        if collecting_pointers and line.startswith("assert_table_length"):
            break
        if collecting_pointers:
            match = re.match(r"dw\s+(\.\w+)", line)
            if match:
                pointer_labels.append(match.group(1))

    label_offsets, bytecode = _assemble_local_bytecode(path)
    terminator = 0xFF
    mapping: dict[str, list[int]] = {}
    for index, tileset_symbol in enumerate(tileset_order):
        if index >= len(pointer_labels):
            break
        label = pointer_labels[index]
        offset = label_offsets.get(label)
        if offset is None:
            mapping[tileset_symbol] = []
            continue
        mapping[tileset_symbol] = _read_until_terminator(bytecode, offset, terminator)
    return mapping


def _parse_door_tile_ids(path: Path) -> dict[str, list[int]]:
    pointer_map: dict[str, list[str]] = defaultdict(list)
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = _strip_comment(raw_line)
        if not line:
            continue
        match = re.match(r"dbw\s+(\w+),\s*(\.\w+)", line)
        if match:
            tileset_symbol, label = match.groups()
            pointer_map[label].append(tileset_symbol)

    label_offsets, bytecode = _assemble_local_bytecode(path)
    terminator = 0
    mapping: dict[str, list[int]] = {}
    for label, symbols in pointer_map.items():
        offset = label_offsets.get(label)
        values = [] if offset is None else _read_until_terminator(bytecode, offset, terminator)
        for symbol in symbols:
            mapping[symbol] = list(values)
    return mapping


def _parse_special_warps(path: Path, maps: dict[str, dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    dungeon_targets: list[tuple[str, int]] = []
    dungeon_coordinates: list[tuple[str, int, int]] = []
    named_warps: list[dict[str, Any]] = []
    fly_pointer_to_label: list[tuple[str, str]] = []
    fly_coordinates: dict[str, tuple[str, int, int]] = {}
    section: str | None = None
    current_label: str | None = None

    for raw_line in lines:
        stripped = _strip_comment(raw_line)
        if not stripped:
            continue
        label_match = re.match(r"(\.?\w+):(?:\s+(.*))?", stripped)
        if label_match:
            current_label = label_match.group(1)
            remainder = label_match.group(2) or ""
            if current_label == "DungeonWarpList":
                section = "dungeon_targets"
                continue
            if current_label == "DungeonWarpData":
                section = "dungeon_coordinates"
                continue
            if current_label == "FlyWarpDataPtr":
                section = "fly_pointers"
                continue
            if current_label.startswith(".") and remainder.startswith("fly_warp "):
                section = None
                match = re.match(r"fly_warp\s+(\w+),\s*(\d+),\s*(\d+)", remainder)
                if match:
                    map_symbol, x, y = match.groups()
                    fly_coordinates[current_label] = (map_symbol, int(x), int(y))
                continue
            if remainder.startswith("special_warp_spec "):
                section = None
                match = re.match(r"special_warp_spec\s+(\w+),\s*(\d+),\s*(\d+),\s*(\w+)", remainder)
                if match:
                    map_symbol, x, y, destination_symbol = match.groups()
                    named_warps.append(
                        {
                            "name": current_label,
                            "map_name": maps.get(map_symbol, {}).get("name"),
                            "map_symbol": map_symbol,
                            "x": int(x),
                            "y": int(y),
                            "destination_symbol": destination_symbol,
                        }
                    )
                continue

        if section == "dungeon_targets":
            if stripped == "db -1":
                section = None
                continue
            match = re.match(r"db\s+(\w+),\s*(\d+)", stripped)
            if match:
                map_symbol, warp_index = match.groups()
                dungeon_targets.append((map_symbol, int(warp_index)))
            continue

        if section == "dungeon_coordinates":
            match = re.match(r"fly_warp\s+(\w+),\s*(\d+),\s*(\d+)", stripped)
            if match:
                map_symbol, x, y = match.groups()
                dungeon_coordinates.append((map_symbol, int(x), int(y)))
            continue

        if section == "fly_pointers":
            match = re.match(r"fly_warp_spec\s+(\w+),\s*(\.\w+)", stripped)
            if match:
                fly_pointer_to_label.append(match.groups())
            continue

    dungeon_warps: list[dict[str, Any]] = []
    for (map_symbol, warp_index), (coord_symbol, x, y) in zip(dungeon_targets, dungeon_coordinates, strict=False):
        dungeon_warps.append(
            {
                "map_name": maps.get(map_symbol, {}).get("name"),
                "map_symbol": map_symbol,
                "warp_index": warp_index,
                "x": x,
                "y": y,
                "coordinate_map_symbol": coord_symbol,
            }
        )

    fly_warps: list[dict[str, Any]] = []
    for map_symbol, label in fly_pointer_to_label:
        coordinate = fly_coordinates.get(label)
        if coordinate is None:
            continue
        _coordinate_map_symbol, x, y = coordinate
        fly_warps.append(
            {
                "map_name": maps.get(map_symbol, {}).get("name"),
                "map_symbol": map_symbol,
                "x": x,
                "y": y,
            }
        )

    return {
        "dungeon_warps": dungeon_warps,
        "fly_warps": fly_warps,
        "named_warps": named_warps,
    }


def _build_map_objects(
    maps: dict[str, dict[str, Any]],
    pret_dir: Path,
    *,
    generated_at: str,
    pret_commit: str,
    sprite_ids: dict[str, int],
    tileset_ids: dict[str, int],
    tileset_blocksets: dict[str, str],
    warp_tile_ids: dict[str, list[int]],
    door_tile_ids: dict[str, list[int]],
    special_warps: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    blockset_cache: dict[str, bytes] = {}
    maps_payload: list[dict[str, Any]] = []
    total_warps = 0
    total_signs = 0
    total_npcs = 0
    tilesets_payload: list[dict[str, Any]] = []

    for tileset_symbol, tileset_id in sorted(tileset_ids.items(), key=lambda item: item[1]):
        tilesets_payload.append(
            {
                "blockset": tileset_blocksets.get(tileset_symbol),
                "door_tile_ids": list(door_tile_ids.get(tileset_symbol, [])),
                "id": tileset_id,
                "symbol": tileset_symbol,
                "warp_tile_ids": list(warp_tile_ids.get(tileset_symbol, [])),
            }
        )

    for symbol in sorted(maps, key=lambda item: maps[item]["id"]):
        payload = maps[symbol]
        warp_entries: list[dict[str, Any]] = []
        door_ids = set(door_tile_ids.get(payload.get("tileset") or "", []))
        warp_ids = set(warp_tile_ids.get(payload.get("tileset") or "", []))
        for warp in payload["warps"]:
            tile_id = _warp_tile_id_for_coordinate(
                pret_dir=pret_dir,
                map_payload=payload,
                tileset_blocksets=tileset_blocksets,
                blockset_cache=blockset_cache,
                x=warp["x"],
                y=warp["y"],
            )
            activation_mode, activation_source = _infer_warp_activation_mode(
                map_payload=payload,
                x=warp["x"],
                y=warp["y"],
                tile_id=tile_id,
                door_tile_ids=door_ids,
                warp_tile_ids=warp_ids,
            )
            warp_entries.append(
                {
                    "activation_mode": activation_mode,
                    "activation_source": activation_source,
                    "destination_map_id": warp.get("resolved_destination_map_id"),
                    "destination_name": warp.get("resolved_destination_name"),
                    "destination_symbol": warp.get("resolved_destination_symbol"),
                    "destination_warp_id": warp["destination_warp_id"],
                    "index": warp["index"],
                    "original_destination_symbol": warp["destination_symbol"],
                    "resolution_method": warp.get("resolution_method"),
                    "tile_id": tile_id,
                    "x": warp["x"],
                    "y": warp["y"],
                }
            )

        sign_entries = [
            {
                "index": event["index"],
                "text_id": event["text_id"],
                "x": event["x"],
                "y": event["y"],
            }
            for event in payload["bg_events"]
        ]
        npc_entries = []
        for event in payload["object_events"]:
            trainer_class = event.get("arg7") if isinstance(event.get("arg7"), str) and event["arg7"].startswith("OPP_") else None
            npc_entries.append(
                {
                    "facing": event["facing"],
                    "index": event["index"],
                    "is_trainer": trainer_class is not None,
                    "movement": event["movement"],
                    "sprite_id": sprite_ids.get(event["sprite"]),
                    "sprite_symbol": event["sprite"],
                    "text_id": event["text_id"],
                    "trainer_class": trainer_class,
                    "trainer_id": event.get("arg8") if trainer_class is not None else None,
                    "x": event["x"],
                    "y": event["y"],
                }
            )

        total_warps += len(warp_entries)
        total_signs += len(sign_entries)
        total_npcs += len(npc_entries)
        maps_payload.append(
            {
                "file": payload["file"],
                "id": payload["id"],
                "name": payload["name"],
                "npcs": npc_entries,
                "routing_enabled": payload["routing_enabled"],
                "signs": sign_entries,
                "slug": payload["slug"],
                "symbol": payload["symbol"],
                "tileset": payload["tileset"],
                "warps": warp_entries,
            }
        )

    return {
        "maps": maps_payload,
        "meta": {
            "generated_at": generated_at,
            "notes": [
                "Warp activation modes are derived from PRET tileset tile IDs when available.",
                "Boundary-based activation is retained as a fallback when a warp tile cannot be resolved to a known tile ID.",
            ],
            "pret_commit": pret_commit,
            "pret_repo": "https://github.com/pret/pokered",
            "stats": {
                "map_count": len(maps_payload),
                "npc_count": total_npcs,
                "sign_count": total_signs,
                "special_warp_count": sum(len(items) for items in special_warps.values()),
                "tileset_count": len(tilesets_payload),
                "warp_count": total_warps,
            },
        },
        "special_warps": special_warps,
        "tilesets": tilesets_payload,
    }


def _build_type_chart(
    type_matchup_path: Path,
    type_constants_path: Path,
    *,
    generated_at: str,
    pret_commit: str,
) -> dict[str, Any]:
    type_values, ordered_types = _parse_sequential_constants(type_constants_path)
    del type_values
    types = [
        _normalize_type_symbol(symbol)
        for symbol in ordered_types
        if symbol not in {"BIRD"}
    ]
    explicit: dict[tuple[str, str], float] = {}
    multipliers = {
        "NO_EFFECT": 0.0,
        "NOT_VERY_EFFECTIVE": 0.5,
        "SUPER_EFFECTIVE": 2.0,
    }
    for raw_line in type_matchup_path.read_text(encoding="utf-8").splitlines():
        line = _strip_comment(raw_line)
        match = re.match(r"db\s+(\w+),\s*(\w+),\s*(\w+)", line)
        if not match:
            continue
        attacker_symbol, defender_symbol, multiplier_symbol = match.groups()
        if attacker_symbol == "-1":
            break
        multiplier = multipliers.get(multiplier_symbol)
        if multiplier is None:
            continue
        explicit[(_normalize_type_symbol(attacker_symbol), _normalize_type_symbol(defender_symbol))] = multiplier

    matchups: list[dict[str, Any]] = []
    for attacker in types:
        for defender in types:
            matchups.append(
                {
                    "attacker": attacker,
                    "defender": defender,
                    "multiplier": explicit.get((attacker, defender), 1.0),
                }
            )

    return {
        "matchups": matchups,
        "meta": {
            "generated_at": generated_at,
            "pret_commit": pret_commit,
            "pret_repo": "https://github.com/pret/pokered",
            "stats": {
                "matchup_count": len(matchups),
                "type_count": len(types),
            },
        },
        "types": types,
    }


def _assemble_local_bytecode(path: Path) -> tuple[dict[str, int], list[int]]:
    labels: dict[str, int] = {}
    bytecode: list[int] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = _strip_comment(raw_line)
        if not line:
            continue
        label_match = re.match(r"(\.\w+|\w+):(?:\s+(.*))?", line)
        if label_match:
            label, remainder = label_match.groups()
            labels[label] = len(bytecode)
            if remainder:
                bytecode.extend(_emit_data_bytes(remainder))
            continue
        bytecode.extend(_emit_data_bytes(line))
    return labels, bytecode


def _emit_data_bytes(line: str) -> list[int]:
    bytes_out: list[int] = []
    macro_match = re.match(r"(warp_tiles|door_tiles)\s*(.*)", line)
    if macro_match:
        macro_name, args = macro_match.groups()
        if args:
            bytes_out.extend(_parse_numeric_arguments(args))
        bytes_out.append(0xFF if macro_name == "warp_tiles" else 0x00)
        return bytes_out
    db_match = re.match(r"db\s+(.+)", line)
    if db_match:
        return _parse_numeric_arguments(db_match.group(1))
    return bytes_out


def _parse_numeric_arguments(arg_text: str) -> list[int]:
    values: list[int] = []
    for raw_token in arg_text.split(","):
        token = raw_token.strip()
        if not token:
            continue
        parsed = _parse_int_token(token)
        if parsed is None:
            return []
        values.append(parsed & 0xFF)
    return values


def _read_until_terminator(bytecode: list[int], offset: int, terminator: int) -> list[int]:
    values: list[int] = []
    for index in range(offset, len(bytecode)):
        value = bytecode[index]
        if value == terminator:
            break
        values.append(value)
    return values


def _warp_tile_id_for_coordinate(
    *,
    pret_dir: Path,
    map_payload: dict[str, Any],
    tileset_blocksets: dict[str, str],
    blockset_cache: dict[str, bytes],
    x: int,
    y: int,
) -> int | None:
    map_file = map_payload.get("file")
    tileset_symbol = map_payload.get("tileset")
    if not map_file or not tileset_symbol:
        return None
    blockset_relative_path = tileset_blocksets.get(tileset_symbol)
    if blockset_relative_path is None:
        return None
    map_path = pret_dir / "maps" / f"{map_file}.blk"
    if not map_path.exists():
        return None
    map_width = map_payload["width"]
    map_height = map_payload["height"]
    if x < 0 or y < 0 or x >= map_width * 2 or y >= map_height * 2:
        return None
    block_bytes = map_path.read_bytes()
    block_index = (y // 2) * map_width + (x // 2)
    if block_index >= len(block_bytes):
        return None
    block_id = block_bytes[block_index]
    blockset_bytes = blockset_cache.get(tileset_symbol)
    if blockset_bytes is None:
        blockset_bytes = (pret_dir / blockset_relative_path).read_bytes()
        blockset_cache[tileset_symbol] = blockset_bytes
    block_offset = block_id * 16
    if block_offset + 16 > len(blockset_bytes):
        return None
    block = blockset_bytes[block_offset : block_offset + 16]
    local_x = x % 2
    local_y = y % 2
    lower_left_index = ((local_y * 2) + 1) * 4 + (local_x * 2)
    if lower_left_index >= len(block):
        return None
    return int(block[lower_left_index])


def _infer_warp_activation_mode(
    *,
    map_payload: dict[str, Any],
    x: int,
    y: int,
    tile_id: int | None,
    door_tile_ids: set[int],
    warp_tile_ids: set[int],
) -> tuple[str, str]:
    if tile_id is not None:
        if tile_id in door_tile_ids:
            return "push", "door_tile_id"
        if tile_id in warp_tile_ids:
            return "step_on", "warp_tile_id"
    if _warp_boundary_side(map_payload, x, y) is not None:
        return "push", "fallback_boundary"
    return "step_on", "fallback_default"


def _warp_boundary_side(map_payload: dict[str, Any], x: int, y: int) -> str | None:
    max_x = max(0, map_payload["width"] * 2 - 1)
    max_y = max(0, map_payload["height"] * 2 - 1)
    if y <= 0:
        return "north"
    if x >= max_x:
        return "east"
    if y >= max_y:
        return "south"
    if x <= 0:
        return "west"
    return None


def _strip_comment(raw_line: str) -> str:
    return raw_line.split(";", 1)[0].strip()


def _parse_int_token(token: str) -> int | None:
    value = token.strip()
    if not value:
        return None
    if value.startswith("$"):
        try:
            return int(value[1:], 16)
        except ValueError:
            return None
    try:
        return int(value, 10)
    except ValueError:
        return None


def _camel_to_symbol(value: str) -> str:
    value = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", value)
    value = re.sub(r"([A-Za-z])(\d)", r"\1_\2", value)
    value = re.sub(r"(\d)([A-Za-z])", r"\1_\2", value)
    return value.upper()


def _normalize_type_symbol(symbol: str) -> str:
    if symbol == "PSYCHIC_TYPE":
        return "PSYCHIC"
    return symbol


def _parse_name_labels(path: Path) -> dict[str, str]:
    labels: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        match = re.match(r"(\w+):\s+db\s+\"([^\"]+)\"", line)
        if not match:
            continue
        label, raw_value = match.groups()
        cleaned = raw_value.replace("@", "")
        cleaned = cleaned.replace("#MON", "Pokemon")
        cleaned = cleaned.replace("<PKMN>", "Pokemon")
        cleaned = cleaned.replace("MT.", "Mt. ")
        cleaned = cleaned.replace("S.S.ANNE", "S.S. Anne")
        cleaned = re.sub(r"\s+", " ", cleaned).strip().title()
        labels[label] = cleaned
    return labels


def _parse_indoor_group_anchors(path: Path, name_labels: dict[str, str]) -> dict[str, str]:
    anchors: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        match = re.match(r"\s*indoor_map\s+(\w+),\s*\d+,\s*\d+,\s*(\w+)", line)
        if not match:
            continue
        group, label = match.groups()
        if group not in anchors and label in name_labels:
            anchors[group] = name_labels[label]
    return anchors


def _resolve_last_map_warps(maps: dict[str, dict[str, Any]]) -> None:
    incoming_by_target: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
    for source_symbol, source_map in maps.items():
        for source_warp in source_map["warps"]:
            destination_symbol = source_warp["destination_symbol"]
            if destination_symbol == LAST_MAP_SYMBOL or destination_symbol not in maps:
                continue
            incoming_by_target[destination_symbol][source_symbol].append(source_warp["destination_warp_id"])

    for target_symbol, target_map in maps.items():
        for warp in target_map["warps"]:
            if warp["destination_symbol"] != LAST_MAP_SYMBOL:
                destination_symbol = warp["destination_symbol"]
                warp["resolved_destination_symbol"] = destination_symbol if destination_symbol in maps else None
                warp["resolution_method"] = "explicit" if destination_symbol in maps else "missing_destination_map"
                if destination_symbol in maps:
                    destination_map = maps[destination_symbol]
                    warp["resolved_destination_name"] = destination_map["name"]
                    warp["resolved_destination_map_id"] = destination_map["id"]
                else:
                    warp["resolved_destination_name"] = None
                    warp["resolved_destination_map_id"] = None
                continue

            source_warp_index = warp["destination_warp_id"]
            candidate_symbols = [
                source_symbol
                for source_symbol, source_map in maps.items()
                if len(source_map["warps"]) >= source_warp_index
                and source_map["warps"][source_warp_index - 1]["destination_symbol"] == target_symbol
            ]
            if not candidate_symbols:
                warp["resolved_destination_symbol"] = None
                warp["resolved_destination_name"] = None
                warp["resolved_destination_map_id"] = None
                warp["resolution_method"] = "unresolved_last_map"
                warp["candidate_destination_symbols"] = []
                continue

            if len(candidate_symbols) == 1:
                chosen_symbol = candidate_symbols[0]
                resolution_method = "last_map_single_candidate"
            else:
                scored_candidates: list[tuple[tuple[int, int, int, str], str]] = []
                for candidate_symbol in candidate_symbols:
                    incoming_indices = incoming_by_target[target_symbol].get(candidate_symbol, [])
                    distance = min(
                        abs(warp["x"] - target_map["warps"][incoming_index - 1]["x"])
                        + abs(warp["y"] - target_map["warps"][incoming_index - 1]["y"])
                        for incoming_index in incoming_indices
                    )
                    candidate_map = maps[candidate_symbol]
                    score = (
                        distance,
                        1 if candidate_map["is_copy"] else 0,
                        1 if candidate_map["is_unused"] else 0,
                        candidate_symbol,
                    )
                    scored_candidates.append((score, candidate_symbol))
                scored_candidates.sort(key=lambda item: item[0])
                chosen_symbol = scored_candidates[0][1]
                resolution_method = "last_map_proximity"
                warp["candidate_destination_symbols"] = sorted(candidate_symbols)

            destination_map = maps[chosen_symbol]
            warp["resolved_destination_symbol"] = chosen_symbol
            warp["resolved_destination_name"] = destination_map["name"]
            warp["resolved_destination_map_id"] = destination_map["id"]
            warp["resolution_method"] = resolution_method


def _build_world_graph(maps: dict[str, dict[str, Any]], pret_dir: Path) -> dict[str, Any]:
    commit = _git_output(pret_dir, "rev-parse", "HEAD")
    maps_payload: list[dict[str, Any]] = []
    connection_count = 0
    resolved_warp_count = 0
    unresolved_warp_count = 0
    for symbol in sorted(maps, key=lambda item: maps[item]["id"]):
        payload = maps[symbol]
        connections: list[dict[str, Any]] = []
        for connection in payload["connections"]:
            destination_symbol = connection["target_symbol"]
            if destination_symbol not in maps:
                continue
            destination_map = maps[destination_symbol]
            connections.append(
                {
                    "direction": connection["direction"],
                    "destination_map_id": destination_map["id"],
                    "destination_name": destination_map["name"],
                    "destination_symbol": destination_symbol,
                    "offset": connection["offset"],
                }
            )
            connection_count += 1

        warps: list[dict[str, Any]] = []
        for warp in payload["warps"]:
            resolved_symbol = warp.get("resolved_destination_symbol")
            if resolved_symbol is None:
                unresolved_warp_count += 1
            else:
                resolved_warp_count += 1
            warps.append(
                {
                    "candidate_destination_symbols": warp.get("candidate_destination_symbols", []),
                    "destination_map_id": warp.get("resolved_destination_map_id"),
                    "destination_name": warp.get("resolved_destination_name"),
                    "destination_symbol": resolved_symbol,
                    "destination_warp_id": warp["destination_warp_id"],
                    "index": warp["index"],
                    "original_destination_symbol": warp["destination_symbol"],
                    "resolution_method": warp.get("resolution_method"),
                    "x": warp["x"],
                    "y": warp["y"],
                }
            )

        bg_events = [
            {
                "index": event["index"],
                "text_id": event["text_id"],
                "x": event["x"],
                "y": event["y"],
            }
            for event in payload["bg_events"]
        ]

        maps_payload.append(
            {
                "anchor_name": payload["anchor_name"],
                "bg_events": bg_events,
                "connections": connections,
                "file": payload["file"],
                "group": payload["group"],
                "height": payload["height"],
                "id": payload["id"],
                "is_copy": payload["is_copy"],
                "is_multiplayer": payload["is_multiplayer"],
                "is_unused": payload["is_unused"],
                "name": payload["name"],
                "routing_enabled": payload["routing_enabled"],
                "slug": payload["slug"],
                "symbol": payload["symbol"],
                "tileset": payload["tileset"],
                "warps": warps,
                "width": payload["width"],
            }
        )

    return {
        "meta": {
            "generated_at": subprocess.run(
                ["date", "-u", "+%Y-%m-%dT%H:%M:%SZ"],
                capture_output=True,
                check=True,
                text=True,
            ).stdout.strip(),
            "notes": [
                "LAST_MAP warps are resolved from PRET warp tables when possible.",
                "A small number of special or unused LAST_MAP warps remain unresolved and are kept with null destinations.",
            ],
            "pret_commit": commit,
            "pret_repo": "https://github.com/pret/pokered",
            "stats": {
                "connection_count": connection_count,
                "map_count": len(maps_payload),
                "resolved_warp_count": resolved_warp_count,
                "unresolved_warp_count": unresolved_warp_count,
            },
        },
        "maps": maps_payload,
    }


def _build_landmarks(maps: dict[str, dict[str, Any]], world_graph: dict[str, Any]) -> dict[str, Any]:
    used_ids: set[str] = set()
    landmarks: list[dict[str, Any]] = []
    for map_payload in world_graph["maps"]:
        source_map = maps[map_payload["symbol"]]
        if not map_payload["routing_enabled"]:
            continue

        for connection in map_payload["connections"]:
            landmark_id = _unique_landmark_id(
                used_ids,
                f"{map_payload['slug']}_{connection['direction']}_exit_{_slug(connection['destination_symbol'])}",
            )
            landmarks.append(
                {
                    "destination_map_id": connection["destination_map_id"],
                    "destination_name": connection["destination_name"],
                    "destination_symbol": connection["destination_symbol"],
                    "direction": connection["direction"],
                    "id": landmark_id,
                    "label": f"{map_payload['name']} {connection['direction'].title()} Exit",
                    "map_id": map_payload["id"],
                    "map_name": map_payload["name"],
                    "map_symbol": map_payload["symbol"],
                    "type": "route_exit",
                    "x": None,
                    "y": None,
                }
            )

        for warp in map_payload["warps"]:
            destination_symbol = warp["destination_symbol"]
            if destination_symbol is None or destination_symbol not in maps:
                continue
            landmark_type = _classify_warp_landmark(source_map, maps[destination_symbol])
            if landmark_type is None:
                continue
            destination_map = maps[destination_symbol]
            landmark_id = _unique_landmark_id(
                used_ids,
                _preferred_landmark_id(map_payload["slug"], destination_map["slug"], landmark_type),
            )
            landmarks.append(
                {
                    "destination_map_id": destination_map["id"],
                    "destination_name": destination_map["name"],
                    "destination_symbol": destination_symbol,
                    "id": landmark_id,
                    "label": destination_map["name"],
                    "map_id": map_payload["id"],
                    "map_name": map_payload["name"],
                    "map_symbol": map_payload["symbol"],
                    "type": landmark_type,
                    "warp_index": warp["index"],
                    "x": warp["x"],
                    "y": warp["y"],
                }
            )

        for event in map_payload["bg_events"]:
            if not _is_sign_like(event["text_id"], source_map):
                continue
            landmark_id = _unique_landmark_id(
                used_ids,
                f"{map_payload['slug']}_{_slug(event['text_id'].replace('TEXT_', ''))}",
            )
            landmarks.append(
                {
                    "id": landmark_id,
                    "label": _humanize_text_id(event["text_id"]),
                    "map_id": map_payload["id"],
                    "map_name": map_payload["name"],
                    "map_symbol": map_payload["symbol"],
                    "text_id": event["text_id"],
                    "type": "sign",
                    "x": event["x"],
                    "y": event["y"],
                }
            )

    return {
        "meta": {
            "generated_at": world_graph["meta"]["generated_at"],
            "landmark_count": len(landmarks),
            "pret_commit": world_graph["meta"]["pret_commit"],
            "pret_repo": world_graph["meta"]["pret_repo"],
        },
        "landmarks": sorted(landmarks, key=lambda item: (item["map_id"], item["type"], item["id"])),
    }


def _classify_warp_landmark(source_map: dict[str, Any], destination_map: dict[str, Any]) -> str | None:
    destination_symbol = destination_map["symbol"]
    destination_name = destination_map["name"]
    destination_tokens = set(_symbol_tokens(destination_symbol))
    source_is_overworld = source_map.get("tileset") == "OVERWORLD" or source_map["id"] < 0x25
    if destination_map["is_unused"] or destination_map["is_multiplayer"]:
        return None
    if "POKECENTER" in destination_symbol or destination_map.get("tileset") == "POKECENTER":
        return "pokecenter"
    if "MART" in destination_symbol or destination_map.get("tileset") == "MART":
        return "mart"
    if "GYM" in destination_symbol or (
        destination_map.get("tileset") == "GYM" and "ROOM" not in destination_tokens and "DOJO" not in destination_tokens
    ):
        return "gym"
    if "GATE" in destination_symbol or destination_name.endswith(" Gate"):
        return "route_exit"
    if source_is_overworld and (CAVE_TOKENS & destination_tokens or destination_map.get("tileset") == "CAVERN"):
        return "cave_entrance"
    if source_is_overworld and (DUNGEON_TOKENS & destination_tokens):
        return "dungeon_entrance"
    if source_is_overworld and IMPORTANT_BUILDING_TOKENS & destination_tokens:
        return "important_building"
    if source_is_overworld and "HOUSE" in destination_tokens and not destination_symbol.endswith(COPY_SUFFIXES):
        return "important_building"
    return None


def _preferred_landmark_id(source_slug: str, destination_slug: str, landmark_type: str) -> str:
    if landmark_type in {"pokecenter", "mart", "gym"}:
        return f"{source_slug}_{landmark_type}"
    return f"{source_slug}_{destination_slug}"


def _is_sign_like(text_id: str, source_map: dict[str, Any]) -> bool:
    if any(token in text_id for token in SIGN_TEXT_TOKENS):
        return True
    if source_map.get("tileset") == "OVERWORLD":
        return text_id.startswith("TEXT_")
    return False


def _humanize_text_id(text_id: str) -> str:
    cleaned = text_id.replace("TEXT_", "")
    cleaned = re.sub(r"_+", " ", cleaned)
    return cleaned.title()


def _unique_landmark_id(used_ids: set[str], base: str) -> str:
    candidate = base
    suffix = 2
    while candidate in used_ids:
        candidate = f"{base}_{suffix}"
        suffix += 1
    used_ids.add(candidate)
    return candidate


def _humanize_symbol(symbol: str) -> str:
    value = symbol.replace("POKECENTER", "Pokecenter")
    value = value.replace("POKEMON", "Pokemon")
    value = value.replace("DIGLETTS", "Diglett's")
    value = value.replace("MR_", "Mr_")
    value = value.replace("SS_", "S.S. ")
    value = value.replace("MT_", "Mt. ")
    value = value.replace("_B1F", " B1F")
    value = value.replace("_B2F", " B2F")
    value = value.replace("_B3F", " B3F")
    value = value.replace("_B4F", " B4F")
    value = value.replace("_1F", " 1F")
    value = value.replace("_2F", " 2F")
    value = value.replace("_3F", " 3F")
    value = value.replace("_4F", " 4F")
    value = value.replace("_5F", " 5F")
    value = value.replace("_6F", " 6F")
    value = value.replace("_7F", " 7F")
    value = value.replace("_8F", " 8F")
    value = value.replace("_9F", " 9F")
    value = value.replace("_10F", " 10F")
    value = value.replace("_11F", " 11F")
    value = value.replace("_", " ")
    value = re.sub(r"\s+", " ", value).strip().title()
    value = value.replace("Ss.", "S.S.")
    value = value.replace("Mt.", "Mt.")
    value = value.replace("Pokemon", "Pokemon")
    return value


def _symbol_tokens(symbol: str) -> list[str]:
    return [token for token in re.findall(r"[A-Z0-9]+", symbol.upper()) if token]


def _slug(value: str) -> str:
    tokens = re.findall(r"[a-z0-9]+", value.lower())
    return "_".join(tokens) or "item"


def _git_output(directory: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(directory), *args],
        capture_output=True,
        check=True,
        text=True,
    ).stdout.strip()


if __name__ == "__main__":
    raise SystemExit(main())
