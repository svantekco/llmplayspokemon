#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
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
    parser.add_argument("--pret-dir", required=True, type=Path, help="Path to a temporary pret/pokered checkout.")
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
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    pret_dir = args.pret_dir.resolve()
    if not pret_dir.exists():
        raise SystemExit(f"PRET checkout not found: {pret_dir}")

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

    args.world_graph_out.parent.mkdir(parents=True, exist_ok=True)
    args.landmarks_out.parent.mkdir(parents=True, exist_ok=True)
    args.world_graph_out.write_text(json.dumps(world_graph, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.landmarks_out.write_text(json.dumps(landmarks, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote {args.world_graph_out}")
    print(f"Wrote {args.landmarks_out}")
    return 0


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
                warp_match = re.match(r"\s*warp_event\s+(\d+),\s*(\d+),\s*(\w+),\s*(\d+)", line)
                if warp_match:
                    x, y, destination_symbol, destination_warp_id = warp_match.groups()
                    warps.append(
                        {
                            "index": len(warps) + 1,
                            "x": int(x),
                            "y": int(y),
                            "destination_symbol": destination_symbol,
                            "destination_warp_id": int(destination_warp_id),
                        }
                    )
            elif section == "bg_events":
                bg_match = re.match(r"\s*bg_event\s+(\d+),\s*(\d+),\s*(\w+)", line)
                if bg_match:
                    x, y, text_id = bg_match.groups()
                    bg_events.append({"index": len(bg_events) + 1, "x": int(x), "y": int(y), "text_id": text_id})
            elif section == "object_events":
                object_match = re.match(
                    r"\s*object_event\s+(\d+),\s*(\d+),\s*(\w+),\s*(\w+),\s*(\w+),\s*(\w+)(?:,\s*(\w+))?(?:,\s*(\d+))?",
                    line,
                )
                if object_match:
                    x, y, sprite, movement, facing, text_id, arg7, arg8 = object_match.groups()
                    object_events.append(
                        {
                            "index": len(object_events) + 1,
                            "x": int(x),
                            "y": int(y),
                            "sprite": sprite,
                            "movement": movement,
                            "facing": facing,
                            "text_id": text_id,
                            "arg7": arg7,
                            "arg8": int(arg8) if arg8 is not None else None,
                        }
                    )
        if map_symbol is not None:
            objects[map_symbol] = {
                "warps": warps,
                "bg_events": bg_events,
                "object_events": object_events,
            }
    return objects


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
