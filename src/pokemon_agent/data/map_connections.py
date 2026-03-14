from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import re


@dataclass(frozen=True, slots=True)
class MapConnection:
    from_map: str
    direction: str
    to_map: str


# Overworld map-to-map connections (boundary exits).
# Direction is the boundary side the player crosses to reach the destination.
# Source: pret/pokered data/maps/connections
MAP_CONNECTIONS: tuple[MapConnection, ...] = (
    # Pallet Town area
    MapConnection("Pallet Town", "north", "Route 1"),
    MapConnection("Pallet Town", "south", "Route 21"),
    MapConnection("Route 1", "south", "Pallet Town"),
    MapConnection("Route 1", "north", "Viridian City"),
    # Viridian City area
    MapConnection("Viridian City", "south", "Route 1"),
    MapConnection("Viridian City", "north", "Route 2"),
    MapConnection("Viridian City", "west", "Route 22"),
    MapConnection("Route 2", "south", "Viridian City"),
    MapConnection("Route 2", "north", "Pewter City"),
    MapConnection("Route 22", "east", "Viridian City"),
    MapConnection("Route 22", "north", "Route 23"),
    # Pewter City area
    MapConnection("Pewter City", "south", "Route 2"),
    MapConnection("Pewter City", "east", "Route 3"),
    # Route 3 / Mt Moon approach
    MapConnection("Route 3", "west", "Pewter City"),
    MapConnection("Route 3", "east", "Route 4"),
    MapConnection("Route 4", "west", "Route 3"),
    MapConnection("Route 4", "east", "Cerulean City"),
    # Cerulean City area
    MapConnection("Cerulean City", "west", "Route 4"),
    MapConnection("Cerulean City", "south", "Route 5"),
    MapConnection("Cerulean City", "east", "Route 9"),
    MapConnection("Cerulean City", "north", "Route 24"),
    MapConnection("Route 24", "south", "Cerulean City"),
    MapConnection("Route 24", "east", "Route 25"),
    MapConnection("Route 25", "west", "Route 24"),
    # Route 5 / Saffron bypass
    MapConnection("Route 5", "north", "Cerulean City"),
    MapConnection("Route 5", "south", "Saffron City"),
    MapConnection("Route 6", "north", "Saffron City"),
    MapConnection("Route 6", "south", "Vermilion City"),
    # Vermilion City area
    MapConnection("Vermilion City", "north", "Route 6"),
    MapConnection("Vermilion City", "east", "Route 11"),
    MapConnection("Route 11", "west", "Vermilion City"),
    MapConnection("Route 11", "east", "Route 12"),
    # Route 9 / Rock Tunnel
    MapConnection("Route 9", "west", "Cerulean City"),
    MapConnection("Route 9", "east", "Route 10"),
    MapConnection("Route 10", "west", "Route 9"),
    MapConnection("Route 10", "south", "Lavender Town"),
    # Lavender Town area
    MapConnection("Lavender Town", "north", "Route 10"),
    MapConnection("Lavender Town", "west", "Route 8"),
    MapConnection("Lavender Town", "south", "Route 12"),
    MapConnection("Route 8", "east", "Lavender Town"),
    MapConnection("Route 8", "west", "Saffron City"),
    # Route 12-15 (Lavender to Fuchsia)
    MapConnection("Route 12", "north", "Lavender Town"),
    MapConnection("Route 12", "south", "Route 13"),
    MapConnection("Route 13", "north", "Route 12"),
    MapConnection("Route 13", "east", "Route 14"),
    MapConnection("Route 13", "west", "Route 14"),
    MapConnection("Route 14", "north", "Route 13"),
    MapConnection("Route 14", "south", "Route 15"),
    MapConnection("Route 15", "north", "Route 14"),
    MapConnection("Route 15", "west", "Fuchsia City"),
    # Saffron City area
    MapConnection("Saffron City", "north", "Route 5"),
    MapConnection("Saffron City", "south", "Route 6"),
    MapConnection("Saffron City", "east", "Route 8"),
    MapConnection("Saffron City", "west", "Route 7"),
    MapConnection("Route 7", "east", "Saffron City"),
    MapConnection("Route 7", "west", "Celadon City"),
    # Celadon City area
    MapConnection("Celadon City", "east", "Route 7"),
    MapConnection("Celadon City", "west", "Route 16"),
    MapConnection("Route 16", "east", "Celadon City"),
    MapConnection("Route 16", "south", "Route 17"),
    MapConnection("Route 17", "north", "Route 16"),
    MapConnection("Route 17", "south", "Route 18"),
    MapConnection("Route 18", "north", "Route 17"),
    MapConnection("Route 18", "east", "Fuchsia City"),
    # Fuchsia City area
    MapConnection("Fuchsia City", "east", "Route 15"),
    MapConnection("Fuchsia City", "west", "Route 18"),
    MapConnection("Fuchsia City", "south", "Route 19"),
    MapConnection("Route 19", "north", "Fuchsia City"),
    MapConnection("Route 19", "south", "Route 20"),
    MapConnection("Route 20", "north", "Route 19"),
    MapConnection("Route 20", "west", "Cinnabar Island"),
    # Cinnabar Island area
    MapConnection("Cinnabar Island", "east", "Route 20"),
    MapConnection("Cinnabar Island", "north", "Route 21"),
    MapConnection("Route 21", "south", "Cinnabar Island"),
    MapConnection("Route 21", "north", "Pallet Town"),
    # Victory Road approach
    MapConnection("Route 23", "south", "Route 22"),
    MapConnection("Route 23", "north", "Indigo Plateau"),
    MapConnection("Indigo Plateau", "south", "Route 23"),
)

# Warp connections (doors, stairs, caves, gates).
# These represent map transitions triggered by walking into a door/warp tile.
# Only story-critical warps are listed to keep the graph manageable.
WARP_CONNECTIONS: tuple[MapConnection, ...] = (
    # Pallet Town buildings
    MapConnection("Pallet Town", "warp", "Red's House 1F"),
    MapConnection("Red's House 1F", "warp", "Pallet Town"),
    MapConnection("Red's House 1F", "warp", "Red's House 2F"),
    MapConnection("Red's House 2F", "warp", "Red's House 1F"),
    MapConnection("Pallet Town", "warp", "Oak's Lab"),
    MapConnection("Oak's Lab", "warp", "Pallet Town"),
    # Viridian City buildings
    MapConnection("Viridian City", "warp", "Viridian Pokecenter"),
    MapConnection("Viridian Pokecenter", "warp", "Viridian City"),
    MapConnection("Viridian City", "warp", "Viridian Mart"),
    MapConnection("Viridian Mart", "warp", "Viridian City"),
    MapConnection("Viridian City", "warp", "Viridian Gym"),
    MapConnection("Viridian Gym", "warp", "Viridian City"),
    # Viridian Forest gates
    MapConnection("Route 2", "warp", "Route 2 Gate"),
    MapConnection("Route 2 Gate", "warp", "Route 2"),
    MapConnection("Route 2 Gate", "warp", "Viridian Forest"),
    MapConnection("Route 2", "warp", "Viridian Forest South Gate"),
    MapConnection("Viridian Forest South Gate", "warp", "Route 2"),
    MapConnection("Viridian Forest South Gate", "warp", "Viridian Forest"),
    MapConnection("Viridian Forest", "warp", "Viridian Forest South Gate"),
    MapConnection("Viridian Forest", "warp", "Viridian Forest North Gate"),
    MapConnection("Viridian Forest North Gate", "warp", "Viridian Forest"),
    MapConnection("Viridian Forest North Gate", "warp", "Pewter City"),
    # Pewter City buildings
    MapConnection("Pewter City", "warp", "Pewter Gym"),
    MapConnection("Pewter Gym", "warp", "Pewter City"),
    MapConnection("Pewter City", "warp", "Pewter Pokecenter"),
    MapConnection("Pewter Pokecenter", "warp", "Pewter City"),
    MapConnection("Pewter City", "warp", "Pewter Mart"),
    MapConnection("Pewter Mart", "warp", "Pewter City"),
    # Mt Moon
    MapConnection("Route 3", "warp", "Mt Moon Pokecenter"),
    MapConnection("Mt Moon Pokecenter", "warp", "Route 3"),
    MapConnection("Route 3", "warp", "Mt Moon 1F"),
    MapConnection("Mt Moon 1F", "warp", "Route 3"),
    MapConnection("Mt Moon 1F", "warp", "Mt Moon B1F"),
    MapConnection("Mt Moon B1F", "warp", "Mt Moon 1F"),
    MapConnection("Mt Moon B1F", "warp", "Mt Moon B2F"),
    MapConnection("Mt Moon B2F", "warp", "Mt Moon B1F"),
    MapConnection("Mt Moon B2F", "warp", "Route 4"),
    # Cerulean City buildings
    MapConnection("Cerulean City", "warp", "Cerulean Pokecenter"),
    MapConnection("Cerulean Pokecenter", "warp", "Cerulean City"),
    MapConnection("Cerulean City", "warp", "Cerulean Gym"),
    MapConnection("Cerulean Gym", "warp", "Cerulean City"),
    MapConnection("Cerulean City", "warp", "Cerulean Mart"),
    MapConnection("Cerulean Mart", "warp", "Cerulean City"),
    # Bill's House
    MapConnection("Route 25", "warp", "Bill's House"),
    MapConnection("Bill's House", "warp", "Route 25"),
    # Underground paths (Route 5 <-> Route 6)
    MapConnection("Route 5", "warp", "Route 5 Gate"),
    MapConnection("Route 5 Gate", "warp", "Route 5"),
    MapConnection("Route 5 Gate", "warp", "Underground Path Route 5"),
    MapConnection("Underground Path Route 5", "warp", "Route 5 Gate"),
    MapConnection("Underground Path North-South", "warp", "Route 5 Gate"),
    MapConnection("Underground Path North-South", "warp", "Route 6 Gate"),
    MapConnection("Route 6 Gate", "warp", "Route 6"),
    MapConnection("Route 6", "warp", "Route 6 Gate"),
    # Underground paths (Route 7 <-> Route 8)
    MapConnection("Route 7", "warp", "Route 7 Gate"),
    MapConnection("Route 7 Gate", "warp", "Route 7"),
    MapConnection("Route 8", "warp", "Route 8 Gate"),
    MapConnection("Route 8 Gate", "warp", "Route 8"),
    MapConnection("Underground Path West-East", "warp", "Route 7 Gate"),
    MapConnection("Underground Path West-East", "warp", "Route 8 Gate"),
    # Vermilion City buildings
    MapConnection("Vermilion City", "warp", "Vermilion Pokecenter"),
    MapConnection("Vermilion Pokecenter", "warp", "Vermilion City"),
    MapConnection("Vermilion City", "warp", "Vermilion Mart"),
    MapConnection("Vermilion Mart", "warp", "Vermilion City"),
    MapConnection("Vermilion City", "warp", "Vermilion Gym"),
    MapConnection("Vermilion Gym", "warp", "Vermilion City"),
    MapConnection("Vermilion City", "warp", "Vermilion Dock"),
    MapConnection("Vermilion Dock", "warp", "Vermilion City"),
    MapConnection("Vermilion Dock", "warp", "S.S. Anne 1F"),
    MapConnection("S.S. Anne 1F", "warp", "Vermilion Dock"),
    MapConnection("S.S. Anne 1F", "warp", "S.S. Anne 2F"),
    MapConnection("S.S. Anne 2F", "warp", "S.S. Anne 1F"),
    MapConnection("S.S. Anne 2F", "warp", "S.S. Anne Captain's Room"),
    MapConnection("S.S. Anne Captain's Room", "warp", "S.S. Anne 2F"),
    # Rock Tunnel
    MapConnection("Route 10", "warp", "Rock Tunnel Pokecenter"),
    MapConnection("Rock Tunnel Pokecenter", "warp", "Route 10"),
    MapConnection("Route 10", "warp", "Rock Tunnel 1F"),
    MapConnection("Rock Tunnel 1F", "warp", "Route 10"),
    MapConnection("Rock Tunnel 1F", "warp", "Rock Tunnel B1F"),
    MapConnection("Rock Tunnel B1F", "warp", "Rock Tunnel 1F"),
    MapConnection("Rock Tunnel B1F", "warp", "Route 10"),
    # Lavender Town buildings
    MapConnection("Lavender Town", "warp", "Lavender Pokecenter"),
    MapConnection("Lavender Pokecenter", "warp", "Lavender Town"),
    MapConnection("Lavender Town", "warp", "Pokemon Tower 1F"),
    MapConnection("Pokemon Tower 1F", "warp", "Lavender Town"),
    MapConnection("Pokemon Tower 1F", "warp", "Pokemon Tower 2F"),
    MapConnection("Pokemon Tower 2F", "warp", "Pokemon Tower 1F"),
    MapConnection("Pokemon Tower 2F", "warp", "Pokemon Tower 3F"),
    MapConnection("Pokemon Tower 3F", "warp", "Pokemon Tower 2F"),
    MapConnection("Pokemon Tower 3F", "warp", "Pokemon Tower 4F"),
    MapConnection("Pokemon Tower 4F", "warp", "Pokemon Tower 3F"),
    MapConnection("Pokemon Tower 4F", "warp", "Pokemon Tower 5F"),
    MapConnection("Pokemon Tower 5F", "warp", "Pokemon Tower 4F"),
    MapConnection("Pokemon Tower 5F", "warp", "Pokemon Tower 6F"),
    MapConnection("Pokemon Tower 6F", "warp", "Pokemon Tower 5F"),
    MapConnection("Pokemon Tower 6F", "warp", "Pokemon Tower 7F"),
    MapConnection("Pokemon Tower 7F", "warp", "Pokemon Tower 6F"),
    MapConnection("Lavender Town", "warp", "Mr. Fuji's House"),
    MapConnection("Mr. Fuji's House", "warp", "Lavender Town"),
    # Celadon City buildings
    MapConnection("Celadon City", "warp", "Celadon Pokecenter"),
    MapConnection("Celadon Pokecenter", "warp", "Celadon City"),
    MapConnection("Celadon City", "warp", "Celadon Gym"),
    MapConnection("Celadon Gym", "warp", "Celadon City"),
    MapConnection("Celadon City", "warp", "Game Corner"),
    MapConnection("Game Corner", "warp", "Celadon City"),
    MapConnection("Game Corner", "warp", "Rocket Hideout B1F"),
    MapConnection("Rocket Hideout B1F", "warp", "Game Corner"),
    MapConnection("Rocket Hideout B1F", "warp", "Rocket Hideout B2F"),
    MapConnection("Rocket Hideout B2F", "warp", "Rocket Hideout B1F"),
    MapConnection("Rocket Hideout B2F", "warp", "Rocket Hideout B3F"),
    MapConnection("Rocket Hideout B3F", "warp", "Rocket Hideout B2F"),
    MapConnection("Rocket Hideout B3F", "warp", "Rocket Hideout B4F"),
    MapConnection("Rocket Hideout B4F", "warp", "Rocket Hideout B3F"),
    MapConnection("Celadon City", "warp", "Celadon Mart 1F"),
    MapConnection("Celadon Mart 1F", "warp", "Celadon City"),
    # Fuchsia City buildings
    MapConnection("Fuchsia City", "warp", "Fuchsia Pokecenter"),
    MapConnection("Fuchsia Pokecenter", "warp", "Fuchsia City"),
    MapConnection("Fuchsia City", "warp", "Fuchsia Gym"),
    MapConnection("Fuchsia Gym", "warp", "Fuchsia City"),
    MapConnection("Fuchsia City", "warp", "Safari Zone Gate"),
    MapConnection("Safari Zone Gate", "warp", "Fuchsia City"),
    MapConnection("Safari Zone Gate", "warp", "Safari Zone Center"),
    MapConnection("Safari Zone Center", "warp", "Safari Zone Gate"),
    MapConnection("Safari Zone Center", "warp", "Safari Zone East"),
    MapConnection("Safari Zone East", "warp", "Safari Zone Center"),
    MapConnection("Safari Zone Center", "warp", "Safari Zone West"),
    MapConnection("Safari Zone West", "warp", "Safari Zone Center"),
    MapConnection("Safari Zone Center", "warp", "Safari Zone North"),
    MapConnection("Safari Zone North", "warp", "Safari Zone Center"),
    MapConnection("Safari Zone West", "warp", "Safari Zone Secret House"),
    MapConnection("Safari Zone Secret House", "warp", "Safari Zone West"),
    MapConnection("Fuchsia City", "warp", "Warden's House"),
    MapConnection("Warden's House", "warp", "Fuchsia City"),
    # Saffron City buildings
    MapConnection("Saffron City", "warp", "Saffron Pokecenter"),
    MapConnection("Saffron Pokecenter", "warp", "Saffron City"),
    MapConnection("Saffron City", "warp", "Saffron Gym"),
    MapConnection("Saffron Gym", "warp", "Saffron City"),
    MapConnection("Saffron City", "warp", "Silph Co. 1F"),
    MapConnection("Silph Co. 1F", "warp", "Saffron City"),
    MapConnection("Saffron City", "warp", "Saffron Mart"),
    MapConnection("Saffron Mart", "warp", "Saffron City"),
    # Cinnabar Island buildings
    MapConnection("Cinnabar Island", "warp", "Cinnabar Pokecenter"),
    MapConnection("Cinnabar Pokecenter", "warp", "Cinnabar Island"),
    MapConnection("Cinnabar Island", "warp", "Cinnabar Gym"),
    MapConnection("Cinnabar Gym", "warp", "Cinnabar Island"),
    MapConnection("Cinnabar Island", "warp", "Pokemon Mansion 1F"),
    MapConnection("Pokemon Mansion 1F", "warp", "Cinnabar Island"),
    MapConnection("Cinnabar Island", "warp", "Cinnabar Lab"),
    MapConnection("Cinnabar Lab", "warp", "Cinnabar Island"),
    MapConnection("Cinnabar Island", "warp", "Cinnabar Mart"),
    MapConnection("Cinnabar Mart", "warp", "Cinnabar Island"),
    # Pokemon Mansion floors
    MapConnection("Pokemon Mansion 1F", "warp", "Pokemon Mansion 2F"),
    MapConnection("Pokemon Mansion 2F", "warp", "Pokemon Mansion 1F"),
    MapConnection("Pokemon Mansion 2F", "warp", "Pokemon Mansion 3F"),
    MapConnection("Pokemon Mansion 3F", "warp", "Pokemon Mansion 2F"),
    MapConnection("Pokemon Mansion 3F", "warp", "Pokemon Mansion B1F"),
    MapConnection("Pokemon Mansion B1F", "warp", "Pokemon Mansion 3F"),
    # Victory Road
    MapConnection("Route 23", "warp", "Victory Road 1F"),
    MapConnection("Victory Road 1F", "warp", "Route 23"),
    MapConnection("Victory Road 1F", "warp", "Victory Road 2F"),
    MapConnection("Victory Road 2F", "warp", "Victory Road 1F"),
    MapConnection("Victory Road 2F", "warp", "Victory Road 3F"),
    MapConnection("Victory Road 3F", "warp", "Victory Road 2F"),
    MapConnection("Victory Road 2F", "warp", "Indigo Plateau"),
    # Indigo Plateau / Elite Four
    MapConnection("Indigo Plateau", "warp", "Indigo Plateau Lobby"),
    MapConnection("Indigo Plateau Lobby", "warp", "Indigo Plateau"),
    MapConnection("Indigo Plateau Lobby", "warp", "Lorelei's Room"),
    MapConnection("Lorelei's Room", "warp", "Bruno's Room"),
    MapConnection("Bruno's Room", "warp", "Agatha's Room"),
    MapConnection("Agatha's Room", "warp", "Lance's Room"),
    MapConnection("Lance's Room", "warp", "Champion's Room"),
    MapConnection("Champion's Room", "warp", "Hall of Fame"),
    # Route 22 gate
    MapConnection("Route 22", "warp", "Route 22 Gate"),
    MapConnection("Route 22 Gate", "warp", "Route 22"),
    MapConnection("Route 22 Gate", "warp", "Route 23"),
    # Diglett's Cave (connects Route 2 and Route 11)
    MapConnection("Route 2", "warp", "Diglett's Cave Route 2"),
    MapConnection("Diglett's Cave Route 2", "warp", "Route 2"),
    MapConnection("Diglett's Cave Route 2", "warp", "Diglett's Cave"),
    MapConnection("Diglett's Cave", "warp", "Diglett's Cave Route 2"),
    MapConnection("Diglett's Cave", "warp", "Diglett's Cave Route 11"),
    MapConnection("Diglett's Cave Route 11", "warp", "Diglett's Cave"),
    MapConnection("Diglett's Cave Route 11", "warp", "Route 11"),
    MapConnection("Route 11", "warp", "Diglett's Cave Route 11"),
    # Route gates
    MapConnection("Route 11", "warp", "Route 11 Gate 1F"),
    MapConnection("Route 11 Gate 1F", "warp", "Route 11"),
    MapConnection("Route 11 Gate 1F", "warp", "Route 12"),
    MapConnection("Route 12", "warp", "Route 12 Gate 1F"),
    MapConnection("Route 12 Gate 1F", "warp", "Route 12"),
    MapConnection("Route 15", "warp", "Route 15 Gate 1F"),
    MapConnection("Route 15 Gate 1F", "warp", "Route 15"),
    MapConnection("Route 15 Gate 1F", "warp", "Fuchsia City"),
    MapConnection("Route 16", "warp", "Route 16 Gate 1F"),
    MapConnection("Route 16 Gate 1F", "warp", "Route 16"),
    MapConnection("Route 16 Gate 1F", "warp", "Route 17"),
    MapConnection("Route 18", "warp", "Route 18 Gate 1F"),
    MapConnection("Route 18 Gate 1F", "warp", "Route 18"),
    MapConnection("Route 18 Gate 1F", "warp", "Fuchsia City"),
    # Silph Co floors (story critical path only)
    MapConnection("Silph Co. 1F", "warp", "Silph Co. 2F"),
    MapConnection("Silph Co. 2F", "warp", "Silph Co. 1F"),
    MapConnection("Silph Co. 2F", "warp", "Silph Co. 3F"),
    MapConnection("Silph Co. 3F", "warp", "Silph Co. 2F"),
    MapConnection("Silph Co. 3F", "warp", "Silph Co. 4F"),
    MapConnection("Silph Co. 4F", "warp", "Silph Co. 3F"),
    MapConnection("Silph Co. 4F", "warp", "Silph Co. 5F"),
    MapConnection("Silph Co. 5F", "warp", "Silph Co. 4F"),
    MapConnection("Silph Co. 5F", "warp", "Silph Co. 6F"),
    MapConnection("Silph Co. 6F", "warp", "Silph Co. 5F"),
    MapConnection("Silph Co. 6F", "warp", "Silph Co. 7F"),
    MapConnection("Silph Co. 7F", "warp", "Silph Co. 6F"),
    MapConnection("Silph Co. 7F", "warp", "Silph Co. 8F"),
    MapConnection("Silph Co. 8F", "warp", "Silph Co. 7F"),
    MapConnection("Silph Co. 8F", "warp", "Silph Co. 9F"),
    MapConnection("Silph Co. 9F", "warp", "Silph Co. 8F"),
    MapConnection("Silph Co. 9F", "warp", "Silph Co. 10F"),
    MapConnection("Silph Co. 10F", "warp", "Silph Co. 9F"),
    MapConnection("Silph Co. 10F", "warp", "Silph Co. 11F"),
    MapConnection("Silph Co. 11F", "warp", "Silph Co. 10F"),
)

# Build adjacency index: map_name -> list of (direction, destination)
_ADJACENCY: dict[str, list[tuple[str, str]]] = {}
for _conn in MAP_CONNECTIONS + WARP_CONNECTIONS:
    _ADJACENCY.setdefault(_conn.from_map, []).append((_conn.direction, _conn.to_map))

_GENERIC_MAP_TOKENS = {
    "city",
    "town",
    "route",
    "road",
    "house",
    "gym",
    "lab",
    "forest",
    "cave",
    "tower",
    "dock",
    "center",
    "pokecenter",
    "room",
    "rooms",
    "island",
    "plateau",
    "gate",
    "hideout",
    "mansion",
    "building",
    "co",
    "lobby",
    "floor",
}


def exits_from(map_name: str) -> list[MapConnection]:
    """Return all known connections leaving the given map."""
    return [
        MapConnection(from_map=map_name, direction=direction, to_map=to_map)
        for direction, to_map in _ADJACENCY.get(map_name, [])
    ]


def destination_for_exit(map_name: str, direction: str) -> str | None:
    """Return the destination map name for a given exit direction, or None if unknown.

    For boundary exits, direction is 'north', 'south', 'east', 'west'.
    Returns the first match found.
    """
    direction_lower = direction.lower()
    for conn_direction, to_map in _ADJACENCY.get(map_name, []):
        if conn_direction.lower() == direction_lower:
            return to_map
    return None


def destinations_for_exit(map_name: str, direction: str) -> list[str]:
    """Return all destination maps for a given exit direction."""
    direction_lower = direction.lower()
    return [
        to_map
        for conn_direction, to_map in _ADJACENCY.get(map_name, [])
        if conn_direction.lower() == direction_lower
    ]


def shortest_map_path(from_map: str, to_map: str) -> list[str] | None:
    """BFS shortest path between two maps. Returns list of map names (excluding from_map).

    Returns None if no path found or if from_map == to_map.
    """
    if _map_matches(from_map, to_map):
        return []
    if from_map not in _ADJACENCY:
        return None

    target_candidates = _resolve_target_candidates(to_map)
    if not target_candidates:
        return None

    visited: set[str] = {from_map}
    queue: deque[tuple[str, list[str]]] = deque()
    for _direction, neighbor in _ADJACENCY.get(from_map, []):
        if neighbor not in visited:
            visited.add(neighbor)
            queue.append((neighbor, [neighbor]))

    while queue:
        current, path = queue.popleft()
        if current in target_candidates:
            return path
        for _direction, neighbor in _ADJACENCY.get(current, []):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, path + [neighbor]))

    return None


def direction_toward(from_map: str, to_map: str) -> str | None:
    """Return the exit direction from from_map that leads toward to_map.

    Uses shortest_map_path to find the route, then returns the direction
    of the first hop. Returns None if no path exists.
    """
    path = shortest_map_path(from_map, to_map)
    if not path:
        return None
    next_map = path[0]
    for conn_direction, dest in _ADJACENCY.get(from_map, []):
        if dest == next_map:
            return conn_direction
    return None


def next_hop_toward(from_map: str, to_map: str) -> MapConnection | None:
    path = shortest_map_path(from_map, to_map)
    if not path:
        return None
    next_map = path[0]
    for conn_direction, dest in _ADJACENCY.get(from_map, []):
        if dest == next_map:
            return MapConnection(from_map=from_map, direction=conn_direction, to_map=dest)
    return None


def map_matches(current_map_name: str, target_map_name: str) -> bool:
    return _map_matches(current_map_name, target_map_name)


def _resolve_target_candidates(target_map_name: str) -> set[str]:
    if target_map_name in _ADJACENCY:
        return {target_map_name}
    candidates = {map_name for map_name in _ADJACENCY if _map_matches(map_name, target_map_name)}
    if candidates:
        return candidates
    return {target_map_name} if target_map_name in _ADJACENCY else set()


def _map_matches(current_map_name: str, target_map_name: str) -> bool:
    current_tokens = _tokenize_name(current_map_name)
    target_tokens = _tokenize_name(target_map_name)
    if not current_tokens or not target_tokens:
        return False
    if target_tokens.issubset(current_tokens) or current_tokens.issubset(target_tokens):
        return True

    current_core = current_tokens - _GENERIC_MAP_TOKENS
    target_core = target_tokens - _GENERIC_MAP_TOKENS
    if not current_core or not target_core:
        return False
    return current_core.issubset(target_core) or target_core.issubset(current_core)


def _tokenize_name(value: str) -> set[str]:
    return {token for token in re.findall(r"[a-z0-9]+", value.lower()) if token}
