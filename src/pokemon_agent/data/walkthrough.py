from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Iterable


@dataclass(frozen=True, slots=True)
class Milestone:
    id: str
    description: str
    completion_flag: str | None
    completion_item: str | None
    prerequisite_flags: list[str]
    prerequisite_items: list[str]
    target_map_name: str
    route_hints: list[str]
    sub_steps: list[str]
    required_hms: list[str]
    next_milestone_id: str | None


MILESTONES: tuple[Milestone, ...] = (
    Milestone(
        id="get_starter",
        description="Collect your starter Pokemon from Professor Oak.",
        completion_flag="got_starter",
        completion_item=None,
        prerequisite_flags=[],
        prerequisite_items=[],
        target_map_name="Oak's Lab",
        route_hints=[
            "Leave your bedroom and head downstairs.",
            "Walk north from Pallet Town into Oak's Lab.",
        ],
        sub_steps=[
            "Exit Red's House 2F and go downstairs to the first floor.",
            "Leave Red's House and step into Pallet Town.",
            "Walk north to Oak's Lab before trying to leave town.",
            "Talk to Professor Oak and wait for him to start the starter sequence.",
            "Choose a starter Pokemon from the Pokeballs on the table.",
            "Finish the rival battle or dialogue that follows the starter choice.",
        ],
        required_hms=[],
        next_milestone_id="deliver_parcel",
    ),
    Milestone(
        id="deliver_parcel",
        description="Deliver Oak's Parcel from Viridian City back to Professor Oak.",
        completion_flag="oak_received_parcel",
        completion_item=None,
        prerequisite_flags=["got_starter"],
        prerequisite_items=[],
        target_map_name="Viridian City",
        route_hints=[
            "Go north from Pallet Town through Route 1 to Viridian City.",
            "Visit the Viridian Mart to trigger the parcel errand.",
            "Return south to Pallet Town and re-enter Oak's Lab.",
        ],
        sub_steps=[
            "Travel north from Pallet Town along Route 1 into Viridian City.",
            "Enter the Viridian Mart and talk to the clerk to receive Oak's Parcel.",
            "Leave Viridian City and head south on Route 1 back to Pallet Town.",
            "Enter Oak's Lab and speak with Professor Oak.",
            "Stay in the conversation until Oak's Parcel is handed over.",
        ],
        required_hms=[],
        next_milestone_id="get_pokedex",
    ),
    Milestone(
        id="get_pokedex",
        description="Receive the Pokedex and Pokeballs from Professor Oak.",
        completion_flag="got_pokedex",
        completion_item=None,
        prerequisite_flags=["oak_received_parcel"],
        prerequisite_items=[],
        target_map_name="Oak's Lab",
        route_hints=[
            "Stay in Oak's Lab after delivering the parcel.",
            "Talk through Oak and Blue's follow-up dialogue.",
        ],
        sub_steps=[
            "Remain in Oak's Lab after the parcel handoff.",
            "Let Oak and Blue finish their scripted dialogue.",
            "Talk to Oak if needed to receive the Pokedex.",
            "Advance the remaining dialogue until you regain control.",
            "Confirm you have the Pokedex and starting Pokeballs before leaving.",
        ],
        required_hms=[],
        next_milestone_id="viridian_forest",
    ),
    Milestone(
        id="viridian_forest",
        description="Cross Viridian Forest and reach Pewter City.",
        completion_flag=None,
        completion_item=None,
        prerequisite_flags=["got_pokedex"],
        prerequisite_items=[],
        target_map_name="Viridian Forest",
        route_hints=[
            "Head north from Viridian City toward Route 2.",
            "Use the south gate to enter Viridian Forest.",
            "Follow the forest's northern path to the Pewter exit.",
        ],
        sub_steps=[
            "Travel north out of Viridian City onto Route 2.",
            "Enter Viridian Forest through the south gate.",
            "Follow the main path while avoiding dead ends and trainers when possible.",
            "Use item balls or safe wild battles only if they help stabilize the run.",
            "Reach the north gate and exit into Pewter City.",
        ],
        required_hms=[],
        next_milestone_id="gym1_brock",
    ),
    Milestone(
        id="gym1_brock",
        description="Defeat Brock in Pewter City Gym and earn the Boulder Badge.",
        completion_flag="beat_brock",
        completion_item="Boulder Badge",
        prerequisite_flags=["got_pokedex"],
        prerequisite_items=[],
        target_map_name="Pewter City",
        route_hints=[
            "Heal and prepare in Pewter City before entering the gym.",
            "Go to the Pewter Gym once your lead Pokemon is ready.",
        ],
        sub_steps=[
            "Enter Pewter City and heal or buy supplies if needed.",
            "Train briefly on nearby wild Pokemon if the team is underleveled.",
            "Walk to the Pewter Gym and talk through the guide dialogue.",
            "Defeat Brock's junior trainer if it blocks the path.",
            "Battle Brock and continue the dialogue until the Boulder Badge is awarded.",
        ],
        required_hms=[],
        next_milestone_id="mt_moon",
    ),
    Milestone(
        id="mt_moon",
        description="Clear Mt. Moon and emerge in Cerulean City's direction.",
        completion_flag=None,
        completion_item=None,
        prerequisite_flags=["beat_brock"],
        prerequisite_items=[],
        target_map_name="Mt Moon",
        route_hints=[
            "Go east from Pewter City across Route 3.",
            "Enter Mt. Moon from the Route 3 exit.",
            "Follow the ladders downward until you reach Route 4.",
        ],
        sub_steps=[
            "Leave Pewter City to the east and cross Route 3.",
            "Heal at the Mt. Moon Pokecenter before entering if the party is weak.",
            "Enter Mt. Moon and work downward through the cave floors.",
            "Beat or route around trainers who block the ladder path.",
            "Choose a fossil when forced and continue toward the eastern exit.",
            "Leave Mt. Moon onto Route 4 and continue toward Cerulean City.",
        ],
        required_hms=[],
        next_milestone_id="gym2_misty",
    ),
    Milestone(
        id="gym2_misty",
        description="Defeat Misty in Cerulean Gym and earn the Cascade Badge.",
        completion_flag="beat_misty",
        completion_item="Cascade Badge",
        prerequisite_flags=["beat_brock"],
        prerequisite_items=[],
        target_map_name="Cerulean City",
        route_hints=[
            "Finish the Route 4 approach and heal in Cerulean City.",
            "Challenge the Cerulean Gym when the team is ready for Water types.",
        ],
        sub_steps=[
            "Arrive in Cerulean City from Route 4.",
            "Heal and restock items before taking on the gym.",
            "Enter Cerulean Gym and clear the junior trainer if needed.",
            "Fight Misty and stay in dialogue until the Cascade Badge is awarded.",
            "Leave the gym ready for the northern route and Bill sidequest.",
        ],
        required_hms=[],
        next_milestone_id="bills_house",
    ),
    Milestone(
        id="bills_house",
        description="Help Bill and receive the S.S. Ticket.",
        completion_flag="got_ss_ticket",
        completion_item="S.S. Ticket",
        prerequisite_flags=["beat_misty"],
        prerequisite_items=[],
        target_map_name="Bill's House",
        route_hints=[
            "Go north from Cerulean City through Route 24 and Route 25.",
            "Follow the cape path to Bill's seaside cottage.",
        ],
        sub_steps=[
            "Leave Cerulean City through the north bridge.",
            "Cross Route 24 and continue east along Route 25.",
            "Reach Bill's House at the end of the cape.",
            "Talk to Bill and help resolve the teleporter accident.",
            "Stay in the conversation until Bill gives you the S.S. Ticket.",
        ],
        required_hms=[],
        next_milestone_id="ss_anne_cut",
    ),
    Milestone(
        id="ss_anne_cut",
        description="Board the S.S. Anne and obtain HM01 Cut.",
        completion_flag="got_hm01_cut",
        completion_item=None,
        prerequisite_flags=["got_ss_ticket"],
        prerequisite_items=["S.S. Ticket"],
        target_map_name="S.S. Anne",
        route_hints=[
            "From Cerulean, go south to Route 5.",
            "Take the Underground Path to Route 6 and enter Vermilion City.",
            "Use the dock to board the S.S. Anne.",
        ],
        sub_steps=[
            "Travel from Cerulean City south through Route 5.",
            "Enter the Underground Path to reach Route 6.",
            "Arrive in Vermilion City and head to the dock.",
            "Board the S.S. Anne using the S.S. Ticket.",
            "Work upward through the ship toward the captain's room.",
            "Talk to the Captain and receive HM01 Cut.",
        ],
        required_hms=[],
        next_milestone_id="gym3_surge",
    ),
    Milestone(
        id="gym3_surge",
        description="Defeat Lt. Surge in Vermilion Gym and earn the Thunder Badge.",
        completion_flag="beat_lt_surge",
        completion_item="Thunder Badge",
        prerequisite_flags=["got_hm01_cut"],
        prerequisite_items=[],
        target_map_name="Vermilion City",
        route_hints=[
            "Teach Cut to a compatible Pokemon if nobody can use it yet.",
            "Cut the small tree near Vermilion Gym to enter.",
        ],
        sub_steps=[
            "Open the party or HM flow and teach Cut if the team still lacks it.",
            "Go to the tree blocking Vermilion Gym and use Cut.",
            "Enter Vermilion Gym and work through the trash can lock puzzle.",
            "Battle Lt. Surge and finish the badge dialogue.",
            "Leave Vermilion City ready to pivot east toward Rock Tunnel.",
        ],
        required_hms=["Cut"],
        next_milestone_id="rock_tunnel",
    ),
    Milestone(
        id="rock_tunnel",
        description="Travel to Rock Tunnel and cross it into Lavender Town.",
        completion_flag=None,
        completion_item=None,
        prerequisite_flags=["beat_lt_surge"],
        prerequisite_items=[],
        target_map_name="Rock Tunnel",
        route_hints=[
            "Return to Cerulean City and go east onto Route 9.",
            "Continue south on Route 10 to Rock Tunnel.",
            "Use Flash if available, then work through the cave to Lavender Town.",
        ],
        sub_steps=[
            "Make your way back to Cerulean City after Vermilion.",
            "Travel east across Route 9 and down Route 10.",
            "Pick up HM05 Flash from the Route 2 aide if the team can meet the requirement.",
            "Enter Rock Tunnel and navigate through the 1F and B1F floors.",
            "Exit Rock Tunnel into Lavender Town.",
        ],
        required_hms=["Flash"],
        next_milestone_id="celadon_city",
    ),
    Milestone(
        id="celadon_city",
        description="Reach Celadon City and unlock the next major set of objectives there.",
        completion_flag=None,
        completion_item=None,
        prerequisite_flags=["beat_lt_surge"],
        prerequisite_items=[],
        target_map_name="Celadon City",
        route_hints=[
            "Go west from Lavender Town through Route 8.",
            "Use the Underground Path to bypass Saffron's gate guards.",
            "Exit onto Route 7 and enter Celadon City.",
        ],
        sub_steps=[
            "Leave Lavender Town heading west along Route 8.",
            "Enter the Underground Path to bypass the blocked Saffron crossing.",
            "Surface on Route 7 and walk into Celadon City.",
            "Heal, shop, and collect any key support items that are easy to grab.",
            "Scout the gym, Game Corner, and department store so later steps are nearby.",
        ],
        required_hms=[],
        next_milestone_id="gym4_erika",
    ),
    Milestone(
        id="gym4_erika",
        description="Defeat Erika in Celadon Gym and earn the Rainbow Badge.",
        completion_flag="beat_erika",
        completion_item="Rainbow Badge",
        prerequisite_flags=["beat_lt_surge"],
        prerequisite_items=[],
        target_map_name="Celadon Gym",
        route_hints=[
            "Use Cut to enter the tree-blocked Celadon Gym.",
            "Prepare for Grass and status-heavy battles before the leader fight.",
        ],
        sub_steps=[
            "Heal or buy status items in Celadon City before entering the gym.",
            "Use Cut on the tree that blocks the Celadon Gym entrance.",
            "Work through the trainer maze toward Erika.",
            "Defeat Erika and finish the badge dialogue before leaving.",
        ],
        required_hms=["Cut"],
        next_milestone_id="rocket_hideout",
    ),
    Milestone(
        id="rocket_hideout",
        description="Clear the Rocket Hideout beneath Celadon Game Corner and obtain the Silph Scope.",
        completion_flag=None,
        completion_item="Silph Scope",
        prerequisite_flags=["beat_erika"],
        prerequisite_items=[],
        target_map_name="Rocket Hideout",
        route_hints=[
            "Inspect Celadon Game Corner to reveal the hidden Rocket Hideout switch.",
            "Descend through the hideout floors until Giovanni is defeated.",
        ],
        sub_steps=[
            "Enter the Celadon Game Corner and inspect the suspicious poster or switch area.",
            "Reveal the Rocket Hideout staircase and head downstairs.",
            "Navigate the spin-tile floors and pick up the Lift Key when it drops.",
            "Use the elevator or stairs to reach Giovanni's floor.",
            "Defeat Giovanni and collect the Silph Scope reward.",
        ],
        required_hms=[],
        next_milestone_id="pokemon_tower",
    ),
    Milestone(
        id="pokemon_tower",
        description="Climb Pokemon Tower and rescue Mr. Fuji to obtain the Poke Flute.",
        completion_flag="got_poke_flute",
        completion_item="Poke Flute",
        prerequisite_flags=[],
        prerequisite_items=["Silph Scope"],
        target_map_name="Pokemon Tower",
        route_hints=[
            "Return east to Lavender Town after getting the Silph Scope.",
            "Climb Pokemon Tower until you reach the top rescue sequence.",
        ],
        sub_steps=[
            "Travel back to Lavender Town from Celadon City.",
            "Enter Pokemon Tower and ascend floor by floor.",
            "Use the Silph Scope to reveal and defeat or bypass the ghost roadblocks.",
            "Reach the upper floors and defeat the Team Rocket rescuers.",
            "Talk to Mr. Fuji after the rescue to receive the Poke Flute.",
        ],
        required_hms=[],
        next_milestone_id="fuchsia_city",
    ),
    Milestone(
        id="fuchsia_city",
        description="Reach Fuchsia City and secure Surf from the Safari Zone arc.",
        completion_flag="got_hm03_surf",
        completion_item=None,
        prerequisite_flags=["got_poke_flute"],
        prerequisite_items=["Poke Flute"],
        target_map_name="Fuchsia City",
        route_hints=[
            "Use Cycling Road from Celadon or the eastern routes from Lavender to reach Fuchsia.",
            "Enter the Safari Zone once you arrive in Fuchsia City.",
        ],
        sub_steps=[
            "Travel to Fuchsia City using the safest available route.",
            "Heal and stock up on balls or healing items before entering the Safari Zone.",
            "Explore the Safari Zone until you find the Secret House.",
            "Talk to the NPC in the Secret House to obtain HM03 Surf.",
            "Pick up the Gold Teeth for the Warden if you can do it on the same pass.",
        ],
        required_hms=[],
        next_milestone_id="gym5_koga",
    ),
    Milestone(
        id="gym5_koga",
        description="Defeat Koga in Fuchsia Gym and earn the Soul Badge.",
        completion_flag="beat_koga",
        completion_item="Soul Badge",
        prerequisite_flags=["got_hm03_surf"],
        prerequisite_items=[],
        target_map_name="Fuchsia Gym",
        route_hints=[
            "Turn in the Gold Teeth to the Warden if you still need Strength support.",
            "Use the invisible-wall clues in Fuchsia Gym to reach Koga.",
        ],
        sub_steps=[
            "Visit the Warden's House if you still need HM04 Strength from the Gold Teeth.",
            "Enter Fuchsia Gym and move carefully through the invisible wall maze.",
            "Defeat the required trainers or route around them efficiently.",
            "Battle Koga and finish the badge dialogue.",
        ],
        required_hms=[],
        next_milestone_id="silph_co",
    ),
    Milestone(
        id="silph_co",
        description="Clear Silph Co. in Saffron City and obtain the Master Ball.",
        completion_flag="got_master_ball",
        completion_item="Master Ball",
        prerequisite_flags=["beat_koga"],
        prerequisite_items=[],
        target_map_name="Silph Co.",
        route_hints=[
            "Reach Saffron City and enter the Silph Co. building.",
            "Use the Card Key and warp pads to reach Giovanni and the president.",
        ],
        sub_steps=[
            "Travel to Saffron City and enter Silph Co.",
            "Work upward through the office floors while collecting the Card Key.",
            "Unlock the needed doors and use warp pads to reach the rival and Giovanni fights.",
            "Defeat Giovanni in Silph Co. and continue to the president's room.",
            "Talk to the president to receive the Master Ball.",
        ],
        required_hms=[],
        next_milestone_id="gym6_sabrina",
    ),
    Milestone(
        id="gym6_sabrina",
        description="Defeat Sabrina in Saffron Gym and earn the Marsh Badge.",
        completion_flag="beat_sabrina",
        completion_item="Marsh Badge",
        prerequisite_flags=["got_master_ball"],
        prerequisite_items=["Master Ball"],
        target_map_name="Saffron Gym",
        route_hints=[
            "Challenge Sabrina after Silph Co. is resolved.",
            "Use the warp pads to route toward the gym leader room.",
        ],
        sub_steps=[
            "Heal and prepare for Psychic-type battles in Saffron City.",
            "Enter Saffron Gym and begin mapping the warp pad sequence.",
            "Route through the warp rooms until Sabrina is reachable.",
            "Defeat Sabrina and stay in dialogue until the Marsh Badge is awarded.",
        ],
        required_hms=[],
        next_milestone_id="cinnabar_island",
    ),
    Milestone(
        id="cinnabar_island",
        description="Reach Cinnabar Island and obtain the Secret Key from Pokemon Mansion.",
        completion_flag=None,
        completion_item="Secret Key",
        prerequisite_flags=["beat_sabrina", "got_hm03_surf"],
        prerequisite_items=[],
        target_map_name="Cinnabar Island",
        route_hints=[
            "Surf south from Pallet Town or west from Fuchsia to reach Cinnabar Island.",
            "Explore Pokemon Mansion to find the Secret Key.",
        ],
        sub_steps=[
            "Use Surf to reach Cinnabar Island.",
            "Heal at the Pokemon Center and restock before entering the mansion.",
            "Enter Pokemon Mansion and use statue switches to open the route forward.",
            "Navigate to the lower floors and pick up the Secret Key.",
            "Return to town ready to open Cinnabar Gym.",
        ],
        required_hms=["Surf"],
        next_milestone_id="gym7_blaine",
    ),
    Milestone(
        id="gym7_blaine",
        description="Defeat Blaine in Cinnabar Gym and earn the Volcano Badge.",
        completion_flag="beat_blaine",
        completion_item="Volcano Badge",
        prerequisite_flags=["beat_sabrina"],
        prerequisite_items=["Secret Key"],
        target_map_name="Cinnabar Gym",
        route_hints=[
            "Unlock the gym with the Secret Key and answer quiz prompts when helpful.",
            "Prepare for Fire-type battles before challenging Blaine.",
        ],
        sub_steps=[
            "Use the Secret Key to unlock Cinnabar Gym.",
            "Move through the gym by answering quiz machines or fighting the trainers.",
            "Reach Blaine and complete the leader battle.",
            "Finish the post-battle dialogue until the Volcano Badge is awarded.",
        ],
        required_hms=[],
        next_milestone_id="gym8_giovanni",
    ),
    Milestone(
        id="gym8_giovanni",
        description="Defeat Giovanni in Viridian Gym and earn the Earth Badge.",
        completion_flag="beat_giovanni",
        completion_item="Earth Badge",
        prerequisite_flags=["beat_blaine"],
        prerequisite_items=[],
        target_map_name="Viridian Gym",
        route_hints=[
            "Return to Viridian City after Cinnabar Island.",
            "Clear the moving-tile gym puzzle and challenge Giovanni.",
        ],
        sub_steps=[
            "Travel back to Viridian City and enter the newly opened gym.",
            "Use the arrow tiles to route through the trainer layout efficiently.",
            "Defeat Giovanni and continue dialogue until the Earth Badge is awarded.",
            "Leave the gym and prepare for the Pokemon League approach.",
        ],
        required_hms=[],
        next_milestone_id="victory_road",
    ),
    Milestone(
        id="victory_road",
        description="Cross Route 22, Route 23, and Victory Road to reach Indigo Plateau.",
        completion_flag=None,
        completion_item=None,
        prerequisite_flags=["beat_giovanni"],
        prerequisite_items=[],
        target_map_name="Victory Road",
        route_hints=[
            "Go west from Viridian City onto Route 22 and then north on Route 23.",
            "Use Strength in Victory Road to solve the boulder switches.",
        ],
        sub_steps=[
            "Travel west from Viridian City onto Route 22 and continue toward the league gate.",
            "Pass the badge checks on Route 23.",
            "Enter Victory Road and push boulders onto the switch plates.",
            "Climb through the cave floors until the Indigo Plateau exit is open.",
            "Leave Victory Road and enter Indigo Plateau.",
        ],
        required_hms=["Strength"],
        next_milestone_id="elite_four_champion",
    ),
    Milestone(
        id="elite_four_champion",
        description="Defeat the Elite Four and the Champion to complete Pokemon Red.",
        completion_flag="beat_champion",
        completion_item=None,
        prerequisite_flags=["beat_giovanni"],
        prerequisite_items=[],
        target_map_name="Indigo Plateau",
        route_hints=[
            "Buy final healing items and restore PP before starting the gauntlet.",
            "Commit to the Elite Four run only when the party is ready.",
        ],
        sub_steps=[
            "Heal and shop in Indigo Plateau Lobby before the league run.",
            "Fight Lorelei, Bruno, Agatha, and Lance in sequence without leaving.",
            "Use level, type, and healing resources carefully through the gauntlet.",
            "Defeat the Champion after the Elite Four.",
            "Advance the Hall of Fame sequence to finish the story.",
        ],
        required_hms=[],
        next_milestone_id=None,
    ),
)

MILESTONES_BY_ID = {milestone.id: milestone for milestone in MILESTONES}
MILESTONES_BY_COMPLETION_FLAG = {
    milestone.completion_flag: milestone
    for milestone in MILESTONES
    if milestone.completion_flag is not None
}
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


def build_progress_inventory_names(
    inventory_names: Iterable[str],
    badges: Iterable[str] = (),
) -> set[str]:
    normalized = {_normalize_name(name) for name in inventory_names if name}
    for badge in badges:
        if not badge:
            continue
        normalized_badge = _normalize_name(badge)
        normalized.add(normalized_badge)
        normalized.add(f"{normalized_badge} badge")
    return normalized


def get_current_milestone(
    active_flags: Iterable[str],
    inventory_names: Iterable[str],
    current_map_name: str | None = None,
    badges: Iterable[str] = (),
) -> Milestone:
    flag_set = {flag for flag in active_flags if flag}
    inventory_index = build_progress_inventory_names(inventory_names, badges)
    first_incomplete = _first_incomplete_index(flag_set, inventory_index)
    if first_incomplete >= len(MILESTONES):
        return MILESTONES[-1]
    active_index = _resolve_active_index(
        first_incomplete,
        current_map_name=current_map_name,
        active_flags=flag_set,
        inventory_index=inventory_index,
    )
    return MILESTONES[active_index]


def get_progress_summary(
    active_flags: Iterable[str],
    inventory_names: Iterable[str] = (),
    current_map_name: str | None = None,
    badges: Iterable[str] = (),
) -> str:
    flag_set = {flag for flag in active_flags if flag}
    inventory_index = build_progress_inventory_names(inventory_names, badges)
    first_incomplete = _first_incomplete_index(flag_set, inventory_index)
    if first_incomplete >= len(MILESTONES):
        completed = len(MILESTONES)
    elif current_map_name:
        completed = _resolve_active_index(
            first_incomplete,
            current_map_name=current_map_name,
            active_flags=flag_set,
            inventory_index=inventory_index,
        )
    else:
        completed = sum(1 for milestone in MILESTONES if _is_complete(milestone, flag_set, inventory_index))
    return f"{completed}/{len(MILESTONES)} milestones complete"


def milestone_for_completion_flag(flag: str) -> Milestone | None:
    return MILESTONES_BY_COMPLETION_FLAG.get(flag)


def _first_incomplete_index(active_flags: set[str], inventory_index: set[str]) -> int:
    highest_explicit_completion = max(
        (
            index
            for index, milestone in enumerate(MILESTONES)
            if _is_complete(milestone, active_flags, inventory_index)
        ),
        default=-1,
    )
    for index, milestone in enumerate(MILESTONES):
        if milestone.completion_flag is None and milestone.completion_item is None and index < highest_explicit_completion:
            continue
        if not _is_complete(milestone, active_flags, inventory_index):
            return index
    return len(MILESTONES)


def _resolve_active_index(
    first_incomplete: int,
    current_map_name: str | None,
    active_flags: set[str],
    inventory_index: set[str],
) -> int:
    if not current_map_name:
        return first_incomplete

    best_index = first_incomplete
    for index in range(first_incomplete, len(MILESTONES)):
        milestone = MILESTONES[index]
        if not _prerequisites_met(milestone, active_flags, inventory_index):
            break
        if _map_matches(current_map_name, milestone.target_map_name):
            best_index = index
    return best_index


def _is_complete(milestone: Milestone, active_flags: set[str], inventory_index: set[str]) -> bool:
    checks: list[bool] = []
    if milestone.completion_flag is not None:
        checks.append(milestone.completion_flag in active_flags)
    if milestone.completion_item is not None:
        checks.append(_normalize_name(milestone.completion_item) in inventory_index)
    if not checks:
        return False
    return any(checks)


def _prerequisites_met(milestone: Milestone, active_flags: set[str], inventory_index: set[str]) -> bool:
    if any(flag not in active_flags for flag in milestone.prerequisite_flags):
        return False
    return all(_normalize_name(item) in inventory_index for item in milestone.prerequisite_items)


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


def _normalize_name(value: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", value.lower()))
