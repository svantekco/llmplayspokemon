from pokemon_agent.data.walkthrough import get_current_milestone
from pokemon_agent.data.walkthrough import get_progress_summary


def test_walkthrough_uses_current_map_to_promote_route_progress() -> None:
    active_flags = ["got_starter", "oak_received_parcel", "got_pokedex"]

    milestone = get_current_milestone(active_flags, [], current_map_name="Pewter City")

    assert milestone.id == "gym1_brock"
    assert milestone.target_map_name == "Pewter City"


def test_walkthrough_progress_summary_counts_promoted_milestones() -> None:
    active_flags = [
        "got_starter",
        "oak_received_parcel",
        "got_pokedex",
        "beat_brock",
        "beat_misty",
        "got_ss_ticket",
        "got_hm01_cut",
        "beat_lt_surge",
    ]

    summary = get_progress_summary(active_flags, [], current_map_name="Rock Tunnel 1F")

    assert summary == "10/24 milestones complete"
