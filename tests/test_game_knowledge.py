from pokemon_agent.agent.game_knowledge import load_game_knowledge


def test_game_knowledge_resolves_known_warp_with_activation_mode() -> None:
    knowledge = load_game_knowledge()

    warp = knowledge.get_warp_at("Pallet Town", 5, 5)

    assert warp is not None
    assert warp.dest_map == "Red's House 1F"
    assert warp.dest_warp_index == 1
    assert warp.activation_mode == "push"


def test_game_knowledge_keeps_edge_stairs_as_step_on() -> None:
    knowledge = load_game_knowledge()

    warp = knowledge.get_warp_at("Red's House 2F", 7, 1)

    assert warp is not None
    assert warp.dest_map == "Red's House 1F"
    assert warp.activation_mode == "step_on"


def test_game_knowledge_indexes_npcs_and_signs() -> None:
    knowledge = load_game_knowledge()

    npcs = knowledge.get_npcs_on_map("Pallet Town")
    signs = knowledge.get_signs_on_map("Pallet Town")

    assert len(npcs) == 3
    assert any(npc.sprite_symbol == "SPRITE_OAK" and npc.is_trainer is False for npc in npcs)
    assert len(signs) == 4
    assert any(sign.text_id == "TEXT_PALLETTOWN_SIGN" for sign in signs)


def test_game_knowledge_type_chart_matches_known_pairs() -> None:
    knowledge = load_game_knowledge()

    assert knowledge.type_effectiveness("FIRE", "GRASS") == 2.0
    assert knowledge.type_effectiveness("NORMAL", "GHOST") == 0.0
    assert knowledge.type_effectiveness("water", "rock") == 2.0


def test_game_knowledge_missing_lookups_fall_back_safely() -> None:
    knowledge = load_game_knowledge()

    assert knowledge.get_warp_at("Missing Map", 1, 1) is None
    assert knowledge.get_npcs_on_map("Missing Map") == []
    assert knowledge.get_signs_on_map("Missing Map") == []
    assert knowledge.type_effectiveness("UNKNOWN", "GRASS") == 1.0
