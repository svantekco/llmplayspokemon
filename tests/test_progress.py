from pokemon_agent.agent.progress import ProgressDetector
from pokemon_agent.models.state import GameMode, StructuredGameState


def test_position_change_is_movement_success():
    detector = ProgressDetector()
    before = StructuredGameState(x=1, y=1)
    after = StructuredGameState(x=2, y=1)
    result = detector.compare(before, after)
    assert result.classification == "movement_success"


def test_map_change_is_major_progress():
    detector = ProgressDetector()
    before = StructuredGameState(map_name="Pallet Town", map_id=0, x=1, y=1)
    after = StructuredGameState(map_name="Route 1", map_id=12, x=1, y=1)
    result = detector.compare(before, after)
    assert result.classification == "major_progress"
    assert "map_id" in result.changed_fields


def test_bootstrap_exit_is_major_progress():
    detector = ProgressDetector()
    before = StructuredGameState(
        map_name="Title Screen",
        mode=GameMode.CUTSCENE,
        metadata={"engine_phase": "bootstrap", "bootstrap_phase": "title_screen"},
    )
    after = StructuredGameState(map_name="Red's House 2F", map_id=0x26, x=7, y=3, mode=GameMode.OVERWORLD)
    result = detector.compare(before, after)
    assert result.classification == "major_progress"
    assert "engine_phase" in result.changed_fields


def test_text_screen_change_counts_as_interaction_success():
    detector = ProgressDetector()
    before = StructuredGameState(
        map_name="Oak's Lab",
        mode=GameMode.TEXT,
        text_box_open=True,
        metadata={"tile_hash": "before", "dialogue": "Hello"},
    )
    after = StructuredGameState(
        map_name="Oak's Lab",
        mode=GameMode.TEXT,
        text_box_open=True,
        metadata={"tile_hash": "after", "dialogue": "Take this"},
    )

    result = detector.compare(before, after)

    assert result.classification == "interaction_success"
    assert "screen_state" in result.changed_fields
