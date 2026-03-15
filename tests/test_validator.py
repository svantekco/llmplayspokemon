import pytest

from pokemon_agent.agent.validator import ActionValidator
from pokemon_agent.agent.stuck_detector import StuckState
from pokemon_agent.models.action import ActionDecision, ActionType
from pokemon_agent.models.state import GameMode, StructuredGameState


def test_validator_clamps_repeat():
    validator = ActionValidator(max_repeat=4)
    result = validator.parse('{"action":"MOVE_UP","repeat":9,"reason":"test"}')
    assert result.repeat == 4


def test_validator_extracts_json_from_wrapped_response():
    validator = ActionValidator(max_repeat=4)
    result = validator.parse('Action: {"action":"PRESS_A","repeat":1,"reason":"advance"}')
    assert result.action.value == "PRESS_A"


def test_validator_parses_coordinate_action():
    validator = ActionValidator(max_repeat=4)
    result = validator.parse(
        '{"action":"MOVE_TO_COORDINATE","repeat":3,"reason":"walk","target_x":2,"target_y":7}'
    )

    assert result.action == ActionType.MOVE_TO_COORDINATE
    assert result.repeat == 1
    assert result.target_x == 2
    assert result.target_y == 7


def test_validator_rejects_coordinate_action_without_target():
    validator = ActionValidator(max_repeat=4)

    with pytest.raises(ValueError, match="target_x and target_y"):
        validator.parse('{"action":"MOVE_TO_COORDINATE","repeat":1,"reason":"walk"}')


def test_validator_prefers_recovery_inputs_when_stuck():
    validator = ActionValidator(max_repeat=4)
    state = StructuredGameState(mode=GameMode.OVERWORLD)
    stuck = StuckState(score=5, recent_failed_actions=["MOVE_UP", "MOVE_RIGHT"])
    result = validator.fallback(state, stuck, "recovery")
    assert result.action.value in {"PRESS_A", "PRESS_START", "PRESS_B", "MOVE_DOWN", "MOVE_LEFT"}


def test_validator_prefers_local_recovery_before_press_start_in_overworld():
    validator = ActionValidator(max_repeat=4)
    state = StructuredGameState(mode=GameMode.OVERWORLD)
    stuck = StuckState(score=44, recent_failed_actions=["PRESS_A", "PRESS_A", "PRESS_A", "PRESS_A", "PRESS_B"])

    result = validator.fallback(state, stuck, "recovery")

    assert result.action == ActionType.PRESS_A


def test_validator_uses_startup_bootstrap_actions():
    validator = ActionValidator(max_repeat=4)
    title_state = StructuredGameState(
        map_name="Title Screen",
        mode=GameMode.CUTSCENE,
        metadata={"engine_phase": "bootstrap", "bootstrap_phase": "title_screen"},
    )
    menu_state = StructuredGameState(
        map_name="Title Menu",
        mode=GameMode.CUTSCENE,
        menu_open=True,
        metadata={"engine_phase": "bootstrap", "bootstrap_phase": "title_menu"},
    )
    intro_state = StructuredGameState(
        map_name="Intro Cutscene",
        mode=GameMode.CUTSCENE,
        text_box_open=True,
        metadata={"engine_phase": "bootstrap", "bootstrap_phase": "intro_cutscene"},
    )

    assert validator.fallback(title_state).action.value == "PRESS_START"
    assert validator.fallback(menu_state).action.value == "PRESS_A"
    assert validator.fallback(intro_state).action.value == "PRESS_A"


def test_validator_converts_coordinate_actions_outside_overworld():
    validator = ActionValidator(max_repeat=4)
    menu_state = StructuredGameState(mode=GameMode.MENU, menu_open=True)

    result = validator.validate(
        ActionDecision(
            action=ActionType.MOVE_TO_COORDINATE,
            repeat=1,
            reason="walk",
            target_x=1,
            target_y=1,
        ),
        menu_state,
    )

    assert result.action == ActionType.PRESS_B
    assert result.repeat == 1
