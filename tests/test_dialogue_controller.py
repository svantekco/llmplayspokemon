from pokemon_agent.agent.controllers.dialogue import DialogueController
from pokemon_agent.agent.controllers.protocol import TurnContext
from pokemon_agent.models.action import ActionType
from pokemon_agent.models.state import GameMode
from pokemon_agent.models.state import StructuredGameState


def _state(
    *,
    text_box_open: bool = True,
    dialogue_text: str | None = None,
    yes_no_prompt: bool = False,
    cursor: str | None = None,
) -> StructuredGameState:
    metadata = {}
    if dialogue_text is not None:
        metadata["dialogue_text"] = dialogue_text
    if yes_no_prompt:
        metadata["yes_no_prompt"] = True
    if cursor is not None:
        metadata["cursor"] = cursor
    return StructuredGameState(
        map_name="Oak's Lab",
        map_id="oaks_lab",
        mode=GameMode.TEXT,
        text_box_open=text_box_open,
        metadata=metadata,
    )


def test_dialogue_controller_advances_plain_text() -> None:
    controller = DialogueController()

    result = controller.step(
        _state(dialogue_text="Oak: Take this.", yes_no_prompt=False),
        TurnContext(turn_index=1),
    )

    assert result.action is not None
    assert result.action.action == ActionType.PRESS_A
    assert result.planner_source == "dialogue_controller"


def test_dialogue_controller_selects_yes_for_heal_prompt() -> None:
    controller = DialogueController()

    result = controller.step(
        _state(
            dialogue_text="Would you like me to heal your Pokemon?\nYES\nNO",
            yes_no_prompt=True,
            cursor="NO",
        ),
        TurnContext(turn_index=1),
    )

    assert result.action is not None
    assert result.action.action == ActionType.MOVE_UP


def test_dialogue_controller_selects_no_for_save_prompt() -> None:
    controller = DialogueController()

    result = controller.step(
        _state(
            dialogue_text="Would you like to save the game?\nYES\nNO",
            yes_no_prompt=True,
            cursor="YES",
        ),
        TurnContext(turn_index=1),
    )

    assert result.action is not None
    assert result.action.action == ActionType.MOVE_DOWN


def test_dialogue_controller_handles_text_transition_without_open_box() -> None:
    controller = DialogueController()

    result = controller.step(_state(text_box_open=False), TurnContext(turn_index=1))

    assert result.action is not None
    assert result.action.action == ActionType.PRESS_A
