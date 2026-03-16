from pokemon_agent.agent.controllers.cutscene import CutsceneController
from pokemon_agent.agent.controllers.protocol import TurnContext
from pokemon_agent.models.action import ActionType
from pokemon_agent.models.state import GameMode
from pokemon_agent.models.state import StructuredGameState


def test_cutscene_controller_advances_with_a() -> None:
    controller = CutsceneController()
    state = StructuredGameState(map_name="Intro", map_id="intro", mode=GameMode.CUTSCENE)

    result = controller.step(state, TurnContext(turn_index=1))

    assert result.action is not None
    assert result.action.action == ActionType.PRESS_A
    assert result.planner_source == "cutscene_controller"
