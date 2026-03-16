from pokemon_agent.agent.controllers.protocol import Controller
from pokemon_agent.agent.controllers.protocol import TurnContext
from pokemon_agent.agent.controllers.stubs import StubBattleController
from pokemon_agent.agent.controllers.stubs import StubCutsceneController
from pokemon_agent.agent.controllers.stubs import StubMenuController
from pokemon_agent.agent.controllers.stubs import StubOverworldController
from pokemon_agent.agent.controllers.stubs import StubTextController
from pokemon_agent.agent.controllers.stubs import StubUnknownController
from pokemon_agent.agent.mode_dispatcher import ModeDispatcher
from pokemon_agent.agent.planning_types import PlanningResult
from pokemon_agent.models.action import ActionDecision
from pokemon_agent.models.action import ActionType
from pokemon_agent.models.state import GameMode
from pokemon_agent.models.state import StructuredGameState


class _FakeController:
    def __init__(self, name: str) -> None:
        self.name = name
        self.step_calls: list[tuple[GameMode, int]] = []
        self.reset_calls = 0

    def step(self, state: StructuredGameState, context: TurnContext) -> PlanningResult:
        self.step_calls.append((state.mode, context.turn_index))
        return PlanningResult(
            action=ActionDecision(action=ActionType.PRESS_A, repeat=1, reason=self.name),
            planner_source=self.name,
        )

    def reset(self) -> None:
        self.reset_calls += 1


def _state(
    *,
    mode: GameMode = GameMode.OVERWORLD,
    menu_open: bool = False,
    text_box_open: bool = False,
    battle_state: dict | None = None,
) -> StructuredGameState:
    return StructuredGameState(
        map_name="Mock Town",
        map_id="mock_town",
        x=5,
        y=5,
        mode=mode,
        menu_open=menu_open,
        text_box_open=text_box_open,
        battle_state=battle_state,
    )


def _controllers() -> dict[GameMode, _FakeController]:
    return {mode: _FakeController(mode.value.lower()) for mode in GameMode}


def test_mode_dispatcher_routes_using_effective_mode():
    controllers = _controllers()
    dispatcher = ModeDispatcher(controllers)
    context = TurnContext(turn_index=7)

    menu_result = dispatcher.dispatch(_state(mode=GameMode.OVERWORLD, menu_open=True), context)
    battle_result = dispatcher.dispatch(
        _state(mode=GameMode.CUTSCENE, battle_state={"kind": "WILD", "opponent": "PIDGEY"}),
        context,
    )

    assert menu_result.planner_source == "menu"
    assert battle_result.planner_source == "battle"
    assert len(controllers[GameMode.OVERWORLD].step_calls) == 0
    assert len(controllers[GameMode.MENU].step_calls) == 1
    assert len(controllers[GameMode.BATTLE].step_calls) == 1


def test_mode_dispatcher_resets_previous_controller_on_effective_mode_change():
    controllers = _controllers()
    dispatcher = ModeDispatcher(controllers)
    context = TurnContext(turn_index=3)

    dispatcher.dispatch(_state(mode=GameMode.OVERWORLD), context)
    dispatcher.dispatch(_state(mode=GameMode.OVERWORLD, text_box_open=True), context)
    dispatcher.dispatch(_state(mode=GameMode.TEXT, text_box_open=True), context)

    assert controllers[GameMode.OVERWORLD].reset_calls == 1
    assert controllers[GameMode.TEXT].reset_calls == 0
    assert len(controllers[GameMode.TEXT].step_calls) == 2


def test_mode_dispatcher_requires_exhaustive_controller_registration():
    partial = _controllers()
    partial.pop(GameMode.UNKNOWN)

    try:
        ModeDispatcher(partial)
    except ValueError as exc:
        assert "UNKNOWN" in str(exc)
    else:  # pragma: no cover - defensive assertion
        raise AssertionError("expected ModeDispatcher to reject missing controller registrations")


def test_stub_controllers_follow_protocol_and_are_distinct_instances():
    result = PlanningResult(
        action=ActionDecision(action=ActionType.PRESS_A, repeat=1, reason="stub result"),
        planner_source="stub",
    )
    controllers = [
        StubOverworldController(lambda _state, _context: result),
        StubMenuController(lambda _state, _context: result),
        StubTextController(lambda _state, _context: result),
        StubBattleController(lambda _state, _context: result),
        StubCutsceneController(lambda _state, _context: result),
        StubUnknownController(lambda _state, _context: result),
    ]
    context = TurnContext(turn_index=1)
    state = _state()

    assert len({id(controller) for controller in controllers}) == len(controllers)
    for controller in controllers:
        assert isinstance(controller, Controller)
        assert controller.step(state, context) is result
        assert controller.reset() is None
