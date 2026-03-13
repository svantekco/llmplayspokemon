from pokemon_agent.agent.stuck_detector import StuckDetector
from pokemon_agent.models.action import ActionDecision, ActionType
from pokemon_agent.models.state import StructuredGameState


def test_stuck_score_increases_on_no_effect():
    detector = StuckDetector()
    state = StructuredGameState(map_name="Test", x=1, y=1)
    action = ActionDecision(action=ActionType.MOVE_UP, repeat=1, reason="test")
    stuck = detector.update(state, action, "no_effect")
    assert stuck.score == 1
    assert stuck.recovery_hint is not None
