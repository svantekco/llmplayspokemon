from pokemon_agent.agent.stuck_detector import StuckDetector
from pokemon_agent.models.action import ActionDecision, ActionType
from pokemon_agent.models.state import GameMode
from pokemon_agent.models.state import StructuredGameState


def test_stuck_score_increases_on_no_effect():
    detector = StuckDetector()
    state = StructuredGameState(map_name="Test", x=1, y=1, mode=GameMode.OVERWORLD)
    action = ActionDecision(action=ActionType.MOVE_UP, repeat=1, reason="test")
    stuck = detector.update(state, action, "no_effect")
    assert stuck.score == 1
    assert stuck.steps_since_progress == 1


def test_stuck_detector_flags_oscillation_as_stuck():
    detector = StuckDetector()
    action = ActionDecision(action=ActionType.MOVE_UP, repeat=1, reason="test")
    states = [
        StructuredGameState(map_name="Test", x=1, y=1, facing="UP", mode=GameMode.OVERWORLD),
        StructuredGameState(map_name="Test", x=1, y=2, facing="DOWN", mode=GameMode.OVERWORLD),
        StructuredGameState(map_name="Test", x=1, y=1, facing="UP", mode=GameMode.OVERWORLD),
        StructuredGameState(map_name="Test", x=1, y=2, facing="DOWN", mode=GameMode.OVERWORLD),
    ]

    for state in states:
        stuck = detector.update(state, action, "no_effect")

    assert stuck.oscillating is True
    assert stuck.score >= detector.threshold
