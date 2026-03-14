from pathlib import Path

from PIL import Image

from pokemon_agent.agent.engine import TurnResult
from pokemon_agent.agent.progress import ProgressResult
from pokemon_agent.agent.stuck_detector import StuckState
from pokemon_agent.models.action import ActionDecision
from pokemon_agent.models.action import ActionType
from pokemon_agent.models.planner import CandidateNextStep
from pokemon_agent.models.planner import ObjectiveTarget
from pokemon_agent.models.state import GameMode
from pokemon_agent.models.state import NavigationSnapshot
from pokemon_agent.models.state import StructuredGameState
from pokemon_agent.models.state import WorldCoordinate
from pokemon_agent.ui.debug_overlay import DebugOverlayWriter


def test_debug_overlay_writer_saves_annotated_turn_image(tmp_path: Path) -> None:
    before = StructuredGameState(
        map_name="Red's House 2F",
        map_id=0x26,
        x=5,
        y=4,
        facing="DOWN",
        mode=GameMode.OVERWORLD,
        navigation=NavigationSnapshot(
            min_x=0,
            min_y=0,
            max_x=7,
            max_y=7,
            player=WorldCoordinate(x=5, y=4),
            walkable=[WorldCoordinate(x=5, y=4), WorldCoordinate(x=6, y=1)],
            blocked=[WorldCoordinate(x=4, y=4)],
            collision_hash="mock",
            coverage="full_map",
            map_width=8,
            map_height=8,
            visible_world_edges=["north", "east", "south", "west"],
            screen_origin_x=-2,
            screen_origin_y=0,
        ),
    )
    turn = TurnResult(
        turn_index=7,
        before=before,
        action=ActionDecision(action=ActionType.MOVE_UP, repeat=1, reason="Head toward the stairs"),
        after=before,
        progress=ProgressResult("movement_success", ["position"], [], ["Moved closer to the stairs"]),
        stuck_state=StuckState(score=3),
        planner_source="auto_candidate",
        candidate_id="static_warp_reds_house_1f_7_1",
        candidates=[
            CandidateNextStep(
                id="static_warp_reds_house_1f_7_1",
                type="ENTER_CONNECTOR",
                target=ObjectiveTarget(
                    kind="connector",
                    map_id=0x26,
                    map_name="Red's House 2F",
                    x=6,
                    y=1,
                    detail="canonical warp to Red's House 1F",
                ),
                why="Use the canonical warp to Red's House 1F.",
                priority=88,
                expected_success_signal="Map changes",
                objective_id="local_explore_for_matching_connector",
            )
        ],
        screen_image=Image.new("RGB", (160, 144), color=(32, 48, 96)),
    )

    writer = DebugOverlayWriter(tmp_path)
    path = writer.write_turn(
        turn,
        {
            "short_term_goal": "Descend the visible staircase.",
            "mid_term_goal": "Leave the house and head to Oak's Lab.",
        },
    )

    assert path == tmp_path / "turn_0007.png"
    assert path.exists()
    assert (tmp_path / "latest.png").exists()
    image = Image.open(path)
    assert image.size[0] > 160
    assert image.size[1] >= 144
