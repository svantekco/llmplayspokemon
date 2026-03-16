from __future__ import annotations

from pathlib import Path
from textwrap import wrap
from typing import TYPE_CHECKING
from typing import Any

from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont

if TYPE_CHECKING:
    from pokemon_agent.agent.engine import TurnResult
    from pokemon_agent.models.memory import NavigationGoal
    from pokemon_agent.models.planner import CandidateNextStep
    from pokemon_agent.models.state import NavigationSnapshot
    from pokemon_agent.models.state import StructuredGameState


SCREEN_WIDTH = 160
SCREEN_HEIGHT = 144
SCREEN_GRID_COLS = 10
SCREEN_GRID_ROWS = 9
RESAMPLE_NEAREST = getattr(Image, "Resampling", Image).NEAREST
PATH_LINE_COLOR = (86, 205, 255, 220)
PATH_FILL_COLOR = (86, 205, 255, 92)
PATH_OUTLINE_COLOR = (86, 205, 255, 200)
NEXT_STEP_FILL_COLOR = (255, 214, 102, 124)
NEXT_STEP_OUTLINE_COLOR = (255, 214, 102, 255)


class WorldPathOverlayPainter:
    def __init__(self, *, scale: int = 1) -> None:
        self.scale = max(1, int(scale))
        self.width = SCREEN_WIDTH * self.scale
        self.height = SCREEN_HEIGHT * self.scale
        self.cell_width = self.width // SCREEN_GRID_COLS
        self.cell_height = self.height // SCREEN_GRID_ROWS

    def render(
        self,
        state: StructuredGameState,
        suggested_path: list[tuple[int, int]],
    ) -> Image.Image:
        overlay = Image.new("RGBA", (self.width, self.height), (0, 0, 0, 0))
        if not suggested_path or state.navigation is None:
            return overlay
        draw = ImageDraw.Draw(overlay, "RGBA")
        player_position = (state.x, state.y) if state.x is not None and state.y is not None else None
        self.draw(draw, state.navigation, suggested_path, start=player_position)
        return overlay

    def draw(
        self,
        draw: ImageDraw.ImageDraw,
        navigation: NavigationSnapshot | None,
        suggested_path: list[tuple[int, int]],
        *,
        start: tuple[int, int] | None = None,
    ) -> None:
        if navigation is None or not suggested_path:
            return
        origin_x = navigation.screen_origin_x if navigation.screen_origin_x is not None else navigation.min_x
        origin_y = navigation.screen_origin_y if navigation.screen_origin_y is not None else navigation.min_y
        centers: list[tuple[int, int]] = []
        if start is not None:
            center = self.world_cell_center(start[0], start[1], origin_x=origin_x, origin_y=origin_y)
            if center is not None:
                centers.append(center)
        for world_x, world_y in suggested_path:
            center = self.world_cell_center(world_x, world_y, origin_x=origin_x, origin_y=origin_y)
            if center is not None:
                centers.append(center)
        if len(centers) >= 2:
            draw.line(centers, fill=PATH_LINE_COLOR, width=max(2, self.scale))
        inset = max(2, min(self.cell_width, self.cell_height) // 4)
        for index, (world_x, world_y) in enumerate(suggested_path):
            rect = self.world_cell_rect(world_x, world_y, origin_x=origin_x, origin_y=origin_y)
            if rect is None:
                continue
            x0, y0, x1, y1 = rect
            fill = PATH_FILL_COLOR
            outline = PATH_OUTLINE_COLOR
            width = 2
            if index == 0:
                fill = NEXT_STEP_FILL_COLOR
                outline = NEXT_STEP_OUTLINE_COLOR
                width = 3
            draw.rectangle(
                (x0 + inset, y0 + inset, x1 - inset, y1 - inset),
                fill=fill,
                outline=outline,
                width=width,
            )

    def world_cell_rect(
        self,
        world_x: int,
        world_y: int,
        *,
        origin_x: int,
        origin_y: int,
    ) -> tuple[int, int, int, int] | None:
        screen_x = world_x - origin_x
        screen_y = world_y - origin_y
        if not (0 <= screen_x < SCREEN_GRID_COLS and 0 <= screen_y < SCREEN_GRID_ROWS):
            return None
        x0 = screen_x * self.cell_width
        y0 = screen_y * self.cell_height
        x1 = x0 + self.cell_width - 1
        y1 = y0 + self.cell_height - 1
        return x0, y0, x1, y1

    def world_cell_center(
        self,
        world_x: int,
        world_y: int,
        *,
        origin_x: int,
        origin_y: int,
    ) -> tuple[int, int] | None:
        rect = self.world_cell_rect(world_x, world_y, origin_x=origin_x, origin_y=origin_y)
        if rect is None:
            return None
        x0, y0, x1, y1 = rect
        return (x0 + x1) // 2, (y0 + y1) // 2


class DebugOverlayWriter:
    def __init__(self, output_dir: str | Path, *, scale: int = 3) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.scale = max(1, int(scale))
        self._font = ImageFont.load_default()
        self._path_painter = WorldPathOverlayPainter(scale=self.scale)

    def write_turn(self, turn: TurnResult, summary: dict[str, Any] | None = None) -> Path | None:
        if turn.screen_image is None:
            return None
        image = self._render_turn_overlay(turn, summary or {})
        target = self.output_dir / f"turn_{turn.turn_index:04d}.png"
        latest = self.output_dir / "latest.png"
        image.save(target)
        image.save(latest)
        return target

    def _render_turn_overlay(self, turn: TurnResult, summary: dict[str, Any]) -> Image.Image:
        frame = turn.screen_image.convert("RGBA").resize(
            (SCREEN_WIDTH * self.scale, SCREEN_HEIGHT * self.scale),
            RESAMPLE_NEAREST,
        )
        annotated = Image.alpha_composite(frame, self._draw_frame_overlay(turn))
        panel = self._build_side_panel(turn, summary)

        canvas = Image.new(
            "RGBA",
            (annotated.width + panel.width, max(annotated.height, panel.height)),
            (17, 20, 24, 255),
        )
        canvas.paste(annotated, (0, 0))
        canvas.paste(panel, (annotated.width, 0))
        return canvas.convert("RGB")

    def _draw_frame_overlay(self, turn: TurnResult) -> Image.Image:
        overlay = Image.new("RGBA", (SCREEN_WIDTH * self.scale, SCREEN_HEIGHT * self.scale), (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay, "RGBA")
        cell_width = overlay.width // SCREEN_GRID_COLS
        cell_height = overlay.height // SCREEN_GRID_ROWS

        self._draw_grid(draw, overlay.width, overlay.height, cell_width, cell_height)

        state = turn.before
        navigation = state.navigation
        chosen = self._chosen_candidate(turn)
        suggested_path = list(turn.suggested_path)
        if navigation is not None:
            origin_x = navigation.screen_origin_x if navigation.screen_origin_x is not None else navigation.min_x
            origin_y = navigation.screen_origin_y if navigation.screen_origin_y is not None else navigation.min_y

            for coordinate in navigation.blocked:
                self._fill_world_cell(
                    draw,
                    coordinate.x,
                    coordinate.y,
                    origin_x=origin_x,
                    origin_y=origin_y,
                    cell_width=cell_width,
                    cell_height=cell_height,
                    fill=(214, 69, 80, 88),
                    outline=(214, 69, 80, 150),
                )
            for coordinate in navigation.walkable:
                self._fill_world_cell(
                    draw,
                    coordinate.x,
                    coordinate.y,
                    origin_x=origin_x,
                    origin_y=origin_y,
                    cell_width=cell_width,
                    cell_height=cell_height,
                    fill=(68, 170, 96, 56),
                    outline=(68, 170, 96, 120),
                )
            if suggested_path:
                overlay = Image.alpha_composite(overlay, self._path_painter.render(state, suggested_path))
                draw = ImageDraw.Draw(overlay, "RGBA")

            if chosen is not None and chosen.target is not None and chosen.target.x is not None and chosen.target.y is not None:
                self._fill_world_cell(
                    draw,
                    chosen.target.x,
                    chosen.target.y,
                    origin_x=origin_x,
                    origin_y=origin_y,
                    cell_width=cell_width,
                    cell_height=cell_height,
                    fill=(71, 123, 255, 40),
                    outline=(71, 123, 255, 255),
                    width=3,
                )

        if state.x is not None and state.y is not None and navigation is not None:
            origin_x = navigation.screen_origin_x if navigation.screen_origin_x is not None else navigation.min_x
            origin_y = navigation.screen_origin_y if navigation.screen_origin_y is not None else navigation.min_y
            self._fill_world_cell(
                draw,
                state.x,
                state.y,
                origin_x=origin_x,
                origin_y=origin_y,
                cell_width=cell_width,
                cell_height=cell_height,
                fill=(248, 216, 80, 120),
                outline=(248, 216, 80, 255),
                width=3,
            )
        return overlay

    def _build_side_panel(self, turn: TurnResult, summary: dict[str, Any]) -> Image.Image:
        panel_width = 360
        panel_height = SCREEN_HEIGHT * self.scale
        panel = Image.new("RGBA", (panel_width, panel_height), (24, 28, 34, 255))
        draw = ImageDraw.Draw(panel)

        y = 12
        y = self._draw_heading(draw, "Debug Overlay", 12, y)
        state = turn.before
        chosen = self._chosen_candidate(turn)
        chosen_label = chosen.id if chosen is not None else turn.candidate_id or "n/a"
        lines = [
            f"Turn: {turn.turn_index}",
            f"Map: {state.map_name}",
            f"Pos: ({state.x}, {state.y}) facing {state.facing or '?'}",
            f"Mode: {state.mode.value}",
            f"Action: {turn.action.action.value} x{turn.action.repeat}",
            f"Planner: {turn.planner_source}",
            f"Chosen: {chosen_label}",
            f"Progress: {turn.progress.classification}",
            f"Stuck: {turn.stuck_state.score}",
        ]
        if turn.suggested_path:
            next_x, next_y = turn.suggested_path[0]
            target_x, target_y = turn.suggested_path[-1]
            lines.append(f"Next: ({next_x}, {next_y})")
            lines.append(f"Path: {len(turn.suggested_path)} steps -> ({target_x}, {target_y})")
        if turn.navigation_goal is not None:
            goal = turn.navigation_goal
            lines.extend(
                [
                    f"Goal: {goal.objective_kind}",
                    f"Target Map: {goal.target_map_name}",
                    f"Next Hop: {goal.next_hop_kind or '-'} -> {goal.next_map_name or '-'}",
                ]
            )
            if goal.final_target_map_name and goal.final_target_map_name != goal.target_map_name:
                lines.append(f"Final Target: {goal.final_target_map_name}")
        if turn.objective is not None:
            lines.extend(
                [
                    f"Objective: {turn.objective.goal}",
                    f"Strategy: {turn.objective.strategy}",
                ]
            )
        for line in lines:
            y = self._draw_wrapped_text(draw, line, 12, y, panel_width - 24)

        y += 6
        y = self._draw_heading(draw, "Candidates", 12, y)
        for index, candidate in enumerate(turn.candidates[:4], start=1):
            prefix = "*" if candidate.id == turn.candidate_id else f"{index}."
            target = ""
            if candidate.target is not None and candidate.target.x is not None and candidate.target.y is not None:
                target = f" @ ({candidate.target.x},{candidate.target.y})"
            line = f"{prefix} {candidate.id} [{candidate.type}] p{candidate.priority}{target}"
            y = self._draw_wrapped_text(draw, line, 12, y, panel_width - 24)
            y = self._draw_wrapped_text(draw, candidate.why, 18, y, panel_width - 30, fill=(198, 204, 214, 255))

        navigation = state.navigation
        if navigation is not None:
            y += 8
            y = self._draw_heading(draw, "Mini Map", 12, y)
            y = self._draw_minimap(draw, 12, y, navigation, state.x, state.y, chosen, turn.suggested_path)

        if summary:
            y += 8
            y = self._draw_heading(draw, "Summary", 12, y)
            for key in ("short_term_goal", "mid_term_goal", "long_term_goal"):
                value = summary.get(key)
                if isinstance(value, str) and value:
                    y = self._draw_wrapped_text(draw, f"{key.replace('_', ' ').title()}: {value}", 12, y, panel_width - 24)
        return panel

    def _draw_minimap(
        self,
        draw: ImageDraw.ImageDraw,
        left: int,
        top: int,
        navigation,
        player_x: int | None,
        player_y: int | None,
        chosen: CandidateNextStep | None,
        suggested_path: list[tuple[int, int]],
    ) -> int:
        bounds_width = navigation.max_x - navigation.min_x + 1
        bounds_height = navigation.max_y - navigation.min_y + 1
        cell = max(10, min(24, 180 // max(1, max(bounds_width, bounds_height))))
        walkable = {(coordinate.x, coordinate.y) for coordinate in navigation.walkable}
        blocked = {(coordinate.x, coordinate.y) for coordinate in navigation.blocked}
        chosen_target = None
        if chosen is not None and chosen.target is not None and chosen.target.x is not None and chosen.target.y is not None:
            chosen_target = (chosen.target.x, chosen.target.y)

        for world_y in range(navigation.min_y, navigation.max_y + 1):
            for world_x in range(navigation.min_x, navigation.max_x + 1):
                x0 = left + (world_x - navigation.min_x) * cell
                y0 = top + (world_y - navigation.min_y) * cell
                x1 = x0 + cell - 1
                y1 = y0 + cell - 1
                fill = (44, 48, 56, 255)
                if (world_x, world_y) in blocked:
                    fill = (164, 64, 72, 255)
                elif (world_x, world_y) in walkable:
                    fill = (66, 130, 84, 255)
                draw.rectangle((x0, y0, x1, y1), fill=fill, outline=(90, 96, 104, 255))
        if suggested_path:
            self._draw_minimap_path(draw, left, top, navigation.min_x, navigation.min_y, cell, suggested_path, player_x, player_y)
        if chosen_target is not None:
            x0 = left + (chosen_target[0] - navigation.min_x) * cell
            y0 = top + (chosen_target[1] - navigation.min_y) * cell
            x1 = x0 + cell - 1
            y1 = y0 + cell - 1
            draw.rectangle((x0 + 1, y0 + 1, x1 - 1, y1 - 1), outline=(71, 123, 255, 255), width=2)
        if player_x is not None and player_y is not None:
            x0 = left + (player_x - navigation.min_x) * cell
            y0 = top + (player_y - navigation.min_y) * cell
            x1 = x0 + cell - 1
            y1 = y0 + cell - 1
            draw.rectangle((x0 + 2, y0 + 2, x1 - 2, y1 - 2), outline=(248, 216, 80, 255), width=2)
        return top + (bounds_height * cell) + 6

    def _draw_grid(self, draw: ImageDraw.ImageDraw, width: int, height: int, cell_width: int, cell_height: int) -> None:
        for index in range(SCREEN_GRID_COLS + 1):
            x = index * cell_width
            draw.line((x, 0, x, height), fill=(255, 255, 255, 50), width=1)
        for index in range(SCREEN_GRID_ROWS + 1):
            y = index * cell_height
            draw.line((0, y, width, y), fill=(255, 255, 255, 50), width=1)

    def _fill_world_cell(
        self,
        draw: ImageDraw.ImageDraw,
        world_x: int,
        world_y: int,
        *,
        origin_x: int,
        origin_y: int,
        cell_width: int,
        cell_height: int,
        fill: tuple[int, int, int, int],
        outline: tuple[int, int, int, int],
        width: int = 1,
    ) -> None:
        rect = self._world_cell_rect(
            world_x,
            world_y,
            origin_x=origin_x,
            origin_y=origin_y,
            cell_width=cell_width,
            cell_height=cell_height,
        )
        if rect is None:
            return
        draw.rectangle(rect, fill=fill, outline=outline, width=width)

    def _world_cell_rect(
        self,
        world_x: int,
        world_y: int,
        *,
        origin_x: int,
        origin_y: int,
        cell_width: int,
        cell_height: int,
    ) -> tuple[int, int, int, int] | None:
        screen_x = world_x - origin_x
        screen_y = world_y - origin_y
        if not (0 <= screen_x < SCREEN_GRID_COLS and 0 <= screen_y < SCREEN_GRID_ROWS):
            return None
        x0 = screen_x * cell_width
        y0 = screen_y * cell_height
        x1 = x0 + cell_width - 1
        y1 = y0 + cell_height - 1
        return x0, y0, x1, y1

    def _world_cell_center(
        self,
        world_x: int,
        world_y: int,
        *,
        origin_x: int,
        origin_y: int,
        cell_width: int,
        cell_height: int,
    ) -> tuple[int, int] | None:
        rect = self._world_cell_rect(
            world_x,
            world_y,
            origin_x=origin_x,
            origin_y=origin_y,
            cell_width=cell_width,
            cell_height=cell_height,
        )
        if rect is None:
            return None
        x0, y0, x1, y1 = rect
        return (x0 + x1) // 2, (y0 + y1) // 2

    def _draw_minimap_path(
        self,
        draw: ImageDraw.ImageDraw,
        left: int,
        top: int,
        min_x: int,
        min_y: int,
        cell: int,
        suggested_path: list[tuple[int, int]],
        player_x: int | None,
        player_y: int | None,
    ) -> None:
        centers: list[tuple[int, int]] = []
        if player_x is not None and player_y is not None:
            centers.append(
                (
                    left + (player_x - min_x) * cell + (cell // 2),
                    top + (player_y - min_y) * cell + (cell // 2),
                )
            )
        for world_x, world_y in suggested_path:
            centers.append(
                (
                    left + (world_x - min_x) * cell + (cell // 2),
                    top + (world_y - min_y) * cell + (cell // 2),
                )
            )
        if len(centers) >= 2:
            draw.line(centers, fill=(86, 205, 255, 255), width=max(2, cell // 4))
        radius = max(2, cell // 5)
        for index, (center_x, center_y) in enumerate(centers[1:]):
            fill = (86, 205, 255, 255)
            if index == 0:
                fill = (255, 214, 102, 255)
            draw.ellipse(
                (center_x - radius, center_y - radius, center_x + radius, center_y + radius),
                fill=fill,
            )

    def _chosen_candidate(self, turn: TurnResult):
        for candidate in turn.candidates:
            if candidate.id == turn.candidate_id:
                return candidate
        return None

    def _draw_heading(self, draw: ImageDraw.ImageDraw, text: str, x: int, y: int) -> int:
        draw.text((x, y), text, fill=(255, 255, 255, 255), font=self._font)
        return y + 16

    def _draw_wrapped_text(
        self,
        draw: ImageDraw.ImageDraw,
        text: str,
        x: int,
        y: int,
        width: int,
        *,
        fill: tuple[int, int, int, int] = (232, 236, 240, 255),
    ) -> int:
        max_chars = max(12, width // 6)
        for line in wrap(text, width=max_chars):
            draw.text((x, y), line, fill=fill, font=self._font)
            y += 12
        return y + 2
