from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from rich import box
from rich.console import Console, Group
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from rich.text import Text

from pokemon_agent.agent.ascii_map import build_ascii_map
from pokemon_agent.agent.engine import TurnResult
from pokemon_agent.models.state import StructuredGameState


class TerminalDashboard:
    def __init__(
        self,
        planner: str,
        continuous: bool,
        target_turns: int | None,
        checkpoint_dir: str | None = None,
        console: Console | None = None,
        history_limit: int = 10,
    ) -> None:
        self.console = console or Console()
        self.planner = planner
        self.continuous = continuous
        self.target_turns = target_turns
        self.checkpoint_dir = checkpoint_dir
        self.history_limit = history_limit
        self.current_state: StructuredGameState | None = None
        self.latest_turn: TurnResult | None = None
        self.turn_history: list[TurnResult] = []
        self.summary: dict[str, Any] = {}
        self.status = "Starting"
        self.status_note: str | None = None
        self.resume_path: str | None = None
        self.resume_turns = 0
        self._live: Live | None = None
        self._live_enabled = bool(self.console.is_terminal and not getattr(self.console, "is_dumb_terminal", False))

    def record_resume(self, resume_path: str, completed_turns: int) -> None:
        self.resume_path = resume_path
        self.resume_turns = completed_turns

    def start(self, initial_state: StructuredGameState, summary: dict[str, Any] | None = None) -> None:
        self.current_state = initial_state
        self.summary = summary or {}
        self.status = "Running"
        if self._live_enabled and self._live is None:
            self._live = Live(
                self.render(),
                console=self.console,
                refresh_per_second=4,
                transient=False,
                vertical_overflow="crop",
            )
            self._live.start()
            return

    def update_turn(self, result: TurnResult, summary: dict[str, Any]) -> None:
        self.current_state = result.after
        self.latest_turn = result
        self.turn_history.append(result)
        self.turn_history = self.turn_history[-self.history_limit :]
        self.summary = summary
        self.status = "Running"
        if self._live is not None:
            self._refresh()

    def finish(self, interrupted: bool = False) -> None:
        self.status = "Interrupted" if interrupted else "Stopped"
        if self._live is not None:
            self._refresh()
            self._live.stop()
            self._live = None
            return
        self.console.print(self.render())

    def request_stop(self) -> None:
        self.status = "Stopping"
        self.status_note = "Press Ctrl+C again to save immediately and exit."
        if self._live is not None:
            self._refresh()

    def render(self) -> Layout:
        layout = Layout(name="root")
        layout.split_column(
            Layout(name="top", size=16),
            Layout(name="turns", ratio=1, minimum_size=6),
            Layout(name="llm", ratio=2, minimum_size=12),
        )
        layout["top"].split_row(
            Layout(name="state"),
            Layout(name="seen_area", size=34),
            Layout(name="status", size=44),
        )
        layout["state"].update(self._render_current_state())
        layout["seen_area"].update(self._render_current_seen_area())
        layout["status"].update(self._render_status())
        layout["turns"].update(self._render_turns())
        layout["llm"].update(self._render_llm_calls())
        return layout

    def _refresh(self) -> None:
        renderable = self.render()
        if self._live is not None:
            self._live.update(renderable, refresh=True)
            return
        self.console.print(renderable)

    def _render_current_state(self) -> Panel:
        if self.current_state is None:
            return Panel("Waiting for emulator state...", title="Current State", border_style="cyan")

        state = self.current_state
        table = Table.grid(expand=True, padding=(0, 1))
        table.add_column(style="cyan", no_wrap=True)
        table.add_column(ratio=1)
        table.add_row("Map", state.map_name)
        table.add_row("Position", self._format_position(state))
        table.add_row("Facing", state.facing or "unknown")
        table.add_row(
            "Mode",
            f"{state.mode.value} | menu={self._yes_no(state.menu_open)} | text={self._yes_no(state.text_box_open)}",
        )
        battle = state.battle_state.get("kind", "active") if state.battle_state else "none"
        table.add_row("Battle", str(battle))
        table.add_row("Party", self._format_party(state))
        table.add_row("Inventory", self._format_inventory(state))
        table.add_row("Step", str(state.step))
        return Panel(table, title="Current State", border_style="cyan")

    def _render_current_seen_area(self) -> Panel:
        if self.current_state is None:
            return Panel("Waiting for overworld map...", title="Current Seen Area", border_style="bright_cyan")

        ascii_map = build_ascii_map(self.current_state)
        body = self._render_ascii_block(ascii_map)
        subtitle_parts: list[str] = []
        discovered_maps = self.summary.get("discovered_maps")
        confirmed = self.summary.get("confirmed_connectors")
        suspected = self.summary.get("suspected_connectors")
        if discovered_maps is not None:
            subtitle_parts.append(f"maps {discovered_maps}")
        if confirmed is not None:
            subtitle_parts.append(f"confirmed {confirmed}")
        if suspected is not None:
            subtitle_parts.append(f"suspected {suspected}")
        subtitle = " | ".join(subtitle_parts) if subtitle_parts else "world minimap hook ready"
        return Panel(body, title="Current Seen Area", subtitle=subtitle, border_style="bright_cyan")

    def _render_status(self) -> Panel:
        table = Table.grid(expand=True, padding=(0, 1))
        table.add_column(style="magenta", no_wrap=True)
        table.add_column(ratio=1)
        turns_completed = self.summary.get("turns", 0)
        if self.continuous:
            turn_progress = f"{turns_completed} / live"
        else:
            target = self.target_turns if self.target_turns is not None else "?"
            turn_progress = f"{turns_completed} / {target}"
        table.add_row("Status", self.status)
        if self.status_note:
            table.add_row("Note", self.status_note)
        table.add_row("Planner", self.planner)
        table.add_row("Turns", turn_progress)
        table.add_row("Fallback", str(self.summary.get("fallback_turns", 0)))
        table.add_row("Route Cache", str(self.summary.get("route_cache_turns", 0)))
        table.add_row("Auto Select", str(self.summary.get("auto_selected_turns", 0)))
        table.add_row("Exec Plan", str(self.summary.get("execution_plan_turns", 0)))
        prompt_chars = self.summary.get("prompt_chars", 0)
        prompt_tokens = self.summary.get("approx_prompt_tokens", 0)
        table.add_row("Prompt", f"{prompt_chars} chars | ~{prompt_tokens} tokens")
        llm_prompt = self.summary.get("llm_prompt_tokens", 0)
        llm_completion = self.summary.get("llm_completion_tokens", 0)
        llm_total = self.summary.get("llm_total_tokens", 0)
        table.add_row("LLM Tokens", f"in {llm_prompt} | out {llm_completion} | total {llm_total}")
        table.add_row("LLM Calls", str(self.summary.get("llm_calls", 0)))
        table.add_row("Turns/Call", str(self.summary.get("turns_per_call", "n/a")))
        table.add_row("Obj Switch", str(self.summary.get("objective_switch_rate", 0.0)))
        if self.resume_path:
            table.add_row("Resumed", f"{self.resume_turns} turns from {self._tail_path(self.resume_path)}")
        if self.checkpoint_dir:
            table.add_row("Session", self._tail_path(self.checkpoint_dir))
        if self.latest_turn is not None:
            table.add_row(
                "Last Turn",
                f"#{self.latest_turn.turn_index} {self.latest_turn.action.action.value} x{self.latest_turn.action.repeat}",
            )
            table.add_row("Source", self.latest_turn.planner_source)
            table.add_row(
                "Outcome",
                f"{self.latest_turn.progress.classification} | stuck={self.latest_turn.stuck_state.score}",
            )
            if self.latest_turn.action.reason:
                table.add_row("Reason", self._truncate_inline(self.latest_turn.action.reason, 90))
            if self.latest_turn.stuck_state.recovery_hint:
                table.add_row("Hint", self._truncate_inline(self.latest_turn.stuck_state.recovery_hint, 90))
        return Panel(table, title="Run Status", border_style="magenta")

    def _render_turns(self) -> Panel:
        if not self.turn_history:
            message = Text("Turns will appear here as the agent makes decisions.", style="dim")
            return Panel(message, title="Turn History", border_style="green")

        table = Table(expand=True, box=box.SIMPLE_HEAVY, show_edge=False)
        table.add_column("#", style="bold", justify="right", no_wrap=True)
        table.add_column("Action", no_wrap=True)
        table.add_column("Result", no_wrap=True)
        table.add_column("State", ratio=1)
        table.add_column("LLM", no_wrap=True)
        table.add_column("Stuck", justify="right", no_wrap=True)
        table.add_column("Events", ratio=1)
        for turn in self.turn_history:
            row_style = "bold cyan" if self.latest_turn and turn.turn_index == self.latest_turn.turn_index else ""
            table.add_row(
                str(turn.turn_index),
                f"{turn.action.action.value} x{turn.action.repeat}",
                turn.progress.classification,
                f"{turn.after.map_name} {self._format_position(turn.after)} {turn.after.mode.value}",
                self._format_llm_status(turn),
                str(turn.stuck_state.score),
                self._format_events(turn),
                style=row_style,
            )
        subtitle = f"showing last {len(self.turn_history)} turn(s)"
        return Panel(table, title="Turn History", subtitle=subtitle, border_style="green")

    def _render_llm_calls(self) -> Panel:
        if self.latest_turn is None:
            message = Text("The latest LLM prompt and response will appear here.", style="dim")
            return Panel(message, title="LLM Calls", border_style="yellow")

        turn = self.latest_turn
        planning_map = build_ascii_map(turn.before)
        prompt_map = self._extract_prompt_visual_map(turn.prompt_messages)
        summary = Table.grid(expand=True, padding=(0, 1))
        summary.add_column(style="yellow", no_wrap=True)
        summary.add_column(ratio=1)
        summary.add_column(style="yellow", no_wrap=True)
        summary.add_column(ratio=1)
        summary.add_row("Turn", str(turn.turn_index), "Attempted", self._yes_no(turn.llm_attempted))
        summary.add_row("Source", turn.planner_source, "Model", turn.llm_model or "n/a")
        summary.add_row("Fallback", self._yes_no(turn.used_fallback), "Action", turn.action.action.value)
        if turn.prompt_metrics is not None:
            summary.add_row(
                "Prompt",
                f"{turn.prompt_metrics.chars} chars | ~{turn.prompt_metrics.approx_tokens} tokens",
                "Compact",
                self._yes_no(turn.prompt_metrics.compact),
            )
        if turn.llm_usage is not None:
            usage = (
                f"in {turn.llm_usage.prompt_tokens or 0} | "
                f"out {turn.llm_usage.completion_tokens or 0} | "
                f"total {turn.llm_usage.total_tokens or 0}"
            )
            summary.add_row("Usage", usage, "Nav", self._format_navigation(turn))
        else:
            summary.add_row("Usage", "n/a", "Nav", self._format_navigation(turn))
        summary.add_row("State Map", self._presence_label(planning_map), "Prompt Map", self._presence_label(prompt_map))
        summary.add_row("Map Match", self._map_match_label(planning_map, prompt_map), "Prompt Sent", self._yes_no(turn.llm_attempted))

        map_panel = self._render_prompt_map_panel(planning_map, prompt_map)

        request_table = Table(expand=True, box=box.SIMPLE, show_header=True, header_style="bold yellow")
        request_table.add_column("Role", width=10, no_wrap=True)
        request_table.add_column("Content", ratio=1)
        for message in turn.prompt_messages:
            request_table.add_row(
                str(message.get("role", "unknown")),
                self._truncate_block(self._format_json_like(message.get("content", ""))),
            )

        if turn.llm_attempted and turn.used_fallback and turn.raw_model_response:
            response_title = "LLM Error"
            response_style = "red"
            response_body = turn.raw_model_response
        elif turn.raw_model_response:
            response_title = "LLM Response"
            response_style = "green"
            response_body = self._format_json_like(turn.raw_model_response)
        else:
            response_title = "Planner Note"
            response_style = "blue"
            response_body = self._planner_note(turn)

        response_panel = Panel(
            self._truncate_block(response_body, max_lines=14, max_chars=1400),
            title=response_title,
            border_style=response_style,
        )
        body = Group(
            summary,
            Rule(style="dim"),
            map_panel,
            Rule(style="dim"),
            Text("Request", style="bold yellow"),
            request_table,
            Rule(style="dim"),
            response_panel,
        )
        return Panel(body, title="LLM Calls", border_style="yellow")

    @staticmethod
    def _format_position(state: StructuredGameState) -> str:
        if state.x is None or state.y is None:
            return "unknown"
        return f"({state.x}, {state.y})"

    @staticmethod
    def _yes_no(value: bool) -> str:
        return "yes" if value else "no"

    @staticmethod
    def _tail_path(path: str, parts: int = 3) -> str:
        path_obj = Path(path)
        tail = path_obj.parts[-parts:]
        return str(Path(*tail))

    @staticmethod
    def _truncate_inline(value: str, limit: int) -> str:
        if len(value) <= limit:
            return value
        return f"{value[: limit - 3]}..."

    @classmethod
    def _truncate_block(cls, value: str, max_lines: int = 12, max_chars: int = 1100) -> str:
        shortened = value if len(value) <= max_chars else f"{value[: max_chars - 3]}..."
        lines = shortened.splitlines()
        if len(lines) <= max_lines:
            return shortened
        visible = lines[:max_lines]
        visible.append("...")
        return "\n".join(visible)

    @staticmethod
    def _format_json(value: Any) -> str:
        try:
            return json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True)
        except TypeError:
            return str(value)

    @classmethod
    def _format_json_like(cls, value: Any) -> str:
        if isinstance(value, (dict, list)):
            return cls._format_json(value)
        text = str(value)
        try:
            parsed = json.loads(text)
        except (TypeError, ValueError):
            return text
        return cls._format_json(parsed)

    @classmethod
    def _format_party(cls, state: StructuredGameState) -> str:
        if not state.party:
            return "empty"
        members = []
        for member in state.party[:3]:
            hp = ""
            if member.hp is not None and member.max_hp is not None:
                hp = f" {member.hp}/{member.max_hp} HP"
            status = f" [{member.status}]" if member.status else ""
            members.append(f"{member.name}{hp}{status}")
        if len(state.party) > 3:
            members.append(f"+{len(state.party) - 3} more")
        return ", ".join(members)

    @classmethod
    def _format_inventory(cls, state: StructuredGameState) -> str:
        if not state.inventory:
            return "empty"
        items = [f"{item.name} x{item.count}" for item in state.inventory[:4]]
        if len(state.inventory) > 4:
            items.append(f"+{len(state.inventory) - 4} more")
        return ", ".join(items)

    @classmethod
    def _format_events(cls, turn: TurnResult) -> str:
        if not turn.events:
            return "none"
        events = [event.summary for event in turn.events[:2]]
        if len(turn.events) > 2:
            events.append(f"+{len(turn.events) - 2} more")
        return cls._truncate_inline(" | ".join(events), 60)

    @staticmethod
    def _format_llm_status(turn: TurnResult) -> str:
        if turn.planner_source == "route_cache":
            return "route"
        if turn.planner_source == "execution_plan":
            return "plan"
        if turn.planner_source == "bootstrap":
            return "boot"
        if turn.planner_source == "auto_candidate":
            return "auto"
        if not turn.llm_attempted:
            return "skip"
        if turn.used_fallback:
            return "err->fb"
        if turn.llm_usage and turn.llm_usage.total_tokens is not None:
            return f"live {turn.llm_usage.total_tokens}t"
        return "live"

    @staticmethod
    def _planner_note(turn: TurnResult) -> str:
        if turn.planner_source == "route_cache":
            return "No network call was made; a cached coordinate route chose the next movement step."
        if turn.planner_source == "execution_plan":
            return "No network call was made; the engine continued an execution plan until a stop condition fired."
        if turn.planner_source == "bootstrap":
            return "No network call was made; deterministic bootstrap handling chose the action."
        if turn.planner_source == "auto_candidate":
            return "No network call was made; one ranked candidate clearly dominated the local decision."
        return "No network call was made; the deterministic fallback planner chose the action."

    @staticmethod
    def _format_navigation(turn: TurnResult) -> str:
        if turn.after.navigation is None:
            return "n/a"
        return f"visible {len(turn.after.navigation.walkable)} walkable"

    @classmethod
    def _render_prompt_map_panel(cls, planning_map: str | None, prompt_map: str | None) -> Panel:
        if planning_map is None and prompt_map is None:
            message = Text("No overworld ASCII map was available for this turn.", style="dim")
            return Panel(message, title="ASCII Map", border_style="cyan")

        table = Table.grid(expand=True, padding=(0, 2))
        table.add_column(ratio=1)
        table.add_column(ratio=1)
        table.add_row(
            Text("Planning State", style="bold cyan"),
            Text("Planner Payload", style="bold yellow"),
        )
        table.add_row(
            cls._render_ascii_block(planning_map),
            cls._render_ascii_block(prompt_map),
        )
        return Panel(table, title="ASCII Map", border_style="cyan")

    @staticmethod
    def _render_ascii_block(value: str | None) -> Text:
        if not value:
            return Text("n/a", style="dim")
        return Text(value)

    @classmethod
    def _extract_prompt_visual_map(cls, messages: list[dict[str, Any]]) -> str | None:
        for message in reversed(messages):
            if str(message.get("role", "")).lower() != "user":
                continue
            payload = cls._parse_message_payload(message.get("content"))
            if payload is None:
                continue
            visual_map = payload.get("context", {}).get("overworld_context", {}).get("visual_map")
            if isinstance(visual_map, str) and visual_map.strip():
                return visual_map
        return None

    @staticmethod
    def _parse_message_payload(content: Any) -> dict[str, Any] | None:
        if isinstance(content, dict):
            return content
        try:
            payload = json.loads(str(content))
        except (TypeError, ValueError):
            return None
        return payload if isinstance(payload, dict) else None

    @staticmethod
    def _presence_label(value: str | None) -> str:
        return "yes" if value else "no"

    @classmethod
    def _map_match_label(cls, planning_map: str | None, prompt_map: str | None) -> str:
        if planning_map is None or prompt_map is None:
            return "n/a"
        return cls._yes_no(planning_map == prompt_map)
