from __future__ import annotations

import re

from pokemon_agent.agent.progress import ProgressResult
from pokemon_agent.agent.stuck_detector import StuckState
from pokemon_agent.agent.world_map import confirm_transition
from pokemon_agent.agent.world_map import observe_state
from pokemon_agent.agent.world_map import world_map_stats
from pokemon_agent.data.walkthrough import Milestone
from pokemon_agent.data.walkthrough import get_current_milestone
from pokemon_agent.data.walkthrough import milestone_for_completion_flag
from pokemon_agent.models.action import ActionDecision
from pokemon_agent.models.events import EventRecord, EventType
from pokemon_agent.models.memory import DiscoveredConnector
from pokemon_agent.models.memory import MemoryState
from pokemon_agent.models.planner import Objective
from pokemon_agent.models.planner import ObjectiveHorizon
from pokemon_agent.models.planner import ObjectiveTarget
from pokemon_agent.models.state import StructuredGameState


class MemoryManager:
    def __init__(self, window: int = 8) -> None:
        self.memory = MemoryState()
        self.window = window

    def add_event(self, event_type: EventType, summary: str, step: int, metadata: dict | None = None) -> None:
        event = EventRecord(type=event_type, summary=summary, step=step, metadata=metadata or {})
        self.memory.recent_events.append(event)
        self.memory.recent_events = self.memory.recent_events[-self.window:]

    def add_event_record(self, event: EventRecord) -> None:
        self.memory.recent_events.append(event)
        self.memory.recent_events = self.memory.recent_events[-self.window:]

    def update_from_transition(
        self,
        previous: StructuredGameState,
        current: StructuredGameState,
        action: ActionDecision,
        progress: ProgressResult,
        stuck_state: StuckState | None = None,
        *,
        active_connector: DiscoveredConnector | None = None,
    ) -> list[EventRecord]:
        events: list[EventRecord] = []
        observe_state(self.memory.long_term.world_map, previous)
        observe_state(self.memory.long_term.world_map, current)

        if not current.is_bootstrap() and (previous.is_bootstrap() or previous.map_id != current.map_id or previous.map_name != current.map_name):
            events.append(
                EventRecord(
                    type=EventType.MAP_CHANGED,
                    summary=f"Entered {current.map_name}",
                    step=current.step,
                    metadata={"map_id": current.map_id},
                )
            )
            if current.map_id is not None:
                self.memory.long_term.known_locations[str(current.map_id)] = current.map_name
            connector = confirm_transition(
                self.memory.long_term.world_map,
                previous,
                action,
                current,
                active_connector=active_connector,
            )
            if connector is not None:
                events.append(
                    EventRecord(
                        type=EventType.CONNECTOR_CONFIRMED,
                        summary=f"Confirmed connector from {previous.map_name} to {current.map_name}",
                        step=current.step,
                        metadata={
                            "connector_id": connector.id,
                            "source_map": previous.map_name,
                            "destination_map": current.map_name,
                            "kind": connector.kind,
                        },
                    )
                )

        if previous.x != current.x or previous.y != current.y:
            events.append(
                EventRecord(
                    type=EventType.MOVED,
                    summary=f"Moved to ({current.x}, {current.y}) on {current.map_name}",
                    step=current.step,
                    metadata={"from": {"x": previous.x, "y": previous.y}, "to": {"x": current.x, "y": current.y}},
                )
            )

        if previous.menu_open != current.menu_open:
            event_type = EventType.MENU_OPENED if current.menu_open else EventType.MENU_CLOSED
            summary = "Opened menu" if current.menu_open else "Closed menu"
            events.append(EventRecord(type=event_type, summary=summary, step=current.step))

        if not previous.text_box_open and current.text_box_open:
            events.append(EventRecord(type=EventType.TEXT_OPENED, summary="Opened dialogue/text box", step=current.step))
        elif previous.text_box_open and current.text_box_open and action.action.value == "PRESS_A":
            events.append(EventRecord(type=EventType.TEXT_ADVANCED, summary="Advanced dialogue", step=current.step))

        if not previous.battle_state and current.battle_state:
            events.append(
                EventRecord(
                    type=EventType.BATTLE_STARTED,
                    summary=f"Battle started ({current.battle_state.get('kind', 'UNKNOWN')})",
                    step=current.step,
                )
            )
        elif previous.battle_state and not current.battle_state:
            events.append(EventRecord(type=EventType.BATTLE_ENDED, summary="Battle ended", step=current.step))

        gained_items = self._inventory_gain(previous, current)
        for item_name in gained_items:
            events.append(
                EventRecord(
                    type=EventType.ITEM_RECEIVED,
                    summary=f"Received {item_name}",
                    step=current.step,
                    metadata={"item_name": item_name},
                )
            )

        if progress.classification == "no_effect" and stuck_state and stuck_state.score >= 3:
            event_type = EventType.BLOCKED_MOVE if action.action.value.startswith("MOVE_") else EventType.NO_EFFECT
            events.append(
                EventRecord(
                    type=event_type,
                    summary=f"{action.action.value} had no effect",
                    step=current.step,
                    metadata={"stuck_score": stuck_state.score},
                )
            )

        events.extend(self._update_goals(current, progress, stuck_state))
        for event in events:
            if self.memory.recent_events and self.memory.recent_events[-1].summary == event.summary:
                continue
            self.add_event_record(event)
        return events

    def summarize_for_prompt(self, state: StructuredGameState) -> dict:
        return {
            "current_state": state.prompt_summary(),
            "goals": self.memory.goals.model_dump(),
            "recent_events": [e.model_dump() for e in self.memory.recent_events],
            "long_term": {
                "known_locations": self.memory.long_term.known_locations,
                "story_flags": self.memory.long_term.story_flags[-8:],
                "navigation_notes": self.memory.long_term.navigation_notes[-6:],
                "heuristics": self.memory.long_term.heuristics[-6:],
                "world_map_stats": world_map_stats(self.memory.long_term.world_map),
                "navigation_goal": (
                    self.memory.long_term.navigation_goal.model_dump()
                    if self.memory.long_term.navigation_goal is not None
                    else None
                ),
                "objective": (
                    self.memory.long_term.objective.model_dump(mode="json")
                    if self.memory.long_term.objective is not None
                    else None
                ),
            },
        }

    def _inventory_gain(self, previous: StructuredGameState, current: StructuredGameState) -> list[str]:
        before = {item.name: item.count for item in previous.inventory}
        after = {item.name: item.count for item in current.inventory}
        gained: list[str] = []
        for name, count in after.items():
            if count > before.get(name, 0):
                gained.append(name)
        return gained

    def _update_goals(
        self,
        state: StructuredGameState,
        progress: ProgressResult,
        stuck_state: StuckState | None,
    ) -> list[EventRecord]:
        events = self._update_story_flags(state)
        goals = self.memory.goals
        old_snapshot = goals.model_dump()
        current_milestone = self._current_milestone(state)
        objective = self.memory.long_term.objective

        if state.is_bootstrap():
            phase = state.bootstrap_phase()
            goals.mid_term_goal = "Finish the intro flow and reach a controllable in-game state"
            goals.short_term_goal = "Reach a controllable gameplay state"
            goals.long_term_goal = "Reach the first controllable in-game scene"
            goals.success_conditions = ["Reach controllable gameplay"]
            if phase == "title_screen":
                goals.current_strategy = "Use PRESS_START on the title screen"
            elif phase == "title_menu":
                goals.current_strategy = "Use PRESS_A to confirm the highlighted title option"
            elif state.text_box_open:
                goals.current_strategy = "Use PRESS_A to advance intro dialogue"
            else:
                goals.current_strategy = "Use single confirm inputs while the startup sequence advances"
        else:
            goals.long_term_goal = current_milestone.description
            goals.mid_term_goal = self._mid_term_goal(state, current_milestone)
            goals.short_term_goal = self._short_term_goal(state, current_milestone)
            goals.current_strategy = self._current_strategy(state, current_milestone)
            goals.success_conditions = self._milestone_success_conditions(current_milestone, progress)
            if objective is not None:
                goals.short_term_goal = objective.goal
                goals.current_strategy = objective.strategy

        if stuck_state and stuck_state.score >= 4 and stuck_state.recovery_hint:
            goals.current_strategy = stuck_state.recovery_hint
            if stuck_state.recovery_hint not in self.memory.long_term.heuristics:
                self.memory.long_term.heuristics.append(stuck_state.recovery_hint)
                self.memory.long_term.heuristics = self.memory.long_term.heuristics[-10:]

        goals.active_objectives = self._build_objective_stack(state, progress, stuck_state, current_milestone)

        if goals.model_dump() != old_snapshot:
            events.append(
                EventRecord(
                    type=EventType.GOAL_UPDATED,
                    summary=f"Goals updated for {state.map_name}",
                    step=state.step,
                    metadata={
                        "short_term_goal": goals.short_term_goal,
                        "strategy": goals.current_strategy,
                        "milestone_id": current_milestone.id,
                    },
                )
            )
        return events

    def _update_story_flags(self, state: StructuredGameState) -> list[EventRecord]:
        current_flags = [flag for flag in state.story_flags if flag]
        previous_flags = set(self.memory.long_term.story_flags)
        self.memory.long_term.story_flags = current_flags

        events: list[EventRecord] = []
        for flag in current_flags:
            if flag in previous_flags:
                continue
            milestone = milestone_for_completion_flag(flag)
            metadata = {"flag": flag}
            if milestone is not None:
                metadata["milestone_id"] = milestone.id
                metadata["milestone_description"] = milestone.description
                summary = f"Completed milestone: {milestone.description}"
            else:
                summary = f"Unlocked story milestone: {flag.replace('_', ' ')}"
            events.append(
                EventRecord(
                    type=EventType.MILESTONE_COMPLETE,
                    summary=summary,
                    step=state.step,
                    metadata=metadata,
                )
            )
        return events

    def _current_milestone(self, state: StructuredGameState) -> Milestone:
        return get_current_milestone(
            state.story_flags,
            [item.name for item in state.inventory],
            current_map_name=state.map_name,
            badges=state.badges,
        )

    def _build_objective_stack(
        self,
        state: StructuredGameState,
        progress: ProgressResult,
        stuck_state: StuckState | None,
        milestone: Milestone,
    ) -> list[Objective]:
        long_term = Objective(
            id=f"long_{milestone.id}",
            horizon=ObjectiveHorizon.LONG_TERM,
            target=ObjectiveTarget(kind="story", map_name=milestone.target_map_name, detail=milestone.id),
        )
        mid_term = Objective(
            id=f"mid_{milestone.id}",
            horizon=ObjectiveHorizon.MID_TERM,
            target=ObjectiveTarget(kind="map", map_id=state.map_id, map_name=milestone.target_map_name),
        )
        short_term = Objective(
            id=f"short_{milestone.id}",
            horizon=ObjectiveHorizon.SHORT_TERM,
            target=self._short_term_target(state),
        )
        return [long_term, mid_term, short_term]

    def _mid_term_success_conditions(
        self,
        state: StructuredGameState,
        progress: ProgressResult,
        milestone: Milestone,
    ) -> list[str]:
        if progress.newly_completed_subgoals:
            return progress.newly_completed_subgoals[-2:]
        if state.is_bootstrap():
            return ["Reach controllable gameplay"]
        if state.battle_state:
            return ["Battle ends"]
        return self._milestone_success_conditions(milestone)[:2]

    def _short_term_target(self, state: StructuredGameState) -> ObjectiveTarget:
        if state.is_bootstrap():
            return ObjectiveTarget(kind="bootstrap", detail=state.bootstrap_phase())
        if state.text_box_open:
            return ObjectiveTarget(kind="text", map_name=state.map_name)
        if state.menu_open:
            return ObjectiveTarget(kind="menu", map_name=state.map_name)
        if state.battle_state:
            return ObjectiveTarget(kind="battle", map_name=state.map_name)
        return ObjectiveTarget(kind="map", map_id=state.map_id, map_name=state.map_name, x=state.x, y=state.y)

    def _short_term_success_conditions(self, state: StructuredGameState) -> list[str]:
        if state.is_bootstrap():
            return ["Bootstrap phase advances"]
        if state.text_box_open:
            return ["Dialogue changes or closes"]
        if state.menu_open:
            return ["Menu closes or selection changes"]
        if state.battle_state:
            return ["Battle state changes or ends"]
        return ["Position changes", "Dialogue opens", "Map changes"]

    def _short_term_invalidation_conditions(
        self,
        state: StructuredGameState,
        stuck_state: StuckState | None,
    ) -> list[str]:
        conditions = ["No effect", "Map changes", "Mode changes"]
        if state.navigation is not None:
            conditions.append("Collision hash changes")
        if stuck_state and stuck_state.score >= 4:
            conditions.append("Stuck recovery overrides the plan")
        return conditions

    def _mid_term_goal(self, state: StructuredGameState, milestone: Milestone) -> str:
        if self._on_target_map(state, milestone):
            return milestone.sub_steps[0] if milestone.sub_steps else f"Work through {milestone.target_map_name}"
        if milestone.route_hints:
            return milestone.route_hints[0]
        return f"Travel toward {milestone.target_map_name}"

    def _short_term_goal(self, state: StructuredGameState, milestone: Milestone) -> str:
        if state.battle_state:
            return f"Finish the current battle so progress toward {milestone.target_map_name} can continue"
        if state.text_box_open:
            return "Advance the current dialogue and watch for walkthrough-relevant cues"
        if state.menu_open:
            return f"Resolve the current menu and return to {milestone.target_map_name}"
        if self._on_target_map(state, milestone):
            return milestone.sub_steps[0] if milestone.sub_steps else f"Advance the objective on {state.map_name}"
        return f"Move toward {milestone.target_map_name}"

    def _current_strategy(self, state: StructuredGameState, milestone: Milestone) -> str:
        if state.battle_state:
            return "Keep battle inputs small, survive the fight, then resume the walkthrough route"
        if state.text_box_open:
            return "Use PRESS_A to advance dialogue unless the visible state or prompt changes"
        if state.menu_open:
            return "Use single inputs while the menu is open and only stay if it helps the milestone"
        if self._on_target_map(state, milestone):
            return milestone.sub_steps[0] if milestone.sub_steps else f"Execute the next step on {milestone.target_map_name}"
        if milestone.route_hints:
            return milestone.route_hints[0]
        return "Prefer deterministic local plans, then choose among ranked candidates"

    def _milestone_success_conditions(
        self,
        milestone: Milestone,
        progress: ProgressResult | None = None,
    ) -> list[str]:
        conditions: list[str] = []
        if progress and progress.newly_completed_subgoals:
            conditions.extend(progress.newly_completed_subgoals[-2:])
        if milestone.completion_flag is not None:
            conditions.append(f"Story flag {milestone.completion_flag} is active")
        if milestone.completion_item is not None:
            conditions.append(f"Obtain {milestone.completion_item}")
        if not conditions:
            conditions.append(f"Reach and clear the {milestone.target_map_name} objective")
        deduped: list[str] = []
        for condition in conditions:
            if condition not in deduped:
                deduped.append(condition)
        return deduped[:5]

    def _on_target_map(self, state: StructuredGameState, milestone: Milestone) -> bool:
        current_tokens = {token for token in re.findall(r"[a-z0-9]+", (state.map_name or "").lower()) if token}
        target_tokens = {token for token in re.findall(r"[a-z0-9]+", milestone.target_map_name.lower()) if token}
        if not current_tokens or not target_tokens:
            return False
        if target_tokens.issubset(current_tokens) or current_tokens.issubset(target_tokens):
            return True
        generic_tokens = {
            "city",
            "town",
            "route",
            "road",
            "house",
            "gym",
            "lab",
            "forest",
            "cave",
            "tower",
            "dock",
            "center",
            "pokecenter",
            "room",
            "rooms",
            "island",
            "plateau",
            "gate",
            "hideout",
            "mansion",
            "co",
            "lobby",
        }
        current_core = current_tokens - generic_tokens
        target_core = target_tokens - generic_tokens
        if not current_core or not target_core:
            return False
        return current_core.issubset(target_core) or target_core.issubset(current_core)
