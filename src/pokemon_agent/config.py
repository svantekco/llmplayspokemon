from __future__ import annotations

import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

load_dotenv()


def _get_env_float(name: str, default: str) -> float:
    return float(os.getenv(name, default))


def _get_env_int(name: str, default: str) -> int:
    return int(os.getenv(name, default))


@dataclass(slots=True)
class OpenRouterConfig:
    api_key: str = field(default_factory=lambda: os.getenv("OPENROUTER_API_KEY", ""))
    model: str = field(default_factory=lambda: os.getenv("OPENROUTER_MODEL", "openai/gpt-5-mini"))
    base_url: str = field(default_factory=lambda: os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"))
    site_url: str = field(default_factory=lambda: os.getenv("OPENROUTER_SITE_URL", "http://localhost"))
    app_name: str = field(default_factory=lambda: os.getenv("OPENROUTER_APP_NAME", "pokemon-llm-agent"))
    timeout_seconds: float = field(default_factory=lambda: _get_env_float("OPENROUTER_TIMEOUT_SECONDS", "30"))
    max_retries: int = field(default_factory=lambda: _get_env_int("OPENROUTER_MAX_RETRIES", "2"))
    min_request_interval_seconds: float = field(
        default_factory=lambda: _get_env_float("OPENROUTER_MIN_REQUEST_INTERVAL_SECONDS", "1.0")
    )


@dataclass(slots=True)
class AppConfig:
    openrouter: OpenRouterConfig = field(default_factory=OpenRouterConfig)
    max_repeat: int = 4
    short_term_window: int = 8
    prompt_budget_tokens: int = field(default_factory=lambda: _get_env_int("POKEMON_AGENT_PROMPT_BUDGET_TOKENS", "2500"))
    context_action_window: int = field(default_factory=lambda: _get_env_int("POKEMON_AGENT_CONTEXT_ACTION_WINDOW", "4"))
    context_event_window: int = field(default_factory=lambda: _get_env_int("POKEMON_AGENT_CONTEXT_EVENT_WINDOW", "4"))
    max_turns: int = field(default_factory=lambda: _get_env_int("POKEMON_AGENT_MAX_TURNS", "8"))
    pyboy_window: str = field(default_factory=lambda: os.getenv("PYBOY_WINDOW", "null"))
    pyboy_press_frames: int = field(default_factory=lambda: _get_env_int("PYBOY_PRESS_FRAMES", "4"))
    pyboy_post_action_frames: int = field(default_factory=lambda: _get_env_int("PYBOY_POST_ACTION_FRAMES", "20"))
    pyboy_boot_frames: int = field(default_factory=lambda: _get_env_int("PYBOY_BOOT_FRAMES", "1440"))
    stuck_threshold: int = field(default_factory=lambda: _get_env_int("POKEMON_AGENT_STUCK_THRESHOLD", "4"))
    default_rom_path: str = field(default_factory=lambda: os.getenv("POKEMON_ROM_PATH", "game.gb"))
