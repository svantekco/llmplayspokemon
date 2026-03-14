from __future__ import annotations

from pokemon_agent.config import AppConfig


def test_app_config_reads_pyboy_window_at_instantiation(monkeypatch):
    monkeypatch.setenv("PYBOY_WINDOW", "SDL2")
    assert AppConfig().pyboy_window == "SDL2"


def test_app_config_reads_context_budget_env(monkeypatch):
    monkeypatch.setenv("POKEMON_AGENT_PROMPT_BUDGET_TOKENS", "777")
    monkeypatch.setenv("POKEMON_AGENT_CONTEXT_ACTION_WINDOW", "5")
    monkeypatch.setenv("POKEMON_AGENT_CONTEXT_EVENT_WINDOW", "6")

    config = AppConfig()

    assert config.prompt_budget_tokens == 777
    assert config.context_action_window == 5
    assert config.context_event_window == 6


def test_app_config_uses_larger_default_prompt_budget(monkeypatch):
    monkeypatch.delenv("POKEMON_AGENT_PROMPT_BUDGET_TOKENS", raising=False)

    assert AppConfig().prompt_budget_tokens == 2500
