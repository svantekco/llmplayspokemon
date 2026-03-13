from __future__ import annotations

from argparse import Namespace

import pytest

from pokemon_agent.main import run_args


def test_run_args_requires_api_key_for_llm(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    args = Namespace(
        mode="mock",
        rom=None,
        turns=1,
        continuous=False,
        planner="llm",
        checkpoint_dir=None,
        resume=None,
        log_mode="quiet",
        eval=False,
    )

    with pytest.raises(SystemExit, match="OPENROUTER_API_KEY"):
        run_args(args)
