from __future__ import annotations

from argparse import Namespace

import pytest

from pokemon_agent.main import _InterruptController, run_args


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


def test_interrupt_controller_requires_two_ctrl_c_presses():
    controller = _InterruptController()

    controller.handle_signal(None, None)

    assert controller.shutdown_requested is True

    with pytest.raises(KeyboardInterrupt):
        controller.handle_signal(None, None)
