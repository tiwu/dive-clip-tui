"""Tests for the .env loader."""

from __future__ import annotations

import os

from utils.config import load_config
from utils.env import load_dotenv


def test_load_dotenv_parses_and_respects_existing(tmp_path, monkeypatch):
    monkeypatch.delenv("A_KEY", raising=False)
    monkeypatch.delenv("B_KEY", raising=False)
    monkeypatch.delenv("C_KEY", raising=False)
    monkeypatch.delenv("D_KEY", raising=False)
    monkeypatch.setenv("KEEP_KEY", "shell")
    env = tmp_path / ".env"
    env.write_text(
        "# comment\n\nA_KEY=one\nexport B_KEY=\"two words\"\nC_KEY='three'\n"
        "D_KEY=four # trailing\nKEEP_KEY=file\nnot a pair\n=novalue\n",
        encoding="utf-8",
    )
    assert load_dotenv(env) is True
    assert os.environ["A_KEY"] == "one"
    assert os.environ["B_KEY"] == "two words"
    assert os.environ["C_KEY"] == "three"
    assert os.environ["D_KEY"] == "four"
    assert os.environ["KEEP_KEY"] == "shell"


def test_load_dotenv_missing_file(tmp_path):
    assert load_dotenv(tmp_path / ".env") is False


def test_load_config_reads_env_next_to_config(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text("OPENROUTER_API_KEY=abc\n", encoding="utf-8")
    cfg = tmp_path / "config.yaml"
    cfg.write_text("vlm:\n  provider: openrouter\n", encoding="utf-8")
    load_config(cfg)
    assert os.environ["OPENROUTER_API_KEY"] == "abc"
    monkeypatch.delenv("OPENROUTER_API_KEY")
