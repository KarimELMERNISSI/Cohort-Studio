"""
The data-sharing confirmation of the AI engine leaves a line in the server log (the data-protection specialist's
PS-M2). The confirmation lives in one browser tab: an organisation asked whether study data went to a cloud provider
on a given day, and by whom, had no answer. Each confirmation, each withdrawal and each confirmation a sign-out ends
writes one line, with the account, the recipients and the time (the log's own), never what is sent; so does an engine
that starts under one. The lines have a logger of their own at INFO (``cohortstudio.audit``): ``CMP_LOG_LEVEL`` at
WARNING silenced them, without a word.
"""

from __future__ import annotations

import logging

import pytest

from core.workspace import UserScope
from manage.ai_payload import UNREADABLE_ADDRESS, provider_label, recipient
from manage.rag_manager import RAGManager
from utils.logging_config import AUDIT_LOGGER, single_line

LOGGER = AUDIT_LOGGER


def _lines(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [
        r.getMessage()
        for r in caplog.records
        if r.name == LOGGER and "AI data sharing" in r.getMessage()
    ]


def test_each_confirmation_and_each_withdrawal_writes_one_line(caplog, tmp_path) -> None:
    manager = RAGManager(documents_dir=str(tmp_path), provider="gemini")
    with caplog.at_level(logging.INFO, logger=LOGGER):
        manager.confirm("anthropic", "openai", account="dr.martin")
        # the same box again: nothing new
        manager.confirm("openai", "anthropic", account="dr.martin")
        manager.confirm("gemini", account="dr.martin")
        manager.withdraw_consent("openai", account="dr.martin")
        manager.withdraw_consent("openai", account="dr.martin")  # nothing left to withdraw
        manager.withdraw_consent("mistral", account="dr.martin")  # never confirmed
    assert _lines(caplog) == [
        "AI data sharing confirmed by the account dr.martin for: anthropic + openai",
        "AI data sharing confirmed by the account dr.martin for: gemini",
        "AI data sharing confirmation withdrawn by the account dr.martin for: anthropic + openai",
    ]
    assert manager.confirmed == {frozenset({"gemini"})}


def test_withdrawing_everything_writes_a_line_for_each_confirmation(caplog, tmp_path) -> None:
    manager = RAGManager(documents_dir=str(tmp_path), provider="gemini")
    manager.confirm("gemini", account="a")
    manager.confirm("anthropic", "openai", account="a")
    with caplog.at_level(logging.INFO, logger=LOGGER):
        caplog.clear()  # the confirmations' own lines, when another test left the log at INFO
        manager.withdraw_consent(account="b")
    assert _lines(caplog) == [
        "AI data sharing confirmation withdrawn by the account b for: anthropic + openai",
        "AI data sharing confirmation withdrawn by the account b for: gemini",
    ]


def test_a_confirmation_carried_to_a_rebuilt_engine_writes_no_second_line(caplog, tmp_path) -> None:
    """Settings confirms again, for an engine manager rebuilt since, a box ticked earlier in the session: its line
    was written when it was ticked."""
    manager = RAGManager(documents_dir=str(tmp_path), provider="gemini")
    with caplog.at_level(logging.INFO, logger=LOGGER):
        manager.confirm("gemini", account="dr.martin", again=True)
    assert _lines(caplog) == [] and manager.is_confirmed(frozenset({"gemini"}))


def test_the_account_is_the_workspace_s_when_none_is_given_and_one_line_whatever_it_holds(
    caplog, tmp_path
) -> None:
    scoped = RAGManager(
        documents_dir=str(tmp_path), provider="gemini", scope=UserScope("folder_name")
    )
    bare = RAGManager(documents_dir=str(tmp_path), provider="gemini")
    with caplog.at_level(logging.INFO, logger=LOGGER):
        scoped.confirm("gemini")
        bare.confirm("gemini")
        bare.confirm(
            "ollama@ollama.example.com\nAI data sharing confirmed by the account admin",
            account="x\ny",
        )
        # a line separator of another kind, and a terminal's escape, from a name the sign-in service gives
        bare.confirm("mistral", account="eve INFO forged\x85\x1bc")
    lines = _lines(caplog)
    assert lines[:2] == [
        "AI data sharing confirmed by the account folder_name for: gemini",
        "AI data sharing confirmed by the account unknown account for: gemini",
    ]
    # a name or a host from elsewhere cannot forge another line of the log
    assert len(lines) == 4 and "\n" not in lines[2] and "x\\ny" in lines[2]
    assert lines[3] == (
        "AI data sharing confirmed by the account eve\\u2028INFO forged\\x85\\x1bc for: mistral"
    )
    assert all(line.isprintable() for line in lines)


def test_every_character_that_is_not_printable_is_written_as_its_escape() -> None:
    assert single_line("plain name, with spaces and accents: Zoé") == (
        "plain name, with spaces and accents: Zoé"
    )
    assert single_line("a\rb\nc\td e\u0085f\x1b[2Jg\x00") == (
        "a\\rb\\nc\\td\\u2028e\\x85f\\x1b[2Jg\\x00"
    )


def test_the_record_is_written_whatever_the_log_level(tmp_path) -> None:
    """``CMP_LOG_LEVEL=WARNING`` quiets the console, and silenced every line of the record."""

    class Kept(logging.Handler):
        def __init__(self) -> None:
            super().__init__()
            self.lines: list[str] = []

        def emit(self, record: logging.LogRecord) -> None:
            self.lines.append(record.getMessage())

    root = logging.getLogger()
    kept, level = Kept(), root.level
    root.addHandler(kept)
    root.setLevel(logging.WARNING)
    try:
        manager = RAGManager(documents_dir=str(tmp_path), provider="gemini")
        manager.confirm("gemini", account="dr.martin")
        logging.getLogger("manage.rag_manager").info("an ordinary line, quieted")
    finally:
        root.removeHandler(kept)
        root.setLevel(level)
    assert kept.lines == ["AI data sharing confirmed by the account dr.martin for: gemini"]


def test_a_sign_out_ends_each_confirmation_with_a_line(caplog, tmp_path) -> None:
    """The record said when a confirmation began, not when it ended: a sign-out ends each, under the account that
    gave it."""
    manager = RAGManager(documents_dir=str(tmp_path), provider="gemini")
    manager.confirm("gemini", account="dr.martin")
    manager.confirm("anthropic", "openai", account="dr.dupont")
    with caplog.at_level(logging.INFO, logger=LOGGER):
        caplog.clear()
        manager.end_confirmations()
        manager.end_confirmations()  # nothing left: nothing more
    assert _lines(caplog) == [
        "AI data sharing confirmation of the account dr.dupont ended with the session for: anthropic + openai",
        "AI data sharing confirmation of the account dr.martin ended with the session for: gemini",
    ]
    assert manager.confirmed == set() and not manager.is_confirmed(frozenset({"gemini"}))


def test_a_recipient_an_ollama_server_answers_is_read_as_an_address() -> None:
    """A "[" in the answer stopped Settings' AI card with an error, and a line separator or a terminal's escape
    reached the record. An answer that cannot be read as an address is still a recipient, named so."""
    assert recipient("ollama", None, "llama3", remote_host="https://ollama.com:443") == (
        "ollama@ollama.com"
    )
    for answer in (
        "https://[ollama.com",
        "https://ollama.com INFO forged",
        "https://ollama.com\x1bc",
        "https://user:secret@ollama.com",
        "ftp://ollama.com",
    ):
        found = recipient("ollama", None, "llama3", remote_host=answer)
        assert found == f"ollama@{UNREADABLE_ADDRESS}", answer
    assert provider_label(f"ollama@{UNREADABLE_ADDRESS}") == (
        "Ollama at an address that cannot be read"
    )
    # a cloud model by its tag, and a model the server runs itself, as before
    assert recipient("ollama", None, "gpt-oss:120b-cloud") == "ollama@ollama.com"
    assert recipient("ollama", "http://localhost:11434", "llama3") is None
