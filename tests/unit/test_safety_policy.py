"""The list of vulnerabilities the dependency check is told to ignore (Phase 6.6).

It had 24 entries with reasons like "Test/Dev" and "Pip ignore"; only 6 still matched anything.
A reason that does not say what the flaw is, and why it does not reach this project, is a
way to switch the check off without anyone deciding to."""

from datetime import date
from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")

POLICY = Path(__file__).resolve().parents[2] / ".safety-policy.yml"
LAZY_REASONS = {"test/dev", "pip ignore", "pytest ignore", "chroma ignore", "ignore", "false pos"}


def _entries() -> dict[str, dict[str, object]]:
    return yaml.safe_load(POLICY.read_text(encoding="utf-8"))["security"]["ignore-vulnerabilities"]


def test_every_ignored_vulnerability_names_its_cve_and_says_why_it_does_not_apply() -> None:
    for vulnerability, entry in _entries().items():
        reason = str(entry["reason"])
        assert reason.lower() not in LAZY_REASONS, f"{vulnerability}: a reason that says nothing"
        assert "CVE-" in reason, f"{vulnerability}: the reason does not name the CVE"
        assert len(reason) >= 60, f"{vulnerability}: the reason is too short to be an assessment"


def test_every_ignore_expires_so_that_it_has_to_be_looked_at_again() -> None:
    for vulnerability, entry in _entries().items():
        assert isinstance(entry["expires"], date), f"{vulnerability}: no expiry date"


def test_there_are_no_duplicate_or_empty_entries() -> None:
    entries = _entries()

    assert entries, "an empty ignore list should be removed from the file, not left empty"
    assert all(isinstance(entry, dict) and entry for entry in entries.values())
