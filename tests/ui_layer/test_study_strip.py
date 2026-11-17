"""
The study definition strip on Reproduce > Study dossier (M8, the study package):
- a revision is saved once for each definition, as written: an unchanged one saves none, a corrected description is
  the next revision (it differs in its labels only), and a revision can be deleted, its number never given again;
- a session that holds no definition is offered to load the latest revision (the next day, a saved version loaded
  again), never to save an empty one;
- the trace records a revision by its numbers and fingerprint only, never its body or a checksum covering its labels,
  and the trace report names it by its number;
- the Word dossier names the study and its revision.
"""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import utils.data_paths as dp
from ui import nav
from ui.components.population import _family_key

ROOT = Path(__file__).resolve().parents[2]
MAIN = str(ROOT / "main.py")
KEY = "dossier_study"


def _ok(at: AppTest) -> None:
    assert not at.exception, [str(e.value)[:300] for e in at.exception]


def _texts(at: AppTest) -> str:
    return " | ".join(
        [m.value for m in at.markdown]
        + [c.value for c in at.caption]
        + [s.value for s in at.success]
        + [i.value for i in at.info]
        + [e.value for e in at.error]
    )


def _buttons(at: AppTest) -> dict[str, str]:
    return {b.key: b.label for b in at.button if b.key and b.key.startswith(KEY)}


@pytest.fixture
def app(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(dp, "DATA_ROOT", str(tmp_path))
    monkeypatch.setitem(os.environ, "CMP_IAM_PROVIDER", "local")
    at = AppTest.from_file(MAIN, default_timeout=300)
    at.run()
    at.session_state["user_authenticated"] = True
    at.session_state["username"] = "study_user"
    at.session_state["user_role"] = "researcher"
    at.run()
    at.switch_page(nav.page_path("start")).run()
    at.button(key="start_tutorial").click().run()
    at.checkbox(key=_family_key("study_inclusion")).check().run()
    _ok(at)
    at.switch_page(nav.page_path("dossier")).run()
    _ok(at)
    return at


def _revisions(root: Path) -> list[Path]:
    return sorted(root.glob("users/study_user/**/revisions/r*.json"))


def _dossier(at: AppTest) -> AppTest:
    at.switch_page(nav.page_path("dossier")).run()
    _ok(at)
    return at


def _choose(at: AppTest, choices) -> None:
    """The study's privacy choices as ``AppState.set_privacy_choices`` keeps them (no screen sets them yet): with
    the id of the study they are of, apart from the configuration."""
    at.session_state["_study_privacy"] = {
        "study": "tutorial_cohort",
        "section": choices.to_section(),
    }


def test_a_revision_is_saved_once_for_each_definition(app: AppTest, tmp_path: Path) -> None:
    shown = _texts(app)
    assert (
        "Study definition" in shown and "tutorial_cohort" in shown and "no revision saved" in shown
    )
    assert "4 of 12 criteria ticked" in shown  # counted as the sidebar counts
    assert _buttons(app) == {f"{KEY}_save": "Save revision 1"}
    app.button(key=f"{KEY}_save").click().run()
    _ok(app)
    shown = _texts(app)
    assert "Revision 1 saved in your workspace: the study definition above." in shown
    assert "revision 1, saved" in shown
    assert _buttons(app) == {f"{KEY}_delete": "Delete revision 1"}  # unchanged: nothing to save
    (saved,) = _revisions(tmp_path)
    assert json.loads(saved.read_text(encoding="utf-8"))["body"]["population"]["ticked"]

    # another tick: changed since revision 1, said where, and revision 2 follows it
    app.switch_page(nav.page_path("start")).run()
    app.checkbox(key=_family_key("clinical_anomalies")).check().run()
    _dossier(app)
    shown = _texts(app)
    assert "changed since revision 1" in shown
    assert "Changed since revision 1: which criteria are ticked." in shown
    assert _buttons(app)[f"{KEY}_save"] == "Save revision 2"
    # unticked back: the definition of revision 1 again
    app.switch_page(nav.page_path("start")).run()
    app.checkbox(key=_family_key("clinical_anomalies")).uncheck().run()
    _dossier(app)
    assert "revision 1, saved" in _texts(app) and f"{KEY}_save" not in _buttons(app)


def test_a_session_without_a_definition_loads_the_latest_revision(app: AppTest) -> None:
    """The next day a saved version comes back without its criteria: the strip offers the revision, never an empty
    one as the next."""
    app.button(key=f"{KEY}_save").click().run()
    kept = app.session_state["config"]
    app.session_state["config"] = {**copy.deepcopy(kept), "mask_families": {}}
    _dossier(app)
    shown = _texts(app)
    assert "revision 1 not loaded" in shown
    assert "This session holds no criteria: revision 1 holds 4 of 12 criteria ticked." in shown
    assert _buttons(app) == {f"{KEY}_load": "Load revision 1"}  # no Save of an empty definition
    app.button(key=f"{KEY}_load").click().run()
    _ok(app)
    assert "Revision 1 loaded" in _texts(app) and "revision 1, saved" in _texts(app)
    assert app.session_state["config"]["mask_families"] == kept["mask_families"]
    assert (
        "_study_privacy" not in app.session_state
    )  # the revision holds no privacy choice: none is kept


def test_loading_a_revision_brings_its_privacy_choices_back(app: AppTest) -> None:
    """They are part of the revision: loaded with its criteria, into the session's own place for them, never into
    the configuration."""
    from core.privacy import StudyPrivacy

    _choose(app, StudyPrivacy(k=20))
    _dossier(app)
    app.button(key=f"{KEY}_save").click().run()
    _ok(app)
    del app.session_state["_study_privacy"]
    app.session_state["config"] = {
        **copy.deepcopy(app.session_state["config"]),
        "mask_families": {},
    }
    _dossier(app)
    app.button(key=f"{KEY}_load").click().run()
    _ok(app)
    assert "the study's privacy choices are this session's again" in _texts(app)
    assert app.session_state["_study_privacy"] == {
        "study": "tutorial_cohort",
        "section": {"k": 20},
    }
    assert "privacy" not in app.session_state["config"]
    assert "revision 1, saved" in _texts(
        app
    )  # the session's definition is the revision's, choices included


def test_a_corrected_description_is_the_next_revision_and_the_first_can_be_deleted(
    app: AppTest, tmp_path: Path
) -> None:
    app.button(key=f"{KEY}_save").click().run()
    config = copy.deepcopy(app.session_state["config"])
    config["mask_families"]["clinical_anomalies"]["Implausible age"]["description"] = (
        "Checked by Dr J. Smith"
    )
    app.session_state["config"] = config
    _dossier(app)
    assert "labels changed since revision 1" in _texts(app)
    assert "descriptions or notes only" in _texts(app)
    app.button(key=f"{KEY}_save").click().run()
    _ok(app)
    assert "revision 2, saved" in _texts(app) and len(_revisions(tmp_path)) == 2
    # the name was a slip: corrected as revision 3, then revisions 2 is deleted, its number kept
    config["mask_families"]["clinical_anomalies"]["Implausible age"]["description"] = "Checked"
    app.session_state["config"] = config
    _dossier(app)
    app.button(key=f"{KEY}_save").click().run()
    _ok(app)
    state_store = app.session_state["_services"].protocols
    assert state_store.delete_revision("tutorial_cohort", 2)
    files = _revisions(tmp_path)
    assert [f.name[:5] for f in files] == ["r0001", "r0003"]
    assert not any("J. Smith" in f.read_text(encoding="utf-8") for f in files)
    assert state_store.next_number("tutorial_cohort") == 4


def test_delete_asks_first(app: AppTest, tmp_path: Path) -> None:
    """Delete opens a confirmation, which names what goes; the revision stays until it is confirmed (the browser
    check confirms it: AppTest reruns the whole page where the browser reruns the dialog alone)."""
    app.button(key=f"{KEY}_save").click().run()
    app.button(key=f"{KEY}_delete").click().run()
    _ok(app)
    confirm = [b for b in app.button if b.label == "Delete revision 1" and b.key != f"{KEY}_delete"]
    assert confirm, [b.label for b in app.button]  # the dialog's own button
    assert "with the descriptions and notes it holds" in " ".join(m.value for m in app.markdown)
    assert len(_revisions(tmp_path)) == 1


def test_the_trace_records_a_revision_by_its_numbers_and_fingerprint_only(app: AppTest) -> None:
    app.button(key=f"{KEY}_save").click().run()
    _ok(app)
    history = app.session_state["transformation_manager"].history
    (step,) = [s for s in history if s.get("function") == "protocol_snapshot"]
    assert set(step["params"]) == {
        "study_id",
        "revision",
        "parent_revision",
        "protocol_hash",
        "schema",
    }
    assert step["params"]["study_id"] == "tutorial_cohort" and step["params"]["revision"] == 1
    assert "mask_families" not in json.dumps(step)  # never the body


def test_the_dossier_names_the_study_definition() -> None:
    from types import SimpleNamespace

    from ui.components.study import StudyStatus, status_text

    latest = SimpleNamespace(meta=SimpleNamespace(revision=3))
    assert (
        status_text(StudyStatus("tutorial_cohort", latest, {}, "saved"))
        == "tutorial_cohort, revision 3"
    )
    assert status_text(StudyStatus("tutorial_cohort", latest, {}, "changed")) == (
        "tutorial_cohort, changed since revision 3 (not saved)"
    )
    assert status_text(StudyStatus("s", None, {}, "none")) == "s, no revision saved"
    assert status_text(None) == "n/a"


def test_the_trace_report_names_a_revision_by_its_number() -> None:
    from manage.trace_documenter import revision_params

    shown = revision_params(
        {
            "study_id": "tutorial_cohort",
            "revision": 2,
            "protocol_hash": "a" * 64,
            "parent_revision": 1,
            "schema": 1,
        }
    )
    assert shown == {"study": "tutorial_cohort", "revision": 2, "follows": "revision 1"}
    assert (
        revision_params({"study_id": "s", "revision": 1})["follows"] == "none (the first revision)"
    )


def test_a_change_of_the_privacy_choices_alone_is_said_and_saved_as_the_next_revision(
    app: AppTest, tmp_path: Path
) -> None:
    """The study's privacy choices select no row: a change of them reads as its own state (it read "its descriptions
    or notes changed", which understates a change of k), and is the next revision."""
    from core.privacy import StudyPrivacy

    app.button(key=f"{KEY}_save").click().run()
    _choose(app, StudyPrivacy(k=20))
    _dossier(app)
    shown = _texts(app)
    assert "privacy choices changed since revision 1" in shown
    assert (
        "Changed since revision 1: the privacy choices (the definition selects the same rows)."
        in shown
    )
    assert "descriptions or notes" not in shown
    assert _buttons(app)[f"{KEY}_save"] == "Save revision 2"
    app.button(key=f"{KEY}_save").click().run()
    _ok(app)
    assert "revision 2, saved" in _texts(app) and len(_revisions(tmp_path)) == 2
    first, second = (json.loads(f.read_text(encoding="utf-8")) for f in _revisions(tmp_path))
    assert second["body"]["privacy"] == {"k": 20} and "privacy" not in first["body"]
    assert second["meta"]["parent"] == first["checksum"]
    # the trace names the revision with the same fingerprint: nothing selects differently
    steps = [
        s
        for s in app.session_state["transformation_manager"].history
        if s.get("function") == "protocol_snapshot"
    ]
    assert [s["params"]["revision"] for s in steps] == [1, 2]
    assert steps[0]["params"]["protocol_hash"] == steps[1]["params"]["protocol_hash"]
    assert "privacy" not in json.dumps(steps)
    # the configuration never holds them: it stays for the next dataset, and is written to files
    assert "privacy" not in app.session_state["config"]
    # a description corrected with them: both are said
    _choose(app, StudyPrivacy(k=30))
    config = copy.deepcopy(app.session_state["config"])
    config["mask_families"]["clinical_anomalies"]["Implausible age"]["description"] = "Checked"
    app.session_state["config"] = config
    _dossier(app)
    shown = _texts(app)
    assert "labels and privacy choices changed since revision 2" in shown
    assert "Changed since revision 2: descriptions or notes, and the privacy choices" in shown
    # with another tick, the choices are named among what changed
    app.switch_page(nav.page_path("start")).run()
    app.checkbox(key=_family_key("clinical_anomalies")).check().run()
    _dossier(app)
    assert "Changed since revision 2: which criteria are ticked and the privacy choices." in _texts(
        app
    )


def test_a_study_s_privacy_choices_do_not_follow_the_user_into_another_study(
    app: AppTest, tmp_path: Path
) -> None:
    """The configuration stays for the next dataset loaded in a session, criteria included, by design. A study's
    privacy choices were kept in it: a mouse study's declaration that its rows are not people would have been saved
    with the first revision of the cohort of people loaded after it."""
    from dataclasses import replace

    from core.privacy import StudyPrivacy

    def saved(study: str) -> list[dict]:
        files = [json.loads(f.read_text(encoding="utf-8")) for f in _revisions(tmp_path)]
        return [one for one in files if one["meta"]["study_id"] == study]

    _choose(app, StudyPrivacy(k=20, rows_not_people="animal_data"))
    _dossier(app)
    app.button(key=f"{KEY}_save").click().run()
    _ok(app)
    assert saved("tutorial_cohort")[-1]["body"]["privacy"] == {
        "k": 20,
        "rows_not_people": "animal_data",
    }
    # a dataset of another study: its definition holds the criteria carried over, and no privacy choice
    app.session_state["dataset_ref"] = replace(
        app.session_state["dataset_ref"], name="other_cohort"
    )
    _dossier(app)
    shown = _texts(app)
    assert "other_cohort" in shown and "no revision saved" in shown
    assert "privacy choices" not in shown
    app.button(key=f"{KEY}_save").click().run()
    _ok(app)
    (other,) = saved("other_cohort")
    assert "privacy" not in other["body"] and other["body"]["mask_families"]
    assert (
        "privacy" in saved("tutorial_cohort")[-1]["body"]
    )  # the first study's revision keeps them
    # loading a dataset drops the choices of another study, and keeps those of its own
    for study, kept in (("another_study", False), ("tutorial_cohort", True)):
        app.session_state["_study_privacy"] = {"study": study, "section": {"k": 20}}
        app.switch_page(nav.page_path("start")).run()
        app.button(key="start_tutorial").click().run()
        _ok(app)
        assert ("_study_privacy" in app.session_state) is kept, study


def test_the_dossier_names_a_change_of_the_privacy_choices() -> None:
    from types import SimpleNamespace

    from ui.components.study import LABELS, PRIVACY, StudyStatus, status_text

    latest = SimpleNamespace(meta=SimpleNamespace(revision=3))
    assert status_text(StudyStatus("s", latest, {}, "privacy", (PRIVACY,))) == (
        "s, revision 3, its privacy choices changed since (not saved)"
    )
    assert status_text(StudyStatus("s", latest, {}, "labels", (LABELS, PRIVACY))) == (
        "s, revision 3, its descriptions or notes and its privacy choices changed since (not saved)"
    )
    assert status_text(StudyStatus("s", latest, {}, "labels", (LABELS,))) == (
        "s, revision 3, its descriptions or notes changed since (not saved)"
    )


def test_a_session_without_the_revision_s_privacy_choices_says_what_a_save_would_drop(
    app: AppTest,
) -> None:
    """Back on a study whose revision holds privacy choices, in a session that holds none (a new session, or one
    that dropped them with another study's dataset), the strip read "Save them as the next revision": the save that
    removes the k the study asked for."""
    from core.privacy import StudyPrivacy

    _choose(app, StudyPrivacy(k=20))
    _dossier(app)
    app.button(key=f"{KEY}_save").click().run()
    _ok(app)
    del app.session_state["_study_privacy"]
    _dossier(app)
    shown = _texts(app)
    assert "privacy choices changed since revision 1" in shown
    assert (
        "Changed since revision 1: this session holds no privacy choices, and revision 1 holds some (the "
        "definition selects the same rows). A revision saved now would not hold them: load revision 1 to take "
        "them back."
    ) in shown
    assert "Save them as the next revision" not in shown
    assert _buttons(app) == {
        f"{KEY}_load": "Load revision 1",
        f"{KEY}_save": "Save revision 2",
        f"{KEY}_delete": "Delete revision 1",
    }
    app.button(key=f"{KEY}_load").click().run()
    _ok(app)
    assert app.session_state["_study_privacy"]["section"] == {"k": 20}
    assert "revision 1, saved" in _texts(app)
    # with another change of the definition, the warning stands without Load, which would undo that change
    del app.session_state["_study_privacy"]
    app.switch_page(nav.page_path("start")).run()
    app.checkbox(key=_family_key("clinical_anomalies")).check().run()
    _dossier(app)
    shown = _texts(app)
    assert (
        "This session holds no privacy choices, and revision 1 holds some: a revision saved now would not hold "
        "them."
    ) in shown
    assert f"{KEY}_load" not in _buttons(app)
    # choices changed in the session, not dropped: the ordinary words
    app.switch_page(nav.page_path("start")).run()
    app.checkbox(key=_family_key("clinical_anomalies")).uncheck().run()
    _choose(app, StudyPrivacy(k=30))
    _dossier(app)
    shown = _texts(app)
    assert "Save them as the next revision." in shown and "holds no privacy choices" not in shown


def test_three_parts_that_changed_are_listed_with_commas() -> None:
    """ "the criteria and the follow-up and the privacy choices" stood for three parts."""
    from ui.components.study import _listed

    assert _listed(("the criteria",)) == "the criteria"
    assert (
        _listed(("the criteria", "the privacy choices")) == "the criteria and the privacy choices"
    )
    assert _listed(("the criteria", "the follow-up", "the privacy choices")) == (
        "the criteria, the follow-up and the privacy choices"
    )
