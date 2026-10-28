"""Per-user workspace: directory isolation and path containment (audit E4)."""

import os
import random
import unicodedata
from pathlib import Path

import pytest

import utils.data_paths as dp
from core.workspace import (
    UserScope,
    all_user_scopes,
    earlier_folders,
    move_earlier_workspace,
    planned_move,
    remove_pdf,
    resolve_within,
    safe_dirname,
    safe_filename,
    validate_username,
    workspace_exists,
)


@pytest.fixture
def root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(dp, "DATA_ROOT", str(tmp_path))
    return tmp_path


def test_username_validation() -> None:
    assert validate_username(" dr.lee_2 ") == "dr.lee_2"
    for bad in ("ab", "../x", "name with space", "", None, "x" * 40):
        with pytest.raises(ValueError):
            validate_username(bad)


def test_a_folder_name_never_leaves_the_users_folder() -> None:
    assert safe_dirname("admin") == "admin"
    assert safe_dirname("Dr Lee/ops") == "_Dr_x20_Lee_x2f_ops"
    assert safe_dirname("../../etc") == "_.._x2f_.._x2f_etc"
    for name in ("..", ".", "./", "a/b", "a\\b", "C:x", "nul", "CON.txt", "drlee.", " lead"):
        folder = safe_dirname(name)
        assert "/" not in folder and "\\" not in folder and ":" not in folder, name
        assert folder not in (".", "..") and not folder.endswith("."), name
    for bad in ("", "  ", None):  # no user
        with pytest.raises(ValueError):
            safe_dirname(bad)


def _decoded(folder: str) -> str:
    """The name a coded folder stands for: what makes the code one name's only."""
    assert folder.startswith("_")
    out, rest = [], folder[1:]
    while rest:
        if rest.startswith("__"):
            out.append("_")
            rest = rest[2:]
        elif rest.startswith("_at_"):
            out.append("@")
            rest = rest[4:]
        elif rest.startswith("_x"):
            code, rest = rest[2:].split("_", 1)
            out.append(chr(int(code, 16)))
        else:
            out.append(rest[0])
            rest = rest[1:]
    return "".join(out)


def test_no_two_names_share_a_folder() -> None:
    """
    Earlier versions replaced each run of characters a folder name cannot hold by "_": "j.smith@chl.lu" (an identity
    provider's e-mail) and "j.smith_chl.lu" (an account created in the application) had one workspace. A name that is
    a folder name keeps it; any other gets "_" and a readable code of it, which no folder name and no other name has.
    """
    long_mail = "firstname.middlename.lastname.department@hospital-group.example.org"  # its code is too long
    names = [
        "j.smith@chl.lu",
        "j.smith_chl.lu",
        "j.smith_at_chl.lu",
        "_j.smith_at_chl.lu",
        "o'neil@x.org",
        "o_neil@x.org",
        "a+b@x.org",
        "a_b@x.org",
        "José",
        "Josè",
        "Jos",
        "Мария1",
        "Иван1",
        "1",
        "drlee.",
        "drlee..",
        "drlee",
        "-admin",
        "admin",
        "nul",
        "_nul",
        "Мария",
        "Мариам",
        "..",
        "./",
        "_",
        long_mail,
        long_mail.upper(),
    ]
    folders = [safe_dirname(name) for name in names]
    assert len(set(folders)) == len(names), sorted(folders)  # one folder per name
    assert safe_dirname("j.smith@chl.lu") == "_j.smith_at_chl.lu"
    assert safe_dirname("o_neil@x.org") == "_o__neil_at_x.org"
    for name, folder in zip(names, folders, strict=True):
        if folder == name:  # a folder name already: an account created in the application
            assert not folder.startswith("_")
        elif len(folder) == 33 and folder.count("_") == 1:  # a code too long: the hash
            assert len(name) > 30
        else:
            assert _decoded(folder) == name
        assert len(folder) <= 64
    assert safe_dirname(" Мария ") == safe_dirname("Мария")  # the same name, the same folder
    assert UserScope.for_user("Мария").username == safe_dirname("Мария")


def test_a_coded_name_is_never_normalised_into_another_names_folder() -> None:
    """Two spellings an identity provider may tell apart keep two folders: at worst one person's files are split,
    never two people's merged."""
    composed = "Ёлка"  # Ё as one character, or as Е and a combining diaeresis
    assert safe_dirname(unicodedata.normalize("NFD", composed)) != safe_dirname(composed)
    assert safe_dirname("ａｌｉｃｅ") != safe_dirname("alice")  # full-width letters
    assert safe_dirname("МАРИЯ") != safe_dirname("Мария")
    assert safe_dirname("José") != safe_dirname(unicodedata.normalize("NFD", "José"))


def _earlier(root: Path, folder: str) -> Path:
    path = root / "users" / folder / "datasets" / "cohort_v1.parquet"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"rows")
    return path


def _never_asked() -> list[str]:
    raise AssertionError("the accounts are listed only when there is a folder to move")


def test_an_earlier_folder_is_moved_once_when_it_is_clearly_the_accounts(root: Path) -> None:
    _earlier(root, "j.smith_chl.lu")
    moved, outcome = move_earlier_workspace("j.smith@chl.lu", lambda: ["alice", "j.smith@chl.lu"])
    assert moved and "moved" in outcome
    there = root / "users" / "_j.smith_at_chl.lu" / "datasets" / "cohort_v1.parquet"
    assert there.read_bytes() == b"rows" and not (root / "users" / "j.smith_chl.lu").exists()
    assert move_earlier_workspace("j.smith@chl.lu", _never_asked) == (False, "")  # once


def test_an_earlier_folder_another_account_could_own_is_left_where_it_is(root: Path) -> None:
    earlier = _earlier(root, "j.smith_chl.lu")
    for accounts in (
        ["j.smith@chl.lu", "j.smith_chl.lu"],  # the account whose folder it is now
        ["j.smith@chl.lu", "j.smith!chl.lu"],  # another account an earlier version gave it
        ["alice"],  # a list without this account: a provider that failed returns none
        [],
    ):
        moved, outcome = move_earlier_workspace(
            "j.smith@chl.lu", lambda accounts=accounts: accounts
        )
        assert not moved and outcome and earlier.exists(), accounts
    assert not (root / "users" / "_j.smith_at_chl.lu").exists()


def test_the_accounts_are_compared_by_folder_as_the_disk_compares_them(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """On Windows and macOS two names differing only by case are one folder: such an account could own it too."""
    earlier = _earlier(root, "j.smith_chl.lu")
    monkeypatch.setattr("core.workspace.os.path.normcase", str.lower)  # as on Windows
    moved, _ = move_earlier_workspace(
        "j.smith@chl.lu", lambda: ["j.smith@chl.lu", "J.Smith_chl.lu"]
    )
    assert not moved and earlier.exists()


def test_a_folder_that_holds_files_is_never_replaced(root: Path) -> None:
    _earlier(root, "j.smith_chl.lu")
    mine = _earlier(root, "_j.smith_at_chl.lu")
    moved, outcome = move_earlier_workspace("j.smith@chl.lu", _never_asked)
    assert not moved and "holds files" in outcome  # the log names the folder left
    assert mine.exists() and (root / "users" / "j.smith_chl.lu").exists()


def test_several_earlier_folders_are_left_for_an_administrator(root: Path) -> None:
    trimmed = _earlier(root, "j.smith_chl.lu")
    hashed = _earlier(root, earlier_folders("j.smith@chl.lu")[1])  # a development version's folder
    moved, outcome = move_earlier_workspace("j.smith@chl.lu", _never_asked)
    assert not moved and "several" in outcome and trimmed.exists() and hashed.exists()


def test_a_link_is_never_moved(root: Path) -> None:
    target = root / "elsewhere"
    target.mkdir()
    link = root / "users" / "j.smith_chl.lu"
    link.parent.mkdir(parents=True)
    try:
        if os.name == "nt":
            import _winapi  # ty: ignore[unresolved-import]

            _winapi.CreateJunction(str(target), str(link))
        else:
            link.symlink_to(target, target_is_directory=True)
        moved, outcome = move_earlier_workspace("j.smith@chl.lu", _never_asked)
        assert not moved and "link" in outcome
        assert not (root / "users" / "_j.smith_at_chl.lu").exists()
    finally:
        if os.name == "nt":
            os.rmdir(link)  # the junction only
        else:
            link.unlink()


def test_a_folder_created_empty_before_the_move_is_replaced(root: Path) -> None:
    _earlier(root, "drlee")
    # created by this visit, before the sign-in moved anything
    (root / "users" / "_drlee_x2e_" / "traces" / "snapshots").mkdir(parents=True)
    moved, _ = move_earlier_workspace("drlee.", lambda: ["drlee."])
    assert moved and (root / "users" / "_drlee_x2e_" / "datasets" / "cohort_v1.parquet").exists()


def test_a_name_that_is_its_own_folder_has_nothing_to_move(root: Path) -> None:
    _earlier(root, "alice")
    assert move_earlier_workspace("alice", _never_asked) == (False, "")
    assert "alice" not in earlier_folders("alice")  # its folder now is never one to move away


def test_planned_moves_name_the_folder_and_move_nothing(root: Path) -> None:
    earlier = _earlier(root, "j.smith_chl.lu")
    folder, reason = planned_move("j.smith@chl.lu", ["j.smith@chl.lu"])
    assert folder is not None and folder.name == "j.smith_chl.lu" and reason == ""
    assert earlier.exists()  # a plan only


def test_no_two_of_many_random_names_share_a_folder() -> None:
    """Names drawn from the characters that make codes: letters of both cases, digits, "_", "@", ".", "-", a space,
    an apostrophe, letters of other scripts; the folders differ unless the names do."""
    rng = random.Random(20261028)
    alphabet = "aAbZ09_@.- 'éЖ_x"
    names = {
        "".join(rng.choice(alphabet) for _ in range(rng.randint(1, 12))).strip()
        for _ in range(20000)
    }
    names.discard("")
    folders = {safe_dirname(name): name for name in names}
    assert len(folders) == len(names)


def test_a_new_account_never_takes_a_folder_that_exists(root: Path) -> None:
    _earlier(root, "j.smith_chl.lu")  # the earlier folder of "j.smith@chl.lu", until it signs in
    assert workspace_exists("j.smith_chl.lu") and not workspace_exists("someone.else")
    assert not workspace_exists("") and not workspace_exists(None)


def test_safe_filename() -> None:
    assert safe_filename("../../x.parquet") == "x.parquet"
    assert safe_filename("my file (v2).csv") == "my file (v2).csv"
    assert safe_filename("...") == "file"
    assert safe_filename(None, default="dataset") == "dataset"


def test_scope_directories_are_per_user(root: Path) -> None:
    alice = UserScope.for_user("alice")
    bob = UserScope.for_user("bob", role="admin")
    assert alice.datasets_dir() == root / "users" / "alice" / "datasets"
    assert bob.datasets_dir() == root / "users" / "bob" / "datasets"
    assert alice.artifacts_dir("session_1").parent == alice.traces_dir() / "artifacts"
    assert alice.snapshots_dir().exists() and alice.vectors_dir().exists()
    assert bob.is_admin and not alice.is_admin
    assert {s.username for s in all_user_scopes()} == {"alice", "bob"}


def test_resolve_within_blocks_escapes(root: Path) -> None:
    alice = UserScope.for_user("alice")
    inside = alice.uploads_dir() / "data.csv"
    inside.write_text("x")
    assert alice.resolve_within("data.csv", "uploads") == inside.resolve()
    assert alice.resolve_within(str(inside)) == inside.resolve()
    with pytest.raises(ValueError):
        alice.resolve_within("../bob/datasets/secret.parquet", "uploads")
    with pytest.raises(ValueError):
        alice.resolve_within(str(root / "users" / "bob" / "datasets" / "secret.parquet"))
    with pytest.raises(ValueError):
        resolve_within(
            alice.root, "C:/Windows/system32" if str(root).startswith("C:") else "/etc/passwd"
        )
    with pytest.raises(ValueError):
        alice.dir("nope")


def test_a_sibling_directory_sharing_the_prefix_is_outside(tmp_path: Path) -> None:
    # a prefix check must stop at a separator: alice2 is not inside alice
    (tmp_path / "alice").mkdir()
    (tmp_path / "alice2").mkdir()
    with pytest.raises(ValueError):
        resolve_within(tmp_path / "alice", tmp_path / "alice2" / "secret.csv")
    with pytest.raises(ValueError):
        resolve_within(tmp_path / "alice", "../alice2/secret.csv")
    assert resolve_within(tmp_path / "alice", ".") == (tmp_path / "alice").resolve()
    assert (
        resolve_within(tmp_path / "alice", "a/../b.csv") == (tmp_path / "alice" / "b.csv").resolve()
    )


def test_only_a_pdf_of_the_library_can_be_removed(root: Path) -> None:
    library = UserScope.for_user("alice").documents_dir()
    (library / "paper.pdf").write_bytes(b"%PDF-1.4")
    (library / "notes.txt").write_text("x")
    (root / "outside.pdf").write_bytes(b"%PDF-1.4")
    assert remove_pdf(library, "paper.pdf") and not (library / "paper.pdf").exists()
    assert not remove_pdf(library, "paper.pdf")  # already gone
    assert not remove_pdf(library, "notes.txt") and (library / "notes.txt").exists()  # not a PDF
    with pytest.raises(ValueError):
        remove_pdf(library, "../../../outside.pdf")
    assert (root / "outside.pdf").exists()


def test_the_administrators_script_plans_then_moves_only_what_is_clear(root: Path) -> None:
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "move_workspaces",
        Path(__file__).resolve().parents[2] / "dev_scripts" / "move_workspaces.py",
    )
    assert spec is not None and spec.loader is not None
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    _earlier(root, "j.smith_chl.lu")
    shared = _earlier(root, "o_neil_x.org")  # the earlier folder of two accounts
    names = ["j.smith@chl.lu", "o'neil@x.org", "o_neil@x.org", "alice"]
    rows = {account: verdict for account, _, verdict in script.plan(names)}
    assert rows["j.smith@chl.lu"] == "move"
    assert "administrator" in rows["o'neil@x.org"] and "administrator" in rows["o_neil@x.org"]
    assert (root / "users" / "j.smith_chl.lu").exists()  # a plan moves nothing
    assert len(script.apply(names)) == 1
    assert (root / "users" / "_j.smith_at_chl.lu" / "datasets" / "cohort_v1.parquet").exists()
    assert shared.exists()


def test_a_link_where_the_folder_goes_is_never_followed(root: Path) -> None:
    _earlier(root, "drlee")
    outside = root / "outside"
    (outside / "empty").mkdir(parents=True)
    link = root / "users" / "_drlee_x2e_"
    try:
        if os.name == "nt":
            import _winapi  # ty: ignore[unresolved-import]

            _winapi.CreateJunction(str(outside), str(link))
        else:
            link.symlink_to(outside, target_is_directory=True)
        moved, outcome = move_earlier_workspace("drlee.", lambda: ["drlee."])
        assert not moved and "link" in outcome
        assert (outside / "empty").exists() and (root / "users" / "drlee").exists()
    finally:
        if os.name == "nt":
            os.rmdir(link)  # the junction only
        else:
            link.unlink()


def test_the_script_counts_the_name_before_an_at_sign_as_an_owner(root: Path) -> None:
    """Better Auth lists an account that signs in as "j.smith_chl.lu" under "j.smith_chl.lu@local.cohort" when it has
    no username field: that account could own the folder "j.smith_chl.lu", which then stays."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "move_workspaces",
        Path(__file__).resolve().parents[2] / "dev_scripts" / "move_workspaces.py",
    )
    assert spec is not None and spec.loader is not None
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    current = _earlier(root, "j.smith_chl.lu")
    names = ["j.smith@chl.lu", "j.smith_chl.lu@local.cohort"]
    rows = {account: verdict for account, _, verdict in script.plan(names)}
    assert "administrator" in rows["j.smith@chl.lu"]
    assert script.apply(names) == [] and current.exists()
