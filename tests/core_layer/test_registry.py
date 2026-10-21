"""The shared kit of the extension points: a catalogue of kinds addressed by stored ids (core/registry.py)."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from core.registry import (
    Availability,
    Catalogue,
    ConnectorRegistry,
    UnknownConnectorError,
    UnknownIdError,
    as_code,
)


@dataclass(frozen=True)
class Colour:
    id: str
    label: str


def _colours() -> Catalogue[Colour]:
    colours: Catalogue[Colour] = Catalogue("colour")
    colours.register(Colour("red", "Red"), aliases=("rouge",))
    colours.register(Colour("green", "Green"))
    colours.register(Colour("mauve", "Mauve"), legacy=True)
    return colours


def test_kinds_are_found_by_id_or_alias_in_registration_order() -> None:
    colours = _colours()
    assert colours.get("red").label == "Red" and colours.get("rouge").label == "Red"
    assert [c.id for c in colours] == ["red", "green", "mauve"] and len(colours) == 3
    assert "green" in colours and "rouge" in colours and "blue" not in colours and 3 not in colours
    assert colours.find("blue") is None


def test_a_legacy_kind_still_reads_but_is_not_offered() -> None:
    colours = _colours()
    assert colours.get("mauve").label == "Mauve" and colours.is_legacy("mauve")
    assert not colours.is_legacy("red") and not colours.is_legacy("rouge")
    assert colours.ids() == ("red", "green", "mauve")
    assert colours.ids(legacy=False) == ("red", "green")


def test_an_unknown_id_names_the_valid_ones() -> None:
    with pytest.raises(UnknownIdError) as raised:
        _colours().get("blue")
    error = raised.value
    assert isinstance(error, LookupError)
    assert (error.catalogue, error.unknown, error.valid) == ("colour", "blue", ("red", "green"))
    assert str(error) == "unknown colour 'blue': the valid ids are red, green"


@pytest.mark.parametrize(
    "kind, aliases",
    [
        (Colour("red", "Again"), ()),
        (Colour("rouge", "Clash with an alias"), ()),
        (Colour("blue", "Blue"), ("green",)),
        (Colour("blue", "Blue"), ("bleu", "bleu")),
    ],
)
def test_an_id_or_alias_is_registered_once(kind: Colour, aliases: tuple[str, ...]) -> None:
    colours = _colours()
    with pytest.raises(ValueError, match="already registered"):
        colours.register(kind, aliases=aliases)
    assert "blue" not in colours and "bleu" not in colours  # nothing was half-registered


# --- connector families (M5b step 4) --------------------------------------------------------------------------------

FAKE = """
from core.registry import Availability


class Good:
    name = "good"
    display_name = "Good"

    @classmethod
    def availability(cls):
        return Availability(True)


class Down:
    name = "down"
    display_name = "Down"

    @classmethod
    def availability(cls):
        return Availability(False, "library too old")


class Boom:
    name = "boom"
    display_name = "Boom"

    @classmethod
    def availability(cls):
        raise RuntimeError("zz9 secret in a broken check")
"""


@pytest.fixture
def fake(tmp_path, monkeypatch: pytest.MonkeyPatch) -> str:
    """Two fake connector modules on the path: one that loads, one whose package is not installed."""
    import sys

    module = f"fake_connectors_{tmp_path.name.replace('-', '_')}"
    (tmp_path / f"{module}.py").write_text(FAKE, encoding="utf-8")
    (tmp_path / f"{module}_extra.py").write_text("import not_installed_pkg_zz\n", encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    yield module
    for name in (module, f"{module}_extra"):
        sys.modules.pop(name, None)


def _registry(module: str) -> ConnectorRegistry:
    registry: ConnectorRegistry = ConnectorRegistry(
        "graph engine", setting="CMP_GRAPH_RENDERER", auto=("good",)
    )
    registry.register("good", f"{module}:Good", aliases=("fine",))
    registry.register("down", f"{module}:Down", extra="graph-yfiles")
    registry.register("boom", f"{module}:Boom")
    registry.register("ext", f"{module}_extra:X", extra="ai")
    return registry


def test_the_names_import_nothing_and_a_class_is_imported_once(fake: str) -> None:
    import sys

    registry = _registry(fake)
    assert registry.names() == ("good", "down", "boom", "ext") and registry.valid()[0] == "auto"
    assert fake not in sys.modules
    assert registry.get("FINE").name == "good" and fake in sys.modules  # an alias, in any case
    assert registry.get("good") is registry.get("fine")
    with pytest.raises(UnknownConnectorError, match="use good, down, boom or ext") as refused:
        registry.get("nope-zz9")
    assert (
        "zz9" not in str(refused.value) and refused.value.unknown is None
    )  # a setting's value is never repeated
    assert registry.target("FINE") == f"{fake}:Good"


def test_availability_never_raises_and_names_the_group_to_install(fake: str) -> None:
    registry = _registry(fake)
    missing = registry.availability("ext")
    assert not missing.ok and "not_installed_pkg_zz is not installed" in missing.reason
    assert missing.extra == "ai" and missing.install_hint == "uv sync --extra ai"
    broken = registry.availability("boom")
    assert not broken.ok and "RuntimeError" in broken.reason and "zz9" not in broken.reason
    # the class's own refusal names its group itself: installing may not help (a server that does not answer)
    assert registry.availability("down") == Availability(False, "library too old")
    assert registry.availability("good").ok and registry.availability("good").install_hint is None
    assert not registry.availability("cohere").ok  # unknown: not offered
    registry.register("typo", f"{fake}_mistyped:Good", extra="ai")
    typo = registry.availability("typo")  # the registration names no module: nothing to install
    assert not typo.ok and "registration error" in typo.reason and typo.install_hint is None


def test_a_connector_that_cannot_load_is_shown_by_its_name(fake: str) -> None:
    registry = _registry(fake)
    assert registry.display_name("FINE") == "Good"
    assert registry.display_name("ext") == "ext"  # its module needs a package that is not installed
    assert (
        registry.display_name("co[here](http://e.x)") == "`co[here](http://e.x)`"
    )  # unknown: as code


def test_a_module_is_found_without_being_imported(
    fake: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    import sys

    assert Availability.of_module(fake).ok and fake not in sys.modules
    absent = Availability.of_module("not_installed_pkg_zz", extra="ai")
    assert not absent.ok and absent.install_hint == "uv sync --extra ai"
    assert not Availability.of_module(
        "not_installed_pkg_zz.sub"
    ).ok  # its parent is looked up first
    monkeypatch.setitem(sys.modules, fake, None)  # blocked, as the contract tests block an SDK
    assert not Availability.of_module(fake).ok


def test_the_choice_follows_the_session_the_preference_the_setting_then_auto(fake: str) -> None:
    registry = _registry(fake)
    assert registry.select(session="good", preference="down", setting="down").source == "session"
    chosen = registry.select(preference="down", setting="good")
    assert (chosen.chosen, chosen.source) == ("good", "setting")
    assert chosen.warnings == (
        "Your saved graph engine `down` is not available here (library too old)",
    )
    assert chosen.reasons == ("the server's setting (CMP_GRAPH_RENDERER)",)
    stale = registry.select(preference="cohere")
    assert (stale.chosen, stale.source) == ("good", "auto")
    assert stale.warnings == ("Your saved graph engine `cohere` is not offered here",)
    assert stale.reasons == ("the first available of good",)
    # "auto" at a level is no choice there: the next level decides (the session's "automatic" = the server's default)
    deferred = registry.select(session="auto", preference="fine", setting="down")
    assert (deferred.chosen, deferred.source, deferred.reasons) == (
        "good",
        "preference",
        ("your saved choice",),
    )
    assert registry.select(session="good").reasons == ("chosen for this session",)
    recommended = registry.select(auto="down", auto_reasons=("recommended here",))
    assert (
        recommended.chosen == "good"
    )  # the recommendation cannot run here: the family's order, and why
    assert recommended.warnings == (
        "The recommended graph engine `down` is not available here (library too old)",
    )
    assert dict(recommended.alternatives)["ext"].extra == "ai"
    failing = registry.select(
        session="good", check=lambda name: 1 / 0
    )  # a family's rule that fails: a reason
    assert failing.chosen is None and "its check failed (ZeroDivisionError)" in failing.warnings[0]


def test_a_setting_naming_nothing_is_reported_without_its_value(fake: str) -> None:
    registry = _registry(fake)
    problem = registry.setting_problem("qdrnt-zz9")
    assert problem == "CMP_GRAPH_RENDERER is not a graph engine: use auto, good, down, boom or ext"
    assert registry.setting_problem("FINE") is None and registry.setting_problem(" Auto ") is None
    assert registry.setting_problem("") is None  # unset
    with pytest.raises(ValueError, match="use auto, good"):
        registry.select(setting="nope-zz9")


def test_a_family_without_auto(fake: str) -> None:
    registry: ConnectorRegistry = ConnectorRegistry(
        "identity provider", setting="CMP_IAM_PROVIDER", auto=None
    )
    registry.register("local", f"{fake}:Good", aliases=("default",))
    assert registry.valid() == ("local",)
    assert (
        registry.setting_problem("auto")
        == "CMP_IAM_PROVIDER is not an identity provider: use local"
    )
    assert registry.select(setting="DEFAULT").chosen == "local"
    assert registry.select().chosen is None  # nothing named: no automatic choice


@pytest.mark.parametrize(
    ("name", "target"),
    [("good", "x:Y"), ("Upper", "x:Y"), ("auto", "x:Y"), ("other", "module_without_class")],
)
def test_a_registration_is_checked(fake: str, name: str, target: str) -> None:
    registry = _registry(fake)
    with pytest.raises(ValueError):
        registry.register(name, target)


def test_a_saved_name_is_shown_as_code() -> None:
    assert (
        as_code("x`y\n z") == "`xy z`" and as_code("[link](http://e.x)") == "`[link](http://e.x)`"
    )
    # bidirectional overrides and zero-width characters cannot reorder or hide part of the sentence
    assert as_code("ab‮cd​ef") == "`abcdef`"
    assert as_code("```") == "`?`"  # nothing left to show
    long = as_code("x" * 300)
    assert len(long) == 66 and long.endswith("…`")  # cut at 64 characters
