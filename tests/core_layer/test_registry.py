"""The shared kit of the extension points: a catalogue of kinds addressed by stored ids (core/registry.py)."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from core.registry import Catalogue, UnknownIdError


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
