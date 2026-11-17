"""
A study's privacy choices (the section ``privacy`` of its definition, ``core.protocol``), and the two numbers every
privacy rule starts from.

A study definition may hold what the study's owner asked for the documents made from it: the smallest count they
give (``k``), the outcomes declared followed to the event (each with the override of the cap when it was asked for),
the declaration that the study's rows are not people, and the word the documents use for the rows. These are what was
*asked*. What applies is decided when an output is made, from the installation's floor and from what its
administrator allows: nothing here reads a setting, and an administrator's later restriction changes no study file.

The section is a file format. Its keys and its closed values are stored ids, never renamed. It holds nothing typed
by a person: whole numbers, ``true``, ids of closed lists, and the names of the outcomes' variables; no reason in
words, no date, no role and no account (a revision says when it was saved, and whose workspace holds it). One
function writes it (:meth:`StudyPrivacy.to_section`), in one order, and writes no default, so that equal choices give
one checksum and an unchanged choice never makes a revision.

As written in a study's body::

    "privacy": {"k": 20,
                "followed_to_the_event": {"death": {}, "visits": {"beyond_the_cap": true}},
                "rows_not_people": "animal_data",
                "rows_word": "animals"}

Of the four, ``k`` and ``followed_to_the_event`` are what a package's policy is made from
(``core.disclosure.DisclosurePolicy.for_study``, which the package's writer calls with them and with the
installation's floor; no output reads the section before that writer exists). ``rows_not_people`` and ``rows_word``
are read, kept when a revision is saved, and applied by nothing in this version: the rules stay on and the documents
say "participants", which is the stricter reading, and a study file written by a version that applies them opens
here.
A key or a closed value this version does not know refuses the whole file as written by a later version; a value it
cannot read refuses it too, never ignored (a ``k`` of 20 written "20" and ignored would apply 10).

The section never travels: a shared study definition and the configuration files leave it out, so that whoever opens
a study inherits nothing of another person's declarations; the package's policy records what applied.
"""

from __future__ import annotations

import unicodedata
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, TypeGuard

PRIVACY_KEY = "privacy"  # the section's key in a study definition's body

DEFAULT_K = 10  # the smallest count a shared document gives, when a study asks none
FLOOR_K = 5  # nothing lower is accepted, whatever a study or an installation asks

# the section's keys, in the order written
K = "k"
FOLLOWED = "followed_to_the_event"
NOT_PEOPLE = "rows_not_people"
ROWS_WORD = "rows_word"
SECTION_KEYS = (K, FOLLOWED, NOT_PEOPLE, ROWS_WORD)
BEYOND_THE_CAP = (
    "beyond_the_cap"  # in a declaration: the override of the cap, written only when asked for
)
OUTCOME_MAX = 256  # characters of an outcome's name

# Why a study's rows are not people: the stored id, and the words an output writes. Animal data first. Not reasons,
# and never here: records of people with their names removed or coded (pseudonymised data are personal data),
# records called "anonymised", synthetic data made from real records, data kept on one machine, a recipient with an
# agreement, public data about people.
NOT_PEOPLE_REASONS: dict[str, str] = {
    "animal_data": "animal data: every row is an animal",
    "simulated_data": "simulated data, made by a program, not from records of real people",
    "tutorial_cohort": "the tutorial cohort shipped with CohortStudio",
    "teaching_set": "a teaching set made up for the purpose",
}
# The word the documents use for a study's rows. "participants" is the default: it is never written, and a word
# changes no rule (only the declaration does).
ROWS_DEFAULT = "participants"
ROWS_WORDS = ("animals", "subjects")

# why a section is refused, in words that quote nothing of the file
NOT_READ = "its privacy section cannot be read by this version"
LATER_VALUE = (
    "written by a later version: its privacy section holds a value this version does not know"
)
BELOW_THE_FLOOR = (
    f"its privacy section asks for a k below {FLOOR_K}, which CohortStudio never applies"
)


def _later_keys(count: int) -> str:
    """Counted, never named: a key of a file could hold anything."""
    return (
        f"written by a later version: its privacy section holds {count} key{'s' if count != 1 else ''} this version "
        "does not read"
    )


def _is_count(value: Any) -> TypeGuard[int]:
    """A whole number as JSON writes one: not a yes or no, not a decimal (20.0), not a text."""
    return isinstance(value, int) and not isinstance(value, bool)


def _is_outcome(value: Any) -> bool:
    """A variable's name: a text of 1 to ``OUTCOME_MAX`` characters that is not blank and holds no control or
    format character."""
    return (
        isinstance(value, str)
        and 0 < len(value) <= OUTCOME_MAX
        and bool(value.strip())
        and not any(unicodedata.category(c) in ("Cc", "Cf") for c in value)
    )


def _as_written(section: Mapping[str, Any]) -> list[tuple[str, Any]]:
    """A section's keys and values in the order written, those of the mappings inside it too (two mappings are
    equal whatever their order)."""
    return [
        (key, _as_written(value) if isinstance(value, Mapping) else value)
        for key, value in section.items()
    ]


@dataclass(frozen=True, order=True)
class Declared:
    """An outcome the study declares followed to the event (``core.disclosure.FollowedToTheEvent`` says what the
    declaration means); ``beyond_the_cap`` is the override of the cap, asked for this outcome by an act of its own."""

    outcome: str
    beyond_the_cap: bool = False

    def __post_init__(self) -> None:
        if not _is_outcome(self.outcome):
            raise ValueError("a declaration names its outcome, a variable's name")
        if not isinstance(self.beyond_the_cap, bool):
            raise ValueError("the override of a declaration's cap is asked for, or not")


@dataclass(frozen=True)
class StudyPrivacy:
    """
    What a study asks for its documents; every choice is optional, and none set is the study that asks nothing.

    ``k``: the smallest count its documents give (None: the default of the version that makes the output; a ``k`` of
    10 is the study asking 10, which is not the same). ``followed_to_the_event``: its declarations, kept sorted by
    outcome, one per outcome. ``rows_not_people``: the reason, an id of ``NOT_PEOPLE_REASONS``, of the declaration
    that its rows are not people. ``rows_word``: ``animals`` or ``subjects`` (None, or ``participants``: the
    default).
    """

    k: int | None = None
    followed_to_the_event: tuple[Declared, ...] = ()
    rows_not_people: str | None = None
    rows_word: str | None = None

    def __post_init__(self) -> None:
        if self.k is not None:
            if not _is_count(self.k):
                raise ValueError("k is a whole number")
            if self.k < FLOOR_K:
                raise ValueError(f"k is never below {FLOOR_K}")
        declared = tuple(self.followed_to_the_event)
        if not all(isinstance(one, Declared) for one in declared):
            raise ValueError("an outcome is declared by a Declared")
        if len({one.outcome for one in declared}) != len(declared):
            raise ValueError("an outcome is declared once")
        object.__setattr__(self, "followed_to_the_event", tuple(sorted(declared)))
        if self.rows_not_people is not None and self.rows_not_people not in NOT_PEOPLE_REASONS:
            raise ValueError("the rows are declared not people for a reason of the closed list")
        if self.rows_word == ROWS_DEFAULT:
            object.__setattr__(self, "rows_word", None)  # the default has one writing: none
        if self.rows_word is not None and self.rows_word not in ROWS_WORDS:
            raise ValueError("the word for the rows is one of the closed list")

    @property
    def word(self) -> str:
        """The word the study chose for its rows ("participants" when it chose none)."""
        return self.rows_word or ROWS_DEFAULT

    def __bool__(self) -> bool:
        return bool(self.to_section())

    def to_section(self) -> dict[str, Any]:
        """The section as a study's body holds it: the keys in their order, the outcomes sorted, no default written;
        empty when the study asks nothing (the section is then not written at all)."""
        section: dict[str, Any] = {}
        if self.k is not None:
            section[K] = self.k
        if self.followed_to_the_event:
            section[FOLLOWED] = {
                one.outcome: ({BEYOND_THE_CAP: True} if one.beyond_the_cap else {})
                for one in self.followed_to_the_event
            }
        if self.rows_not_people is not None:
            section[NOT_PEOPLE] = self.rows_not_people
        if self.rows_word is not None:
            section[ROWS_WORD] = self.rows_word
        return section

    @classmethod
    def from_section(cls, section: Any) -> tuple[StudyPrivacy | None, list[str]]:
        """
        The choices a section holds, or none with why it is refused. The reasons quote nothing of the file: no key,
        no value and no outcome. Nothing here reads a setting: a ``k`` looser than an installation's floor is read
        as asked, and the floor is applied where an output is made.
        """
        if section is None:
            return cls(), []
        if not isinstance(section, Mapping):
            return None, [NOT_READ]
        later = len(set(section) - set(SECTION_KEYS))
        if later:
            return None, [_later_keys(later)]
        k = section.get(K)
        if K in section:
            if not _is_count(k):
                return None, [NOT_READ]
            if k < FLOOR_K:
                return None, [BELOW_THE_FLOOR]
        declared: list[Declared] = []
        if FOLLOWED in section:
            entries = section[FOLLOWED]
            if not isinstance(entries, Mapping) or not entries:
                return None, [NOT_READ]  # an empty object is another writing of "none declared"
            for outcome, entry in entries.items():
                if not _is_outcome(outcome) or not isinstance(entry, Mapping):
                    return None, [NOT_READ]
                later = len(set(entry) - {BEYOND_THE_CAP})
                if later:
                    return None, [_later_keys(later)]
                # written only when asked for: false is another writing of its absence
                if BEYOND_THE_CAP in entry and entry[BEYOND_THE_CAP] is not True:
                    return None, [NOT_READ]
                declared.append(Declared(outcome, beyond_the_cap=BEYOND_THE_CAP in entry))
        reason = section.get(NOT_PEOPLE)
        if NOT_PEOPLE in section:
            if not isinstance(reason, str):
                return None, [NOT_READ]
            if reason not in NOT_PEOPLE_REASONS:
                return None, [LATER_VALUE]
        word = section.get(ROWS_WORD)
        if ROWS_WORD in section:
            # the default is never written: "participants" there is another writing of its absence
            if not isinstance(word, str) or word == ROWS_DEFAULT:
                return None, [NOT_READ]
            if word not in ROWS_WORDS:
                return None, [LATER_VALUE]
        choices = cls(
            k=k, followed_to_the_event=tuple(declared), rows_not_people=reason, rows_word=word
        )
        # one writing per meaning, the order included: the same choices with their keys or their outcomes in another
        # order have another checksum, and saved again unchanged they would read as changed and make a revision
        if _as_written(section) != _as_written(choices.to_section()):
            return None, [NOT_READ]
        return choices, []


def study_privacy(body: Mapping[str, Any] | None) -> StudyPrivacy:
    """The privacy choices of a study definition's body (none set when it holds no section). ValueError, with the
    reason, for a section this version cannot read."""
    choices, reasons = StudyPrivacy.from_section((body or {}).get(PRIVACY_KEY))
    if choices is None:
        raise ValueError(reasons[0])
    return choices


def with_privacy(config: Mapping[str, Any] | None, choices: StudyPrivacy) -> dict[str, Any]:
    """A configuration holding these choices as its privacy section, written by the one writer; without the section
    when nothing is set. The other keys are kept as they are."""
    out = {key: value for key, value in (config or {}).items() if key != PRIVACY_KEY}
    section = choices.to_section()
    if section:
        out[PRIVACY_KEY] = section
    return out


__all__ = [
    "BELOW_THE_FLOOR",
    "BEYOND_THE_CAP",
    "DEFAULT_K",
    "FLOOR_K",
    "FOLLOWED",
    "K",
    "LATER_VALUE",
    "NOT_PEOPLE",
    "NOT_PEOPLE_REASONS",
    "NOT_READ",
    "OUTCOME_MAX",
    "PRIVACY_KEY",
    "ROWS_DEFAULT",
    "ROWS_WORD",
    "ROWS_WORDS",
    "SECTION_KEYS",
    "Declared",
    "StudyPrivacy",
    "study_privacy",
    "with_privacy",
]
