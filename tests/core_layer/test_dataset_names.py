"""
A dataset's name from its file name (core.state.dataset_base_name): it names the dataset's saved versions and, since
M8, its study (core.protocol.study_id_for), so accents are folded rather than dropped.
"""

from __future__ import annotations

import pytest

from core.protocol import study_id_for
from core.state import dataset_base_name


@pytest.mark.parametrize(
    ("filename", "name", "study"),
    [
        ("My Cohort (2024).xlsx", "My_Cohort_2024", "my_cohort_2024"),
        ("Étude Cohorte 2024.csv", "Etude_Cohorte_2024", "etude_cohorte_2024"),
        ("données.csv", "donnees", "donnees"),
        (
            "Ærø ﬁle.csv",
            "r_file",
            "r_file",
        ),  # a letter with no ASCII form goes; a ligature is folded
        (
            "Данные.csv",
            "dataset",
            "dataset",
        ),  # nothing of its own: the fallback (AppState adds the file's checksum)
    ],
)
def test_a_name_folds_its_accents(filename: str, name: str, study: str) -> None:
    assert dataset_base_name(filename) == name
    assert study_id_for(name) == study
