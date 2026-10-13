"""The product is called CohortStudio everywhere users see it (the former name must not come back)."""

import tomllib
from pathlib import Path

from ui.theme import APP_NAME, APP_TAGLINE

ROOT = Path(__file__).resolve().parents[2]
FORMER = ("Cohort Monitoring", "cohort-monitoring", "cohort_monitoring")


def test_name_and_package() -> None:
    assert APP_NAME == "CohortStudio" and APP_TAGLINE
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    assert project["name"] == "cohortstudio" and project["description"].startswith(APP_NAME)


def test_former_name_is_gone_from_the_application() -> None:
    files = [
        ROOT / "main.py",
        ROOT / "ui" / "styles.css",
        ROOT / "docker-compose.yml",
        ROOT / "Dockerfile",
    ]
    files += sorted((ROOT / "ui").rglob("*.py")) + sorted((ROOT / "app_pages").rglob("*.py"))
    offenders = [
        str(f.relative_to(ROOT))
        for f in files
        if "__pycache__" not in f.parts
        and any(old in f.read_text(encoding="utf-8") for old in FORMER)
    ]
    assert not offenders, offenders
