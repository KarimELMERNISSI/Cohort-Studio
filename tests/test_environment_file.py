"""
The conda environment (environments/environment.yml) installs the locked packages of requirements.txt.

It kept a list of its own, which drifted from pyproject.toml: it still installed ChromaDB and three LangChain
packages that the application no longer uses, and none of the vector databases (LanceDB, Qdrant), the OpenAI,
Anthropic and Ollama (httpx) connectors, pyarrow, xlrd or lifelines.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
ENVIRONMENT = ROOT / "environments" / "environment.yml"


def test_the_conda_environment_installs_the_locked_requirements() -> None:
    env = yaml.safe_load(ENVIRONMENT.read_text(encoding="utf-8"))
    conda = [d for d in env["dependencies"] if isinstance(d, str)]
    pip = [d["pip"] for d in env["dependencies"] if isinstance(d, dict)]
    assert env["channels"] == ["conda-forge", "nodefaults"]  # never the user's defaults channel
    assert pip == [["-r ../requirements.txt"]]  # conda runs pip from the environment file's folder
    assert (ENVIRONMENT.parent / ".." / "requirements.txt").resolve().is_file()
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    assert project["requires-python"] == ">=3.14" and sorted(conda) == ["pip", "python=3.14"]
