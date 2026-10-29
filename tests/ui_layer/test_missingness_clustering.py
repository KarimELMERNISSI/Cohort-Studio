"""
The missingness correlation of Validate's dashboard orders the columns by a hierarchical clustering. It showed "Could
not cluster variables" instead: SciPy 1.18 no longer lends `scipy.spatial.distance` to `scipy.cluster.hierarchy`, and,
with "Filter Complete Columns" unticked, a column whose missingness never varies put a 1 on the distance diagonal.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import scipy.cluster.hierarchy as sch
from streamlit.testing.v1 import AppTest

import utils.data_paths as dp
from app_pages.data_quality_dashboard import missingness_linkage
from ui import nav

MAIN = str(Path(__file__).resolve().parents[2] / "main.py")


def _nullity_corr() -> pd.DataFrame:
    """Two columns missing together, two missing apart, one complete and one empty."""
    rng = np.random.default_rng(7)
    together = rng.random(200) < 0.3
    frame = pd.DataFrame(
        {
            "weight": np.where(together, np.nan, 70.0),
            "height": np.where(together, np.nan, 1.7),
            "sbp": np.where(rng.random(200) < 0.2, np.nan, 130.0),
            "ldl": np.where(rng.random(200) < 0.4, np.nan, 3.1),
            "age": 60.0,
            "note": np.nan,
        }
    )
    return frame.isnull().astype(int).corr()


def test_columns_missing_together_are_neighbours() -> None:
    corr = _nullity_corr()
    linkage = missingness_linkage(corr)
    assert sch.is_valid_linkage(linkage)
    order = list(corr.columns[sch.leaves_list(linkage)])
    assert abs(order.index("weight") - order.index("height")) == 1
    # the first merge joins the two at distance 0
    assert {corr.columns[int(i)] for i in linkage[0, :2]} == {"weight", "height"}
    assert linkage[0, 2] == pytest.approx(0)


def test_a_column_whose_missingness_never_varies_is_clustered_too() -> None:
    corr = _nullity_corr()
    assert corr["age"].isna().all() and corr["note"].isna().all()
    linkage = missingness_linkage(corr)
    assert sorted(sch.leaves_list(linkage)) == list(range(len(corr)))
    # the matrix it was given is left as it was
    assert np.isnan(corr.loc["age", "age"])


@pytest.fixture(scope="module")
def app(tmp_path_factory: pytest.TempPathFactory):
    previous = dp.DATA_ROOT
    dp.DATA_ROOT = str(tmp_path_factory.mktemp("data"))
    os.environ.setdefault("CMP_IAM_PROVIDER", "local")
    at = AppTest.from_file(MAIN, default_timeout=240)
    at.run()
    at.session_state["user_authenticated"] = True
    at.session_state["username"] = "clustering_user"
    at.session_state["user_role"] = "researcher"
    at.run()
    at.switch_page(nav.page_path("start")).run()
    at.button(key="start_sample").click().run()
    assert not at.exception, [str(e.value)[:300] for e in at.exception]
    yield at
    dp.DATA_ROOT = previous


def test_the_dashboard_clusters_and_draws_the_dendrogram(app: AppTest) -> None:
    """The sample cohort has 2 columns with missing values: unticking the filter keeps its 17 complete ones."""
    app.switch_page(nav.page_path("validate")).run()
    app.checkbox(key="dq_nullity_filter_complete").uncheck()
    app.checkbox(key="dq_nullity_dendrogram").check()
    app.run()
    assert not app.exception, [str(e.value)[:300] for e in app.exception]
    shown = [w.value for w in app.warning] + [e.value for e in app.error]
    assert not [message for message in shown if "Could not cluster" in message], shown
    titles = [chart.proto.spec for chart in app.get("plotly_chart")]
    assert any("Hierarchical Clustering Dendrogram" in spec for spec in titles)
