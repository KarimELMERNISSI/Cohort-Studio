"""What the AI features send about a dataset: a summary of each column, never a row or an extreme value (M5)."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from manage.ai_payload import (
    MIN_COUNT,
    TRACING_VARIABLES,
    column_profile,
    disable_remote_tracing,
    document_name,
    frequent_values,
    safe_stats,
    taxonomy_for_prompt,
)


@pytest.mark.parametrize(
    ("source", "name"),
    [
        (r"C:\data\users\alice\documents\Smith_J_discharge.pdf", "Smith_J_discharge.pdf"),
        ("/srv/cohortstudio/data/users/alice/documents/guide.pdf", "guide.pdf"),  # read on any system
        ("guide.pdf", "guide.pdf"),
        ("C:\\data\\users\\alice\\", "Unknown"),  # a folder is never named
        ("", "Unknown"),
        (None, "Unknown"),
    ],
)
def test_a_passage_names_its_document_never_its_path(source: str | None, name: str) -> None:
    """A prompt names a passage's document by the file's name: the path names the user's account and the server's
    folders."""
    assert document_name({"source": source, "page": 2}) == name
    assert document_name(None) == "Unknown"


def _cohort() -> pd.DataFrame:
    rng = np.random.default_rng(7)
    n = 200
    df = pd.DataFrame(
        {
            "creatinine": rng.normal(90, 20, n).round(1),
            "nyha": rng.choice(["I", "II", "III"], n),
            "rare_note": ["common"] * (n - 2) + ["Mrs Dupont, room 12", "unique entry"],
            "visit": pd.date_range("2019-03-01", periods=n, freq="7D"),
            "visit_text": [d.strftime("%d/%m/%Y") for d in pd.date_range("2020-01-01", periods=n, freq="3D")],
            "smoker": rng.integers(0, 2, n),
        }
    )
    df.loc[17, "creatinine"] = 912.0  # one extreme value
    return df


def test_no_row_and_no_extreme_value_is_sent() -> None:
    df = _cohort()
    sent = json.dumps(column_profile(df, date_columns=["visit_text"]))
    assert "912" not in sent  # the maximum of creatinine singles out one participant
    assert "Mrs Dupont" not in sent and "unique entry" not in sent  # values of one row
    for value in df.iloc[3].astype(str):
        if value not in ("I", "II", "III", "0", "1", "common"):  # frequent values may be sent
            assert value not in sent, value


def test_each_kind_of_column_is_summarised() -> None:
    profile = {p["name"]: p for p in column_profile(_cohort(), date_columns=["visit_text"])}
    assert profile["creatinine"]["type"] == "Numeric"
    assert set(profile["creatinine"]["stats"]) == {"p5", "p25", "p50", "p75", "p95"}
    assert profile["nyha"]["type"] == "Categorical" and set(profile["nyha"]["top_values"]) == {"I", "II", "III"}
    assert all(n >= MIN_COUNT for n in profile["rare_note"]["top_values"].values())
    assert profile["visit"]["range"] == "2019 to 2022" and profile["visit_text"]["range"] == "2020 to 2021"
    assert profile["smoker"]["type"] == "Categorical"  # 0/1: described by its values


def test_a_column_of_names_codes_or_dates_on_visits_sends_no_value() -> None:
    """The summary sent the five most frequent names, participant codes and birth dates of a table of visits, each held
    by 5 rows or more, where Data > Clean reads the same columns as identifiers and sends nothing. The column keeps its
    name and type; categories keep their values."""
    rng = np.random.default_rng(7)
    visits = rng.integers(1, 9, size=40)  # 40 participants, 1 to 8 visits each
    people = np.repeat(np.arange(40), visits)
    df = pd.DataFrame(
        {
            "name": [f"Participant {chr(65 + p % 26)}{p}" for p in people],
            "code": [f"P-{p:04d}" for p in people],
            "birth": [f"19{50 + p:02d}-01-01" for p in people],
            "sex": np.where(people % 2 == 0, "F", "M"),
            "site": np.where(people % 3 == 0, "Esch", "Luxembourg"),
        }
    )
    profile = {p["name"]: p for p in column_profile(df)}
    for column in ("name", "code", "birth"):
        assert profile[column]["top_values"] == {}, column
        assert profile[column]["type"] == "Categorical"
    assert set(profile["sex"]["top_values"]) == {"F", "M"}
    assert set(profile["site"]["top_values"]) == {"Esch", "Luxembourg"}
    sent = json.dumps(column_profile(df))
    assert "P-00" not in sent and "Participant" not in sent and "-01-01" not in sent


def _born(people: int, visits: np.ndarray) -> pd.DataFrame:
    """Birth dates of ``people`` 397 days apart from 1930, each on its rows (``visits``), as numbers: yyyymmdd and days
    since 1960 (SAS; an Excel serial counts days too)."""
    born = pd.date_range("1930-01-01", periods=people, freq="397D")
    days = (born - pd.Timestamp("1960-01-01")).days.to_numpy()
    return pd.DataFrame(
        {
            "dob": np.repeat(born.strftime("%Y%m%d").astype(int), visits),
            "birth_sas": np.repeat(days, visits),
            "value_since_1960": np.repeat(days, visits),  # a name that does not say it
        }
    )


def test_a_birth_date_held_as_a_number_never_leaves_through_its_percentiles() -> None:
    """On visits, one person's rows sit side by side once sorted: 4 of the 5 percentiles of a birth date written
    yyyymmdd, 40 people on 1 to 8 visits, were participants' own birth dates, and those of a count of days the same."""
    rng = np.random.default_rng(1)
    df = _born(40, rng.integers(1, 9, 40))
    df["sbp"] = rng.normal(132, 18, len(df)).round(1)
    profile = {p["name"]: p for p in column_profile(df)}
    assert profile["dob"] == {"name": "dob", "type": "Date", "range": "1930 to 1972"}  # read as a date
    assert profile["birth_sas"] == {"name": "birth_sas", "type": "Numeric"}  # named as one
    # a count of days whose name does not say so: rounded to one step, two digits of the largest (near 1960 too)
    stats = profile["value_since_1960"]["stats"]
    assert stats == {"p5": -10000.0, "p25": -7000.0, "p50": -3000.0, "p75": 0.0, "p95": 4000.0}
    assert not set(stats.values()) & set(df["value_since_1960"].astype(float))
    assert all(v % 10 == 0 for v in profile["sbp"]["stats"].values())  # 100 to 160: the scale and the unit
    few = {p["name"]: p for p in column_profile(_born(8, np.full(8, 6)))}  # 10 values or fewer
    assert few["dob"]["type"] == "Date" and few["birth_sas"] == {"name": "birth_sas", "type": "Numeric"}
    death = column_profile(pd.DataFrame({"death": [0, 1] * 30}))  # a flag, whatever its name
    assert death[0]["top_values"] == {"0": 30, "1": 30}


def test_names_on_the_visits_of_30_people_or_fewer_are_counted_by_their_rows() -> None:
    """What the notice says: rows are counted, not people, until the dataset says which column names a participant.
    A name on the 6 visits of each of 25 people is a value of 6 rows; with more than 30 people, none is sent."""
    (few,) = column_profile(pd.DataFrame({"name": np.repeat([f"Person {i:02d}" for i in range(25)], 6)}))
    assert len(few["top_values"]) == 5 and set(few["top_values"].values()) == {6}
    (many,) = column_profile(pd.DataFrame({"name": np.repeat([f"Person {i:02d}" for i in range(31)], 6)}))
    assert many["top_values"] == {}


def test_statistics_outside_the_notice_are_dropped() -> None:
    # an imported or edited variable map may hold anything: only the percentiles are kept (an allow-list)
    assert safe_stats({"min": 31.5, "max": 912.0, "mean": 88.4}) == {}
    assert safe_stats({"minimum": 18, "maximum": 97}) == {} and safe_stats({"0%": 18, "100%": 97}) == {}
    kept = safe_stats({"p5": 60.0, "p50": 88.0, "p95": 130.0, "max": 912.0})
    assert kept == {"p5": 60.0, "p50": 90.0, "p95": 130.0}  # rounded as column_profile rounds them
    assert safe_stats({"p50": "88", "p95": True}) == {}  # a percentile that is no number
    assert safe_stats("min 31.5, max 912") == {} and safe_stats(None) == {}
    # the frequent values of column_profile: values shared by fewer than 5 rows are not sent
    assert frequent_values({"II": 80, "III": 40, "Mrs Dupont": 1}) == {"II": 80, "III": 40}
    taxonomy = {
        "creatinine": {"unit": "umol/L", "stats": {"min": 31.5, "p50": 88.0}, "range": "31.5 to 912"},
        "nyha": {"description": "class", "top_values": {"Mrs Dupont": 1}, "levels": ["Mrs Dupont"]},
        "note": "min 18, max 97",
    }
    # only the keys that describe a variable (an allow-list), and only mappings
    assert taxonomy_for_prompt(taxonomy) == {
        "creatinine": {"unit": "umol/L", "stats": {"p50": 88.0}},
        "nyha": {"description": "class"},
    }
    assert taxonomy["creatinine"]["stats"]["min"] == 31.5  # the user's map itself is unchanged


@pytest.mark.parametrize("variable", TRACING_VARIABLES)
def test_remote_tracing_of_the_prompts_is_switched_off(monkeypatch, variable: str) -> None:
    # a LangSmith switch left in .env would send every prompt to a tracing service
    import os

    for name in TRACING_VARIABLES:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv(variable, "true")
    assert disable_remote_tracing() == [variable]
    assert all(os.environ[name] == "false" for name in TRACING_VARIABLES)
    try:
        from langsmith import utils as langsmith_utils
    except ImportError:  # no AI stack: nothing could trace
        return
    monkeypatch.setenv("LANGSMITH_TRACING_V2", "true")  # read first by LangSmith: its global switch still wins
    langsmith_utils.get_env_var.cache_clear()
    assert not langsmith_utils.tracing_is_enabled()


@pytest.mark.parametrize(
    ("url", "found"),
    [
        ("https://abc.eu-central.aws.cloud.qdrant.io:6333", "qdrant@abc.eu-central.aws.cloud.qdrant.io"),
        ("http://qdrant:6333", None),  # a Docker service on the private network
        ("http://localhost:6333", None),
        ("http://192.168.1.20:6333", None),
        (None, None),  # embedded in the workspace
    ],
)
def test_a_qdrant_server_outside_the_network_is_a_recipient(url: str | None, found: str | None) -> None:
    from manage.ai_payload import provider_label, store_recipient

    assert store_recipient(url) == found
    if found:
        assert provider_label(found) == "Qdrant at abc.eu-central.aws.cloud.qdrant.io"


@pytest.mark.parametrize(
    "url",
    [
        "http://evil.example.com" + chr(92) + "@localhost:6333",  # a backslash: the client goes to evil.example.com
        "http://134744072:6333",  # 8.8.8.8, as the resolver reads it
    ],
)
def test_an_address_is_read_as_the_client_connects_to_it(url: str) -> None:
    """Each passed a public server for a local one, so an Ollama or a Qdrant server there skipped the confirmation."""
    from manage.ai_payload import outside_private_network, recipient, store_recipient

    assert outside_private_network(url)
    assert store_recipient(url) is not None and recipient("ollama", url, None) is not None


def test_the_vector_database_named_here_is_the_registry_s() -> None:
    from manage.ai_payload import QDRANT
    from manage.vectorstores.base import VectorEngine

    assert VectorEngine.QDRANT.value == QDRANT


def test_an_ollama_model_run_elsewhere_is_a_recipient_whatever_its_name() -> None:
    """A copy of a cloud model (ollama cp gpt-oss:120b-cloud gpt-oss:120b) reads local by its name: the server says
    where it runs it. Behind an Ollama outside the network, both receive the prompts."""
    from manage.ai_payload import recipient, server_recipient

    here = "http://localhost:11434"
    assert recipient("ollama", here, "gpt-oss:120b") is None
    assert recipient("ollama", here, "gpt-oss:120b", remote_host="https://ollama.com:443") == "ollama@ollama.com"
    assert recipient("ollama", here, "x" * 90 + "-cloud") == "ollama@ollama.com"  # the whole name decides
    outside = "https://ollama.example.com"
    assert recipient("ollama", outside, "glm-4.6:cloud") == "ollama@ollama.com"
    assert server_recipient("ollama", outside) == "ollama@ollama.example.com"
    assert server_recipient("ollama", here) is None and server_recipient("anthropic", None) is None


def test_the_ollama_server_says_which_models_it_runs_elsewhere(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("httpx", reason="the ai extra")
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    monkeypatch.undo()  # the suite's stand-in answer (tests/conftest.py): the server's own here
    from manage.rag.providers.ollama_server import NoAnswer, remote_host, remote_models, runs_at

    class Handler(BaseHTTPRequestHandler):
        def _answer(self, body: object) -> None:
            data = json.dumps(body).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self) -> None:
            self._answer({"models": [{"name": "llama3.2"}, {"name": "gpt-oss:120b", "remote_host": "https://ollama.com:443"}]})

        def do_POST(self) -> None:
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            remote = body.get("model") == "gpt-oss:120b"
            self._answer({"remote_host": "https://ollama.com:443", "remote_model": "gpt-oss:120b"} if remote else {})

        def log_message(self, *args) -> None:
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        url = f"http://127.0.0.1:{server.server_address[1]}"
        assert remote_host(url, "gpt-oss:120b") == "https://ollama.com:443"
        assert remote_host(url, "llama3.2") is None and remote_host(url, None) is None
        assert remote_models(url) == {"gpt-oss:120b": "https://ollama.com:443"}
        assert runs_at(url, "gpt-oss:120b") == "https://ollama.com:443" and runs_at(url, "llama3.2") is None
    finally:
        server.shutdown()
    assert remote_host("http://127.0.0.1:9", "gpt-oss:120b") is None  # no server: nothing known for Settings
    with pytest.raises(NoAnswer) as silence:  # and a decision on what is sent never reads a silence as "it runs here"
        runs_at("http://127.0.0.1:9", "gpt-oss:120b")
    from manage.rag.providers.ollama_server import UNREACHABLE

    assert silence.value.reason == UNREACHABLE  # nothing listens there: Settings says "did not answer at"


def test_names_with_near_names_on_visits_are_still_an_identifier() -> None:
    """A name misspelt once (a near-name) counts as that name, as a category's typos do: the names of 40 people on 6
    visits, five of them misspelt once, still read as an identifier, and none is sent."""
    names = [f"Person {chr(65 + i // 26)}{chr(97 + i % 26)}" for i in range(40)]
    rows = [name for name in names for _ in range(6)]
    rows += [name.replace("Person", "Persn") for name in names[:5]]  # one typo each, on one more visit
    (info,) = column_profile(pd.DataFrame({"name": rows}))
    assert info["top_values"] == {}
