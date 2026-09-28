"""Automated checks: statistics match an independent pandas calculation, and the
program degrades gracefully (no numeric columns, no categorical columns, LLM off/unavailable).

Run from the repository root:  python -m pytest -q
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from profiler import generate_profile, verify_llm_text  # noqa: E402


def _run(tmp_path, df, name, **kw):
    csv = tmp_path / f"{name}.csv"
    df.to_csv(csv, index=False)
    out = generate_profile(csv, tmp_path / "out", dataset_name=name, **kw)
    return out, json.loads((out / "analysis_summary.json").read_text())


def test_numeric_stats_match_pandas(tmp_path):
    rng = np.random.default_rng(0)
    x = rng.normal(50, 10, 300)
    df = pd.DataFrame({"x": x, "y": x * 2 + rng.normal(0, 1, 300), "grp": rng.choice(["a", "b"], 300)})
    out, s = _run(tmp_path, df, "num", use_llm=False)
    st = [n for n in s["numeric_statistics"] if n["column"] == "x"][0]
    q1, q3 = np.percentile(x, [25, 75])
    iqr = q3 - q1
    assert abs(st["mean"] - x.mean()) < 1e-3
    assert abs(st["median"] - np.median(x)) < 1e-3
    assert abs(st["std"] - x.std(ddof=1)) < 1e-3
    assert st["outlier_count"] == int(((x < q1 - 1.5 * iqr) | (x > q3 + 1.5 * iqr)).sum())
    for f in ("report.md", "column_profile.csv", "analysis_summary.json", "llm_prompt.txt", "llm_response.txt"):
        assert (out / f).exists()
    assert len(list((out / "plots").glob("plot_*.png"))) >= 5


def test_no_numeric_columns(tmp_path):
    df = pd.DataFrame({"a": ["x", "y", "z"] * 10, "b": ["p", "q", "r"] * 10})
    out, s = _run(tmp_path, df, "cat", use_llm=False)
    assert s["numeric_statistics"] == []
    assert "no numeric measure columns" in (out / "report.md").read_text()


def test_no_categorical_columns(tmp_path):
    df = pd.DataFrame({"a": np.arange(40) * 1.5, "b": np.arange(40) ** 0.5})
    out, s = _run(tmp_path, df, "numonly", use_llm=False)
    assert s["categorical_statistics"] == []
    assert s["relationships"]["performed"]


def test_llm_unavailable_still_reports(tmp_path):
    df = pd.DataFrame({"a": np.arange(30), "b": ["x", "y", "z"] * 10})
    out, s = _run(tmp_path, df, "down", use_llm=True, config={"ollama_url": "http://127.0.0.1:9/api/generate",
                                                              "llm_timeout_sec": 2})
    assert s["llm"]["status"] == "skipped"
    assert "skipped" in (out / "report.md").read_text()


def test_verifier_flags_invented_numbers_mislabeled_stats_and_causation(tmp_path):
    rng = np.random.default_rng(1)
    df = pd.DataFrame({"x": rng.normal(50, 10, 200), "y": rng.normal(0, 1, 200),
                       "district": rng.choice([41, 1, 45], 200).astype(float)})
    _, s = _run(tmp_path, df, "verify", use_llm=False)
    mean_x = [n for n in s["numeric_statistics"] if n["column"] == "x"][0]["mean"]
    text = (f"1. `x` has a mean of {mean_x}.\n"
            "2. `x` causes `y`, with 99.9% missing.\n"
            "3. `district` has a mean of 41.0.\n")
    checks = verify_llm_text(text, s)
    assert checks[0]["all_numbers_verified"] is True and not checks[0]["statistic_issues"]
    assert checks[1]["all_numbers_verified"] is False
    assert "causes" in checks[1]["causal_language"]
    # district is a code (categorical), so Python computed no mean: must be flagged
    assert any("no mean" in i for i in checks[2]["statistic_issues"])
