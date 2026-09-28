"""
Automated CSV Profiler and Verified Insight Generator
CSCI 5502 - Assignment 2: From CSV to Evidence

Architecture:
    CSV file -> Python analysis -> verified summary (JSON) -> LLM explanation -> generated report

Python computes every statistic. The LLM (optional, via a local Ollama server)
only explains the verified summary. Every number in the LLM response is then
checked automatically against the Python summary.

Entry point:
    generate_profile(csv_path, output_dir, use_llm=True, ...)

Command line:
    python src/profiler.py data/my_file.csv --name dataset_a
    python src/profiler.py data/my_file.csv --name dataset_a --no-llm
"""

from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import sys
import textwrap
import urllib.error
import urllib.request
import warnings
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # render to files, no window needed
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

warnings.filterwarnings("ignore", category=UserWarning)

# ---------------------------------------------------------------------------
# Configuration (documented in README). Change these, never the analysis logic.
# ---------------------------------------------------------------------------
DEFAULT_CONFIG = {
    # A column is flagged as "high missingness" when >= this % of values are missing.
    # 30% is a common rule of thumb: above it, simple imputation becomes unreliable
    # and any analysis of that column rests on less than ~70% of the rows.
    "missing_threshold_pct": 30.0,
    # Share of non-missing values that must parse as numbers / dates for a text
    # column to be treated as numeric / date-like.
    "numeric_parse_threshold": 0.95,
    "date_parse_threshold": 0.90,
    # A column whose unique values cover >= this share of non-missing rows is "high cardinality".
    "high_cardinality_ratio": 0.90,
    # Text columns with at most this many distinct values are treated as categorical.
    "max_categories": 50,
    # Average length (characters) above which a mostly-unique text column is "free text".
    "free_text_min_avg_len": 30,
    # How many categories to list per categorical column.
    "top_k_categories": 10,
    # Plot budget.
    "max_plots": 8,
    # Extra strings to treat as missing (e.g. ["-999", "unknown", "not reported"]).
    "extra_na_values": [],
    # Outlier rule multiplier (1.5 x IQR is required by the assignment).
    "iqr_multiplier": 1.5,
    # Ollama settings.
    "llm_model": "llama3.2",
    "ollama_url": "http://localhost:11434/api/generate",
    "llm_timeout_sec": 300,
    # Ollama's default context window is small; longer prompts are silently cut from the START,
    # which would drop the instructions. 8192 tokens fits the summary comfortably.
    "llm_num_ctx": 8192,
    # Retry once if the reply is not a numbered list of insights.
    "llm_max_attempts": 2,
}

ROLE_NUMERIC = "Numeric measure"
ROLE_CATEGORICAL = "Categorical attribute"
ROLE_BOOLEAN = "Boolean field"
ROLE_DATE = "Date-like field"
ROLE_ID = "Identifier-like field"
ROLE_TEXT = "Free-text field"
ROLE_UNKNOWN = "Unknown or mixed type"

BOOL_TOKENS = {"true", "false", "yes", "no", "y", "n", "t", "f", "0", "1", "1.0", "0.0"}
CODE_NAME_PATTERN = re.compile(r"(zip|postal|fips|geoid|phone|census|tract|district|precinct|ward|block_?group|(^|[_\s])code$|_cd$)", re.I)
ID_NAME_PATTERN = re.compile(r"(^id$|_id$|^id_|\bid\b|identifier|uuid|guid|key$|_no$|number$|^record)", re.I)

SENSITIVE_NAME_PATTERNS = {
    "person name": r"(^|_|\b)(first|last|full)?_?name($|_|\b)|surname",
    "email": r"e-?mail",
    "phone": r"phone|mobile|cell|fax",
    "government ID": r"ssn|social_?security|passport|driver|license|licence|tax_?id|ein",
    "date of birth / age": r"dob|birth|(^|_)age($|_)",
    "street address": r"address|street|addr",
    "postal code": r"zip|postal|postcode",
    "precise location": r"(^|_)(lat|latitude|lon|lng|longitude)($|_)|coordinates|geolocation",
    "health / medical": r"diagnos|patient|medical|health|icd",
    "financial": r"salary|income|account|credit|bank|iban",
    "demographic": r"race|ethnic|gender|sex($|_)|religion",
    "network identifier": r"(^|_)ip(_|$)|ip_?address|mac_?address",
}
SENSITIVE_VALUE_PATTERNS = {
    "email": r"^[\w.+-]+@[\w-]+\.[\w.-]+$",
    "US phone number": r"^\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}$",
    "SSN-like": r"^\d{3}-\d{2}-\d{4}$",
    "IP address": r"^\d{1,3}(\.\d{1,3}){3}$",
}

# Chart palette (validated categorical/sequential/diverging values, light theme).
C_BLUE = "#2a78d6"
C_ORANGE = "#eb6834"
C_RED = "#e34948"
C_INK = "#0b0b0b"
C_INK2 = "#52514e"
C_MUTED = "#898781"
C_GRID = "#e1e0d9"
C_AXIS = "#c3c2b7"
C_SURFACE = "#fcfcfb"
C_NEUTRAL = "#f0efec"

# Horizontal boxplots: Matplotlib >= 3.10 uses orientation=, older versions use vert=
_MPL = tuple(int(p) for p in matplotlib.__version__.split(".")[:2])
HBOX = {"orientation": "horizontal"} if _MPL >= (3, 10) else {"vert": False}
CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def _r(x, nd=4):
    """Round to a JSON-safe Python float (or None)."""
    if x is None:
        return None
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    if math.isnan(x) or math.isinf(x):
        return None
    return round(x, nd)


def _pct(part, whole, nd=2):
    return round(100.0 * part / whole, nd) if whole else 0.0


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return _r(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, (pd.Timestamp, datetime)):
        return o.isoformat()
    return str(o)


def _safe_name(s):
    return re.sub(r"[^A-Za-z0-9_-]+", "_", str(s)).strip("_") or "dataset"


def _short(label, n=28):
    label = str(label)
    return label if len(label) <= n else label[: n - 1] + "…"


def _md_escape(s):
    return str(s).replace("|", "\\|").replace("\n", " ")


# ---------------------------------------------------------------------------
# 1. Loading
# ---------------------------------------------------------------------------
def load_csv(csv_path, cfg):
    """Load a single-table CSV without changing any values."""
    na_values = list(cfg.get("extra_na_values") or [])
    last_err = None
    for enc in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            df = pd.read_csv(csv_path, low_memory=False, encoding=enc,
                             na_values=na_values or None, keep_default_na=True)
            return df, enc
        except UnicodeDecodeError as e:
            last_err = e
    raise last_err


# ---------------------------------------------------------------------------
# 2. Type / role inference
# ---------------------------------------------------------------------------
def _looks_like_date_strings(s: pd.Series) -> bool:
    """Cheap pre-check so plain words/numbers are not sent to the date parser."""
    sample = s.astype(str).head(200)
    has_sep = sample.str.contains(r"[-/:]|\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)", case=False, regex=True)
    has_digit = sample.str.contains(r"\d", regex=True)
    return (has_sep & has_digit).mean() >= 0.8


def infer_column(name, s: pd.Series, n_rows, cfg):
    """Return a dict describing technical type, role, and parse evidence for one column."""
    non_null = s.dropna()
    n_valid = int(non_null.shape[0])
    n_unique = int(non_null.nunique())
    info = {
        "column": name,
        "pandas_dtype": str(s.dtype),
        "non_missing": n_valid,
        "missing": int(n_rows - n_valid),
        "missing_pct": _pct(n_rows - n_valid, n_rows),
        "unique": n_unique,
        "unique_ratio": _r(n_unique / n_valid, 4) if n_valid else 0.0,
        "inferred_type": None,
        "role": None,
        "notes": [],
    }

    if n_valid == 0:
        info.update(inferred_type="empty", role=ROLE_UNKNOWN)
        info["notes"].append("all values missing")
        return info

    # --- Boolean (native bool, or exactly two boolean-like tokens)
    lowered = non_null.astype(str).str.strip().str.lower()
    if pd.api.types.is_bool_dtype(s) or (n_unique == 2 and set(lowered.unique()) <= BOOL_TOKENS):
        info.update(inferred_type="boolean", role=ROLE_BOOLEAN)
        return info

    # --- Native numeric dtype
    if pd.api.types.is_numeric_dtype(s):
        is_int = pd.api.types.is_integer_dtype(s) or bool(np.all(np.mod(non_null.astype(float), 1) == 0))
        info["inferred_type"] = "integer" if is_int else "float"
        if is_int and CODE_NAME_PATTERN.search(str(name)):
            info["role"] = ROLE_CATEGORICAL if n_unique <= cfg["max_categories"] else ROLE_ID
            info["notes"].append("numeric-looking code (name suggests ZIP/FIPS/census/district/phone/code); arithmetic statistics not meaningful")
            return info
        idx_like = is_int and n_unique == n_valid and n_valid > 20 and (
            non_null.is_monotonic_increasing or ID_NAME_PATTERN.search(str(name)))
        if idx_like or (ID_NAME_PATTERN.search(str(name)) and info["unique_ratio"] >= cfg["high_cardinality_ratio"]):
            info["role"] = ROLE_ID
            info["notes"].append("unique integer sequence or ID-like name; excluded from statistics that assume a measurement")
        else:
            info["role"] = ROLE_NUMERIC
            if is_int and n_unique <= 10:
                info["notes"].append(f"only {n_unique} distinct integer values; may be a coded category")
        return info

    # --- Text columns: try numeric, then date, then categorical / id / free text
    as_num = pd.to_numeric(non_null.astype(str).str.replace(",", "", regex=False).str.strip(), errors="coerce")
    num_rate = float(as_num.notna().mean())
    info["numeric_parse_rate"] = _r(num_rate, 4)

    if num_rate >= cfg["numeric_parse_threshold"] and CODE_NAME_PATTERN.search(str(name)):
        info.update(inferred_type="numeric-looking code (text)",
                    role=ROLE_CATEGORICAL if n_unique <= cfg["max_categories"] else ROLE_ID)
        info["notes"].append("name suggests ZIP/FIPS/census/district/phone/code; treated as a label, not a measurement")
        return info

    if num_rate >= cfg["numeric_parse_threshold"]:
        info.update(inferred_type="numeric stored as text", role=ROLE_NUMERIC)
        if num_rate < 1.0:
            bad = non_null[as_num.isna()].astype(str).value_counts().head(5).index.tolist()
            info["notes"].append(f"{_pct((1 - num_rate) * n_valid, n_valid)}% of values are non-numeric (e.g. {bad})")
        return info

    date_rate = 0.0
    if _looks_like_date_strings(non_null):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                parsed = pd.to_datetime(non_null.astype(str), errors="coerce", format="mixed")
            except (TypeError, ValueError):
                parsed = pd.to_datetime(non_null.astype(str), errors="coerce")
        date_rate = float(parsed.notna().mean())
    info["date_parse_rate"] = _r(date_rate, 4)
    if date_rate >= cfg["date_parse_threshold"]:
        info.update(inferred_type="datetime (text)", role=ROLE_DATE)
        return info

    avg_len = float(lowered.str.len().mean())
    info["avg_length"] = _r(avg_len, 1)

    # Mixed: a meaningful minority of numbers mixed with words
    if 0.20 <= num_rate < cfg["numeric_parse_threshold"]:
        info.update(inferred_type="mixed (numbers and text)", role=ROLE_UNKNOWN)
        info["notes"].append(f"{_pct(num_rate * n_valid, n_valid)}% of values parse as numbers, the rest do not")
        return info

    if n_unique <= cfg["max_categories"] or info["unique_ratio"] < 0.05:
        info.update(inferred_type="string", role=ROLE_CATEGORICAL)
        return info

    if info["unique_ratio"] >= 0.5 and avg_len >= cfg["free_text_min_avg_len"]:
        info.update(inferred_type="string", role=ROLE_TEXT)
        return info

    if info["unique_ratio"] >= cfg["high_cardinality_ratio"]:
        info.update(inferred_type="string", role=ROLE_ID)
        info["notes"].append("nearly every value is unique; behaves like an identifier")
        return info

    info.update(inferred_type="string", role=ROLE_CATEGORICAL)
    info["notes"].append(f"high-cardinality categorical ({n_unique} categories); only top values reported")
    return info


def numeric_view(s: pd.Series) -> pd.Series:
    """Numeric version of a column for statistics (text numbers are parsed, not altered in the data)."""
    if pd.api.types.is_numeric_dtype(s):
        return s.astype(float)
    return pd.to_numeric(s.astype(str).str.replace(",", "", regex=False).str.strip(), errors="coerce").where(s.notna())


def date_view(s: pd.Series) -> pd.Series:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            return pd.to_datetime(s, errors="coerce", format="mixed")
        except (TypeError, ValueError):
            return pd.to_datetime(s, errors="coerce")


# ---------------------------------------------------------------------------
# 3. Data quality
# ---------------------------------------------------------------------------
def quality_checks(df, profiles, cfg):
    n = len(df)
    dup = int(df.duplicated().sum())
    q = {
        "duplicate_rows": dup,
        "duplicate_rows_pct": _pct(dup, n),
        "constant_columns": [p["column"] for p in profiles if p["non_missing"] > 0 and p["unique"] <= 1],
        "empty_columns": [p["column"] for p in profiles if p["non_missing"] == 0],
        "high_missing_threshold_pct": cfg["missing_threshold_pct"],
        "high_missing_columns": [
            {"column": p["column"], "missing_pct": p["missing_pct"], "missing": p["missing"]}
            for p in sorted(profiles, key=lambda p: -p["missing_pct"])
            if p["missing_pct"] >= cfg["missing_threshold_pct"]
        ],
        "total_missing_cells": int(df.isna().sum().sum()),
        "total_missing_cells_pct": _pct(int(df.isna().sum().sum()), df.size),
        "columns_with_any_missing": int((df.isna().sum() > 0).sum()),
        "mixed_type_columns": [],
        "identifier_or_high_cardinality_columns": [],
        "potentially_sensitive_columns": [],
        "sensitive_note": ("Sensitive-field detection is a name/pattern heuristic only. "
                           "The absence of a warning does NOT mean the dataset is free of sensitive information."),
    }
    for p in profiles:
        col = p["column"]
        if p["role"] == ROLE_UNKNOWN and p["non_missing"] > 0 or (
                p.get("numeric_parse_rate") is not None and 0 < p["numeric_parse_rate"] < 1 and p["role"] == ROLE_NUMERIC):
            q["mixed_type_columns"].append({"column": col, "detail": "; ".join(p["notes"]) or p["inferred_type"]})
        if p["role"] == ROLE_ID or (p["non_missing"] > 0 and p["unique_ratio"] >= cfg["high_cardinality_ratio"]
                                    and p["role"] not in (ROLE_NUMERIC, ROLE_DATE)):
            q["identifier_or_high_cardinality_columns"].append(
                {"column": col, "unique": p["unique"], "unique_ratio": p["unique_ratio"], "role": p["role"]})

        reasons = []
        for label, pat in SENSITIVE_NAME_PATTERNS.items():
            if re.search(pat, str(col), re.I):
                reasons.append(f"column name suggests {label}")
        if df[col].dtype == object:
            sample = df[col].dropna().astype(str).str.strip().head(500)
            if len(sample):
                for label, pat in SENSITIVE_VALUE_PATTERNS.items():
                    rate = sample.str.match(pat).mean()
                    if rate >= 0.5:
                        reasons.append(f"{_pct(rate, 1)}% of sampled values match a {label} pattern")
        if reasons:
            q["potentially_sensitive_columns"].append({"column": col, "reasons": reasons})
    return q


# ---------------------------------------------------------------------------
# 4. Descriptive statistics
# ---------------------------------------------------------------------------
def numeric_stats(name, s: pd.Series, n_rows, cfg):
    x = numeric_view(s).dropna()
    n = int(x.shape[0])
    out = {"column": name, "valid_count": n, "missing_count": int(n_rows - n),
           "missing_pct": _pct(n_rows - n, n_rows)}
    if n == 0:
        return out
    q1, med, q3 = x.quantile([0.25, 0.5, 0.75]).tolist()
    iqr = q3 - q1
    k = cfg["iqr_multiplier"]
    lo, hi = q1 - k * iqr, q3 + k * iqr
    n_out = int(((x < lo) | (x > hi)).sum())
    vc = x.value_counts()
    # Mode is only meaningful when some value actually repeats.
    if vc.iloc[0] > 1 and x.nunique() < 0.5 * n:
        mode_val, mode_note = _r(vc.index[0]), f"occurs {int(vc.iloc[0])} times"
    else:
        mode_val, mode_note = None, "not meaningful (values are mostly unique)"
    out.update({
        "min": _r(x.min()), "max": _r(x.max()), "mean": _r(x.mean()), "median": _r(med),
        "mode": mode_val, "mode_note": mode_note,
        "std": _r(x.std(ddof=1)) if n > 1 else None,
        "q1": _r(q1), "q3": _r(q3), "iqr": _r(iqr),
        "lower_fence": _r(lo), "upper_fence": _r(hi),
        "outlier_count": n_out, "outlier_pct": _pct(n_out, n),
        "skewness": _r(x.skew()) if n > 2 else None,
        "zero_count": int((x == 0).sum()), "negative_count": int((x < 0).sum()),
    })
    return out


def categorical_stats(name, s: pd.Series, cfg):
    x = s.dropna().astype(str).str.strip()
    n = int(x.shape[0])
    vc = x.value_counts()
    top_k = cfg["top_k_categories"]
    top = [{"value": str(v), "count": int(c), "pct": _pct(c, n)} for v, c in vc.head(top_k).items()]
    max_count = int(vc.iloc[0]) if len(vc) else 0
    modes = [str(v) for v, c in vc.items() if c == max_count][:5]
    return {
        "column": name, "valid_count": n, "unique_categories": int(vc.shape[0]),
        "most_frequent": modes, "most_frequent_count": max_count,
        "most_frequent_pct": _pct(max_count, n),
        "top_categories": top, "shown_categories": len(top),
        "other_categories_count": int(max(vc.shape[0] - top_k, 0)),
        "other_categories_rows": int(vc.iloc[top_k:].sum()) if vc.shape[0] > top_k else 0,
        "singleton_categories": int((vc == 1).sum()),
    }


def date_stats(name, s: pd.Series):
    d = date_view(s).dropna()
    if d.empty:
        return {"column": name, "valid_count": 0}
    span = (d.max() - d.min()).days
    return {"column": name, "valid_count": int(d.shape[0]), "min": d.min().isoformat(),
            "max": d.max().isoformat(), "span_days": int(span), "unique_dates": int(d.nunique())}


def relationships(df, num_cols, cfg):
    usable = []
    for c in num_cols:
        x = numeric_view(df[c])
        if x.notna().sum() >= 3 and x.nunique() > 1:
            usable.append(c)
    if len(usable) < 2:
        return {"performed": False,
                "reason": f"needs at least 2 non-constant numeric measure columns; found {len(usable)}",
                "columns": usable}
    num = pd.DataFrame({c: numeric_view(df[c]) for c in usable})
    corr = num.corr(method="pearson", min_periods=3)
    pairs = []
    for i, a in enumerate(usable):
        for b in usable[i + 1:]:
            r = corr.loc[a, b]
            if pd.notna(r):
                n_pair = int(num[[a, b]].dropna().shape[0])
                ar = abs(round(float(r), 3))
                strength = "very weak" if ar < 0.1 else "weak" if ar < 0.3 else "moderate" if ar < 0.7 else "strong"
                pairs.append({"x": a, "y": b, "pearson_r": _r(r, 3), "n_pairs": n_pair, "strength": strength})
    pos = sorted([p for p in pairs if p["pearson_r"] > 0], key=lambda p: -p["pearson_r"])[:5]
    neg = sorted([p for p in pairs if p["pearson_r"] < 0], key=lambda p: p["pearson_r"])[:5]
    return {
        "performed": True, "method": "Pearson correlation (pairwise complete observations)",
        "columns": usable,
        "matrix": {a: {b: _r(corr.loc[a, b], 3) for b in usable} for a in usable},
        "strongest_positive": pos, "strongest_negative": neg,
        "strength_scale": "|r| < 0.1 very weak, < 0.3 weak, < 0.7 moderate, >= 0.7 strong",
        "note": "Correlation measures linear association only and does not imply causation.",
    }


# ---------------------------------------------------------------------------
# 5. Adaptive visualizations
# ---------------------------------------------------------------------------
def _style(ax, title, xlabel, ylabel):
    ax.set_title(title, fontsize=12, color=C_INK, loc="left", pad=10)
    ax.set_xlabel(xlabel, fontsize=10, color=C_INK2)
    ax.set_ylabel(ylabel, fontsize=10, color=C_INK2)
    ax.set_facecolor(C_SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(C_AXIS)
    ax.tick_params(colors=C_MUTED, labelsize=9)
    ax.grid(True, color=C_GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def _save(fig, plots_dir, idx):
    path = plots_dir / f"plot_{idx:02d}.png"
    fig.patch.set_facecolor(C_SURFACE)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def make_plots(df, profiles, num_stats, cat_stats, rel, plots_dir, cfg):
    """Choose plots based on which column roles exist. Returns (made, skipped)."""
    roles = {p["column"]: p["role"] for p in profiles}
    num_cols = [s["column"] for s in num_stats if s.get("valid_count", 0) > 1 and (s.get("iqr") is not None)]
    num_cols_var = [c for c in num_cols if numeric_view(df[c]).nunique() > 1]
    cat_cols = [s["column"] for s in cat_stats if 2 <= s["unique_categories"]]
    date_cols = [c for c, r in roles.items() if r == ROLE_DATE]
    # Prefer the most complete columns
    comp = {p["column"]: p["missing_pct"] for p in profiles}
    num_cols_var.sort(key=lambda c: comp[c])
    cat_cols.sort(key=lambda c: (comp[c], [s for s in cat_stats if s["column"] == c][0]["unique_categories"] > 20))

    candidates, skipped = [], []

    # (a) Missing values
    miss = df.isna().mean().mul(100)
    miss = miss[miss > 0].sort_values(ascending=False)
    if len(miss):
        candidates.append(("missing", None))
    else:
        skipped.append({"plot": "Missing-value bar chart", "reason": "no column has missing values"})

    # (b) Histogram(s)
    if num_cols_var:
        for c in num_cols_var[:2]:
            candidates.append(("hist", c))
    else:
        skipped.append({"plot": "Histogram", "reason": "no non-constant numeric measure columns"})

    # (c) Categorical frequency bars
    if cat_cols:
        for c in cat_cols[:2]:
            candidates.append(("catbar", c))
    else:
        skipped.append({"plot": "Categorical frequency bar chart", "reason": "no categorical or Boolean columns with 2+ categories"})

    # (d) Correlation heatmap + scatter of strongest pair
    if rel.get("performed"):
        candidates.append(("heatmap", None))
        allpairs = rel["strongest_positive"] + rel["strongest_negative"]
        if allpairs:
            best = max(allpairs, key=lambda p: abs(p["pearson_r"]))
            candidates.append(("scatter", best))
    else:
        skipped.append({"plot": "Correlation heatmap / scatterplot", "reason": rel.get("reason", "not supported")})

    # (e) Boxplot of the column with the most outliers
    with_out = sorted([s for s in num_stats if s["column"] in num_cols_var], key=lambda s: -s.get("outlier_count", 0))
    if with_out:
        candidates.append(("box", with_out[0]["column"]))
    else:
        skipped.append({"plot": "Boxplot", "reason": "no non-constant numeric measure columns"})

    # (f) Numeric by category
    grp_cat = [c for c in cat_cols if 2 <= [s for s in cat_stats if s["column"] == c][0]["unique_categories"] <= 12]
    if num_cols_var and grp_cat:
        candidates.append(("bycat", (num_cols_var[0], grp_cat[0])))
    else:
        skipped.append({"plot": "Numeric distribution by category",
                        "reason": "needs a numeric measure and a categorical column with 2-12 categories"})

    # (g) Time series
    if date_cols:
        candidates.append(("time", date_cols[0]))
    else:
        skipped.append({"plot": "Time-series plot", "reason": "no date-like column detected"})

    # Priority order so the budget keeps a variety of plot types
    priority = ["missing", "hist", "catbar", "heatmap", "scatter", "box", "time", "bycat"]
    first_of_each, rest = [], []
    seen = set()
    for kind in priority:
        for c in candidates:
            if c[0] == kind:
                (rest if kind in seen else first_of_each).append(c)
                seen.add(kind)
    ordered = first_of_each + rest
    chosen = ordered[: cfg["max_plots"]]
    for c in ordered[cfg["max_plots"]:]:
        skipped.append({"plot": f"{c[0]} ({c[1]})", "reason": f"plot budget of {cfg['max_plots']} reached"})

    made = []
    idx = 0
    for kind, arg in chosen:
        try:
            idx += 1
            if kind == "missing":
                top = miss.head(20)
                fig, ax = plt.subplots(figsize=(8, max(3, 0.35 * len(top) + 1.2)))
                ax.barh([_short(c) for c in top.index[::-1]], top.values[::-1], color=C_BLUE, height=0.6)
                ax.axvline(cfg["missing_threshold_pct"], color=C_RED, linestyle="--", linewidth=1.2,
                           label=f"{cfg['missing_threshold_pct']:.0f}% high-missing threshold")
                ax.legend(frameon=False, fontsize=9)
                _style(ax, "Missing values by column", "Missing (% of rows)", "Column")
                ax.set_xlim(0, 100)
                why = "Shows which columns are incomplete and which cross the high-missingness threshold."
                title = "Missing values by column"
            elif kind == "hist":
                x = numeric_view(df[arg]).dropna()
                fig, ax = plt.subplots(figsize=(8, 4.5))
                bins = min(40, max(10, int(np.sqrt(len(x)))))
                ax.hist(x, bins=bins, color=C_BLUE, edgecolor=C_SURFACE, linewidth=1)
                st = [s for s in num_stats if s["column"] == arg][0]
                ax.axvline(st["mean"], color=C_ORANGE, linewidth=2, label=f"mean = {st['mean']:.4g}")
                ax.axvline(st["median"], color=C_INK2, linewidth=2, linestyle="--", label=f"median = {st['median']:.4g}")
                ax.legend(frameon=False, fontsize=9)
                title = f"Distribution of {arg}"
                _style(ax, title, f"{arg} (units not documented)", "Count of rows")
                why = "A histogram suits a continuous numeric measure: it shows shape, skew, and spread."
            elif kind == "catbar":
                st = [s for s in cat_stats if s["column"] == arg][0]
                vals = [t["value"] for t in st["top_categories"]][::-1]
                cnts = [t["count"] for t in st["top_categories"]][::-1]
                fig, ax = plt.subplots(figsize=(8, max(3, 0.4 * len(vals) + 1.2)))
                ax.barh([_short(v, 32) for v in vals], cnts, color=C_BLUE, height=0.6)
                for yi, (c, t) in enumerate(zip(cnts, st["top_categories"][::-1])):
                    ax.text(c, yi, f" {t['pct']:.1f}%", va="center", fontsize=8, color=C_INK2)
                suffix = f" (top {len(vals)} of {st['unique_categories']})" if st["unique_categories"] > len(vals) else ""
                title = f"Most frequent values of {arg}{suffix}"
                _style(ax, title, "Count of rows", arg)
                why = "A sorted bar chart compares category frequencies for a categorical attribute."
            elif kind == "heatmap":
                cols = rel["columns"][:15]
                m = np.array([[rel["matrix"][a][b] if rel["matrix"][a][b] is not None else np.nan for b in cols] for a in cols])
                from matplotlib.colors import LinearSegmentedColormap
                cmap = LinearSegmentedColormap.from_list("div", [C_BLUE, C_NEUTRAL, C_RED])
                size = max(5, 0.55 * len(cols) + 2.5)
                fig, ax = plt.subplots(figsize=(size + 1, size))
                im = ax.imshow(m, cmap=cmap, vmin=-1, vmax=1)
                ax.set_xticks(range(len(cols)), [_short(c, 18) for c in cols], rotation=45, ha="right")
                ax.set_yticks(range(len(cols)), [_short(c, 18) for c in cols])
                if len(cols) <= 10:
                    for i in range(len(cols)):
                        for j in range(len(cols)):
                            if not np.isnan(m[i, j]):
                                ax.text(j, i, f"{m[i, j]:.2f}", ha="center", va="center", fontsize=8, color=C_INK)
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Pearson r")
                title = "Correlation matrix of numeric measures"
                _style(ax, title, "Numeric column", "Numeric column")
                ax.grid(False)
                why = "A heatmap summarizes all pairwise linear correlations among numeric measures at once."
            elif kind == "scatter":
                a, b = arg["x"], arg["y"]
                d = pd.DataFrame({"a": numeric_view(df[a]), "b": numeric_view(df[b])}).dropna()
                if len(d) > 5000:
                    d = d.sample(5000, random_state=0)
                fig, ax = plt.subplots(figsize=(7, 5))
                ax.scatter(d["a"], d["b"], s=14, alpha=0.45, color=C_BLUE, edgecolors="none")
                title = f"{b} vs {a} (r = {arg['pearson_r']:.2f})"
                _style(ax, title, a, b)
                why = "A scatterplot shows the strongest numeric pair so the correlation can be checked visually (linearity, clusters, outliers)."
            elif kind == "box":
                x = numeric_view(df[arg]).dropna()
                fig, ax = plt.subplots(figsize=(8, 3.2))
                ax.boxplot(x, **HBOX, whis=cfg["iqr_multiplier"], widths=0.5, patch_artist=True,
                           boxprops=dict(facecolor="#cde2fb", color=C_BLUE),
                           medianprops=dict(color=C_INK, linewidth=2),
                           whiskerprops=dict(color=C_BLUE), capprops=dict(color=C_BLUE),
                           flierprops=dict(marker="o", markersize=4, markerfacecolor=C_ORANGE, markeredgecolor="none", alpha=0.6))
                ax.set_yticks([1], [_short(arg)])
                st = [s for s in num_stats if s["column"] == arg][0]
                title = f"Boxplot of {arg}: {st['outlier_count']} potential outliers (1.5×IQR)"
                _style(ax, title, f"{arg} (units not documented)", "Column")
                why = "A boxplot shows the median, IQR, and points beyond the 1.5×IQR fences (potential outliers)."
            elif kind == "bycat":
                nc, cc = arg
                st = [s for s in cat_stats if s["column"] == cc][0]
                cats = [t["value"] for t in st["top_categories"]][:12]
                xs = numeric_view(df[nc])
                groups = [xs[df[cc].astype(str).str.strip() == c].dropna() for c in cats]
                keep = [(c, g) for c, g in zip(cats, groups) if len(g) > 0]
                fig, ax = plt.subplots(figsize=(8, max(3.5, 0.45 * len(keep) + 1.5)))
                ax.boxplot([g for _, g in keep], **HBOX, whis=cfg["iqr_multiplier"], widths=0.55, patch_artist=True,
                           boxprops=dict(facecolor="#cde2fb", color=C_BLUE), medianprops=dict(color=C_INK, linewidth=2),
                           whiskerprops=dict(color=C_BLUE), capprops=dict(color=C_BLUE),
                           flierprops=dict(marker="o", markersize=3, markerfacecolor=C_ORANGE, markeredgecolor="none", alpha=0.5))
                ax.set_yticks(range(1, len(keep) + 1), [_short(c, 24) for c, _ in keep])
                title = f"{nc} by {cc}"
                _style(ax, title, nc, cc)
                why = "Side-by-side boxplots compare a numeric measure across the categories of a categorical attribute."
            elif kind == "time":
                d = date_view(df[arg]).dropna()
                span = (d.max() - d.min()).days if len(d) else 0
                freq, lab = ("D", "day") if span <= 90 else (("MS", "month") if span <= 365 * 6 else ("YS", "year"))
                counts = d.dt.to_period(freq[0]).value_counts().sort_index()
                fig, ax = plt.subplots(figsize=(9, 4.5))
                ax.plot(counts.index.to_timestamp(), counts.values, color=C_BLUE, linewidth=2, marker="o", markersize=3)
                title = f"Rows per {lab} by {arg}"
                _style(ax, title, f"{arg} ({lab})", "Count of rows")
                fig.autofmt_xdate()
                why = "A time-series line shows how record volume changes over the date range."
            else:
                continue
            path = _save(fig, plots_dir, idx)
            made.append({"file": f"plots/{path.name}", "type": kind, "title": title, "why": why,
                         "columns": list(arg) if isinstance(arg, tuple) else ([arg["x"], arg["y"]] if isinstance(arg, dict) else ([arg] if arg else []))})
        except Exception as e:  # never let one plot kill the run
            plt.close("all")
            idx -= 1
            skipped.append({"plot": f"{kind} ({arg})", "reason": f"plotting error: {e}"})
    return made, skipped


# ---------------------------------------------------------------------------
# 6. Deterministic (Python-written) findings - always produced
# ---------------------------------------------------------------------------
def computed_findings(summary):
    """Template-based findings. Every number comes straight from the summary."""
    f = []
    q = summary["data_quality"]
    ov = summary["overview"]
    n = ov["rows"]

    # Data quality
    if q["high_missing_columns"]:
        h = q["high_missing_columns"][0]
        f.append({"type": "data quality",
                  "text": f"`{h['column']}` is missing for {h['missing']:,} of {n:,} rows ({h['missing_pct']}%), "
                          f"above the {q['high_missing_threshold_pct']}% high-missingness threshold; "
                          f"{len(q['high_missing_columns'])} column(s) exceed it in total.",
                  "evidence": "data_quality.high_missing_columns"})
    elif q["total_missing_cells"]:
        f.append({"type": "data quality",
                  "text": f"{q['total_missing_cells']:,} cells ({q['total_missing_cells_pct']}%) are missing across "
                          f"{q['columns_with_any_missing']} column(s); none exceeds the {q['high_missing_threshold_pct']}% threshold.",
                  "evidence": "data_quality.total_missing_cells"})
    else:
        f.append({"type": "data quality", "text": "No missing values were detected in any column (after applying the configured missing-value codes).",
                  "evidence": "data_quality.total_missing_cells"})
    if q["duplicate_rows"]:
        f.append({"type": "data quality",
                  "text": f"{q['duplicate_rows']:,} rows ({q['duplicate_rows_pct']}%) are exact duplicates of another row.",
                  "evidence": "data_quality.duplicate_rows"})

    # Distribution
    ns = [s for s in summary["numeric_statistics"] if s.get("valid_count")]
    if ns:
        s = max(ns, key=lambda s: s.get("outlier_pct", 0))
        f.append({"type": "distribution",
                  "text": f"`{s['column']}` has median {s['median']:,} and mean {s['mean']:,} "
                          f"(IQR {s['q1']:,} to {s['q3']:,}); {s['outlier_count']:,} values ({s['outlier_pct']}%) "
                          f"fall outside the 1.5×IQR fences [{s['lower_fence']:,}, {s['upper_fence']:,}] and are flagged, not removed.",
                  "evidence": f"numeric_statistics[{s['column']}]"})

    if len(ns) > 1:
        sk = [s for s in ns if s.get("skewness") is not None]
        if sk:
            s2 = max(sk, key=lambda s: abs(s["skewness"]))
            shape = "right-skewed (long upper tail)" if s2["skewness"] > 0 else "left-skewed (long lower tail)"
            if abs(s2["skewness"]) >= 1:
                f.append({"type": "distribution",
                          "text": f"`{s2['column']}` is strongly {shape}: skewness {s2['skewness']}, mean {s2['mean']:,} vs "
                                  f"median {s2['median']:,}, range {s2['min']:,} to {s2['max']:,}. The median is the more robust center.",
                          "evidence": f"numeric_statistics[{s2['column']}].skewness"})

    # Categorical
    cs = [s for s in summary["categorical_statistics"] if s["valid_count"] and s["unique_categories"] >= 2]
    if cs:
        s = max(cs, key=lambda s: s["most_frequent_pct"])
        f.append({"type": "categorical",
                  "text": f"In `{s['column']}`, the most frequent value '{s['most_frequent'][0]}' accounts for "
                          f"{s['most_frequent_count']:,} of {s['valid_count']:,} non-missing rows ({s['most_frequent_pct']}%) "
                          f"across {s['unique_categories']} categories.",
                  "evidence": f"categorical_statistics[{s['column']}]"})

    # Relationship
    rel = summary["relationships"]
    if rel.get("performed"):
        pairs = rel["strongest_positive"] + rel["strongest_negative"]
        if pairs:
            p = max(pairs, key=lambda p: abs(p["pearson_r"]))
            f.append({"type": "relationship",
                      "text": f"The strongest linear relationship found is between `{p['x']}` and `{p['y']}` "
                              f"(Pearson r = {p['pearson_r']}, {p['strength']}, n = {p['n_pairs']:,} complete pairs). "
                              f"This is an association, not evidence of causation.",
                      "evidence": "relationships.strongest_positive / strongest_negative"})

    # Limitation
    lim = []
    if q["potentially_sensitive_columns"]:
        lim.append(f"{len(q['potentially_sensitive_columns'])} column(s) were flagged as potentially sensitive by name/pattern heuristics")
    unk = [c["column"] for c in summary["columns"] if c["role"] == ROLE_UNKNOWN]
    if unk:
        lim.append(f"{len(unk)} column(s) have unknown or mixed types ({', '.join('`'+u+'`' for u in unk[:3])})")
    lim.append("column meanings and units are not documented in the file, so no units or definitions are assumed")
    f.append({"type": "limitation", "text": "Limitation: " + "; ".join(lim) + ".", "evidence": "columns, data_quality"})

    # Follow-up question
    if q["high_missing_columns"]:
        c = q["high_missing_columns"][0]["column"]
        qtext = f"Is missingness in `{c}` random, or concentrated in particular categories or time periods?"
    elif rel.get("performed") and (rel["strongest_positive"] or rel["strongest_negative"]):
        p = max(rel["strongest_positive"] + rel["strongest_negative"], key=lambda p: abs(p["pearson_r"]))
        qtext = f"Does the association between `{p['x']}` and `{p['y']}` hold within subgroups, or is it driven by a third variable?"
    elif ns:
        qtext = f"Are the flagged outliers in `{ns[0]['column']}` data-entry errors or genuine extreme cases?"
    else:
        qtext = "What documentation (data dictionary) exists to define each column's meaning and valid range?"
    f.append({"type": "follow-up question", "text": "Question for further investigation: " + qtext, "evidence": "derived from the findings above"})
    return f[:8]


# ---------------------------------------------------------------------------
# 7. LLM step (optional) + automatic verification
# ---------------------------------------------------------------------------
LLM_RULES = """You are a careful data analyst writing for a data mining course report.
You are given a JSON summary of a CSV file. EVERY statistic in it was computed by Python and verified.

Write 5 to 8 numbered insights about this dataset. Rules:
1. Use ONLY numbers that appear in the JSON. Copy them exactly as written. Never calculate, estimate, or round new numbers.
2. Every insight that uses a number must name the column in backticks, e.g. `column_name`.
3. Include at least: one data-quality finding, one distribution finding, one categorical finding (if categorical_statistics is not empty),
   one relationship finding (if relationships.performed is true), one limitation or potential-bias warning, and one question for further investigation.
4. Do NOT guess what a column means, its units, or what abbreviations stand for. If the meaning is unclear, say it is undocumented.
5. Correlation is NOT causation. Never say one variable causes, drives, or leads to another.
6. Outliers are flagged, not errors. Do not say they should be removed.
7. Label each insight with its type in brackets, e.g. [Data quality].
Output only the numbered list."""


def build_llm_payload(summary, max_cols=40):
    """A compact copy of the verified summary for the prompt (keeps small models on track)."""
    s = summary
    slim = {
        "overview": s["overview"],
        "columns": [{k: c[k] for k in ("column", "inferred_type", "role", "missing_pct", "unique")} for c in s["columns"][:max_cols]],
        "data_quality": {k: v for k, v in s["data_quality"].items() if k != "sensitive_note"},
        "numeric_statistics": [{k: v for k, v in n.items() if k in (
            "column", "valid_count", "missing_pct", "min", "max", "mean", "median", "std", "q1", "q3", "iqr",
            "outlier_count", "outlier_pct", "skewness")} for n in s["numeric_statistics"][:max_cols]],
        "categorical_statistics": [{"column": c["column"], "unique_categories": c["unique_categories"],
                                    "top_categories": c["top_categories"][:5]} for c in s["categorical_statistics"][:max_cols]],
        "date_statistics": s["date_statistics"],
        "relationships": {k: v for k, v in s["relationships"].items() if k != "matrix"},
    }
    return slim


TASK_REMINDER = ("TASK: Using ONLY the JSON above, write 5 to 8 numbered insights following the rules "
                 "(each labeled [Data quality], [Distribution], [Categorical], [Relationship], [Limitation], or [Question]). "
                 "Do not describe the JSON structure. Output only the numbered list.")


def build_llm_prompt(payload):
    """User prompt: data first, task repeated at the END so it survives any truncation."""
    data = json.dumps(payload, separators=(",", ":"), default=_json_default)
    return "VERIFIED SUMMARY (JSON):\n" + data + "\n\n" + TASK_REMINDER


def looks_like_insight_list(text):
    return len(re.findall(r"(?m)^\s*\d+[.)]\s+\S", text)) >= 3


def call_ollama(prompt, cfg, system=None):
    body = json.dumps({"model": cfg["llm_model"], "prompt": prompt, "system": system or LLM_RULES,
                       "stream": False,
                       "options": {"temperature": 0.2, "num_ctx": cfg["llm_num_ctx"]}}).encode()
    req = urllib.request.Request(cfg["ollama_url"], data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=cfg["llm_timeout_sec"]) as resp:
        data = json.loads(resp.read().decode())
    if "error" in data:
        raise RuntimeError(data["error"])
    return data.get("response", "").strip()


def _collect_numbers(obj, acc, include_strings=True):
    if isinstance(obj, bool):
        return
    if isinstance(obj, (int, float)) and not (isinstance(obj, float) and math.isnan(obj)):
        acc.append(float(obj))
    elif isinstance(obj, dict):
        for v in obj.values():
            _collect_numbers(v, acc, include_strings)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            _collect_numbers(v, acc, include_strings)
    elif isinstance(obj, str) and include_strings:
        for m in re.findall(r"-?\d[\d,]*\.?\d*", obj):
            try:
                acc.append(float(m.replace(",", "")))
            except ValueError:
                pass


CAUSAL_PATTERN = re.compile(r"\b(causes?|caused by|causing|leads? to|results? in|drives?|driven by|due to|because of|impacts?|effect of)\b", re.I)

# Statistic words the model may use, mapped to the field Python computed.
STAT_KEYWORDS = [
    (r"standard deviation|\bstd\b|\bsd\b", "std"),
    (r"\bmean\b|\baverage\b", "mean"),
    (r"\bmedian\b", "median"),
    (r"\bminimum\b|\bmin\b", "min"),
    (r"\bmaximum\b|\bmax\b", "max"),
    (r"\bskew(?:ness)?\b", "skewness"),
    (r"\biqr\b|interquartile range", "iqr"),
    (r"correlation|pearson|\br\s*=", "pearson_r"),
]
NUM_RE = re.compile(r"(?<![\w.])-?\d[\d,]*(?:\.\d+)?")


def _matches(v, raw, pool):
    """True if v equals a value in pool at the precision it was written (or as a percentage of a fraction)."""
    if not len(pool):
        return False
    nd = len(raw.split(".")[1]) if "." in raw else 0
    tol = 0.5 * 10 ** (-nd) + 1e-9
    return bool(np.any(np.abs(pool - v) <= tol) or np.any(np.abs(pool * 100 - v) <= tol))


def _column_index(summary):
    """Numbers Python computed for each column, plus dataset-wide numbers."""
    idx = {}
    for c in summary["columns"]:
        acc = []
        _collect_numbers(c, acc)
        idx[c["column"]] = acc
    for key in ("numeric_statistics", "categorical_statistics", "date_statistics"):
        for st in summary[key]:
            _collect_numbers(st, idx.setdefault(st["column"], []))
    rel = summary["relationships"]
    if rel.get("performed"):
        for a, row in rel["matrix"].items():
            for b, r in row.items():
                if r is not None and a != b:
                    idx.setdefault(a, []).append(r)
                    idx.setdefault(b, []).append(r)
    glob = []
    _collect_numbers(summary["overview"], glob, include_strings=False)
    _collect_numbers({k: v for k, v in summary["data_quality"].items()
                      if not isinstance(v, (list, dict))}, glob, include_strings=False)
    glob.append(float(len(summary["columns"])))
    return {k: np.array(v) for k, v in idx.items()}, np.array(glob)


def verify_llm_text(text, summary):
    """Check the LLM text against the Python summary, claim by claim.

    1. Numbers: each number must match a value Python computed for the column(s) the claim names
       (or a dataset-wide value such as row count). A number that only exists elsewhere in the
       summary does not count, which stops coincidental matches.
    2. Statistic words: 'mean 41.0' must equal the mean Python computed for that column; if the
       column has no mean (e.g. it is categorical), the claim is flagged.
    3. Causal wording is flagged.
    """
    col_idx, glob = _column_index(summary)
    lower_map = {c.lower(): c for c in col_idx}
    num_stats = {s["column"]: s for s in summary["numeric_statistics"]}
    role = {c["column"]: c["role"] for c in summary["columns"]}
    rel = summary["relationships"]
    checks = []
    for line in [l.strip() for l in text.splitlines() if l.strip()]:
        body = re.sub(r"^\s*(\d+[.)]|[-*])\s*", "", line)
        named = []
        for t in re.findall(r"`([^`]+)`", body):
            c = lower_map.get(t.strip().lower())
            if c and c not in named:
                named.append(c)
        pool = np.concatenate([glob] + [col_idx.get(c, np.array([])) for c in named]) if named else None
        if pool is None:  # no column named: fall back to all numbers, marked as weak
            allnums = []
            _collect_numbers(summary, allnums)
            pool = np.array(allnums)
        body_no_cols = re.sub(r"`[^`]*`", " ", body)
        results = []
        for m in NUM_RE.finditer(body_no_cols):
            tok = m.group()
            raw = tok.replace(",", "")
            try:
                v = float(raw)
            except ValueError:
                continue
            if abs(v) <= 10 and "." not in raw:
                continue
            results.append({"value": tok, "found_in_summary": _matches(v, raw, pool)})

        stat_issues = []
        for pat, field in STAT_KEYWORDS:
            for km in re.finditer(pat, body_no_cols, re.I):
                nm = NUM_RE.search(body_no_cols[km.end(): km.end() + 60])
                if not nm or not named:
                    continue
                raw = nm.group().replace(",", "")
                v = float(raw)
                if field == "pearson_r":
                    if len(named) >= 2 and rel.get("performed"):
                        vals = [rel["matrix"].get(a, {}).get(b) for a in named for b in named if a != b]
                        vals = np.array([x for x in vals if x is not None])
                        if not len(vals):
                            stat_issues.append(f"no correlation was computed between {', '.join(named)}")
                        elif not _matches(v, raw, vals):
                            stat_issues.append(f"correlation {nm.group()} does not match Python ({', '.join(str(x) for x in vals[:2])})")
                    continue
                expected = [num_stats[c].get(field) for c in named if c in num_stats]
                expected = [x for x in expected if x is not None]
                if not expected:
                    stat_issues.append(f"Python computed no {field} for {', '.join(named)} (role: {', '.join(role[c] for c in named)})")
                elif not _matches(v, raw, np.array(expected)):
                    stat_issues.append(f"{field} {nm.group()} does not match Python ({', '.join(str(x) for x in expected)})")
                break  # one check per statistic word is enough
        stat_issues = sorted(set(stat_issues))
        causal = CAUSAL_PATTERN.findall(body)
        if results or causal or stat_issues:
            checks.append({
                "claim": body[:300],
                "columns_named": named,
                "numbers": results,
                "all_numbers_verified": all(r["found_in_summary"] for r in results) if results else None,
                "statistic_issues": stat_issues,
                "causal_language": sorted(set(w.lower() for w in causal)),
            })
    return checks


# ---------------------------------------------------------------------------
# 8. Report writer
# ---------------------------------------------------------------------------
def write_report(out, summary, profile_df, findings, llm, plots, skipped_plots):
    ov, q = summary["overview"], summary["data_quality"]
    L = []
    L.append(f"# Automated Data Profile: `{ov['filename']}`\n")
    L.append(f"*Generated by `profiler.py` on {summary['generated_at']}. All statistics were computed by Python; "
             f"the AI narrative (if present) only explains them and is automatically checked.*\n")

    L.append("## 1. Dataset overview\n")
    L.append("| Item | Value |\n|---|---|")
    L.append(f"| File | `{ov['filename']}` |")
    L.append(f"| Rows | {ov['rows']:,} |")
    L.append(f"| Columns | {ov['columns']} |")
    L.append(f"| Encoding used to read file | {ov['encoding']} |")
    L.append(f"| Role counts | " + ", ".join(f"{k}: {v}" for k, v in ov["role_counts"].items()) + " |\n")

    L.append("### Column profile\n")
    L.append("| Column | Inferred type | Probable role | Non-missing | Missing % | Unique | Notes |")
    L.append("|---|---|---|---:|---:|---:|---|")
    for c in summary["columns"]:
        L.append(f"| `{_md_escape(c['column'])}` | {c['inferred_type']} | {c['role']} | {c['non_missing']:,} | "
                 f"{c['missing_pct']} | {c['unique']:,} | {_md_escape('; '.join(c['notes']))} |")
    L.append("\n*Roles are inferred from values and names only. Column meanings, units, and valid ranges are not "
             "documented in the CSV, so they are not assumed.*\n")

    L.append("## 2. Data quality\n")
    L.append(f"- **Duplicate rows:** {q['duplicate_rows']:,} ({q['duplicate_rows_pct']}%)")
    L.append(f"- **Missing cells overall:** {q['total_missing_cells']:,} ({q['total_missing_cells_pct']}% of all cells), in {q['columns_with_any_missing']} column(s)")
    L.append(f"- **High-missingness columns (≥ {q['high_missing_threshold_pct']}%):** " +
             (", ".join(f"`{h['column']}` ({h['missing_pct']}%)" for h in q["high_missing_columns"]) or "none"))
    L.append(f"- **Constant (single-value) columns:** " + (", ".join(f"`{c}`" for c in q["constant_columns"]) or "none"))
    L.append(f"- **Completely empty columns:** " + (", ".join(f"`{c}`" for c in q["empty_columns"]) or "none"))
    L.append(f"- **Mixed / inconsistent types:** " +
             ("; ".join(f"`{m['column']}` ({_md_escape(m['detail'])})" for m in q["mixed_type_columns"]) or "none detected"))
    L.append(f"- **Identifier-like / high-cardinality columns:** " +
             (", ".join(f"`{m['column']}` ({m['unique']:,} unique)" for m in q["identifier_or_high_cardinality_columns"]) or "none"))
    L.append(f"- **Potentially sensitive columns (heuristic):** " +
             ("; ".join(f"`{m['column']}` ({', '.join(m['reasons'])})" for m in q["potentially_sensitive_columns"]) or "none flagged"))
    L.append(f"\n> {q['sensitive_note']}\n")
    L.append("> No rows, values, or outliers were removed or changed. Problems are flagged only.\n")

    L.append("## 3. Descriptive statistics\n")
    L.append("### Numeric columns\n")
    ns = summary["numeric_statistics"]
    if ns:
        L.append("| Column | Valid | Missing (%) | Min | Q1 | Median | Mean | Q3 | Max | Std | IQR | Mode | Outliers (%) |")
        L.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|")
        for s in ns:
            if not s.get("valid_count"):
                L.append(f"| `{s['column']}` | 0 | {s['missing_count']} ({s['missing_pct']}) | – | – | – | – | – | – | – | – | – | – |")
                continue
            mode = s["mode"] if s["mode"] is not None else "n/a"
            L.append(f"| `{_md_escape(s['column'])}` | {s['valid_count']:,} | {s['missing_count']:,} ({s['missing_pct']}) | "
                     f"{s['min']} | {s['q1']} | {s['median']} | {s['mean']} | {s['q3']} | {s['max']} | {s['std']} | "
                     f"{s['iqr']} | {mode} | {s['outlier_count']:,} ({s['outlier_pct']}) |")
        L.append(f"\n*Outliers use the 1.5×IQR rule: values below Q1 − 1.5·IQR or above Q3 + 1.5·IQR. "
                 f"They are potential outliers to investigate, not errors. Std is the sample standard deviation (n − 1). "
                 f"Mode is shown only when values repeat.*\n")
    else:
        L.append("**Skipped:** no numeric measure columns were detected, so numeric statistics, outlier analysis, "
                 "histograms, and boxplots were not produced.\n")

    L.append("### Categorical and Boolean columns\n")
    cs = summary["categorical_statistics"]
    if cs:
        for s in cs:
            L.append(f"**`{_md_escape(s['column'])}`** — {s['unique_categories']:,} categories, {s['valid_count']:,} non-missing. "
                     f"Most frequent: {', '.join(repr(m) for m in s['most_frequent'])} ({s['most_frequent_count']:,} rows, {s['most_frequent_pct']}%).\n")
            L.append("| Value | Count | % of non-missing |\n|---|---:|---:|")
            for t in s["top_categories"]:
                L.append(f"| {_md_escape(_short(t['value'], 60))} | {t['count']:,} | {t['pct']} |")
            if s["other_categories_count"]:
                L.append(f"| *({s['other_categories_count']:,} other categories)* | {s['other_categories_rows']:,} | "
                         f"{_pct(s['other_categories_rows'], s['valid_count'])} |")
            L.append("")
    else:
        L.append("**Skipped:** no categorical or Boolean columns were detected, so frequency tables and categorical bar charts were not produced.\n")

    ds = summary["date_statistics"]
    if ds:
        L.append("### Date-like columns\n")
        L.append("| Column | Valid | Earliest | Latest | Span (days) | Unique dates |\n|---|---:|---|---|---:|---:|")
        for d in ds:
            L.append(f"| `{d['column']}` | {d['valid_count']:,} | {d.get('min', '–')} | {d.get('max', '–')} | {d.get('span_days', '–')} | {d.get('unique_dates', '–')} |")
        L.append("")

    L.append("## 4. Relationships between variables\n")
    rel = summary["relationships"]
    if rel.get("performed"):
        L.append(f"Method: {rel['method']} on {len(rel['columns'])} numeric measure columns (identifier-like columns excluded).\n")
        L.append("| Direction | Column A | Column B | Pearson r | Strength | Complete pairs |\n|---|---|---|---:|---|---:|")
        for p in rel["strongest_positive"][:3]:
            L.append(f"| Positive | `{p['x']}` | `{p['y']}` | {p['pearson_r']} | {p['strength']} | {p['n_pairs']:,} |")
        for p in rel["strongest_negative"][:3]:
            L.append(f"| Negative | `{p['x']}` | `{p['y']}` | {p['pearson_r']} | {p['strength']} | {p['n_pairs']:,} |")
        if not rel["strongest_negative"]:
            L.append("| Negative | – | – | none found | – | – |")
        L.append(f"\n*Strength scale: {rel['strength_scale']}. {rel['note']}*\n")
    else:
        L.append(f"**Skipped:** {rel['reason']}.\n")

    L.append("## 5. Visualizations\n")
    for i, p in enumerate(plots, 1):
        L.append(f"### Figure {i}. {p['title']}\n")
        L.append(f"![{p['title']}]({p['file']})\n")
        L.append(f"*Why this plot:* {p['why']}\n")
    if len(plots) < 5:
        L.append(f"> Only {len(plots)} meaningful plot(s) could be generated for this dataset (fewer than 5). See skipped plots below.\n")
    if skipped_plots:
        L.append("**Plots skipped and why:**\n")
        for s in skipped_plots:
            L.append(f"- {s['plot']}: {s['reason']}")
        L.append("")

    L.append("## 6. Key findings computed by Python\n")
    L.append("*These findings are generated from templates filled directly with values from `analysis_summary.json`; they are always produced, even without an LLM.*\n")
    for i, f in enumerate(findings, 1):
        L.append(f"{i}. **[{f['type'].title()}]** {f['text']}  \n   <sub>Evidence: `{f['evidence']}`</sub>")
    L.append("")

    L.append("## 7. AI-assisted narrative insights\n")
    if llm["status"] == "ok":
        L.append(f"*Model: `{llm['model']}` via Ollama. Prompt saved in `llm_prompt.txt`, raw response in `llm_response.txt`. "
                 f"The model received only the Python-verified summary. Attempts: {llm.get('attempts', 1)}.*\n")
        if not llm.get("format_ok", True):
            L.append("> ⚠ The model did not return a numbered list of insights even after a retry, so its output below "
                     "does not meet the required format. Rely on the Python-computed findings in section 6.\n")
        L.append(llm["response"] + "\n")
        L.append("### Automatic verification of the AI narrative\n")
        L.append("Every number in the response was compared with the values in `analysis_summary.json` "
                 "(matching at the precision the model wrote). Each number must match a value Python computed for the column(s) "
                 "the claim names, or a dataset-wide value such as the row count. Statistic words (mean, median, std, min, max, "
                 "skewness, IQR, correlation) are checked against that exact statistic. \"✓ numbers verified\" does NOT mean the "
                 "wording or reasoning is correct; a person must still read every claim.\n")
        L.append("| # | Claim (truncated) | Columns named | Numbers checked | Numbers not found for those columns | Statistic mismatches | Causal wording | Status |")
        L.append("|---|---|---|---:|---|---|---|---|")
        for i, c in enumerate(llm["verification"], 1):
            bad = [n["value"] for n in c["numbers"] if not n["found_in_summary"]]
            status = "⚠ review" if bad or c["causal_language"] or c["statistic_issues"] else "✓ numbers verified"
            cols = ", ".join(f"`{x}`" for x in c["columns_named"]) or "none (weak check)"
            L.append(f"| {i} | {_md_escape(_short(c['claim'], 90))} | {_md_escape(cols)} | {len(c['numbers'])} | {', '.join(bad) or '–'} | "
                     f"{_md_escape('; '.join(c['statistic_issues'])) or '–'} | {', '.join(c['causal_language']) or '–'} | {status} |")
        v = llm["verification_summary"]
        L.append(f"\n**Verification summary:** {v['numbers_checked']} numbers checked, {v['numbers_verified']} matched the Python summary, "
                 f"{v['numbers_unverified']} did not; {v['claims_with_statistic_mismatch']} claim(s) had a statistic mismatch; "
                 f"{v['claims_with_causal_language']} claim(s) used causal wording.\n")
    else:
        L.append(f"**AI-generated narrative insights were skipped.** Reason: {llm['reason']}\n")
        L.append("The dataset overview, data-quality results, descriptive statistics, visualizations, structured summary "
                 "(`analysis_summary.json`), and the Python-computed findings above were still produced. "
                 "The prompt that would have been sent is saved in `llm_prompt.txt`.\n")

    L.append("## 8. Limitations of this automated analysis\n")
    L.append("- Type and role inference is heuristic (value parsing and column-name patterns); it can misclassify coded numbers, ZIP codes, or IDs.")
    L.append("- Column meanings, units, and valid ranges are not known unless documented; the report does not assume them.")
    L.append("- Outliers are flagged with the 1.5×IQR rule, which assumes a roughly symmetric distribution and over-flags skewed data.")
    L.append("- Pearson correlation captures only linear association, is sensitive to outliers, and never establishes causation.")
    L.append("- Sensitive-data detection is a heuristic; no warning does not mean the data is safe.")
    L.append("- The LLM narrative may still contain wording problems that the automatic number check cannot catch; it must be reviewed by a person.")
    L.append(f"\n## 9. Configuration used\n\n```json\n{json.dumps(summary['config'], indent=2)}\n```\n")

    (out / "report.md").write_text("\n".join(L), encoding="utf-8")


# ---------------------------------------------------------------------------
# 9. Main entry point
# ---------------------------------------------------------------------------
def generate_profile(csv_path, output_dir="output", use_llm=True, dataset_name=None, config=None):
    """Profile one CSV and write report.md, column_profile.csv, analysis_summary.json,
    plots/, llm_prompt.txt, and llm_response.txt into output_dir/dataset_name/.

    Returns the path of the dataset's output folder.
    """
    cfg = {**DEFAULT_CONFIG, **(config or {})}
    csv_path = Path(csv_path)
    name = _safe_name(dataset_name or csv_path.stem)
    out = Path(output_dir) / name
    if out.exists():
        shutil.rmtree(out)  # regenerate cleanly so old plots never linger
    plots_dir = out / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    print(f"[1/7] Loading {csv_path}")
    df, enc = load_csv(csv_path, cfg)
    n_rows, n_cols = df.shape

    print(f"[2/7] Inferring types for {n_cols} columns")
    profiles = [infer_column(c, df[c], n_rows, cfg) for c in df.columns]
    roles = {p["column"]: p["role"] for p in profiles}

    print("[3/7] Data-quality checks")
    quality = quality_checks(df, profiles, cfg)

    print("[4/7] Descriptive statistics and relationships")
    num_cols = [c for c, r in roles.items() if r == ROLE_NUMERIC]
    cat_cols = [c for c, r in roles.items() if r in (ROLE_CATEGORICAL, ROLE_BOOLEAN)]
    date_cols = [c for c, r in roles.items() if r == ROLE_DATE]
    nstats = [numeric_stats(c, df[c], n_rows, cfg) for c in num_cols]
    cstats = [categorical_stats(c, df[c], cfg) for c in cat_cols]
    dstats = [date_stats(c, df[c]) for c in date_cols]
    rel = relationships(df, num_cols, cfg)

    skipped_analyses = []
    if not num_cols:
        skipped_analyses.append("Numeric statistics and outlier analysis skipped: no numeric measure columns.")
    if not cat_cols:
        skipped_analyses.append("Categorical frequency analysis skipped: no categorical or Boolean columns.")
    if not rel.get("performed"):
        skipped_analyses.append(f"Correlation analysis skipped: {rel['reason']}.")
    if not date_cols:
        skipped_analyses.append("Time-based analysis skipped: no date-like columns.")

    print("[5/7] Creating visualizations")
    plots, skipped_plots = make_plots(df, profiles, nstats, cstats, rel, plots_dir, cfg)

    role_counts = {}
    for p in profiles:
        role_counts[p["role"]] = role_counts.get(p["role"], 0) + 1

    summary = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "overview": {"filename": csv_path.name, "rows": int(n_rows), "columns": int(n_cols),
                     "column_names": [str(c) for c in df.columns], "encoding": enc, "role_counts": role_counts},
        "columns": profiles,
        "data_quality": quality,
        "numeric_statistics": nstats,
        "categorical_statistics": cstats,
        "date_statistics": dstats,
        "relationships": rel,
        "plots": plots,
        "skipped_plots": skipped_plots,
        "skipped_analyses": skipped_analyses,
        "config": {k: v for k, v in cfg.items()},
    }
    findings = computed_findings(summary)
    summary["computed_findings"] = findings

    # column_profile.csv
    prof_rows = []
    nmap = {s["column"]: s for s in nstats}
    for p in profiles:
        row = {k: p.get(k) for k in ("column", "pandas_dtype", "inferred_type", "role", "non_missing", "missing",
                                     "missing_pct", "unique", "unique_ratio")}
        row["notes"] = "; ".join(p["notes"])
        for k in ("min", "q1", "median", "mean", "q3", "max", "std", "iqr", "outlier_count", "outlier_pct"):
            row[k] = nmap.get(p["column"], {}).get(k)
        prof_rows.append(row)
    profile_df = pd.DataFrame(prof_rows)
    profile_df.to_csv(out / "column_profile.csv", index=False)

    print("[6/7] LLM narrative" + ("" if use_llm else " (disabled)"))
    payload = build_llm_payload(summary)
    prompt = build_llm_prompt(payload)
    (out / "llm_prompt.txt").write_text(
        "=== SYSTEM PROMPT (rules) ===\n" + LLM_RULES + "\n\n=== USER PROMPT (verified summary + task) ===\n" + prompt,
        encoding="utf-8")
    llm = {"status": "skipped", "model": cfg["llm_model"]}
    if not use_llm:
        llm["reason"] = "the LLM component was disabled by the user (use_llm=False / --no-llm)."
    else:
        try:
            attempts, text = 0, ""
            while attempts < cfg["llm_max_attempts"]:
                attempts += 1
                p = prompt if attempts == 1 else prompt + (
                    "\n\nYour previous reply described the JSON instead of listing insights. "
                    "Reply ONLY with a numbered list: 1. ... 2. ... 3. ...")
                text = call_ollama(p, cfg)
                if text and looks_like_insight_list(text):
                    break
            if not text:
                raise RuntimeError("model returned an empty response")
            llm["attempts"] = attempts
            llm["format_ok"] = looks_like_insight_list(text)
            checks = verify_llm_text(text, summary)
            allnums = [n for c in checks for n in c["numbers"]]
            llm.update(status="ok", response=text, verification=checks, verification_summary={
                "numbers_checked": len(allnums),
                "numbers_verified": sum(n["found_in_summary"] for n in allnums),
                "numbers_unverified": sum(not n["found_in_summary"] for n in allnums),
                "claims_with_statistic_mismatch": sum(bool(c["statistic_issues"]) for c in checks),
                "claims_with_causal_language": sum(bool(c["causal_language"]) for c in checks),
            })
        except (urllib.error.URLError, ConnectionError, TimeoutError, RuntimeError, OSError, json.JSONDecodeError) as e:
            llm["reason"] = (f"the model was unavailable ({type(e).__name__}: {e}). Check that Ollama is running "
                             f"(`ollama serve`) and the model is pulled (`ollama pull {cfg['llm_model']}`).")
    (out / "llm_response.txt").write_text(
        llm["response"] if llm["status"] == "ok" else f"[LLM SKIPPED] {llm['reason']}\n", encoding="utf-8")
    summary["llm"] = {k: v for k, v in llm.items() if k != "response"}

    print("[7/7] Writing analysis_summary.json and report.md")
    (out / "analysis_summary.json").write_text(json.dumps(summary, indent=2, default=_json_default), encoding="utf-8")
    write_report(out, summary, profile_df, findings, llm, plots, skipped_plots)
    print(f"Done. Report: {out / 'report.md'}")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description="Automated CSV profiler and verified insight generator.")
    ap.add_argument("csv_path", help="Path to the CSV file to analyze")
    ap.add_argument("--output-dir", default="output", help="Parent output folder (default: output)")
    ap.add_argument("--name", default=None, help="Output subfolder name, e.g. dataset_a (default: CSV file name)")
    ap.add_argument("--no-llm", action="store_true", help="Disable the LLM narrative step")
    ap.add_argument("--model", default=DEFAULT_CONFIG["llm_model"], help="Ollama model name (default: llama3.2)")
    ap.add_argument("--ollama-url", default=DEFAULT_CONFIG["ollama_url"])
    ap.add_argument("--missing-threshold", type=float, default=DEFAULT_CONFIG["missing_threshold_pct"],
                    help="High-missingness threshold in percent (default: 30)")
    ap.add_argument("--na-values", nargs="*", default=[],
                    help='Extra strings to treat as missing, e.g. --na-values -999 unknown "not reported"')
    ap.add_argument("--num-ctx", type=int, default=DEFAULT_CONFIG["llm_num_ctx"],
                    help="Ollama context window in tokens (default: 8192)")
    ap.add_argument("--max-plots", type=int, default=DEFAULT_CONFIG["max_plots"])
    a = ap.parse_args(argv)
    cfg = {"llm_model": a.model, "ollama_url": a.ollama_url, "missing_threshold_pct": a.missing_threshold,
           "extra_na_values": a.na_values, "max_plots": a.max_plots, "llm_num_ctx": a.num_ctx}
    generate_profile(a.csv_path, a.output_dir, use_llm=not a.no_llm, dataset_name=a.name, config=cfg)


if __name__ == "__main__":
    sys.exit(main())
