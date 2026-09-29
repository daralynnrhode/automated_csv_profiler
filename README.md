# Automated CSV Profiler and Verified Insight Generator

CSCI 5502 – AI-Enhanced Data Mining · Assignment #2: *From CSV to Evidence*
Author: Daralynn Rhode

```
CSV file → Python analysis → verified summary (JSON) → LLM explanation → generated report
```

## Purpose

This system takes any reasonably well-formed, single-table CSV and automatically produces a data profile *before* any modeling: a dataset overview, inferred column types and roles, data-quality checks, descriptive statistics with 1.5×IQR outlier flags, correlations, up to 8 adaptive plots, and a Markdown report.

**Python computes every number.** An optional local LLM (Ollama) receives only the verified JSON summary and writes a narrative. Every number in the LLM's response is then checked against the values Python computed for the column(s) that claim names, statistic words (mean, std, correlation, …) are checked against that exact statistic, and causal wording ("causes", "leads to", …) is flagged. The program never deletes rows, fills missing values, or removes outliers — it only flags them.

## Environment

- Python:  3.13.2 (developed on macOS, Apple M1 Pro)
- Libraries: `pandas`, `numpy`, `matplotlib` (see `requirements.txt`). `pytest` is only for the optional tests.
- Standard library only for the LLM call (`urllib`), so no API client or API key is needed.
- LLM: **Ollama**, model **`llama3.2`** (3B) by default, running locally on a MacBook with Apple M1 Pro. Any Ollama model can be selected with `--model`.

## Installation

Get the code, either by cloning the repository or by unzipping the submitted `automated_csv_profiler.zip`:

```bash
git clone https://github.com/daralynnrhode/automated_csv_profiler.git
cd automated_csv_profiler
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Optional LLM component
# Install Ollama from https://ollama.com, then:
ollama pull llama3.2
```

## Running the program

Run every command from the repository root, with the virtual environment active (`source .venv/bin/activate`). The two CSVs are in `data/`. If they are missing, download them from the links in **Data sources** below and save them as `data/dataset_a.csv` and `data/dataset_b.csv`.

```bash
# Dataset A (development dataset)
python src/profiler.py data/dataset_a.csv --name dataset_a

# Dataset B (new dataset) – same code, only the path and folder name change
python src/profiler.py data/dataset_b.csv --name dataset_b
```

Output is written to `output/<name>/`:

```
output/dataset_a/
    report.md               # full generated report (open in VS Code / GitHub to see plots)
    column_profile.csv      # one row per column: type, role, missingness, uniques, numeric stats
    analysis_summary.json   # the verified structured summary (what the LLM receives, plus more)
    plots/plot_01.png ...   # up to 8 adaptive visualizations
    llm_prompt.txt          # exact prompt sent to the model
    llm_response.txt        # raw model response (or the reason it was skipped)
```

Running again overwrites that dataset's folder so old plots never linger.

### Selecting a CSV

Pass the path as the first argument. Use `--name` to choose the output folder name (default: the CSV's file name). From Python or a notebook, the single entry point is:

```python
import sys; sys.path.insert(0, "src")
from profiler import generate_profile
generate_profile("data/dataset_a.csv", "output", use_llm=True, dataset_name="dataset_a")
```

### Enabling / disabling the LLM

- Enabled by default. Ollama must be running (the desktop app, or `ollama serve`).
- Disable with `--no-llm` (or `use_llm=False`).
- Choose a model with `--model llama3.2:1b` (or any pulled model).
- If Ollama is not reachable, the program still produces everything except the narrative, and the report states that AI insights were skipped and why.

### Configuration options (no code changes needed)

| Option | Default | Meaning |
|---|---|---|
| `--missing-threshold` | `30` | % missing at which a column is flagged "high missingness". 30% is a common rule of thumb: beyond it, the column rests on under ~70% of rows and simple imputation becomes unreliable. |
| `--na-values` | none | Extra strings to treat as missing, e.g. `--na-values -999 unknown "not reported"` |
| `--max-plots` | `8` | Plot budget per dataset |
| `--model` | `llama3.2` | Ollama model |
| `--num-ctx` | `8192` | Ollama context window (tokens). Ollama's default is small and silently cuts the *start* of long prompts, so it is raised here. |
| `--ollama-url` | `http://localhost:11434/api/generate` | Ollama endpoint |
| `--output-dir` | `output` | Parent output folder |

Other thresholds (type-inference parse rates, high-cardinality ratio, top-k categories) are in `DEFAULT_CONFIG` at the top of `src/profiler.py`, and the values used are printed in section 9 of every report.

### Tests (optional extension)

```bash
python -m pytest -q
```

The tests check that the mean, median, standard deviation, and IQR outlier count match an independent NumPy calculation, that the program still runs with no numeric columns, no categorical columns, and no LLM, and that the verifier flags invented numbers, statistics mislabeled for the wrong column (e.g. a "mean" for a categorical column), and causal language.

## How it works

1. **Load** with pandas (UTF-8, falling back to Latin-1). Values are never modified.
2. **Infer type and role** for each column: Boolean (two boolean-like tokens), numeric (native or ≥95% of text values parse as numbers), date-like (≥90% parse as dates), identifier-like (unique integer sequence, ID-like name, or ≥90% unique strings), free text (mostly unique and long), categorical, or unknown/mixed. Numeric-looking codes whose names suggest ZIP, FIPS, census tract, district, precinct, ward, phone, or "code" are treated as labels, not measurements.
3. **Quality checks:** duplicate rows, constant and empty columns, high missingness, mixed types, identifier-like/high-cardinality columns, and potentially sensitive columns (name and value-pattern heuristics).
4. **Statistics:** numeric (count, missing, min, Q1, median, mean, Q3, max, sample std, IQR, mode when values repeat, 1.5×IQR outliers, skewness); categorical (unique count, mode(s), top 10 counts and %).
5. **Relationships:** Pearson correlation on numeric measures (identifiers excluded), strongest positive and negative pairs with a strength label.
6. **Adaptive plots:** chosen from the roles present — missing-value bar chart, histograms, categorical bar charts, correlation heatmap, scatterplot of the strongest pair, boxplot of the column with most outliers, time series for date columns, numeric-by-category boxplots. Skipped plot types are listed with reasons.
7. **Python-computed findings:** 5–8 template findings filled directly from the summary (always produced).
8. **LLM narrative (optional):** the rules go in Ollama's system prompt, and a compact JSON summary plus a closing task reminder go in the user prompt (context window 8192 tokens). If the reply is not a numbered list, it retries once. Each claim is then verified against the column(s) it names, and statistic words are checked against the matching Python statistic.
9. **Report:** `report.md` is written entirely by the program.

## Known limitations

- Type/role inference is heuristic and can misclassify coded numbers (e.g. a 0/1/2 class label is treated as numeric, but noted), ZIP codes without a telling column name, or IDs.
- Column meanings, units, and valid ranges are not known and are never assumed.
- The 1.5×IQR rule assumes roughly symmetric data and over-flags skewed columns.
- Pearson correlation captures only linear association and is sensitive to outliers; it never shows causation.
- Sensitive-data detection is a heuristic; no warning does not mean the data is safe.
- The number verifier checks each number against the columns a claim names and checks statistic words (mean, std, correlation, …) against that exact statistic, but it cannot judge wording or reasoning. Examples it passed: "0.02% of rows" (the value is really a share of cells) and a correct r = −0.544 that is actually caused by placeholder zeros. A person must still review every claim.
- Placeholder codes are treated as real values unless passed with `--na-values`, which applies to every column at once. Example: `Electric Range` = 0 means "not researched" in Dataset A, and (0, 0) coordinates mean "no location" in Dataset B.
- Small local models may ignore instructions (e.g., produce fewer insights); the verification table makes this visible.
- Only single-table CSV files that fit in memory are supported.

## Data sources

| | Dataset A (development) | Dataset B (new) |
|---|---|---|
| Agency / organization | Washington State Department of Licensing | Globe at Night (NSF NOIRLab) |
| Dataset title | Electric Vehicle Population Data | Globe at Night 2024 Observations |
| Source link | https://data.wa.gov/Transportation/Electric-Vehicle-Population-Data/f6w7-q2d2 | https://globeatnight.org/maps-data/ |
| Date accessed | September 27, 2026 | September 27, 2026 |
| Local file | `data/dataset_a.csv` | `data/dataset_b.csv` |
| Rows × columns | 271,113 × 16 | 14,373 × 17 |
| How they differ | Large, nearly complete, mostly categorical vehicle records | Smaller citizen-science sky measurements; heavy missingness, placeholder coordinates, extreme outliers |

See `data/README.md` for details.

## Repository structure

```
automated_csv_profiler/
    README.md
    requirements.txt
    .gitignore
    src/profiler.py
    tests/test_profiler.py
    data/README.md
    output/dataset_a/  output/dataset_b/
    video/video_link.txt
    logs/genai_log.md
```

No API keys, passwords, or tokens are used or stored in this repository.
