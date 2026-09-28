# GenAI Development Log

Assignment #2 is Green Zone Level 4 (full GenAI use allowed). This log records the meaningful ways generative AI supported development. Runtime prompts and responses for each report are saved in `output/<dataset>/llm_prompt.txt` and `llm_response.txt`.

| Date | Tool and model | Purpose | Prompt or summary | Output used | What I verified or changed |
|---|---|---|---|---|---|
| 2026-09-25 | Claude (claude.ai; configured model `claude-opus-5-5`) | Design and write the initial profiler | Shared the full assignment PDF text and asked for help building the system, with every command and required documentation | Initial `src/profiler.py` (type inference, quality checks, statistics, adaptive plots, JSON summary, Ollama call, number verifier, report writer), `tests/test_profiler.py`, README and log templates | TODO: describe what you read through and understood, e.g. "Read every function; re-ran the tests; checked mean/median/std/outlier count for one column by hand in pandas" |
| TODO | Claude | Debugging / changes | TODO (e.g. "Asked why column X was typed as Unknown") | TODO | TODO |
| TODO | Ollama `llama3.2` | Generate report narrative for Dataset A | Structured JSON summary + rules (see `output/dataset_a/llm_prompt.txt`) | TODO (e.g. "6 insights; used in report section 7") | TODO (e.g. "Verification table flagged 1 unverified number (…) and 1 causal phrase; I confirmed against analysis_summary.json that …") |
| TODO | Ollama `llama3.2` | Generate report narrative for Dataset B | Same prompt template, Dataset B summary | TODO | TODO |

## Checks I performed on AI-generated content

- TODO: Recomputed at least one statistic independently (e.g., `df["col"].median()` in a Python shell) and compared it with the report.
- TODO: Reviewed each ⚠ row in the report's verification table and explain what was wrong.
- TODO: Confirmed the program ran on Dataset B with no changes to analysis code (only the path and `--name`).

## Unsupported or hallucinated claims removed or flagged

- TODO: list any, e.g. "The model said `score` causes `violations`; flagged as causal wording and not used as a finding."
