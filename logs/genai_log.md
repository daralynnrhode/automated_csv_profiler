# GenAI Development Log

Assignment #2 is Green Zone Level 4 (full GenAI use allowed). This log records the meaningful ways generative AI supported development. Runtime prompts and responses for each report are saved in `output/<dataset>/llm_prompt.txt` and `llm_response.txt`.
 
| Date | Tool and model | Purpose | Prompt or summary | Output used | What I verified or changed |
|---|---|---|---|---|---|
| 2026-09-25 | Claude (claude.ai; configured model `claude-opus-5-5`) | Design and write the initial profiler | Shared the full assignment text and asked for help building the system, with every command and required documentation | Initial `src/profiler.py`, `tests/test_profiler.py`, README and log templates | Read through each function to understand the pipeline; ran the tests (5 passed); ran it on both datasets and checked results by hand in pandas (see Checks below) |
| 2026-09-27 | Claude | Fix test warnings | Shared pytest warnings (regex match-group warning; Matplotlib `vert` deprecation) | Non-capturing regex group; version-aware boxplot orientation | Reran pytest: 5 passed, no warnings |
| 2026-09-27 | Ollama `llama3.2` | Narrative for Dataset A (run 1) | Rules followed by the JSON summary in one prompt | Not used: the model described the JSON instead of writing insights | Cause: Ollama's small default context window cut off the instructions at the start of the prompt |
| 2026-09-27 | Claude | Fix the LLM prompt | Shared the bad response | Rules moved to the system prompt, context raised to 8192, task repeated at the end, one retry added | Reran Dataset A: the response became a numbered list |
| 2026-09-27 | Claude | Treat geographic codes as labels | Showed insights analyzing `Legislative District` and `2020 Census Tract` as measurements (mean, skewness, correlation) | Wider code-name pattern (census, tract, district, precinct, ward, FIPS, code) | Checked `column_profile.csv`: district became categorical, census tract identifier-like |
| 2026-09-27 | Ollama `llama3.2` | Narrative for Dataset A (run 2) | Same verified summary | Not used | The old verifier marked "`Legislative District` mean 41.0, std 4.34" as verified, but Python computes no mean for a categorical column. It matched by coincidence: 41 is a district value and 4.34 is its percentage |
| 2026-09-27 | Claude | Make the verifier column- and statistic-aware | Shared the verification table that missed the hallucination | New `verify_llm_text()` and an updated test | New test fails on the old verifier and passes on the new one; reran Dataset A |
| 2026-09-27 | Ollama `llama3.2` | Narrative for Dataset A (final) | Same verified summary | 6 insights in report section 7, reviewed below | Every number matched; wording problems and one misleading correlation noted below |
| 2026-09-27 | Ollama `llama3.2` | Narrative for Dataset B | Same prompt template, Dataset B summary | 8 insights in report section 7, reviewed below | Every number matched, but 2 claims were unsupported and 2 were misleading (below) |

## Checks I performed on AI-generated content

- **Dataset A, insight 4:** recomputed the `Model Year`–`Electric Range` Pearson correlation in pandas: −0.544, matching the report. 63.01% of `Electric Range` values are 0, the same as the share of "Eligibility unknown as battery range has not been researched" in the CAFV column. Only 0.1% of pre-2021 vehicles have range 0, compared with 82.3% of 2021-and-newer vehicles. The correlation reflects placeholder coding, not a real relationship.
- **Dataset B, insight 7:** `SQMReading` mean 37,651,172 is verified, but the pandas median is 18.83, 71 readings exceed 25, and the maximum is 9.46×10¹⁰. These are data-entry errors; the median is the appropriate center.
- **Dataset B, insight 6:** `SQMReading`–`LimitingMag` r = 0.029 is verified, but using only plausible readings (10–25) gives r = 0.974. The weak correlation comes from extreme values, which shows Pearson's sensitivity to outliers. The 0.974 is unusually high; one value may be derived from the other, which is undocumented.
- **Dataset B:** 1,111 rows (7.7%) have (0, 0) coordinates, a placeholder for a missing location.
- Confirmed Dataset B ran with no changes to the analysis code (only the path and `--name`).

## Unsupported or hallucinated claims removed or flagged

- **Dataset A, run 2:** "`Legislative District` has a mean of 41.0 and a standard deviation of 4.34". Mislabeled numbers; not used.
- **Dataset A, run 2:** "Unique `DOL Vehicle ID` values may indicate duplication or errors". The reasoning is wrong: unique IDs are expected.
- **Dataset A, final, insight 1:** "0.02% of the total rows". The value is the share of missing cells, not rows.
- **Dataset B, insight 2:** "`Latitude` … relatively normal distribution". A mean and std cannot show normality.
- **Dataset B, insight 8:** "top `Country` categories account for a small percentage". False: the top 3 cover 38.2%. The column also mixes countries and US states.
- **Dataset B, insight 5:** a generic limitation that isn't specific to this data. It was kept, but the report's Python findings give the data-specific limitations.