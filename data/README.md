# Data

Both datasets are public and non-sensitive. The CSV files are placed here and analyzed without modification.

## Dataset A – development dataset

- **Agency / organization:** Washington State Department of Licensing (published on data.wa.gov)
- **Dataset title:** Electric Vehicle Population Data
- **Source link:** https://data.wa.gov/Transportation/Electric-Vehicle-Population-Data/f6w7-q2d2
- **Date accessed:** Sep. 27th, 2026
- **Local file name:** `dataset_a.csv`
- **Size:** 271,113 rows × 16 columns
- **License / terms of use:** Open Data Commons Open Database License (ODbL) v1.0 — https://opendatacommons.org/licenses/odbl/1-0/
- **Notes:** The government CSV examined in class. Vehicle registration records: mostly categorical attributes (make, model, vehicle type, eligibility), a few numeric fields (model year, electric range, base MSRP), and geographic codes (postal code, legislative district, census tract).

## Dataset B – new dataset

- **Agency / organization:** Globe at Night (NSF NOIRLab)
- **Dataset title:** Globe at Night 2024 Observations
- **Source link:** https://globeatnight.org/maps-data/
- **Date accessed:** September 27, 2026
- **Local file name:** `dataset_b.csv`
- **Size:** 14,373 rows × 17 columns
- **License / terms of use:** Creative Commons Attribution 4.0 International (CC BY 4.0)
- **Why it is structurally different from Dataset A:** Dataset A is a large, nearly complete set of vehicle records dominated by categorical fields. Dataset B is a smaller citizen-science dataset of night-sky measurements with a date-time field, several numeric measurements, heavy missingness (`SQMReading`, `SQMSerial`), placeholder (0, 0) coordinates, and extreme numeric outliers from data entry.

## Handling

- Files were downloaded as CSV and renamed only; no rows or values were edited.
- Missing-value codes used via `--na-values`: none.
- Placeholder values were found but deliberately left in place:
  - Dataset A: `Electric Range` = 0 means "range not researched" (63.01% of rows), not a range of zero miles.
  - Dataset B: (0, 0) `Latitude`/`Longitude` means no location was entered (1,111 rows, 7.7%).
  - `--na-values` applies to every column at once, so passing `0` would also erase legitimate zeros elsewhere. These placeholders are documented as a limitation instead.