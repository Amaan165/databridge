# DataBridge: Cross-City Restaurant Inspection Data Integration

**NYU DS-GA 1019 — Data Engineering, Spring 2026**
**Team:** Amaan Mansuri, Rithujaa Rajendrakumar, Vishwa Raval

---

> **Status:** Working draft. The Introduction, Data Sources, Data Quality, and
> Pipeline Architecture sections below are first-pass prose. `[BRACKETS]`
> mark places where final numbers will be substituted from the analysis
> outputs at report-assembly time. The Methods + Evaluation sections are
> drafted separately (`report/methods_draft.md`); Results, Discussion, and
> Conclusion are added later in Week 4.

---

## 1. Introduction

City-level restaurant inspection programs in the United States operate
independently. Each jurisdiction defines its own violation taxonomy,
grading scheme, inspection cadence, and disclosure format. The result is
that even though the underlying public-health regulations are very
similar across the country, the data exhaust they produce is not. A
"failed" inspection in New York City, a "Pass w/ Conditions" in Chicago,
and an `HE_FailExt` in Boston are not directly comparable — neither are
their violation codes, severity flags, or follow-up procedures.

This makes it surprisingly hard to answer basic comparative questions.
Are restaurants in one city more likely to be cited for pest issues than
in another? Do failing restaurants recover at similar rates after a
re-inspection? Which violation categories appear universally, and which
are peculiar to one city's regulatory style? To ask these questions one
first has to build a bridge.

DataBridge integrates the publicly available inspection records from
NYC, Chicago, and Boston into a unified schema designed for cross-city
analysis. The contributions are threefold:

1. **A reproducible cleaning pipeline** that reconciles the three cities'
   formats into a common per-violation representation, including
   restoration of inspection outcomes from numeric scores when grades
   are missing, parsing of Chicago's pipe-delimited free-text
   violations, and inference of Boston re-inspection events that the
   raw data does not label.

2. **A unified violation taxonomy** that uses sentence embeddings,
   density-based clustering, and LLM validation to map the cities'
   `[N_NYC=168]`, `[N_CHI=64]`, and `[N_BOS=313]` distinct violation
   descriptions to `[N_CATEGORIES=64]` shared categories — and a
   crosswalk table that records the cross-city pairs the system
   considers semantically equivalent.

3. **A DuckDB star-schema warehouse** loaded with `[N_INSPECTIONS=148807]`
   inspections and `[N_VIOLATIONS=544641]` violation instances, against
   which we run four research questions (RQ1–RQ4) covering compliance
   rates, violation patterns, re-inspection effectiveness, and an
   evaluation of the crosswalk itself.

The remainder of the report covers the data sources (§2), data quality
findings encountered during cleaning (§3), the pipeline architecture
that turns raw CSVs into the warehouse (§4), the taxonomy and crosswalk
methods (§5), the crosswalk evaluation (§6), and the cross-city results
(§7), with limitations and conclusions in §8 and §9.

---

## 2. Data Sources

The pipeline ingests three city datasets, all retrieved from each
city's open-data portal. Inspections fall in the 2022-01-01 to
2025-12-31 window; this window is set by NYC's rolling three-year
disclosure rule, which is the binding constraint on cross-city
overlap.

### 2.1 New York City

- **Source.** NYC Department of Health and Mental Hygiene (DOHMH),
  *Restaurant Inspection Results* dataset (Socrata id `43nn-pn8j`).
- **Granularity.** One row per violation found in an inspection. Clean
  inspections (no violations) appear as a single row with null
  violation fields.
- **Volume.** 297,134 raw rows × 27 columns. After cleaning:
  `[ROWS_NYC_CLEANED=272597]` rows × 20 columns.
- **Key fields.** `CAMIS` (restaurant id), `INSPECTION DATE`,
  `INSPECTION TYPE`, `VIOLATION CODE`, `VIOLATION DESCRIPTION`,
  `CRITICAL FLAG`, `SCORE`, `GRADE`, plus borough and lat/long.
- **Grading.** A letter grade A/B/C is issued during cycle inspections,
  derived from a numeric score using DOHMH thresholds: A = 0–13,
  B = 14–27, C = 28+.

### 2.2 Chicago

- **Source.** City of Chicago, *Food Inspections* dataset
  (Socrata id `4ijn-s7e5`).
- **Granularity.** One row per inspection. Violations are stored in a
  single free-text field with pipe-delimited entries
  (`32. CATEGORY - Comments: ...`).
- **Volume.** 307,307 raw rows × 17 columns. After filtering to
  restaurant facilities and to 2022-2025 the cleaned file is
  `[ROWS_CHI_CLEANED=42894]` rows; the parsed-violations sibling file
  is `[ROWS_CHI_VIOLATIONS=179641]` rows × 21 columns.
- **Key fields.** `Inspection ID`, `License #`, `Inspection Date`,
  `Inspection Type`, `Results` (Pass / Pass w/ Conditions / Fail / Out
  of Business / No Entry / Not Ready / Business Not Located),
  `Violations` (the free-text blob).
- **Grading.** No numeric score; outcome is the categorical
  `Results` field.

### 2.3 Boston

- **Source.** City of Boston, *Food Establishment Inspections*
  dataset.
- **Granularity.** One row per violation, with severity flags inline
  (`(C)` = Core, `(P)` = Priority, `(Pf)` = Priority Foundation).
- **Volume.** 874,899 raw rows × 26 columns. After filtering to FS
  (Food Service) and RF (Retail Food) license categories and to
  2022–2025: `[ROWS_BOS_CLEANED=106800]` rows × 29 columns.
- **Key fields.** `licenseno` (restaurant id), `resultdttm`,
  `result` (one of 20 result codes; we map `HE_Pass`, `HE_Fail`,
  `HE_FailExt`, `HE_Filed`, `HE_Hearing` to outcome tiers), `violation`
  + `violdesc`, `viol_level`.
- **Grading.** No numeric score and no explicit re-inspection label.

### 2.4 Unified outcome tiers

Across the three cities we map the native outcome representations to a
common three-tier vocabulary used throughout DataBridge:

| Outcome tier | NYC                | Chicago                | Boston                          |
|--------------|--------------------|------------------------|---------------------------------|
| **Pass**     | grade A / score 0–13 | `Pass`               | `HE_Pass`                       |
| **Conditional** | grade B / score 14–27 | `Pass w/ Conditions` | `HE_Filed`, `HE_Hearing`     |
| **Fail**     | grade C / score 28+  | `Fail`                | `HE_Fail`, `HE_FailExt`         |

Boston's `HE_FailExt` is mapped to **Fail**, not Conditional.
Inspection-level analysis showed `HE_FailExt` rows have the same
violation severity profile as `HE_Fail` (and a slightly higher mean
violation count), and that `HE_FailExt` appears almost exclusively
on follow-up inspections — its meaning is "still failing on follow-up,
extension to comply granted," not a conditional pass.

---

## 3. Data Quality

This section catalogues the issues discovered during cleaning and the
choices made to handle them. Three rounds of audits were performed
on the cleaning scripts; the third round in particular surfaced two
NYC-side problems that earlier passes missed (§3.1 below).

### 3.1 NYC

- **Duplicates.** 87 exact duplicate rows were dropped.
- **Sentinel dates.** 3,388 rows had `INSPECTION DATE = 01/01/1900`,
  the DOHMH placeholder for "no recent inspection." These were
  removed.
- **Sentinel coordinates.** 3,154 rows had `Latitude` or `Longitude`
  exactly equal to 0.0; these were replaced with NaN rather than
  dropped.
- **Score-based outcome recovery.** A naïve mapping of letter grades
  to outcome tiers yields tier coverage of about 45% — the rest of the
  rows have a numeric `SCORE` but no letter grade because the grade
  has not yet been issued or was administrative. We recover the
  outcome from the score using DOHMH's published thresholds, lifting
  coverage to `[NYC_TIER_COVERAGE_PCT=99.1]` % for standard
  inspections.
- **Score = 0 with violations present.** A subtler issue: roughly
  2,300 rows have `SCORE = 0` but a non-null violation code. These
  are *not* clean inspections; they are pre-finalization placeholders
  where the score has not yet been computed. The first version of the
  cleaning script mapped them to Pass, which incorrectly classified
  many critical-flag violations. The third audit added an explicit
  guard so that score = 0 is only mapped to Pass when no violation
  is on the row.
- **Non-standard inspection types.** Five inspection types
  (Administrative Miscellaneous, Smoke-Free Air Act, Trans Fat,
  Calorie Posting, Sodium Warning) never receive scores or grades and
  have no equivalent in Chicago or Boston. These ~12,000 rows are
  flagged with `is_standard_inspection = FALSE` rather than dropped,
  preserving them for taxonomy work while excluding them from
  outcome analysis.
- **Re-inspections.** NYC explicitly labels follow-ups in
  `INSPECTION TYPE` (`... / Re-inspection`); we add an
  `is_reinspection` boolean.
- **Violation description normalization.** Near-duplicate phrasings
  (case differences, extra whitespace) collapsed 219 unique strings
  to 168 by canonicalizing to the most frequent variant per uppercase
  key.

### 3.2 Chicago

- **Facility filter.** The raw file mixes restaurants with grocery
  stores, schools, hospitals, etc. We filter to facility types
  containing "restaurant," reducing the file from 307K to ~208K rows
  before further cleaning.
- **City typos.** The `City` field had 14 distinct misspellings of
  "Chicago" (including `CCHICAGO`, `CHICAGOO`, `CHCHICAGO`,
  `312CHICAGO`). These were corrected rather than dropped. Rows
  with NaN city but Chicago-bounds coordinates were also reassigned.
- **Out-of-state rows.** A small number of rows had `State` other
  than `IL`; these were removed.
- **Non-inspection results.** Rows with `Results` in
  `{Out of Business, No Entry, Not Ready, Business Not Located}` are
  not actual inspections and were dropped before outcome mapping.
- **Free-text violation parsing.** Each inspection's pipe-delimited
  violation blob was parsed into structured `(violation_number,
  violation_category, violation_comment)` triples and exploded to
  one row per violation, matching NYC and Boston granularity.
  Category text was further normalized (whitespace collapse + case
  canonicalization).
- **Re-inspections.** Chicago labels follow-ups explicitly in
  `Inspection Type` (e.g., `Canvass Re-Inspection`,
  `License Re-Inspection`); flagged via `is_reinspection`.

### 3.3 Boston

- **License category filter.** Reduced to FS (Food Service) and
  RF (Retail Food); other categories (mobile food, common victualler
  variants) were removed.
- **`viol_level` cleanup.** A handful of rows had a stray `1919` or
  blank value in the severity column; these were dropped.
- **`dbaname` field.** ~99% null and unused; dropped entirely.
- **`location` parsing.** Boston's `location` field is a single
  string `"(lat, lon)"`; parsed into separate `latitude` and
  `longitude` columns to align with NYC and Chicago.
- **Re-inspections (inferred).** Boston has no explicit re-inspection
  label. We infer one: an inspection within 30 days of a prior
  `HE_Fail` or `HE_FailExt` for the same license is flagged as a
  re-inspection. By this rule about 39% of Boston inspections are
  re-inspections, of which 71% subsequently Pass — consistent with
  the expected pattern of restaurants fixing issues after a fail.
  `HE_FailExt` appears on 18.5% of these inferred re-inspections vs
  1.2% of initial inspections, which is the empirical evidence that
  this label means "still failing on follow-up."

### 3.4 Cross-city quality summary

Because the three cities' raw schemas differ so substantially, a
single one-sentence quality statement is misleading. Instead we report
each city's row attrition and final size:

| City    | Raw rows | Cleaned rows | Inspection-level rows | Initial std. inspections |
|---------|---------:|-------------:|----------------------:|--------------------------:|
| NYC     |  297,134 | `[NYC_CLEAN]` | `[NYC_INSP]`         | `[NYC_INIT_STD]`          |
| Chicago |  307,307 | `[CHI_CLEAN]` | `[CHI_INSP]`         | `[CHI_INIT_STD]`          |
| Boston  |  874,899 | `[BOS_CLEAN]` | `[BOS_INSP]`         | `[BOS_INIT_STD]`          |

---

## 4. Pipeline Architecture

The pipeline is implemented in Python with pandas, scikit-learn,
sentence-transformers, HDBSCAN, and DuckDB. It is reproducible via
`pipeline/run_all.py`, which orchestrates the stages described below
with logging, timing, and graceful skips for LLM steps when no API key
is configured.

### 4.1 Stage diagram (data flow)

```
   data/raw/                                pipeline/
   ────────────                              ───────────
   NYC.csv      ──> clean_nyc.py        ──>  data/cleaned/nyc_cleaned.csv
   Chicago.csv  ──> clean_chicago.py    ──>  data/cleaned/chicago_cleaned.csv
                    parse_chicago...    ──>  data/cleaned/chicago_violations_parsed.csv
   Boston.csv   ──> clean_boston.py     ──>  data/cleaned/boston_cleaned.csv

                                              │
                                              ▼
                                         inspection_summaries.py
                                              │
                                              ▼
                                         data/cleaned/inspections_unified.csv

   ── Taxonomy ───────────────────────────────────────────────
   cleaned CSVs ─> taxonomy.py
                   (sentence-transformer embed
                    -> HDBSCAN cluster
                    -> Chicago-anchor labels
                    -> [optional] GPT-4o relabel)
                                              │
                                              ▼
                                       violation_taxonomy.csv
                                              │
                                              ▼
                                       crosswalk.py
                                       (cosine sim -> [optional] GPT-4o validate)
                                              │
                                              ▼
                                       violation_crosswalk.csv

   ── Warehouse ──────────────────────────────────────────────
   cleaned CSVs + taxonomy + crosswalk ──>  load_duckdb.py
                                            load_taxonomy.py
                                              │
                                              ▼
                                       data/integrated/databridge.duckdb
                                       (star schema; see §4.3)
                                              │
                                              ▼
                                       analysis/ ──> RQ1, RQ2, RQ3, RQ4
```

### 4.2 Cleaning stage

Each city has its own cleaning script (§3) producing one cleaned CSV.
Chicago has an additional parsing step that turns the pipe-delimited
violation blob into per-violation rows; this aligns Chicago's
granularity with NYC's and Boston's. `inspection_summaries.py` then
aggregates each cleaning output up to one row per inspection and
produces a unified per-city CSV for downstream consumption.

### 4.3 Star-schema warehouse

The cleaned per-violation rows are loaded into a DuckDB warehouse with
the following star schema:

- **`fact_inspections`** — one row per inspection event
  (`[N_INSPECTIONS=148807]` rows). Columns: surrogate `inspection_id`,
  `restaurant_id` (FK), `inspection_date`, `inspection_type`,
  `outcome_tier`, `outcome_source` (`grade` vs. `score` for NYC),
  `is_reinspection`, `is_standard_inspection`, `violation_count`,
  `critical_violation_count`, `city`, plus city-native fields
  (`score`, `grade`, `result_code`).

- **`dim_violations`** — one row per violation instance
  (`[N_VIOLATIONS=544641]` rows). Joined to `fact_inspections` on
  `inspection_id` and to `dim_violation_taxonomy` on
  `taxonomy_category_id`.

- **`dim_restaurants`** — deduplicated by `(city, source_id)`,
  `[N_RESTAURANTS=40916]` rows. Source IDs: `camis` (NYC),
  `license_no` (Chicago, Boston).

- **`dim_geography`** — borough (NYC), neighborhood (Boston), or
  city (Chicago) reference, with centroid coordinates.

- **`dim_violation_taxonomy`** — `[N_CATEGORIES=64]` rows produced by
  Rithujaa's HDBSCAN + LLM labeling pipeline.

- **`dim_violation_crosswalk`** — cross-city violation pairs above the
  cosine similarity threshold, with LLM-validated flag.

Surrogate integer primary keys are assigned at load time; original
source IDs (`camis`, `license_no`, `inspection_id`) are preserved on
the fact and dimension tables for traceability. Indexes are created
on the columns most commonly used in analytical queries
(`city`, `inspection_date`, `outcome_tier`, `restaurant_id`,
`is_reinspection`, `taxonomy_category_id`).

### 4.4 Reproducibility

The pipeline can be re-run end-to-end with a single command:

```
python pipeline/run_all.py
```

which executes the cleaning, taxonomy, crosswalk, warehouse-load, and
RQ3 analysis stages in order. LLM-dependent steps (taxonomy relabeling,
crosswalk validation in the 0.70–0.85 similarity band) are skipped
automatically when `OPENAI_API_KEY` is unset, falling back to
Chicago-anchored cluster labels and to embedding-only crosswalk
candidates respectively. Pinned dependency versions live in
`requirements.txt`. Verification queries (row counts, FK integrity,
violation-count consistency, sample analytical queries) live in
`pipeline/verify_duckdb.py` and are run as the last pipeline stage
before analysis.

---

## 5. Methods *(drafted in `report/methods_draft.md`; integrated at assembly)*

Cross-references the violation taxonomy construction (HDBSCAN +
Chicago-anchored labeling, optional LLM relabeling) and the semantic
crosswalk methodology (pairwise cosine similarity, LLM validation in
the 0.70–0.85 band, auto-validation above 0.85).

---

## 6. Evaluation *(drafted in `report/methods_draft.md`; integrated at assembly)*

Cross-references precision / recall on the 50-pair manually-labeled
ground-truth sample, the precision–recall curve across cosine
similarity thresholds 0.5–0.95, error analysis of false positives and
false negatives, and the embedding-only vs. embedding + LLM
comparison.

---

## 7. Results *(to be drafted in Week 4)*

Will draw from `outputs/`: `rq1_compliance_rates.{csv,png}`,
`rq2_taxonomy_heatmap.png`, `rq2_top_categories_by_city.csv`,
`rq3_recovery_by_city.png`, `rq3_summary.csv`, and the RQ4
precision–recall outputs.

---

## 8. Discussion + 9. Conclusion *(to be drafted in Week 4)*

Will cover implications for public-health enforcement comparability,
which city is "strictest" and the nuances behind that claim, known
limitations (NYC rolling-window scope, taxonomy LLM costs, Boston
neighborhood naming variability, the inferred-rather-than-labeled
nature of Boston re-inspections), and extensibility to a fourth city
(San Francisco).

---

## References *(to be added at assembly)*

NYC DOHMH dataset, Chicago Food Inspections dataset, Boston Food
Establishment Inspections dataset, sentence-transformers paper,
HDBSCAN paper, DuckDB.
