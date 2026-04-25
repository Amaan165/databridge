# DataBridge: Cross-City Restaurant Inspection Data Integration

A data engineering pipeline that integrates restaurant health inspection data from **New York City**, **Chicago**, and **Boston** into a unified, analysis-ready schema for cross-city food safety analysis.

## Team

- **Amaan Mansuri** — Pipeline & infrastructure lead (cleaning scripts, schema, DuckDB warehouse, orchestration, RQ3)
- **Rithujaa Rajendrakumar** — NLP & semantic intelligence lead (violation taxonomy, semantic crosswalk, RQ4 evaluation, methods writeup)
- **Vishwa Raval** — Analysis & delivery lead (Boston cleaning, inspection summaries, RQ1+RQ2, report, presentation)

## Course

NYU DS-GA 1019: Data Engineering — Spring 2026

---

## Problem

Each city independently defines its own violation taxonomy, grading system, and inspection procedures. NYC uses 168 specific violation descriptions, Chicago uses 64 broad ALL-CAPS categories, and Boston uses 313 medium-length entries with severity codes. There is no standard crosswalk between them. DataBridge builds one, using a combination of NLP embeddings, clustering, and LLM validation, so that food safety outcomes can be compared across jurisdictions.

## Data Sources

| City    | Raw rows | Cleaned rows | Source |
|---------|---------:|-------------:|--------|
| NYC     | 297,134  | 272,597      | [DOHMH Restaurant Inspections](https://data.cityofnewyork.us/Health/DOHMH-New-York-City-Restaurant-Inspection-Results/43nn-pn8j) |
| Chicago | 307,307  | 42,894       | [Chicago Food Inspections](https://data.cityofchicago.org/Health-Human-Services/Food-Inspections/4ijn-s7e5) |
| Boston  | 874,899  | 106,800      | [Boston Food Establishment Inspections](https://data.boston.gov/dataset/food-establishment-inspections) |

Window: **2022-01-01 to 2025-12-31**, set by NYC's rolling three-year disclosure rule.

## Research Questions

1. **RQ1.** How do compliance outcomes (Pass / Conditional / Fail) compare across cities on initial standard inspections?
2. **RQ2.** Which violation categories dominate in each city, and which are universal vs. city-specific?
3. **RQ3.** When restaurants fail, how often are they re-inspected, and what fraction recover on the follow-up?
4. **RQ4.** How well does the LLM-assisted semantic crosswalk align with human-labeled cross-city pairs?

---

## Pipeline overview

```
data/raw/  ──> cleaning ──> data/cleaned/  ──> warehouse ──> data/integrated/databridge.duckdb
                  │
                  ├── clean_nyc.py
                  ├── clean_chicago.py + parse_chicago_violations.py
                  ├── clean_boston.py
                  └── inspection_summaries.py

cleaned ──> taxonomy.py ──> violation_taxonomy.csv ──> crosswalk.py ──> violation_crosswalk.csv
                                                                            │
                                                                            ▼
                                                                       load_taxonomy.py
                                                                            │
                                                                            ▼
                                                                       DuckDB warehouse
                                                                            │
                                                                            ▼
                                                                       analysis/ (RQ1-RQ4)
```

The full sequence is wired up in `pipeline/run_all.py`.

## Star schema (DuckDB)

```
fact_inspections (148,807 rows) ──┬── dim_restaurants  (40,916)
                                   ├── dim_violations  (544,641) ──── dim_violation_taxonomy (64)
                                   └── dim_geography   (30)               │
                                                                          │
                                  dim_violation_crosswalk ────────────────┘
```

See [`sql/schema.sql`](sql/schema.sql) for the full DDL.

---

## Setup

```bash
# 1. Clone
git clone https://github.com/Amaan165/databridge.git
cd databridge

# 2. Install dependencies (Python 3.9-3.11)
pip install -r requirements.txt

# 3. Place raw CSVs
#    data/raw/NYC.csv
#    data/raw/Chicago.csv
#    data/raw/Boston.csv

# 4. (Optional) Set OpenAI key for LLM steps in taxonomy + crosswalk
export OPENAI_API_KEY=sk-...

# 5. Run the full pipeline
python pipeline/run_all.py
```

`run_all.py` runs cleaning, summaries, taxonomy, crosswalk, warehouse load, taxonomy load, verification, and the RQ3 analysis. Without `OPENAI_API_KEY` it falls back to Chicago-anchored cluster labels and embedding-only crosswalk candidates — the pipeline still completes end-to-end.

## Selective execution

```bash
python pipeline/run_all.py --only warehouse         # just (re)build the DuckDB
python pipeline/run_all.py --skip taxonomy --skip crosswalk
python pipeline/run_all.py --dry-run                # print the plan, don't execute
```

## Analysis

- **RQ1, RQ2:** [`analysis/cross_city_analysis.ipynb`](analysis/cross_city_analysis.ipynb)
- **RQ3:** [`analysis/rq3_reinspections.py`](analysis/rq3_reinspections.py)
- **RQ4:** [`analysis/rq4_crosswalk_eval.py`](analysis/rq4_crosswalk_eval.py)

Outputs (charts + CSVs) land in `outputs/`.

## Repository layout

```
databridge/
├── data/
│   ├── raw/                 # Original CSVs (gitignored)
│   ├── cleaned/             # Per-city cleaned + summary CSVs
│   └── integrated/          # databridge.duckdb + taxonomy + crosswalk
├── pipeline/
│   ├── clean_nyc.py
│   ├── clean_chicago.py
│   ├── parse_chicago_violations.py
│   ├── clean_boston.py
│   ├── add_reinspection_flag.py
│   ├── inspection_summaries.py
│   ├── taxonomy.py
│   ├── crosswalk.py
│   ├── load_duckdb.py
│   ├── load_taxonomy.py
│   ├── verify_duckdb.py
│   └── run_all.py
├── analysis/
│   ├── cross_city_analysis.ipynb   # RQ1, RQ2
│   ├── rq3_reinspections.py        # RQ3
│   └── rq4_crosswalk_eval.py       # RQ4
├── violation_taxonomy/
│   ├── violation_exploration.ipynb
│   ├── violation_inventory.csv
│   ├── violation_embeddings.npy
│   ├── taxonomy_categories.csv
│   └── chicago_fanout_analysis.csv
├── sql/
│   └── schema.sql
├── outputs/                 # RQ figures + CSVs
├── report/                  # Final report + slides
├── requirements.txt
└── README.md
```

## Tech Stack

- **Python** — pandas, NumPy, scikit-learn, scipy
- **sentence-transformers** — Semantic embeddings (all-MiniLM-L6-v2)
- **HDBSCAN** — Density-based clustering for violation grouping
- **GPT-4o** — LLM relabeling + crosswalk validation (optional)
- **DuckDB** — Analytical database (star schema)
- **matplotlib + seaborn** — Charts

## Key design decisions

- **Window 2022–2025.** NYC's rolling three-year disclosure rule means ~98.5% of NYC's data is post-2022; using this as the cross-city window preserves comparability without throwing away most data.
- **`is_standard_inspection` flag (NYC).** Five non-standard NYC inspection types (Smoke-Free, Trans Fat, Calorie Posting, Sodium Warning, Administrative Miscellaneous) never receive scores or grades and have no Chicago/Boston equivalents. Flagged rather than dropped, so taxonomy work can use them.
- **`is_reinspection` flag.** NYC and Chicago label re-inspections explicitly in `inspection_type`. Boston has no label; we infer one (an inspection within 30 days of a prior `HE_Fail`/`HE_FailExt` for the same license). All cross-city outcome comparisons use `is_reinspection = FALSE`.
- **Score-based outcome recovery (NYC).** Naïve grade-only mapping leaves ~55% of NYC rows without an outcome tier. Recovering from the numeric score using DOHMH thresholds (A = 0–13, B = 14–27, C = 28+) lifts coverage to ~99.1% on standard inspections. Score = 0 with violations present is treated as "not finalized," not Pass.
- **`HE_FailExt` ↦ Fail.** Boston's "extended fail" code maps to Fail, not Conditional — its severity profile matches `HE_Fail` and it appears almost exclusively on follow-up inspections.
