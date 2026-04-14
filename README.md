# DataBridge: Cross-City Restaurant Inspection Data Integration

A data engineering pipeline that integrates restaurant health inspection data from **New York City**, **Chicago**, and **Boston** into a unified, analysis-ready schema for cross-city food safety analysis.

## Team

- **Amaan Mansuri**
- **Rithujaa Rajendrakumar**
- **Vishwa Raval**

## Data Sources

| City | Records | Source |
|------|--------:|--------|
| NYC | 297,134 | [DOHMH Restaurant Inspections](https://data.cityofnewyork.us/Health/DOHMH-New-York-City-Restaurant-Inspection-Results/43nn-pn8j) |
| Chicago | 307,307 | [Chicago Food Inspections](https://data.cityofchicago.org/Health-Human-Services/Food-Inspections/4ijn-s7e5) |
| Boston | 874,899 | [Boston Food Establishment Inspections](https://data.boston.gov/dataset/food-establishment-inspections) |

## Problem

Each city independently defines its own violation taxonomy, grading system, and inspection procedures. NYC uses 168 specific violation descriptions, Chicago uses 64 broad ALL-CAPS categories, and Boston uses 313 medium-length entries with severity codes. There is no standard crosswalk between them. DataBridge builds one, using a combination of NLP embeddings, clustering, and LLM validation, so that food safety outcomes can be compared across jurisdictions.

## Pipeline

1. **Ingest** — Load raw inspection datasets from all three cities
2. **Clean** — Deduplicate, normalize fields, parse violations, and assign unified outcome tiers (Pass / Conditional / Fail)
3. **Harmonize** — Build a unified violation taxonomy using sentence-transformer embeddings, HDBSCAN clustering, and GPT-4o-mini validation
4. **Analyze** — Load into a DuckDB star schema and answer cross-city research questions

## Research Questions

1. How do inspection failure rates compare across cities after controlling for cuisine type and establishment size?
2. Are there seasonal patterns in violation frequency, and do they differ by city?
3. Which violation categories are most predictive of repeated failures?
4. How well do automated embedding-based crosswalks align with expert-curated mappings?
5. Do geographic clusters of violations exist within each city?

## Tech Stack

- **Python** — pandas, NumPy, scikit-learn
- **sentence-transformers** — Semantic embeddings (all-MiniLM-L6-v2)
- **HDBSCAN** — Density-based clustering for violation grouping
- **GPT-4o-mini** — LLM validation of taxonomy crosswalk
- **DuckDB** — Analytical database (star schema)

## Course

NYU DS-GA 1019: Data Engineering — Spring 2026
