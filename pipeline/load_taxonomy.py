"""
load_taxonomy.py — Task 2.2: Load Violation Taxonomy & Crosswalk into DuckDB
DataBridge Project | Data Engineering Spring 2026

Extends the DuckDB star schema with Rithujaa's violation taxonomy
(HDBSCAN clusters + LLM labels) and cross-city crosswalk table.
Also backfills dim_violations.taxonomy_category_id with the FK.

Depends on:
  - Task 2.1 complete (databridge.duckdb exists with schema)
  - Rithujaa's Task 2.3 output: data/integrated/violation_taxonomy.csv
  - Rithujaa's Task 2.4 output: data/integrated/violation_crosswalk.csv

Expected input formats:

  violation_taxonomy.csv columns:
    violation_text, city, cluster_id, taxonomy_category_id, category_name

  violation_crosswalk.csv columns:
    violation_desc_city_a, city_a, violation_desc_city_b, city_b,
    cosine_similarity, llm_validated, taxonomy_category_id

Usage:
  cd databridge/
  python pipeline/load_taxonomy.py

  Options:
    --db PATH              DuckDB path (default: data/integrated/databridge.duckdb)
    --taxonomy PATH        Taxonomy CSV (default: data/integrated/violation_taxonomy.csv)
    --crosswalk PATH       Crosswalk CSV (default: data/integrated/violation_crosswalk.csv)
    --skip-backfill        Skip updating dim_violations FK (for testing)
"""

import argparse
import os
import sys
import time

import duckdb
import pandas as pd
import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

DB_PATH = os.path.join("data", "integrated", "databridge.duckdb")
TAXONOMY_PATH = os.path.join("data", "integrated", "violation_taxonomy.csv")
CROSSWALK_PATH = os.path.join("data", "integrated", "violation_crosswalk.csv")


# ─────────────────────────────────────────────────────────────────────────────
# Step 1: Load and validate taxonomy categories
# ─────────────────────────────────────────────────────────────────────────────

def load_taxonomy(path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load taxonomy CSV and extract unique categories for dim_violation_taxonomy.

    Returns:
        categories_df: unique (taxonomy_category_id, category_name) for the dim table
        taxonomy_df: full taxonomy with violation_text → category mapping
    """
    print(f"  Loading {path}...")
    df = pd.read_csv(path)
    print(f"  {len(df):,} violation-to-category mappings loaded")

    # Expected columns
    required = {"violation_text", "city", "taxonomy_category_id", "category_name"}
    missing = required - set(df.columns)
    if missing:
        print(f"  ERROR: Missing columns: {missing}")
        print(f"  Found columns: {list(df.columns)}")
        sys.exit(1)

    # Extract unique categories
    categories = (
        df[["taxonomy_category_id", "category_name"]]
        .drop_duplicates()
        .sort_values("taxonomy_category_id")
        .reset_index(drop=True)
    )
    # Add description placeholder (can be enriched later)
    categories["category_description"] = None

    print(f"  {len(categories)} unique taxonomy categories:")
    for _, row in categories.iterrows():
        count = (df["taxonomy_category_id"] == row["taxonomy_category_id"]).sum()
        print(f"    [{row['taxonomy_category_id']:>3}] {row['category_name']:40s} "
              f"({count:,} violations)")

    # Coverage by city
    print(f"\n  Coverage by city:")
    for city in sorted(df["city"].unique()):
        city_total = len(df[df["city"] == city])
        city_mapped = df[(df["city"] == city) & df["taxonomy_category_id"].notna()].shape[0]
        print(f"    {city:15s} {city_mapped:>6,} / {city_total:>6,} "
              f"({city_mapped / city_total * 100:.1f}%)")

    return categories, df


# ─────────────────────────────────────────────────────────────────────────────
# Step 2: Load crosswalk
# ─────────────────────────────────────────────────────────────────────────────

def load_crosswalk(path: str) -> pd.DataFrame:
    """Load crosswalk CSV for dim_violation_crosswalk."""
    print(f"\n  Loading {path}...")
    df = pd.read_csv(path)
    print(f"  {len(df):,} crosswalk pairs loaded")

    required = {
        "violation_desc_city_a", "city_a",
        "violation_desc_city_b", "city_b",
    }
    missing = required - set(df.columns)
    if missing:
        print(f"  ERROR: Missing columns: {missing}")
        sys.exit(1)

    # Ensure expected columns exist with defaults
    if "cosine_similarity" not in df.columns:
        df["cosine_similarity"] = np.nan
    if "match_validated" not in df.columns:
        df["match_validated"] = None
    if "taxonomy_category_id" not in df.columns:
        df["taxonomy_category_id"] = None

    # Convert match_validated to boolean if string
    if df["match_validated"].dtype == object:
        df["match_validated"] = df["match_validated"].str.lower().map(
            {"yes": True, "true": True, "no": False, "false": False}
        )

    # Add surrogate PK
    df.insert(0, "crosswalk_id", range(1, len(df) + 1))

    # Summary
    validated = df["match_validated"].sum() if df["match_validated"].notna().any() else 0
    print(f"  LLM-validated matches: {validated:,}")
    print(f"  Mean cosine similarity: {df['cosine_similarity'].mean():.3f}")

    # City pair breakdown
    print(f"\n  Crosswalk pairs by city pair:")
    pairs = df.groupby(["city_a", "city_b"]).size().reset_index(name="count")
    for _, row in pairs.iterrows():
        print(f"    {row['city_a']:>10s} ↔ {row['city_b']:<10s} {row['count']:>6,}")

    return df


# ─────────────────────────────────────────────────────────────────────────────
# Step 3: Insert into DuckDB and backfill FKs
# ─────────────────────────────────────────────────────────────────────────────

def insert_into_duckdb(db_path: str, categories_df: pd.DataFrame,
                        taxonomy_df: pd.DataFrame, crosswalk_df: pd.DataFrame,
                        skip_backfill: bool = False):
    """Insert taxonomy and crosswalk into DuckDB, backfill dim_violations FK."""
    con = duckdb.connect(db_path)

    # ── Clear existing data (idempotent reload) ────────────────────────────
    con.execute("DELETE FROM dim_violation_crosswalk")
    con.execute("DELETE FROM dim_violation_taxonomy")
    if not skip_backfill:
        con.execute("UPDATE dim_violations SET taxonomy_category_id = NULL")
    print(f"\n  Cleared existing taxonomy/crosswalk data")

    # ── Load dim_violation_taxonomy ────────────────────────────────────────
    cat_load = categories_df[["taxonomy_category_id", "category_name",
                               "category_description"]].copy()
    con.execute("INSERT INTO dim_violation_taxonomy SELECT * FROM cat_load")
    count = con.execute("SELECT COUNT(*) FROM dim_violation_taxonomy").fetchone()[0]
    print(f"  dim_violation_taxonomy: {count} categories loaded")

    # ── Load dim_violation_crosswalk ───────────────────────────────────────
    cw_load = crosswalk_df[["crosswalk_id", "violation_desc_city_a", "city_a",
                             "violation_desc_city_b", "city_b",
                             "cosine_similarity", "match_validated",
                             "taxonomy_category_id"]].copy()
    con.execute("INSERT INTO dim_violation_crosswalk SELECT * FROM cw_load")
    count = con.execute("SELECT COUNT(*) FROM dim_violation_crosswalk").fetchone()[0]
    print(f"  dim_violation_crosswalk: {count} pairs loaded")

    # ── Backfill dim_violations.taxonomy_category_id ───────────────────────
    if not skip_backfill:
        print(f"\n  Backfilling dim_violations.taxonomy_category_id...")

        # Build lookup: (violation_description, city) → taxonomy_category_id
        tax_lookup = taxonomy_df[["violation_text", "city", "taxonomy_category_id"]].copy()
        tax_lookup = tax_lookup.rename(columns={"violation_text": "violation_description"})
        tax_lookup = tax_lookup.dropna(subset=["taxonomy_category_id"])

        # Register as temp table and update via JOIN
        con.execute("CREATE TEMP TABLE tax_lookup AS SELECT * FROM tax_lookup")
        updated = con.execute("""
            UPDATE dim_violations v
            SET taxonomy_category_id = t.taxonomy_category_id
            FROM tax_lookup t
            WHERE v.violation_description = t.violation_description
              AND v.city = t.city
        """)

        # Report coverage
        total = con.execute("SELECT COUNT(*) FROM dim_violations").fetchone()[0]
        filled = con.execute(
            "SELECT COUNT(*) FROM dim_violations WHERE taxonomy_category_id IS NOT NULL"
        ).fetchone()[0]
        print(f"  Backfill complete: {filled:,} / {total:,} violations "
              f"have taxonomy_category_id ({filled / total * 100:.1f}%)")

        # Per-city coverage
        print(f"\n  Backfill coverage by city:")
        for row in con.execute("""
            SELECT city,
                   COUNT(*) AS total,
                   SUM(CASE WHEN taxonomy_category_id IS NOT NULL THEN 1 ELSE 0 END) AS filled
            FROM dim_violations
            GROUP BY city ORDER BY city
        """).fetchall():
            pct = row[2] / row[1] * 100 if row[1] > 0 else 0
            print(f"    {row[0]:15s} {row[2]:>10,} / {row[1]:>10,}  ({pct:.1f}%)")

        con.execute("DROP TABLE IF EXISTS tax_lookup")

    # ── Final summary ─────────────────────────────────────────────────────
    print(f"\n" + "=" * 65)
    print("TAXONOMY & CROSSWALK LOAD SUMMARY")
    print("=" * 65)
    for table in ["dim_violation_taxonomy", "dim_violation_crosswalk"]:
        count = con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        print(f"  {table:30s} {count:>8,} rows")

    if not skip_backfill:
        filled = con.execute(
            "SELECT COUNT(*) FROM dim_violations WHERE taxonomy_category_id IS NOT NULL"
        ).fetchone()[0]
        total = con.execute("SELECT COUNT(*) FROM dim_violations").fetchone()[0]
        print(f"  dim_violations FK coverage:  {filled:,} / {total:,} ({filled/total*100:.1f}%)")
    print("=" * 65)

    con.close()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Load violation taxonomy and crosswalk into DuckDB."
    )
    parser.add_argument("--db", default=DB_PATH, help="DuckDB database path")
    parser.add_argument("--taxonomy", default=TAXONOMY_PATH,
                        help="Taxonomy CSV path")
    parser.add_argument("--crosswalk", default=CROSSWALK_PATH,
                        help="Crosswalk CSV path")
    parser.add_argument("--skip-backfill", action="store_true",
                        help="Skip updating dim_violations FK")
    args = parser.parse_args()

    print("=" * 65)
    print("DataBridge — Load Taxonomy & Crosswalk into DuckDB")
    print("=" * 65)

    # Verify database exists
    if not os.path.exists(args.db):
        print(f"ERROR: Database not found: {args.db}")
        print("Run load_duckdb.py first to create the base schema.")
        sys.exit(1)

    # Check for input files
    for label, path in [("Taxonomy", args.taxonomy),
                        ("Crosswalk", args.crosswalk)]:
        if not os.path.exists(path):
            print(f"ERROR: {label} file not found: {path}")
            print("Ensure Rithujaa's taxonomy/crosswalk scripts have been run.")
            sys.exit(1)

    start = time.time()

    # Load and validate
    categories_df, taxonomy_df = load_taxonomy(args.taxonomy)
    crosswalk_df = load_crosswalk(args.crosswalk)

    # Insert into DuckDB
    insert_into_duckdb(args.db, categories_df, taxonomy_df, crosswalk_df,
                        skip_backfill=args.skip_backfill)

    elapsed = time.time() - start
    print(f"\nTotal time: {elapsed:.1f}s")
    print("Done.")


if __name__ == "__main__":
    main()
