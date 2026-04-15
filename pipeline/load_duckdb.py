"""
load_duckdb.py — Task 2.1: Load Cleaned CSVs into DuckDB Star Schema
DataBridge Project | Data Engineering Spring 2026

Reads the three cleaned city CSVs (+ Chicago parsed violations),
transforms them into a unified star schema, and loads into DuckDB.

Input files:
  data/cleaned/nyc_cleaned.csv               (272,597 × 20)
  data/cleaned/chicago_cleaned.csv           (42,894 × 19)
  data/cleaned/chicago_violations_parsed.csv (179,641 × 21)
  data/cleaned/boston_cleaned.csv            (106,800 × 29)

Output:
  data/integrated/databridge.duckdb

Tables created (see sql/schema.sql for DDL):
  dim_geography           — city + sub-geography reference
  dim_restaurants         — deduplicated restaurant dimension
  fact_inspections        — one row per inspection (aggregated)
  dim_violations          — one row per violation instance
  dim_violation_taxonomy  — empty placeholder (Rithujaa Week 2)
  dim_violation_crosswalk — empty placeholder (Rithujaa Week 2)

Usage:
  cd databridge/
  python pipeline/load_duckdb.py

  Options:
    --db PATH       DuckDB output path (default: data/integrated/databridge.duckdb)
    --cleaned-dir   Directory with cleaned CSVs (default: data/cleaned)
    --schema        Path to schema SQL (default: sql/schema.sql)
"""

import argparse
import os
import sys
import time
from pathlib import Path

import duckdb
import pandas as pd
import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

CLEANED_DIR = os.path.join("data", "cleaned")
DB_PATH = os.path.join("data", "integrated", "databridge.duckdb")
SCHEMA_PATH = os.path.join("sql", "schema.sql")


# ─────────────────────────────────────────────────────────────────────────────
# Step 1: Load raw cleaned CSVs
# ─────────────────────────────────────────────────────────────────────────────

def load_cleaned_csvs(cleaned_dir: str) -> dict:
    """Load all cleaned CSVs into DataFrames."""
    files = {
        "nyc":               os.path.join(cleaned_dir, "nyc_cleaned.csv"),
        "chicago":           os.path.join(cleaned_dir, "chicago_cleaned.csv"),
        "chicago_violations": os.path.join(cleaned_dir, "chicago_violations_parsed.csv"),
        "boston":             os.path.join(cleaned_dir, "boston_cleaned.csv"),
    }

    dfs = {}
    for key, path in files.items():
        if not os.path.exists(path):
            print(f"  WARNING: {path} not found. Skipping {key}.")
            continue
        print(f"  Loading {path}...", end="", flush=True)
        df = pd.read_csv(path, low_memory=False)
        print(f" {len(df):,} rows × {df.shape[1]} cols")
        dfs[key] = df

    return dfs


# ─────────────────────────────────────────────────────────────────────────────
# Step 2: Build dim_geography
# ─────────────────────────────────────────────────────────────────────────────

def build_dim_geography(dfs: dict) -> pd.DataFrame:
    """Build geography dimension from all cities."""
    geo_parts = []

    # NYC — boroughs
    if "nyc" in dfs:
        nyc = dfs["nyc"]
        nyc_geo = (
            nyc.groupby("boro")
            .agg(latitude_centroid=("latitude", "mean"),
                 longitude_centroid=("longitude", "mean"))
            .reset_index()
            .rename(columns={"boro": "sub_geography"})
        )
        nyc_geo["city"] = "nyc"
        geo_parts.append(nyc_geo)

    # Chicago — single city-level entry
    if "chicago" in dfs:
        chi = dfs["chicago"]
        geo_parts.append(pd.DataFrame([{
            "city": "chicago",
            "sub_geography": "CHICAGO",
            "latitude_centroid": chi["latitude"].mean(),
            "longitude_centroid": chi["longitude"].mean(),
        }]))

    # Boston — neighborhoods
    if "boston" in dfs:
        bos = dfs["boston"]
        bos_geo = (
            bos.groupby("neighborhood")
            .agg(latitude_centroid=("latitude", "mean"),
                 longitude_centroid=("longitude", "mean"))
            .reset_index()
            .rename(columns={"neighborhood": "sub_geography"})
        )
        bos_geo["city"] = "boston"
        geo_parts.append(bos_geo)

    geo_df = pd.concat(geo_parts, ignore_index=True)
    geo_df.insert(0, "geo_id", range(1, len(geo_df) + 1))
    # Match schema column order: geo_id, city, sub_geography, lat, lon
    geo_df = geo_df[["geo_id", "city", "sub_geography",
                      "latitude_centroid", "longitude_centroid"]]
    return geo_df


# ─────────────────────────────────────────────────────────────────────────────
# Step 3: Build dim_restaurants
# ─────────────────────────────────────────────────────────────────────────────

def normalize_zipcode(val) -> str | None:
    """Safely convert a zipcode to a 5-digit string.

    Handles: int (10024), float (10024.0), str ('02119'),
    ZIP+4 ('02119-3212'), and NaN/None.
    """
    if pd.isna(val):
        return None
    s = str(val).strip().split("-")[0].split(".")[0]  # drop +4 suffix and .0
    if not s or not s.isdigit():
        return None
    return s.zfill(5)  # left-pad with zeros (Boston: 2124 → 02124)


def build_dim_restaurants(dfs: dict, geo_df: pd.DataFrame) -> pd.DataFrame:
    """Build restaurant dimension, deduplicated by (city, source_id)."""
    geo_lookup = dict(zip(
        zip(geo_df["city"], geo_df["sub_geography"]),
        geo_df["geo_id"]
    ))
    rest_parts = []

    # NYC — deduplicate by camis, keep most recent row's metadata
    if "nyc" in dfs:
        nyc = dfs["nyc"].sort_values("inspection_date", ascending=False)
        nyc_rest = nyc.drop_duplicates(subset=["camis"], keep="first").copy()
        nyc_rest["source_id"] = nyc_rest["camis"].astype(int).astype(str)
        nyc_rest["zipcode_str"] = nyc_rest["zipcode"].apply(normalize_zipcode)
        nyc_rest["geo_id"] = nyc_rest["boro"].map(
            lambda b: geo_lookup.get(("nyc", b))
        )
        # Use address if available (added in clean_nyc.py v4), else None
        if "address" not in nyc_rest.columns:
            nyc_rest["address"] = None
        rest_parts.append(nyc_rest[[
            "source_id", "dba", "address", "latitude", "longitude",
            "boro", "zipcode_str", "cuisine_description", "geo_id"
        ]].rename(columns={
            "dba": "name", "boro": "sub_geography",
            "zipcode_str": "zipcode", "cuisine_description": "cuisine_type",
        }).assign(city="nyc"))

    # Chicago — deduplicate by license_no
    if "chicago" in dfs:
        chi = dfs["chicago"].copy()
        chi["license_no"] = chi["license_no"].astype("Int64")
        chi_sorted = chi.sort_values("inspection_date", ascending=False)
        chi_rest = chi_sorted.drop_duplicates(subset=["license_no"], keep="first").copy()
        chi_rest["source_id"] = chi_rest["license_no"].astype(str)
        chi_rest["zipcode_str"] = chi_rest["zipcode"].apply(normalize_zipcode)
        chi_geo_id = geo_lookup.get(("chicago", "CHICAGO"))
        rest_parts.append(chi_rest[[
            "source_id", "dba_name", "address", "latitude", "longitude",
            "zipcode_str"
        ]].rename(columns={
            "dba_name": "name", "zipcode_str": "zipcode",
        }).assign(city="chicago", sub_geography="CHICAGO",
                  cuisine_type=None, geo_id=chi_geo_id))

    # Boston — deduplicate by license_no
    if "boston" in dfs:
        bos = dfs["boston"].sort_values("inspection_date", ascending=False)
        bos_rest = bos.drop_duplicates(subset=["license_no"], keep="first").copy()
        bos_rest["source_id"] = bos_rest["license_no"].astype(str)
        bos_rest["zipcode_str"] = bos_rest["zipcode"].apply(normalize_zipcode)
        bos_rest["geo_id"] = bos_rest["neighborhood"].map(
            lambda n: geo_lookup.get(("boston", n))
        )
        rest_parts.append(bos_rest[[
            "source_id", "business_name", "address", "latitude", "longitude",
            "neighborhood", "zipcode_str", "geo_id"
        ]].rename(columns={
            "business_name": "name", "neighborhood": "sub_geography",
            "zipcode_str": "zipcode",
        }).assign(city="boston", cuisine_type=None))

    # Combine and assign surrogate IDs
    rest_df = pd.concat(rest_parts, ignore_index=True)
    rest_df.insert(0, "restaurant_id", range(1, len(rest_df) + 1))

    # Ensure column order matches schema
    rest_df = rest_df[[
        "restaurant_id", "source_id", "name", "address",
        "latitude", "longitude", "city", "sub_geography",
        "zipcode", "cuisine_type", "geo_id"
    ]]

    return rest_df


# ─────────────────────────────────────────────────────────────────────────────
# Step 4: Build fact_inspections
# ─────────────────────────────────────────────────────────────────────────────

def build_fact_inspections(dfs: dict, rest_df: pd.DataFrame) -> pd.DataFrame:
    """Build inspection fact table — one row per inspection event."""
    rest_lookup = dict(zip(
        zip(rest_df["city"], rest_df["source_id"]),
        rest_df["restaurant_id"]
    ))
    insp_parts = []

    # ── NYC ────────────────────────────────────────────────────────────────
    if "nyc" in dfs:
        nyc = dfs["nyc"].copy()
        nyc_g = nyc.groupby(["camis", "inspection_date"], sort=False).agg(
            inspection_type=("inspection_type", "first"),
            is_reinspection=("is_reinspection", "first"),
            is_standard_inspection=("is_standard_inspection", "first"),
            outcome_tier=("outcome_tier", "first"),
            outcome_source=("outcome_source", "first"),
            score=("score", "first"),
            grade=("grade", "first"),
            violation_count=("violation_code", lambda x: x.notna().sum()),
            critical_violation_count=("critical_flag", lambda x: (x == "Critical").sum()),
        ).reset_index()

        nyc_g["source_id"] = nyc_g["camis"].astype(int).astype(str)
        nyc_g["restaurant_id"] = [
            rest_lookup.get(("nyc", sid)) for sid in nyc_g["source_id"]
        ]
        nyc_g["source_inspection_id"] = (
            "nyc_" + nyc_g["source_id"] + "_" + nyc_g["inspection_date"]
        )
        nyc_g["city"] = "nyc"
        nyc_g["result_code"] = None

        insp_parts.append(nyc_g[[
            "restaurant_id", "source_inspection_id", "inspection_date",
            "inspection_type", "is_reinspection", "is_standard_inspection",
            "outcome_tier", "outcome_source", "score", "grade",
            "result_code", "violation_count", "critical_violation_count", "city"
        ]])

    # ── Chicago ────────────────────────────────────────────────────────────
    if "chicago" in dfs:
        chi = dfs["chicago"].copy()
        chi["license_no"] = chi["license_no"].astype("Int64")

        # Get violation counts from parsed file
        chi_viol_counts = pd.Series(dtype=int)
        if "chicago_violations" in dfs:
            cv = dfs["chicago_violations"]
            chi_viol_counts = (
                cv[cv["violation_category"].notna()]
                .groupby("inspection_id")
                .size()
            )

        chi["source_id"] = chi["license_no"].astype(str)
        chi["restaurant_id"] = [
            rest_lookup.get(("chicago", sid)) for sid in chi["source_id"]
        ]
        chi["source_inspection_id"] = "chi_" + chi["inspection_id"].astype(str)
        chi["violation_count"] = (
            chi["inspection_id"].map(chi_viol_counts).fillna(0).astype(int)
        )
        chi["critical_violation_count"] = 0
        chi["is_standard_inspection"] = True
        chi["outcome_source"] = None
        chi["score"] = np.nan
        chi["grade"] = None
        chi["city_col"] = "chicago"

        insp_parts.append(chi[[
            "restaurant_id", "source_inspection_id", "inspection_date",
            "inspection_type", "is_reinspection", "is_standard_inspection",
            "outcome_tier", "outcome_source", "score", "grade",
            "results", "violation_count", "critical_violation_count", "city_col"
        ]].rename(columns={"results": "result_code", "city_col": "city"}))

    # ── Boston ─────────────────────────────────────────────────────────────
    if "boston" in dfs:
        bos = dfs["boston"].copy()
        bos_g = bos.groupby(["license_no", "inspection_date"], sort=False).agg(
            is_reinspection=("is_reinspection", "first"),
            outcome_tier=("outcome_tier", "first"),
            result_code=("result_code", "first"),
            violation_count=("violation_code", lambda x: x.notna().sum()),
            critical_violation_count=(
                "violation_severity", lambda x: (x == "***").sum()
            ),
        ).reset_index()

        bos_g["source_id"] = bos_g["license_no"].astype(str)
        bos_g["restaurant_id"] = [
            rest_lookup.get(("boston", sid)) for sid in bos_g["source_id"]
        ]
        bos_g["source_inspection_id"] = (
            "bos_" + bos_g["source_id"] + "_" + bos_g["inspection_date"]
        )
        bos_g["inspection_type"] = None
        bos_g["is_standard_inspection"] = True
        bos_g["outcome_source"] = None
        bos_g["score"] = np.nan
        bos_g["grade"] = None
        bos_g["city"] = "boston"

        insp_parts.append(bos_g[[
            "restaurant_id", "source_inspection_id", "inspection_date",
            "inspection_type", "is_reinspection", "is_standard_inspection",
            "outcome_tier", "outcome_source", "score", "grade",
            "result_code", "violation_count", "critical_violation_count", "city"
        ]])

    # Combine and assign surrogate IDs
    insp_df = pd.concat(insp_parts, ignore_index=True)
    insp_df.insert(0, "inspection_id", range(1, len(insp_df) + 1))

    # Cast types
    insp_df["violation_count"] = insp_df["violation_count"].astype(int)
    insp_df["critical_violation_count"] = (
        insp_df["critical_violation_count"].astype(int)
    )

    return insp_df


# ─────────────────────────────────────────────────────────────────────────────
# Step 5: Build dim_violations
# ─────────────────────────────────────────────────────────────────────────────

def build_dim_violations(dfs: dict, insp_df: pd.DataFrame) -> pd.DataFrame:
    """Build violation dimension — one row per violation instance."""

    # Inspection lookup: source_inspection_id → surrogate inspection_id
    insp_lookup = dict(zip(
        insp_df["source_inspection_id"],
        insp_df["inspection_id"]
    ))

    viol_parts = []

    # ── NYC ────────────────────────────────────────────────────────────────
    if "nyc" in dfs:
        nyc = dfs["nyc"].copy()
        nyc_v = nyc[nyc["violation_code"].notna()].copy()
        nyc_v["source_inspection_id"] = (
            "nyc_" + nyc_v["camis"].astype(int).astype(str)
            + "_" + nyc_v["inspection_date"]
        )
        nyc_v["insp_id"] = nyc_v["source_inspection_id"].map(insp_lookup)
        nyc_v = nyc_v[nyc_v["insp_id"].notna()].copy()
        nyc_v["insp_id"] = nyc_v["insp_id"].astype(int)

        viol_parts.append(pd.DataFrame({
            "inspection_id": nyc_v["insp_id"].values,
            "violation_code": nyc_v["violation_code"].values,
            "violation_description": nyc_v["violation_description"].values,
            "violation_comment": np.nan,
            "severity": nyc_v["critical_flag"].values,
            "city": "nyc",
            "taxonomy_category_id": np.nan,
        }))

    # ── Chicago ────────────────────────────────────────────────────────────
    if "chicago_violations" in dfs:
        cv = dfs["chicago_violations"].copy()
        cv_v = cv[cv["violation_category"].notna()].copy()
        cv_v["source_inspection_id"] = "chi_" + cv_v["inspection_id"].astype(str)
        cv_v["insp_id"] = cv_v["source_inspection_id"].map(insp_lookup)
        cv_v = cv_v[cv_v["insp_id"].notna()].copy()
        cv_v["insp_id"] = cv_v["insp_id"].astype(int)

        cv_v["viol_code"] = cv_v["violation_number"].apply(
            lambda x: str(int(x)) if pd.notna(x) else None
        )

        viol_parts.append(pd.DataFrame({
            "inspection_id": cv_v["insp_id"].values,
            "violation_code": cv_v["viol_code"].values,
            "violation_description": cv_v["violation_category"].values,
            "violation_comment": cv_v["violation_comment"].values,
            "severity": np.nan,
            "city": "chicago",
            "taxonomy_category_id": np.nan,
        }))

    # ── Boston ─────────────────────────────────────────────────────────────
    if "boston" in dfs:
        bos = dfs["boston"].copy()
        bos_v = bos[bos["violation_code"].notna()].copy()
        bos_v["source_inspection_id"] = (
            "bos_" + bos_v["license_no"].astype(str)
            + "_" + bos_v["inspection_date"]
        )
        bos_v["insp_id"] = bos_v["source_inspection_id"].map(insp_lookup)
        bos_v = bos_v[bos_v["insp_id"].notna()].copy()
        bos_v["insp_id"] = bos_v["insp_id"].astype(int)

        viol_parts.append(pd.DataFrame({
            "inspection_id": bos_v["insp_id"].values,
            "violation_code": bos_v["violation_code"].values,
            "violation_description": bos_v["violation_description"].values,
            "violation_comment": bos_v["comments"].values,
            "severity": bos_v["violation_severity"].values,
            "city": "boston",
            "taxonomy_category_id": np.nan,
        }))

    # Combine and assign surrogate IDs
    viol_df = pd.concat(viol_parts, ignore_index=True)
    viol_df.insert(0, "violation_id", range(1, len(viol_df) + 1))

    return viol_df


# ─────────────────────────────────────────────────────────────────────────────
# Step 6: Load into DuckDB
# ─────────────────────────────────────────────────────────────────────────────

def load_into_duckdb(db_path: str, schema_path: str,
                     geo_df: pd.DataFrame, rest_df: pd.DataFrame,
                     insp_df: pd.DataFrame, viol_df: pd.DataFrame):
    """Create DuckDB database and load all dimension/fact tables."""

    os.makedirs(os.path.dirname(db_path), exist_ok=True)

    # Remove existing database for clean load
    if os.path.exists(db_path):
        os.remove(db_path)
        print(f"  Removed existing database: {db_path}")

    con = duckdb.connect(db_path)

    # Create schema
    print(f"  Executing schema from {schema_path}...")
    with open(schema_path, "r") as f:
        schema_sql = f.read()
    con.execute(schema_sql)

    # Load tables via DuckDB's fast DataFrame ingestion
    print("  Loading dim_geography...")
    con.execute("INSERT INTO dim_geography SELECT * FROM geo_df")

    print("  Loading dim_restaurants...")
    con.execute("INSERT INTO dim_restaurants SELECT * FROM rest_df")

    # Ensure inspection_date is DATE (handles mixed formats:
    # NYC/Chicago = '2023-06-21', Boston = '2022-03-08 17:47:58+00:00')
    insp_load = insp_df.copy()
    insp_load["inspection_date"] = pd.to_datetime(
        insp_load["inspection_date"], format="mixed", utc=True
    ).dt.date
    print("  Loading fact_inspections...")
    con.execute("INSERT INTO fact_inspections SELECT * FROM insp_load")

    print("  Loading dim_violations...")
    con.execute("INSERT INTO dim_violations SELECT * FROM viol_df")

    # ── Verification & Summary ─────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("DUCKDB LOAD SUMMARY")
    print("=" * 65)
    for table in ["dim_geography", "dim_restaurants", "fact_inspections",
                  "dim_violations", "dim_violation_taxonomy",
                  "dim_violation_crosswalk"]:
        count = con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        print(f"  {table:30s} {count:>10,} rows")

    # City-level breakdown
    print(f"\n  fact_inspections by city:")
    for row in con.execute(
        "SELECT city, COUNT(*) AS n "
        "FROM fact_inspections GROUP BY city ORDER BY city"
    ).fetchall():
        print(f"    {row[0]:15s} {row[1]:>8,}")

    # Outcome distribution (initial + standard only)
    print(f"\n  Outcome tier (initial, standard inspections):")
    city_totals = {}
    result = con.execute("""
        SELECT city, outcome_tier, COUNT(*) AS n
        FROM fact_inspections
        WHERE is_reinspection = FALSE
          AND is_standard_inspection = TRUE
          AND outcome_tier IS NOT NULL
        GROUP BY city, outcome_tier
        ORDER BY city, outcome_tier
    """).fetchall()

    for row in result:
        city_totals[row[0]] = city_totals.get(row[0], 0) + row[2]

    current_city = None
    for row in result:
        if row[0] != current_city:
            current_city = row[0]
            total = city_totals[current_city]
            print(f"    {current_city} ({total:,}):")
        pct = row[2] / city_totals[row[0]] * 100
        print(f"      {row[1]:15s} {row[2]:>8,}  ({pct:.1f}%)")

    # Violation counts
    print(f"\n  dim_violations by city:")
    for row in con.execute(
        "SELECT city, COUNT(*) FROM dim_violations GROUP BY city ORDER BY city"
    ).fetchall():
        print(f"    {row[0]:15s} {row[1]:>10,}")

    # Avg violations per inspection
    print(f"\n  Avg violations/inspection (initial, standard):")
    for row in con.execute("""
        SELECT city, ROUND(AVG(violation_count), 1) AS avg_viols
        FROM fact_inspections
        WHERE is_reinspection = FALSE AND is_standard_inspection = TRUE
        GROUP BY city ORDER BY city
    """).fetchall():
        print(f"    {row[0]:15s} {row[1]}")

    print("=" * 65)

    con.close()
    db_size = os.path.getsize(db_path) / (1024 * 1024)
    print(f"\n  Database saved: {db_path} ({db_size:.1f} MB)")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Load cleaned CSVs into DuckDB star schema."
    )
    parser.add_argument("--db", default=DB_PATH, help="DuckDB output path")
    parser.add_argument("--cleaned-dir", default=CLEANED_DIR,
                        help="Directory with cleaned CSVs")
    parser.add_argument("--schema", default=SCHEMA_PATH,
                        help="Path to schema SQL file")
    args = parser.parse_args()

    print("=" * 65)
    print("DataBridge — DuckDB Star Schema Loader")
    print("=" * 65)
    start = time.time()

    # Step 1: Load CSVs
    print("\n[1/6] Loading cleaned CSVs...")
    dfs = load_cleaned_csvs(args.cleaned_dir)
    if not dfs:
        print("ERROR: No data files found. Exiting.")
        sys.exit(1)

    # Step 2: Build dim_geography
    print("\n[2/6] Building dim_geography...")
    geo_df = build_dim_geography(dfs)
    print(f"  {len(geo_df)} geography entries")

    # Step 3: Build dim_restaurants
    print("\n[3/6] Building dim_restaurants...")
    rest_df = build_dim_restaurants(dfs, geo_df)
    print(f"  {len(rest_df):,} unique restaurants")
    for city in ["nyc", "chicago", "boston"]:
        n = (rest_df["city"] == city).sum()
        if n > 0:
            print(f"    {city:15s} {n:>8,}")

    # Step 4: Build fact_inspections
    print("\n[4/6] Building fact_inspections...")
    insp_df = build_fact_inspections(dfs, rest_df)
    print(f"  {len(insp_df):,} inspections")
    for city in ["nyc", "chicago", "boston"]:
        n = (insp_df["city"] == city).sum()
        if n > 0:
            print(f"    {city:15s} {n:>8,}")

    # Step 5: Build dim_violations
    print("\n[5/6] Building dim_violations...")
    viol_df = build_dim_violations(dfs, insp_df)
    print(f"  {len(viol_df):,} violations")
    for city in ["nyc", "chicago", "boston"]:
        n = (viol_df["city"] == city).sum()
        if n > 0:
            print(f"    {city:15s} {n:>10,}")

    # Step 6: Load into DuckDB
    print(f"\n[6/6] Loading into DuckDB ({args.db})...")
    load_into_duckdb(args.db, args.schema, geo_df, rest_df, insp_df, viol_df)

    elapsed = time.time() - start
    print(f"\nTotal time: {elapsed:.1f}s")
    print("Done.")


if __name__ == "__main__":
    main()
