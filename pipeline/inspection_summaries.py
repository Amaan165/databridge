"""
inspection_summaries.py — Task 2.5: Inspection-Level Summary Tables
DataBridge Project | Data Engineering Spring 2026

Aggregates the cleaned violation-level data from each city up to the
inspection level (one row per inspection) and produces a unified
cross-city CSV with common columns.

Input files:
  data/cleaned/nyc_cleaned.csv               (one row per violation, NYC)
  data/cleaned/chicago_violations_parsed.csv (one row per violation, Chicago)
  data/cleaned/boston_cleaned.csv            (one row per violation, Boston)

Outputs:
  data/cleaned/inspections_nyc_summary.csv      Per-city, full granularity
  data/cleaned/inspections_chicago_summary.csv
  data/cleaned/inspections_boston_summary.csv
  data/cleaned/inspections_unified.csv          Common columns across all 3 cities

Aggregation rules (per the Week 2 roadmap):
  NYC:     group by camis + inspection_date
           -> violation_count, critical_count, score, grade, outcome_tier
  Chicago: group by inspection_id (pre-aggregated parse)
           -> violation_count, outcome_tier
  Boston:  group by license_no + inspection_date
           -> violation_count, severe/moderate/minor counts, outcome_tier

The unified CSV keeps only columns that are meaningful for all three cities:
  city, source_inspection_id, restaurant_source_id, inspection_date,
  outcome_tier, is_reinspection, is_standard_inspection,
  violation_count, critical_violation_count, neighborhood_or_boro

Usage:
  cd databridge/
  python pipeline/inspection_summaries.py

  Options:
    --cleaned-dir DIR   Directory with cleaned CSVs (default: data/cleaned)
    --out-dir DIR       Where to write summaries     (default: data/cleaned)
"""

from pathlib import Path
import argparse
import sys
import pandas as pd
import numpy as np


# ── Configuration ────────────────────────────────────────────────────────────

CLEANED_DIR = Path("data") / "cleaned"


# ── NYC summary ──────────────────────────────────────────────────────────────

def summarize_nyc(cleaned_dir: Path) -> pd.DataFrame:
    """One row per (camis, inspection_date)."""
    path = cleaned_dir / "nyc_cleaned.csv"
    print(f"[nyc]      loading {path}...")
    df = pd.read_csv(path, low_memory=False)
    print(f"[nyc]      {len(df):,} violation-level rows")

    g = df.groupby(["camis", "inspection_date"], sort=False).agg(
        dba=("dba", "first"),
        boro=("boro", "first"),
        cuisine_description=("cuisine_description", "first"),
        zipcode=("zipcode", "first"),
        inspection_type=("inspection_type", "first"),
        is_standard_inspection=("is_standard_inspection", "first"),
        is_reinspection=("is_reinspection", "first"),
        outcome_tier=("outcome_tier", "first"),
        outcome_source=("outcome_source", "first"),
        score=("score", "first"),
        grade=("grade", "first"),
        violation_count=("violation_code", lambda x: x.notna().sum()),
        critical_violation_count=("critical_flag",
                                  lambda x: (x == "Critical").sum()),
    ).reset_index()

    g.insert(0, "city", "nyc")
    g["source_inspection_id"] = (
        "nyc_" + g["camis"].astype(int).astype(str)
        + "_" + g["inspection_date"].astype(str)
    )
    print(f"[nyc]      {len(g):,} inspection-level rows")
    return g


# ── Chicago summary ──────────────────────────────────────────────────────────

def summarize_chicago(cleaned_dir: Path) -> pd.DataFrame:
    """One row per inspection_id. Reads the parsed violations file
    so violation_count reflects the structured parse, not raw text blob."""
    path = cleaned_dir / "chicago_violations_parsed.csv"
    print(f"[chicago]  loading {path}...")
    df = pd.read_csv(path, low_memory=False)
    print(f"[chicago]  {len(df):,} violation-level rows")

    g = df.groupby("inspection_id", sort=False).agg(
        dba_name=("dba_name", "first"),
        license_no=("license_no", "first"),
        facility_type=("facility_type", "first"),
        risk=("risk", "first"),
        zipcode=("zipcode", "first"),
        inspection_date=("inspection_date", "first"),
        inspection_type=("inspection_type", "first"),
        is_reinspection=("is_reinspection", "first"),
        results=("results", "first"),
        outcome_tier=("outcome_tier", "first"),
        violation_count=("violation_category",
                         lambda x: x.notna().sum()),
    ).reset_index()

    # Chicago has no critical/non-critical distinction in the parsed text
    g["critical_violation_count"] = 0
    g.insert(0, "city", "chicago")
    g["source_inspection_id"] = "chi_" + g["inspection_id"].astype(str)
    print(f"[chicago]  {len(g):,} inspection-level rows")
    return g


# ── Boston summary ───────────────────────────────────────────────────────────

def summarize_boston(cleaned_dir: Path) -> pd.DataFrame:
    """One row per (license_no, inspection_date).

    Severity: Boston uses *=Core, **=Priority, ***=Priority Foundation.
    *** is treated as 'critical' for cross-city comparability.
    """
    path = cleaned_dir / "boston_cleaned.csv"
    print(f"[boston]   loading {path}...")
    df = pd.read_csv(path, low_memory=False)
    print(f"[boston]   {len(df):,} violation-level rows")

    g = df.groupby(["license_no", "inspection_date"], sort=False).agg(
        business_name=("business_name", "first"),
        neighborhood=("neighborhood", "first"),
        zipcode=("zipcode", "first"),
        result_code=("result_code", "first"),
        outcome_tier=("outcome_tier", "first"),
        is_reinspection=("is_reinspection", "first"),
        violation_count=("violation_code",
                         lambda x: x.notna().sum()),
        severe_count=("violation_severity",
                      lambda x: (x == "***").sum()),
        moderate_count=("violation_severity",
                        lambda x: (x == "**").sum()),
        minor_count=("violation_severity",
                     lambda x: (x == "*").sum()),
    ).reset_index()

    g["critical_violation_count"] = g["severe_count"]
    g.insert(0, "city", "boston")
    g["source_inspection_id"] = (
        "bos_" + g["license_no"].astype(str)
        + "_" + g["inspection_date"].astype(str)
    )
    print(f"[boston]   {len(g):,} inspection-level rows")
    return g


# ── Unified table ────────────────────────────────────────────────────────────

UNIFIED_COLUMNS = [
    "city",
    "source_inspection_id",
    "restaurant_source_id",
    "inspection_date",
    "outcome_tier",
    "is_reinspection",
    "is_standard_inspection",
    "violation_count",
    "critical_violation_count",
    "geography",
]


def build_unified(nyc: pd.DataFrame,
                  chi: pd.DataFrame,
                  bos: pd.DataFrame) -> pd.DataFrame:
    """Combine the three city summaries into a common-column table."""

    nyc_u = pd.DataFrame({
        "city":                     nyc["city"],
        "source_inspection_id":     nyc["source_inspection_id"],
        "restaurant_source_id":     nyc["camis"].astype(str),
        "inspection_date":          nyc["inspection_date"],
        "outcome_tier":             nyc["outcome_tier"],
        "is_reinspection":          nyc["is_reinspection"],
        "is_standard_inspection":   nyc["is_standard_inspection"],
        "violation_count":          nyc["violation_count"],
        "critical_violation_count": nyc["critical_violation_count"],
        "geography":                nyc["boro"],
    })

    chi_u = pd.DataFrame({
        "city":                     chi["city"],
        "source_inspection_id":     chi["source_inspection_id"],
        "restaurant_source_id":     chi["license_no"].astype(str),
        "inspection_date":          chi["inspection_date"],
        "outcome_tier":             chi["outcome_tier"],
        "is_reinspection":          chi["is_reinspection"],
        # Chicago has no Admin/Smoke-Free split; everything is comparable
        "is_standard_inspection":   True,
        "violation_count":          chi["violation_count"],
        "critical_violation_count": chi["critical_violation_count"],
        "geography":                "CHICAGO",
    })

    bos_u = pd.DataFrame({
        "city":                     bos["city"],
        "source_inspection_id":     bos["source_inspection_id"],
        "restaurant_source_id":     bos["license_no"].astype(str),
        "inspection_date":          bos["inspection_date"],
        "outcome_tier":             bos["outcome_tier"],
        "is_reinspection":          bos["is_reinspection"],
        "is_standard_inspection":   True,
        "violation_count":          bos["violation_count"],
        "critical_violation_count": bos["critical_violation_count"],
        "geography":                bos["neighborhood"],
    })

    unified = pd.concat([nyc_u, chi_u, bos_u], ignore_index=True)
    unified = unified[UNIFIED_COLUMNS]
    return unified


# ── Summary printing ─────────────────────────────────────────────────────────

def print_summary(unified: pd.DataFrame) -> None:
    print("\n" + "=" * 65)
    print("INSPECTION SUMMARIES — UNIFIED TABLE")
    print("=" * 65)
    print(f"  total inspections: {len(unified):,}")
    print()

    print("  Per-city counts (all inspections):")
    for city in ["nyc", "chicago", "boston"]:
        n = (unified["city"] == city).sum()
        print(f"    {city:10s} {n:>8,}")

    print("\n  Outcome tier (initial + standard only):")
    mask = (~unified["is_reinspection"]) & unified["is_standard_inspection"]
    init = unified[mask]
    for city in ["nyc", "chicago", "boston"]:
        sub = init[init["city"] == city]
        if len(sub) == 0:
            continue
        print(f"\n    {city} ({len(sub):,} inspections):")
        for tier in ["Pass", "Conditional", "Fail"]:
            c = (sub["outcome_tier"] == tier).sum()
            pct = c / len(sub) * 100 if len(sub) else 0
            print(f"      {tier:15s} {c:>8,}  ({pct:5.1f}%)")

    print("\n  Re-inspection share by city:")
    for city in ["nyc", "chicago", "boston"]:
        sub = unified[unified["city"] == city]
        if len(sub) == 0:
            continue
        n_re = sub["is_reinspection"].sum()
        print(f"    {city:10s} {n_re:>8,} of {len(sub):>8,} "
              f"({n_re / len(sub) * 100:5.1f}%)")

    print("\n  Avg violations per inspection (initial + standard):")
    for city in ["nyc", "chicago", "boston"]:
        sub = init[init["city"] == city]
        if len(sub) == 0:
            continue
        avg = sub["violation_count"].mean()
        print(f"    {city:10s} {avg:5.2f}")
    print("=" * 65)


# ── Entry point ──────────────────────────────────────────────────────────────

def main(cleaned_dir: Path, out_dir: Path) -> None:
    print(f"Cleaned dir: {cleaned_dir}")
    print(f"Output dir:  {out_dir}\n")

    nyc = summarize_nyc(cleaned_dir)
    chi = summarize_chicago(cleaned_dir)
    bos = summarize_boston(cleaned_dir)

    out_dir.mkdir(parents=True, exist_ok=True)
    nyc.to_csv(out_dir / "inspections_nyc_summary.csv", index=False)
    chi.to_csv(out_dir / "inspections_chicago_summary.csv", index=False)
    bos.to_csv(out_dir / "inspections_boston_summary.csv", index=False)
    print(f"\n  Wrote per-city summaries to {out_dir}/")

    unified = build_unified(nyc, chi, bos)
    unified.to_csv(out_dir / "inspections_unified.csv", index=False)
    print(f"  Wrote inspections_unified.csv ({len(unified):,} rows)")

    print_summary(unified)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Aggregate violation-level data into inspection summaries."
    )
    parser.add_argument("--cleaned-dir", default=str(CLEANED_DIR),
                        help="Directory with cleaned CSVs")
    parser.add_argument("--out-dir", default=str(CLEANED_DIR),
                        help="Directory for summary outputs")
    args = parser.parse_args()
    try:
        main(Path(args.cleaned_dir), Path(args.out_dir))
    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)
