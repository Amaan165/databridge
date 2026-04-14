"""
clean_boston.py
----------------
DataBridge — Week 1, Task 1.5 (Vishwa)

Cleans the raw Boston Food Establishment Inspections dataset into a
standardized per-city CSV ready for schema harmonization in Week 2.

Input : data/raw/Boston.csv          (~874K rows x 26 cols)
Output: data/cleaned/boston_cleaned.csv

Cleaning steps (per project roadmap):
  1. Filter licensecat to FS (Food Service) and RF (Retail Food) only
  2. Parse resultdttm to datetime; filter to 2022-2025 inspections
  3. Normalize state casing (MA/Ma/ma -> MA)
  4. Drop dbaname column (~99% null)
  5. Clean viol_level: drop the 2 stray rows ('1919' and blank)
  6. Map 20 result codes to outcome_tier (Pass/Conditional/Fail);
     drop non-inspection codes
  7. Rename 'city' (neighborhood names) -> 'neighborhood'; add city='boston'
  8. Standardize columns to snake_case
"""

from pathlib import Path
import argparse
import sys
import pandas as pd


# --- Configuration ------------------------------------------------------------

LICENSE_CATEGORIES_KEEP = {"FS", "RF"}

# Map Boston's 20 result codes to the unified outcome tier.
# Anything NOT in this map is treated as a non-inspection code and dropped.
RESULT_TO_OUTCOME_TIER = {
    "HE_Pass":    "Pass",
    "HE_Fail":    "Fail",
    "HE_FailExt": "Fail",
    "HE_Filed":   "Conditional",
    "HE_Hearing": "Conditional",
}

# viol_level cleanup: drop rows where the value is one of these stray tokens.
# Legitimate values are '*', '**', '***', or NaN (for clean inspections).
VIOL_LEVEL_BAD = {"1919", " ", ""}

DATE_MIN = pd.Timestamp("2022-01-01", tz="UTC")
DATE_MAX = pd.Timestamp("2026-01-01", tz="UTC")  # exclusive upper bound


# --- Cleaning steps -----------------------------------------------------------

def load_raw(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    print(f"[load]         raw rows={len(df):,}  cols={df.shape[1]}")
    return df


def filter_license_category(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df[df["licensecat"].isin(LICENSE_CATEGORIES_KEEP)].copy()
    print(f"[licensecat]   kept FS/RF: {len(df):,}  (dropped {before - len(df):,})")
    return df


def parse_and_filter_dates(df: pd.DataFrame) -> pd.DataFrame:
    df["resultdttm"] = pd.to_datetime(df["resultdttm"], errors="coerce", utc=True)
    df["issdttm"]    = pd.to_datetime(df["issdttm"],    errors="coerce", utc=True)
    df["expdttm"]    = pd.to_datetime(df["expdttm"],    errors="coerce", utc=True)
    df["violdttm"]   = pd.to_datetime(df["violdttm"],   errors="coerce", utc=True)
    df["status_date"] = pd.to_datetime(df["status_date"], errors="coerce", utc=True)

    before = len(df)
    mask = df["resultdttm"].notna() & \
           (df["resultdttm"] >= DATE_MIN) & \
           (df["resultdttm"] <  DATE_MAX)
    df = df[mask].copy()
    print(f"[dates]        kept 2022-2025: {len(df):,}  (dropped {before - len(df):,})")
    return df


def normalize_state(df: pd.DataFrame) -> pd.DataFrame:
    df["state"] = df["state"].astype("string").str.strip().str.upper()
    print(f"[state]        normalized -> {sorted(df['state'].dropna().unique().tolist())}")
    return df


def drop_dbaname(df: pd.DataFrame) -> pd.DataFrame:
    if "dbaname" in df.columns:
        null_pct = df["dbaname"].isna().mean() * 100
        df = df.drop(columns=["dbaname"])
        print(f"[dbaname]      dropped (was {null_pct:.1f}% null)")
    return df


def clean_viol_level(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    # Normalize empty-string/whitespace to NaN for consistency
    df["viol_level"] = df["viol_level"].astype("string").str.strip()
    df.loc[df["viol_level"] == "", "viol_level"] = pd.NA

    # Drop rows with stray values. Keep NaN (clean inspections legitimately
    # have no violation severity) and keep '*'/'**'/'***' and '-'.
    bad_mask = df["viol_level"].isin({"1919"})
    df = df[~bad_mask].copy()
    print(f"[viol_level]   dropped {before - len(df):,} stray rows "
          f"(values={sorted(df['viol_level'].dropna().unique().tolist())})")
    return df


def map_outcome_tier(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df["outcome_tier"] = df["result"].map(RESULT_TO_OUTCOME_TIER)
    df = df[df["outcome_tier"].notna()].copy()
    print(f"[outcome_tier] kept mappable inspections: {len(df):,}  "
          f"(dropped {before - len(df):,} non-inspection codes)")
    print(f"               distribution: "
          f"{df['outcome_tier'].value_counts().to_dict()}")
    return df


def rename_city_to_neighborhood(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns={"city": "neighborhood"})
    df["neighborhood"] = df["neighborhood"].astype("string").str.strip().str.title()
    df["city"] = "boston"
    print(f"[neighborhood] {df['neighborhood'].nunique()} unique values; "
          f"city='boston' added")
    return df


def split_location(df: pd.DataFrame) -> pd.DataFrame:
    """Boston's `location` is a string like '(42.359, -71.058)'.
    Split into separate latitude/longitude columns to match NYC/Chicago schema."""
    loc = df["location"].astype("string").str.strip("() ")
    parts = loc.str.split(",", n=1, expand=True)
    df["latitude"]  = pd.to_numeric(parts[0], errors="coerce")
    df["longitude"] = pd.to_numeric(parts[1], errors="coerce")
    n_parsed = df["latitude"].notna().sum()
    print(f"[location]     parsed lat/long for {n_parsed:,} rows "
          f"(null: {len(df) - n_parsed:,})")
    return df


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    # Snake-case + consistent naming aligned with NYC/Chicago cleaned outputs.
    rename_map = {
        "businessname": "business_name",
        "legalowner":   "legal_owner",
        "namelast":     "owner_last_name",
        "namefirst":    "owner_first_name",
        "licenseno":    "license_no",
        "issdttm":      "license_issue_date",
        "expdttm":      "license_expiry_date",
        "licstatus":    "license_status",
        "licensecat":   "license_category",
        "descript":     "license_description",
        "result":       "result_code",
        "resultdttm":   "inspection_date",
        "violation":    "violation_code",
        "violdesc":     "violation_description",
        "violdttm":     "violation_date",
        "viol_status":  "violation_status",
        "viol_level":   "violation_severity",
        "zip":          "zipcode",       # align with NYC/Chicago
        "property_id":  "property_id",
    }
    df = df.rename(columns=rename_map)

    # Column ordering for readability. `location` dropped (replaced by lat/long).
    ordered = [
        "city", "neighborhood", "address", "state", "zipcode",
        "latitude", "longitude",
        "property_id", "license_no", "business_name", "legal_owner",
        "owner_first_name", "owner_last_name",
        "license_category", "license_description", "license_status",
        "license_issue_date", "license_expiry_date",
        "inspection_date", "result_code", "outcome_tier",
        "violation_code", "violation_description", "violation_severity",
        "violation_date", "violation_status", "status_date", "comments",
    ]
    ordered = [c for c in ordered if c in df.columns]
    df = df[ordered]
    return df


# --- Summary ------------------------------------------------------------------

def print_summary(df: pd.DataFrame) -> None:
    print("\n" + "=" * 60)
    print("BOSTON CLEANED — SUMMARY")
    print("=" * 60)
    print(f"rows:            {len(df):,}")
    print(f"columns:         {df.shape[1]}")
    print(f"date range:      {df['inspection_date'].min()}  ->  "
          f"{df['inspection_date'].max()}")
    print(f"unique licenses: {df['license_no'].nunique():,}")
    print(f"unique inspections (license_no + inspection_date): "
          f"{df.groupby(['license_no', 'inspection_date']).ngroups:,}")
    print(f"\noutcome_tier distribution:")
    print(df["outcome_tier"].value_counts().to_string())
    print(f"\nlicense_category distribution:")
    print(df["license_category"].value_counts().to_string())
    print(f"\nrows per year:")
    print(df["inspection_date"].dt.year.value_counts().sort_index().to_string())
    print("=" * 60 + "\n")


# --- Entry point --------------------------------------------------------------

def main(input_path: Path, output_path: Path) -> None:
    print(f"Input:  {input_path}")
    print(f"Output: {output_path}\n")

    df = load_raw(input_path)
    df = filter_license_category(df)
    df = parse_and_filter_dates(df)
    df = normalize_state(df)
    df = drop_dbaname(df)
    df = clean_viol_level(df)
    df = map_outcome_tier(df)
    df = rename_city_to_neighborhood(df)
    df = split_location(df)
    df = standardize_columns(df)

    print_summary(df)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Wrote {len(df):,} rows -> {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clean Boston inspection data.")
    parser.add_argument(
        "--input",  default="data/raw/Boston.csv",
        help="Path to raw Boston.csv",
    )
    parser.add_argument(
        "--output", default="data/cleaned/boston_cleaned.csv",
        help="Path to write cleaned CSV",
    )
    args = parser.parse_args()
    try:
        main(Path(args.input), Path(args.output))
    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)