"""
clean_nyc.py — Task 1.1: Clean NYC DOHMH Restaurant Inspection Data
DataBridge Project | Data Engineering Spring 2026

Input:  data/raw/NYC.csv (297,134 rows × 27 cols)
Output: data/cleaned/nyc_cleaned.csv

Cleaning steps:
  1. Drop exact duplicate rows
  2. Parse dates; remove sentinel dates (01/01/1900); filter to 2022-2025
  3. Replace lat/long = 0.0 with NaN
  4. Flag BORO = '0' as 'Unknown'; flag out-of-range zip codes
  5. Standardize columns to snake_case; add city = 'nyc'
  6. Map grades to outcome_tier (A→Pass, B→Conditional, C→Fail),
     then recover outcome_tier from numeric scores for ungraded rows
     using NYC's official thresholds: 0-13=A, 14-27=B, 28+=C
  7. Normalize violation description text (strip, case-normalize near-dupes)
  8. Include inspection_type column for downstream filtering
  9. Save cleaned CSV with summary stats

v2 — Improvements over v1:
  - Score-based outcome_tier recovery (45% → 96% coverage)
  - Inspection type preserved for downstream filtering
  - Violation description text normalized (220 → 219 unique)
  - outcome_source column tracks whether tier came from grade or score
"""

import pandas as pd
import numpy as np
import logging
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
log = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────────────────────
RAW_PATH = os.path.join('data', 'raw', 'NYC.csv')
OUT_PATH = os.path.join('data', 'cleaned', 'nyc_cleaned.csv')

# ── Valid NYC zip code range ───────────────────────────────────────────────────
NYC_ZIP_MIN, NYC_ZIP_MAX = 10001, 11697

# ── NYC official grade thresholds (score-based) ───────────────────────────────
# A = 0-13, B = 14-27, C = 28+
# Verified: 99.5% consistency with actual A/B/C grades in the data
SCORE_BINS = [-1, 13, 27, float('inf')]
SCORE_LABELS = ['Pass', 'Conditional', 'Fail']


def clean_nyc():
    # ── Load ───────────────────────────────────────────────────────────────────
    log.info(f"Loading {RAW_PATH}")
    df = pd.read_csv(RAW_PATH, low_memory=False)
    log.info(f"  Loaded: {len(df):,} rows × {df.shape[1]} cols")
    initial_count = len(df)

    # ── 1. Deduplicate ─────────────────────────────────────────────────────────
    dupes = df.duplicated().sum()
    df = df.drop_duplicates()
    log.info(f"  Step 1 — Dropped {dupes} exact duplicates → {len(df):,} rows")

    # ── 2. Date cleaning ───────────────────────────────────────────────────────
    df['INSPECTION DATE'] = pd.to_datetime(df['INSPECTION DATE'], format='%m/%d/%Y')

    sentinel_mask = df['INSPECTION DATE'] == pd.Timestamp('1900-01-01')
    sentinel_count = sentinel_mask.sum()
    df = df[~sentinel_mask]
    log.info(f"  Step 2a — Removed {sentinel_count:,} sentinel dates (01/01/1900) → {len(df):,} rows")

    pre_filter = len(df)
    df = df[(df['INSPECTION DATE'] >= '2022-01-01') & (df['INSPECTION DATE'] <= '2025-12-31')]
    log.info(f"  Step 2b — Filtered to 2022-2025: dropped {pre_filter - len(df):,} → {len(df):,} rows")

    # ── 3. Coordinate masking ──────────────────────────────────────────────────
    zero_lat = (df['Latitude'] == 0).sum()
    zero_lon = (df['Longitude'] == 0).sum()
    df.loc[df['Latitude'] == 0, 'Latitude'] = np.nan
    df.loc[df['Longitude'] == 0, 'Longitude'] = np.nan
    log.info(f"  Step 3 — Replaced {zero_lat:,} zero-lat and {zero_lon:,} zero-lon with NaN")

    # ── 4. Borough and zip code flags ──────────────────────────────────────────
    boro_zero = (df['BORO'] == '0').sum()
    df.loc[df['BORO'] == '0', 'BORO'] = 'Unknown'
    log.info(f"  Step 4a — Flagged {boro_zero} rows with BORO='0' as 'Unknown'")

    df['ZIPCODE'] = pd.to_numeric(df['ZIPCODE'], errors='coerce')
    invalid_zip_mask = df['ZIPCODE'].notna() & (
        (df['ZIPCODE'] < NYC_ZIP_MIN) | (df['ZIPCODE'] > NYC_ZIP_MAX)
    )
    df['zip_flag'] = ''
    df.loc[invalid_zip_mask, 'zip_flag'] = 'out_of_range'
    log.info(f"  Step 4b — Flagged {invalid_zip_mask.sum()} out-of-range zip codes")

    # ── 5. Standardize columns ─────────────────────────────────────────────────
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(' ', '_', regex=False)
    )
    df['city'] = 'nyc'
    log.info(f"  Step 5 — Columns standardized to snake_case; city='nyc' added")

    # ── 6. Outcome tier: grade-first, then score-based recovery ───────────────

    # 6a: Map from letter grades (A/B/C)
    grade_map = {'A': 'Pass', 'B': 'Conditional', 'C': 'Fail'}
    df['outcome_tier'] = df['grade'].map(grade_map)
    df['outcome_source'] = pd.Series(np.nan, index=df.index, dtype='object')
    df.loc[df['outcome_tier'].notna(), 'outcome_source'] = 'grade'

    grade_count = df['outcome_tier'].notna().sum()
    log.info(f"  Step 6a — Mapped from letter grades: {grade_count:,} rows "
             f"({grade_count / len(df) * 100:.1f}%)")

    # 6b: For ungraded rows WITH scores, recover from score thresholds
    # NYC official: A = 0-13, B = 14-27, C = 28+
    # Validation: 99.5% consistent with actual grades (565 mismatches out of 123,657)
    ungraded_with_score = df['outcome_tier'].isna() & df['score'].notna()
    score_tiers = pd.cut(
        df.loc[ungraded_with_score, 'score'],
        bins=SCORE_BINS,
        labels=SCORE_LABELS
    )
    df.loc[ungraded_with_score, 'outcome_tier'] = score_tiers
    df.loc[ungraded_with_score, 'outcome_source'] = 'score'

    recovered = ungraded_with_score.sum()
    total_tiered = df['outcome_tier'].notna().sum()
    remaining_null = df['outcome_tier'].isna().sum()
    log.info(f"  Step 6b — Recovered {recovered:,} outcome_tiers from scores → "
             f"{total_tiered:,} total ({total_tiered / len(df) * 100:.1f}% coverage, "
             f"{remaining_null:,} still null)")

    # ── 7. Violation description normalization ─────────────────────────────────
    # Fix near-duplicates caused by casing differences
    # e.g., "Alcohol and Pregnancy" vs "Alcohol and pregnancy" (2,082 rows)
    pre_unique = df['violation_description'].dropna().nunique()
    df['violation_description'] = df['violation_description'].str.strip()

    # Build canonical mapping: for each uppercased key, keep the most frequent form
    viol_desc = df.loc[df['violation_description'].notna(), 'violation_description']
    desc_freq = viol_desc.value_counts()
    upper_to_canonical = {}
    for desc in desc_freq.index:
        upper = desc.upper()
        if upper not in upper_to_canonical:
            upper_to_canonical[upper] = desc  # most frequent variant wins

    df.loc[df['violation_description'].notna(), 'violation_description'] = (
        df.loc[df['violation_description'].notna(), 'violation_description']
        .map(lambda x: upper_to_canonical.get(x.upper(), x))
    )
    post_unique = df['violation_description'].dropna().nunique()
    normalized_count = pre_unique - post_unique
    log.info(f"  Step 7 — Normalized violation descriptions: {pre_unique} → "
             f"{post_unique} unique ({normalized_count} near-duplicates merged)")

    # ── 8. Select output columns & save ────────────────────────────────────────
    output_cols = [
        'camis', 'dba', 'boro', 'cuisine_description',
        'inspection_date', 'inspection_type',
        'violation_code', 'violation_description',
        'critical_flag', 'score', 'grade', 'outcome_tier', 'outcome_source',
        'latitude', 'longitude', 'zipcode', 'zip_flag', 'city'
    ]
    df_out = df[output_cols].copy()

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    df_out.to_csv(OUT_PATH, index=False)
    log.info(f"  Step 8 — Saved {len(df_out):,} rows to {OUT_PATH}")

    # ── Summary ────────────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("NYC CLEANING SUMMARY (v2)")
    print("=" * 65)
    print(f"  Input rows:              {initial_count:,}")
    print(f"  Output rows:             {len(df_out):,}")
    print(f"  Rows dropped:            {initial_count - len(df_out):,}")
    print(f"  Date range:              {df_out['inspection_date'].min()} → "
          f"{df_out['inspection_date'].max()}")
    print(f"  Unique restaurants:      {df_out['camis'].nunique():,}")
    print(f"  Unique violation descs:  {df_out['violation_description'].dropna().nunique()}")
    print()
    print("  Outcome tier distribution:")
    tier_counts = df_out['outcome_tier'].value_counts(dropna=False)
    for tier, count in tier_counts.items():
        label = tier if pd.notna(tier) else 'NaN (no grade or score)'
        print(f"    {label:30s} {count:>8,}  ({count / len(df_out) * 100:5.1f}%)")
    print()
    print("  Outcome source (how tier was determined):")
    src_counts = df_out['outcome_source'].value_counts(dropna=False)
    for src, count in src_counts.items():
        label = src if pd.notna(src) else 'NaN (no tier assigned)'
        print(f"    {label:30s} {count:>8,}  ({count / len(df_out) * 100:5.1f}%)")
    print()
    print("  Borough distribution:")
    for boro, count in df_out['boro'].value_counts().items():
        print(f"    {boro:20s} {count:>8,}")
    print()
    print("  Top 10 inspection types:")
    for itype, count in df_out['inspection_type'].value_counts().head(10).items():
        print(f"    {itype:55s} {count:>7,}")
    print()
    print("  Null rates:")
    for col in ['violation_code', 'violation_description', 'score', 'grade',
                'outcome_tier', 'latitude', 'longitude']:
        null_pct = df_out[col].isna().mean() * 100
        print(f"    {col:25s} {null_pct:5.1f}%")
    print("=" * 65)

    return df_out


if __name__ == '__main__':
    clean_nyc()
