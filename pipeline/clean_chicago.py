"""
clean_chicago.py — Task 1.2: Clean Chicago Food Inspection Data
DataBridge Project | Data Engineering Spring 2026

Input:  data/raw/Chicago.csv (307,307 rows × 17 cols)
Output: data/cleaned/chicago_cleaned.csv

Cleaning steps:
  1. Filter to restaurant-type facilities (~208K rows)
  2. Parse dates; filter to 2022-2025
  3. Normalize city casing + fix typos; filter to IL only
  4. Trim address whitespace
  5. Map results to outcome_tier; drop non-inspection outcomes
  6. Flag re-inspections from inspection_type labels
  7. Standardize columns to snake_case; add city = 'chicago'
  8. Save cleaned CSV with summary stats

v3 — Improvements over v2:
  - is_reinspection flag (step 6) derived from inspection_type
    for fair cross-city comparison

v2 — Improvements over v1:
  - City typos now corrected to CHICAGO (CHICAGOO, 312CHICAGO, CCHICAGO, etc.)
    instead of just uppercasing and flagging
"""

import pandas as pd
import numpy as np
import re
import logging
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
log = logging.getLogger(__name__)

RAW_PATH = os.path.join('data', 'raw', 'Chicago.csv')
OUT_PATH = os.path.join('data', 'cleaned', 'chicago_cleaned.csv')

# ── Known Chicago city typos → correction ─────────────────────────────────────
# These are clearly Chicago records with mangled city names
CHICAGO_TYPOS = {
    'CCHICAGO', 'CHICAGOO', 'CHICAGO.', '312CHICAGO', 'CHICAGOCHICAGO',
    'CHCHICAGO', 'CHCAGO', 'CHIAGO', 'CHICAG', 'CHICAGP',
    'CHICGO', 'CHICAHO', 'CHICAGOI', 'CHICAO',
}


def clean_chicago():
    # ── Load ───────────────────────────────────────────────────────────────────
    log.info(f"Loading {RAW_PATH}")
    df = pd.read_csv(RAW_PATH, low_memory=False)
    log.info(f"  Loaded: {len(df):,} rows × {df.shape[1]} cols")
    initial_count = len(df)

    # ── 1. Filter to restaurant facilities ─────────────────────────────────────
    restaurant_mask = df['Facility Type'].str.lower().str.contains('restaurant', na=False)
    df = df[restaurant_mask].copy()
    log.info(f"  Step 1 — Filtered to restaurant facilities: {len(df):,} rows "
             f"(dropped {initial_count - len(df):,} non-restaurant)")

    # ── 2. Date cleaning ───────────────────────────────────────────────────────
    df['Inspection Date'] = pd.to_datetime(df['Inspection Date'], format='%m/%d/%Y')

    pre_filter = len(df)
    df = df[(df['Inspection Date'] >= '2022-01-01') & (df['Inspection Date'] <= '2025-12-31')]
    log.info(f"  Step 2 — Filtered to 2022-2025: dropped {pre_filter - len(df):,} → {len(df):,} rows")

    # ── 3. City and state normalization ────────────────────────────────────────
    # Log city values before normalization
    city_counts = df['City'].value_counts().head(15)
    log.info(f"  Step 3a — City values before normalization (top 15):")
    for city_val, cnt in city_counts.items():
        log.info(f"    {city_val!r:25s} {cnt:>6,}")

    # Normalize: strip, uppercase
    df['City'] = df['City'].str.strip().str.upper()

    # Fix known Chicago typos
    typo_mask = df['City'].isin(CHICAGO_TYPOS)
    typo_count = typo_mask.sum()
    df.loc[typo_mask, 'City'] = 'CHICAGO'
    log.info(f"  Step 3b — Corrected {typo_count} Chicago typos to 'CHICAGO'")

    # Also catch anything that looks like Chicago with extra/missing chars
    # Pattern: starts with "CHI" and has "CAG" or "CGO" somewhere
    fuzzy_chi = df['City'].str.match(r'^CH.*C.*G', na=False) & (df['City'] != 'CHICAGO')
    if fuzzy_chi.sum() > 0:
        remaining_typos = df.loc[fuzzy_chi, 'City'].value_counts()
        log.info(f"  Step 3b+ — Potential remaining Chicago typos found:")
        for city_val, cnt in remaining_typos.items():
            log.info(f"    {city_val!r:25s} {cnt:>6,}")
        df.loc[fuzzy_chi, 'City'] = 'CHICAGO'
        log.info(f"  Step 3b+ — Corrected {fuzzy_chi.sum()} additional fuzzy matches")

    # Flag genuine non-Chicago cities (suburbs, other towns)
    # First: fix NaN city rows that are within Chicago coordinates
    null_city = df['City'].isna()
    if null_city.sum() > 0:
        in_chi_bounds = (
            null_city &
            df['Latitude'].between(41.6, 42.1) &
            df['Longitude'].between(-87.95, -87.5)
        )
        df.loc[in_chi_bounds, 'City'] = 'CHICAGO'
        log.info(f"  Step 3c — Set {in_chi_bounds.sum()} NaN-city rows to 'CHICAGO' "
                 f"(coordinates within Chicago bounds)")

    non_chicago = df[df['City'] != 'CHICAGO']
    if len(non_chicago) > 0:
        log.info(f"  Step 3d — Non-Chicago cities remaining ({len(non_chicago)} rows):")
        for city_val, cnt in non_chicago['City'].value_counts(dropna=False).head(10).items():
            label = city_val if pd.notna(city_val) else 'NaN'
            log.info(f"    {label:25s} {cnt:>6,}")

    # Filter to IL only
    pre_state = len(df)
    non_il = df[df['State'] != 'IL']['State'].value_counts().to_dict()
    df = df[df['State'] == 'IL']
    if non_il:
        log.info(f"  Step 3d — Filtered to IL only: dropped {pre_state - len(df)} "
                 f"out-of-state rows {non_il}")
    log.info(f"  Step 3 complete → {len(df):,} rows")

    # ── 4. Address trimming ────────────────────────────────────────────────────
    for col in ['Address', 'City']:
        df[col] = df[col].str.strip()
    log.info(f"  Step 4 — Trimmed whitespace from Address and City")

    # ── 5. Outcome tier mapping ────────────────────────────────────────────────
    results_map = {
        'Pass': 'Pass',
        'Fail': 'Fail',
        'Pass w/ Conditions': 'Conditional',
    }
    drop_results = ['Out of Business', 'No Entry', 'Not Ready', 'Business Not Located']
    for dr in drop_results:
        cnt = (df['Results'] == dr).sum()
        if cnt > 0:
            log.info(f"  Step 5 — Dropping {cnt:,} rows with Results='{dr}'")

    df['outcome_tier'] = df['Results'].map(results_map)
    pre_drop = len(df)
    df = df[df['outcome_tier'].notna()]
    log.info(f"  Step 5 — Mapped results to outcome_tier: dropped {pre_drop - len(df):,} "
             f"non-inspection outcomes → {len(df):,} rows")

    # ── 6. Flag re-inspections ─────────────────────────────────────────────────
    # Chicago labels re-inspections explicitly in Inspection Type:
    #   "Canvass Re-Inspection", "Complaint Re-Inspection",
    #   "License Re-Inspection", "Suspected Food Poisoning Re-inspection"
    # This flag enables fair cross-city comparison (RQ1) and
    # re-inspection effectiveness analysis (RQ3).
    df['is_reinspection'] = df['Inspection Type'].str.contains(
        r'Re-Inspect', case=False, na=False
    )
    reinsp_count = df['is_reinspection'].sum()
    log.info(f"  Step 6 — Flagged {reinsp_count:,} re-inspection rows "
             f"({reinsp_count / len(df) * 100:.1f}%)")

    # ── 7. Standardize columns ─────────────────────────────────────────────────
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(' ', '_', regex=False)
        .str.replace('#', '', regex=False)
    )

    # Rename for clarity
    df = df.rename(columns={
        'license_': 'license_no',
        'city': 'original_city',
    })
    df['city'] = 'chicago'
    log.info(f"  Step 7 — Columns standardized to snake_case")

    # ── 8. Select output columns & save ────────────────────────────────────────
    output_cols = [
        'inspection_id', 'dba_name', 'aka_name', 'license_no',
        'facility_type', 'risk', 'address', 'original_city', 'state', 'zipcode',
        'inspection_date', 'inspection_type', 'is_reinspection',
        'results', 'outcome_tier',
        'violations', 'latitude', 'longitude', 'city'
    ]

    # Handle case where 'zipcode' might be named 'zip' after snake_case
    if 'zip' in df.columns and 'zipcode' not in df.columns:
        df = df.rename(columns={'zip': 'zipcode'})

    df_out = df[output_cols].copy()

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    df_out.to_csv(OUT_PATH, index=False)
    log.info(f"  Step 8 — Saved {len(df_out):,} rows to {OUT_PATH}")

    # ── Summary ────────────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("CHICAGO CLEANING SUMMARY (v3)")
    print("=" * 65)
    print(f"  Input rows:              {initial_count:,}")
    print(f"  Output rows:             {len(df_out):,}")
    print(f"  Rows dropped:            {initial_count - len(df_out):,}")
    print(f"  Date range:              {df_out['inspection_date'].min()} → "
          f"{df_out['inspection_date'].max()}")
    print(f"  Unique inspections:      {df_out['inspection_id'].nunique():,}")
    print(f"  Unique businesses (DBA): {df_out['dba_name'].nunique():,}")
    print(f"  Unique licenses:         {df_out['license_no'].nunique():,}")
    print()
    reinsp = df_out['is_reinspection'].sum()
    initial = len(df_out) - reinsp
    print(f"  Initial inspection rows: {initial:,} ({initial/len(df_out)*100:.1f}%)")
    print(f"  Re-inspection rows:      {reinsp:,} ({reinsp/len(df_out)*100:.1f}%)")
    print()
    print("  Outcome tier distribution:")
    tier_counts = df_out['outcome_tier'].value_counts()
    for tier, count in tier_counts.items():
        print(f"    {tier:20s} {count:>8,}  ({count / len(df_out) * 100:5.1f}%)")
    print()
    print("  Risk distribution:")
    for risk, count in df_out['risk'].value_counts().items():
        print(f"    {str(risk):20s} {count:>8,}")
    print()
    print(f"  Violations null:         {df_out['violations'].isna().sum():,} "
          f"({df_out['violations'].isna().mean() * 100:.1f}%)")
    print()
    print("  Top facility types:")
    for ft, count in df_out['facility_type'].value_counts().head(10).items():
        print(f"    {ft:35s} {count:>6,}")
    print("=" * 65)

    return df_out


if __name__ == '__main__':
    clean_chicago()
