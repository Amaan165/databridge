"""
add_reinspection_flag.py — Patch: Add is_reinspection flag to all cleaned CSVs
DataBridge Project | Data Engineering Spring 2026

Adds a boolean `is_reinspection` column to each city's cleaned output.
Can be run standalone on existing cleaned CSVs, or the logic can be
integrated into each city's cleaning script (see docstrings below).

Detection method per city:
  - NYC:     inspection_type contains "Re-inspection" (already labeled)
  - Chicago: inspection_type contains "Re-Inspect" (already labeled)
  - Boston:  Inferred from temporal sequence — an inspection within 30 days
             of a prior Fail/FailExt for the same license is a re-inspection.

Usage:
  python add_reinspection_flag.py

  Reads from data/cleaned/{city}_cleaned.csv (+ chicago_violations_parsed.csv)
  Writes updated files in-place with the new column appended.
"""

import pandas as pd
import numpy as np
import os
import sys

CLEANED_DIR = os.path.join('data', 'cleaned')


# ─────────────────────────────────────────────────────────────────────────────
# NYC
# ─────────────────────────────────────────────────────────────────────────────

def flag_reinspections_nyc(df: pd.DataFrame) -> pd.DataFrame:
    """
    Flag re-inspections in NYC data using the inspection_type column.

    NYC labels re-inspections explicitly in the inspection_type field:
      "Cycle Inspection / Re-inspection"
      "Pre-permit (Operational) / Re-inspection"
      "Administrative Miscellaneous / Re-inspection"
      etc.

    ── To integrate into clean_nyc.py ──
    Add after step 6c (is_standard_inspection flag), before step 7:

        # ── 6d. Flag re-inspections ────────────────────────────────────
        df['is_reinspection'] = df['inspection_type'].str.contains(
            r'Re-inspection', case=False, na=False
        )
        reinsp_count = df['is_reinspection'].sum()
        log.info(f"  Step 6d — Flagged {reinsp_count:,} re-inspection rows "
                 f"({reinsp_count / len(df) * 100:.1f}%)")

    Then add 'is_reinspection' to the output_cols list.
    """
    df['is_reinspection'] = df['inspection_type'].str.contains(
        r'Re-inspection', case=False, na=False
    )
    return df


# ─────────────────────────────────────────────────────────────────────────────
# CHICAGO
# ─────────────────────────────────────────────────────────────────────────────

def flag_reinspections_chicago(df: pd.DataFrame) -> pd.DataFrame:
    """
    Flag re-inspections in Chicago data using the inspection_type column.

    Chicago labels re-inspections explicitly:
      "Canvass Re-Inspection"
      "Complaint Re-Inspection"
      "License Re-Inspection"
      "Suspected Food Poisoning Re-inspection"

    ── To integrate into clean_chicago.py ──
    Add after step 5 (outcome tier mapping), before step 6:

        # ── 5b. Flag re-inspections ───────────────────────────────────
        df['is_reinspection'] = df['Inspection Type'].str.contains(
            r'Re-Inspect', case=False, na=False
        )
        reinsp_count = df['is_reinspection'].sum()
        log.info(f"  Step 5b — Flagged {reinsp_count:,} re-inspection rows "
                 f"({reinsp_count / len(df) * 100:.1f}%)")

    Then add 'is_reinspection' to the output_cols list.

    ── To integrate into parse_chicago_violations.py ──
    No changes needed — 'is_reinspection' will carry through
    automatically since it's in the inspection-level columns.
    """
    df['is_reinspection'] = df['inspection_type'].str.contains(
        r'Re-Inspect', case=False, na=False
    )
    return df


# ─────────────────────────────────────────────────────────────────────────────
# BOSTON
# ─────────────────────────────────────────────────────────────────────────────

def flag_reinspections_boston(df: pd.DataFrame) -> pd.DataFrame:
    """
    Flag re-inspections in Boston data using temporal sequencing.

    Boston has no explicit re-inspection label. Instead, we infer:
    an inspection is a re-inspection if it occurs within 30 days of
    a prior Fail (HE_Fail) or FailExt (HE_FailExt) for the same license.

    Rationale (from data analysis):
      - 39% of Boston inspections are re-inspections by this definition
      - 71% of re-inspections result in Pass (establishment fixed issues)
      - HE_FailExt appears almost exclusively on re-inspections (18.5%)
        vs initial inspections (1.2%), confirming it means "still failing
        on follow-up, given extension"
      - Median time to re-inspection: 7 days (Fail), 11 days (FailExt)

    ── To integrate into clean_boston.py ──
    Add as a new function and call it after map_outcome_tier(), before
    rename_city_to_neighborhood():

        def flag_reinspections(df: pd.DataFrame) -> pd.DataFrame:
            '''Infer re-inspections from temporal sequence.
            An inspection within 30 days of a Fail/FailExt for the same
            license is flagged as a re-inspection.'''
            df = df.sort_values(['license_no', 'resultdttm'])

            # Work at inspection level (one row per license+date)
            insp = df.groupby(['licenseno', 'resultdttm']).agg(
                result=('result', 'first')
            ).reset_index().sort_values(['licenseno', 'resultdttm'])

            insp['prev_date'] = insp.groupby('licenseno')['resultdttm'].shift(1)
            insp['prev_result'] = insp.groupby('licenseno')['result'].shift(1)
            insp['days_since_prev'] = (
                insp['resultdttm'] - insp['prev_date']
            ).dt.days

            insp['is_reinspection'] = (
                insp['days_since_prev'].notna()
                & (insp['days_since_prev'] <= 30)
                & insp['prev_result'].isin(['HE_Fail', 'HE_FailExt'])
            )

            # Merge back to row level
            reinsp_map = insp.set_index(
                ['licenseno', 'resultdttm']
            )['is_reinspection']
            df['is_reinspection'] = df.set_index(
                ['licenseno', 'resultdttm']
            ).index.map(reinsp_map).values

            reinsp_count = df['is_reinspection'].sum()
            n_insp = insp['is_reinspection'].sum()
            print(f"[reinspection] flagged {n_insp:,} re-inspections "
                  f"({reinsp_count:,} rows, "
                  f"{n_insp / len(insp) * 100:.1f}% of inspections)")
            return df

    Then add 'is_reinspection' to the ordered columns list in
    standardize_columns().

    NOTE: This function uses the PRE-RENAME column names (licenseno,
    resultdttm, result). When integrating, place it before
    standardize_columns() in the pipeline.
    """
    date_col = 'inspection_date'
    license_col = 'license_no'
    result_col = 'result_code'

    df[date_col] = pd.to_datetime(df[date_col], utc=True)
    df = df.sort_values([license_col, date_col])

    # Work at inspection level
    insp = df.groupby([license_col, date_col]).agg(
        result=(result_col, 'first')
    ).reset_index().sort_values([license_col, date_col])

    insp['prev_date'] = insp.groupby(license_col)[date_col].shift(1)
    insp['prev_result'] = insp.groupby(license_col)['result'].shift(1)
    insp['days_since_prev'] = (insp[date_col] - insp['prev_date']).dt.days

    insp['is_reinspection'] = (
        insp['days_since_prev'].notna()
        & (insp['days_since_prev'] <= 30)
        & insp['prev_result'].isin(['HE_Fail', 'HE_FailExt'])
    )

    # Merge back to row level
    reinsp_lookup = insp.set_index([license_col, date_col])['is_reinspection']
    df['is_reinspection'] = (
        df.set_index([license_col, date_col])
        .index.map(reinsp_lookup)
        .values
    )

    return df


# ─────────────────────────────────────────────────────────────────────────────
# Summary & verification
# ─────────────────────────────────────────────────────────────────────────────

def print_summary(city: str, df: pd.DataFrame, inspection_id_cols: list):
    """Print re-inspection flag summary at both row and inspection level."""
    n_rows = len(df)
    n_reinsp_rows = df['is_reinspection'].sum()

    # Inspection-level stats
    insp = df.groupby(inspection_id_cols).agg(
        outcome_tier=('outcome_tier', 'first'),
        is_reinspection=('is_reinspection', 'first'),
    ).reset_index()

    n_insp = len(insp)
    n_reinsp_insp = insp['is_reinspection'].sum()
    n_initial_insp = n_insp - n_reinsp_insp

    print(f"\n{'='*65}")
    print(f"{city.upper()} — RE-INSPECTION FLAG SUMMARY")
    print(f"{'='*65}")
    print(f"  Total rows:         {n_rows:>8,}")
    print(f"  Re-inspection rows: {n_reinsp_rows:>8,} ({n_reinsp_rows/n_rows*100:.1f}%)")
    print(f"  Total inspections:  {n_insp:>8,}")
    print(f"  Initial:            {n_initial_insp:>8,} ({n_initial_insp/n_insp*100:.1f}%)")
    print(f"  Re-inspections:     {n_reinsp_insp:>8,} ({n_reinsp_insp/n_insp*100:.1f}%)")

    for label, sub in [("INITIAL", insp[~insp['is_reinspection']]),
                       ("RE-INSPECTION", insp[insp['is_reinspection']])]:
        total = len(sub)
        if total == 0:
            continue
        print(f"\n  {label} ({total:,} inspections):")
        for tier in ['Pass', 'Conditional', 'Fail']:
            c = (sub['outcome_tier'] == tier).sum()
            print(f"    {tier:15s} {c:>6,}  ({c/total*100:.1f}%)")
    print(f"{'='*65}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("DataBridge — Adding is_reinspection flag to all cleaned CSVs\n")

    # ── NYC ────────────────────────────────────────────────────────────────
    nyc_path = os.path.join(CLEANED_DIR, 'nyc_cleaned.csv')
    print(f"Loading {nyc_path}...")
    nyc = pd.read_csv(nyc_path, low_memory=False)
    nyc = flag_reinspections_nyc(nyc)
    print_summary('NYC', nyc, ['camis', 'inspection_date'])
    nyc.to_csv(nyc_path, index=False)
    print(f"  Saved → {nyc_path}\n")

    # ── Chicago ────────────────────────────────────────────────────────────
    chi_path = os.path.join(CLEANED_DIR, 'chicago_cleaned.csv')
    print(f"Loading {chi_path}...")
    chi = pd.read_csv(chi_path, low_memory=False)
    chi = flag_reinspections_chicago(chi)
    print_summary('Chicago', chi, ['inspection_id'])
    chi.to_csv(chi_path, index=False)
    print(f"  Saved → {chi_path}\n")

    # ── Chicago violations parsed (carry flag through) ─────────────────────
    chi_v_path = os.path.join(CLEANED_DIR, 'chicago_violations_parsed.csv')
    print(f"Loading {chi_v_path}...")
    chi_v = pd.read_csv(chi_v_path, low_memory=False)
    chi_v = flag_reinspections_chicago(chi_v)
    chi_v.to_csv(chi_v_path, index=False)
    print(f"  Saved → {chi_v_path} ({len(chi_v):,} rows)\n")

    # ── Boston ─────────────────────────────────────────────────────────────
    bos_path = os.path.join(CLEANED_DIR, 'boston_cleaned.csv')
    print(f"Loading {bos_path}...")
    bos = pd.read_csv(bos_path, low_memory=False)
    bos = flag_reinspections_boston(bos)
    print_summary('Boston', bos, ['license_no', 'inspection_date'])
    bos.to_csv(bos_path, index=False)
    print(f"  Saved → {bos_path}\n")

    # ── Cross-city comparison ──────────────────────────────────────────────
    print("\n" + "="*65)
    print("CROSS-CITY COMPARISON — INITIAL INSPECTIONS ONLY")
    print("="*65)

    # NYC: standard + initial only
    nyc_std = nyc[nyc['is_standard_inspection'] == True]
    nyc_init = nyc_std[~nyc_std['is_reinspection']]
    nyc_init_insp = nyc_init.groupby(['camis', 'inspection_date']).agg(
        outcome_tier=('outcome_tier', 'first')
    ).reset_index()

    # Chicago: initial only
    chi_init = chi[~chi['is_reinspection']]

    # Boston: initial only
    bos_init = bos[~bos['is_reinspection']]
    bos_init_insp = bos_init.groupby(['license_no', 'inspection_date']).agg(
        outcome_tier=('outcome_tier', 'first')
    ).reset_index()

    for label, insp_df in [("NYC (standard, initial)",  nyc_init_insp),
                           ("Chicago (initial)",         chi_init),
                           ("Boston (initial)",          bos_init_insp)]:
        total = len(insp_df)
        print(f"\n  {label} ({total:,} inspections):")
        for tier in ['Pass', 'Conditional', 'Fail']:
            c = (insp_df['outcome_tier'] == tier).sum()
            pct = c / total * 100 if total > 0 else 0
            print(f"    {tier:15s} {c:>6,}  ({pct:.1f}%)")

    print(f"\n{'='*65}")
    print("Done. All cleaned CSVs updated with is_reinspection column.")


if __name__ == '__main__':
    main()
