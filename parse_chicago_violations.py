"""
parse_chicago_violations.py — Task 1.3: Parse Chicago Free-Text Violations
DataBridge Project | Data Engineering Spring 2026

Input:  data/cleaned/chicago_cleaned.csv
Output: data/cleaned/chicago_violations_parsed.csv

Chicago violations are stored as a single free-text field per inspection,
with pipe-delimited entries in this format:

  "32. FOOD AND NON-FOOD CONTACT SURFACES... - Comments: OBSERVED BOX FREEZER...
   | 33. FOOD AND NON-FOOD CONTACT EQUIPMENT... - Comments: NEED TO CLEAN..."

This script parses each violation into structured fields and explodes
so each violation becomes its own row, matching NYC/Boston granularity.

v2 — Improvements over v1:
  - Violation category text normalized (strip, title case for consistency)
  - Near-duplicate category detection and merging
"""

import pandas as pd
import numpy as np
import re
import logging
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
log = logging.getLogger(__name__)

IN_PATH = os.path.join('data', 'cleaned', 'chicago_cleaned.csv')
OUT_PATH = os.path.join('data', 'cleaned', 'chicago_violations_parsed.csv')


def parse_violation_text(text):
    """
    Parse a single violation text blob into a list of dicts.

    Each violation follows: {number}. {CATEGORY} - Comments: {comment text}
    Violations are separated by ' | '.

    Returns list of dicts with keys:
      violation_number, violation_category, violation_comment
    """
    if pd.isna(text) or not text.strip():
        return []

    violations = []

    # Split on pipe delimiter — handle variations in spacing
    parts = re.split(r'\s*\|\s*', text.strip())

    for part in parts:
        part = part.strip()
        if not part:
            continue

        # Pattern: "32. CATEGORY TEXT - Comments: comment text"
        match = re.match(
            r'^(\d+)\.\s*(.+?)(?:\s*-\s*Comments?\s*:\s*(.*))?$',
            part,
            re.DOTALL | re.IGNORECASE
        )

        if match:
            viol_num = match.group(1).strip()
            category = match.group(2).strip()
            comment = match.group(3).strip() if match.group(3) else ''

            # Handle case where " - Comments:" is embedded in the category match
            if ' - Comments:' in category:
                cat_parts = category.split(' - Comments:', 1)
                category = cat_parts[0].strip()
                comment = cat_parts[1].strip() + (' ' + comment if comment else '')
        else:
            # Fallback: no leading number
            viol_num = ''
            if ' - Comments:' in part or ' - comments:' in part.lower():
                split_idx = part.lower().index(' - comments:')
                category = part[:split_idx].strip()
                comment = part[split_idx + len(' - Comments:'):].strip()
            else:
                category = part
                comment = ''

        violations.append({
            'violation_number': viol_num,
            'violation_category': category,
            'violation_comment': comment,
        })

    return violations


def normalize_categories(series):
    """
    Normalize violation category text:
    - Strip whitespace
    - Collapse multiple spaces
    - Merge near-duplicates via uppercase canonical mapping
    """
    cleaned = (
        series
        .str.strip()
        .str.replace(r'\s+', ' ', regex=True)  # collapse multiple spaces
    )

    # Build canonical mapping: uppercase key → most frequent original form
    non_null = cleaned.dropna()
    freq = non_null.value_counts()
    upper_to_canonical = {}
    for val in freq.index:
        key = val.upper()
        if key not in upper_to_canonical:
            upper_to_canonical[key] = val

    # Apply mapping
    result = cleaned.copy()
    mask = result.notna()
    result.loc[mask] = result.loc[mask].map(lambda x: upper_to_canonical.get(x.upper(), x))

    return result


def parse_chicago_violations():
    # ── Load ───────────────────────────────────────────────────────────────────
    log.info(f"Loading {IN_PATH}")
    df = pd.read_csv(IN_PATH, low_memory=False)
    log.info(f"  Loaded: {len(df):,} inspections")

    total_inspections = len(df)
    null_violations = df['violations'].isna().sum()
    has_violations = total_inspections - null_violations
    log.info(f"  Inspections with violations: {has_violations:,}")
    log.info(f"  Clean inspections (null):    {null_violations:,} "
             f"({null_violations / total_inspections * 100:.1f}%)")

    # ── Parse violations ───────────────────────────────────────────────────────
    log.info("  Parsing violation text blobs...")

    df_clean = df[df['violations'].isna()].copy()
    df_has_viol = df[df['violations'].notna()].copy()

    # Parse each row's violations into a list of dicts
    df_has_viol['parsed'] = df_has_viol['violations'].apply(parse_violation_text)

    # Count parsing failures
    parse_failures = (df_has_viol['parsed'].apply(len) == 0).sum()
    if parse_failures > 0:
        log.warning(f"  ⚠ {parse_failures} rows had violation text but yielded 0 parsed records")

    # Explode: each violation becomes its own row
    df_exploded = df_has_viol.explode('parsed')
    df_exploded = df_exploded[df_exploded['parsed'].notna()]

    # Unpack parsed dicts into columns
    parsed_df = pd.json_normalize(df_exploded['parsed'])
    parsed_df.index = df_exploded.index

    # Merge back with inspection-level columns
    inspection_cols = [c for c in df.columns if c != 'violations']
    df_violations = df_exploded[inspection_cols].copy()
    df_violations['violation_number'] = parsed_df['violation_number'].values
    df_violations['violation_category'] = parsed_df['violation_category'].values
    df_violations['violation_comment'] = parsed_df['violation_comment'].values

    # For clean inspections, keep as single row with null violation fields
    df_clean_out = df_clean[inspection_cols].copy()
    df_clean_out['violation_number'] = np.nan
    df_clean_out['violation_category'] = np.nan
    df_clean_out['violation_comment'] = np.nan

    # Combine
    df_out = pd.concat([df_violations, df_clean_out], ignore_index=True)
    df_out = df_out.sort_values(['inspection_id', 'violation_number']).reset_index(drop=True)

    # ── Normalize category text ────────────────────────────────────────────────
    pre_unique = df_out['violation_category'].dropna().nunique()
    df_out['violation_category'] = normalize_categories(df_out['violation_category'])
    post_unique = df_out['violation_category'].dropna().nunique()
    log.info(f"  Normalized categories: {pre_unique} → {post_unique} unique")

    # ── Save ───────────────────────────────────────────────────────────────────
    df_out.to_csv(OUT_PATH, index=False)
    log.info(f"  Saved {len(df_out):,} rows to {OUT_PATH}")

    # ── Summary ────────────────────────────────────────────────────────────────
    total_violations = df_violations.shape[0]
    viols_per_inspection = total_violations / has_violations if has_violations > 0 else 0

    print("\n" + "=" * 65)
    print("CHICAGO VIOLATION PARSING SUMMARY (v2)")
    print("=" * 65)
    print(f"  Input inspections:       {total_inspections:,}")
    print(f"  With violations:         {has_violations:,}")
    print(f"  Clean (null violations): {null_violations:,}")
    print(f"  Total violation records: {total_violations:,}")
    print(f"  Output rows:             {len(df_out):,}")
    print(f"  Avg violations/insp:     {viols_per_inspection:.1f}")
    print(f"  Unique categories:       {post_unique}")
    if parse_failures > 0:
        print(f"  Parse failures:          {parse_failures}")
    print()
    print("  Top 15 violation categories by frequency:")
    cat_counts = (
        df_out['violation_category']
        .value_counts()
        .head(15)
    )
    for cat, count in cat_counts.items():
        display_cat = cat[:60] + '...' if len(cat) > 60 else cat
        print(f"    {display_cat:65s} {count:>6,}")
    print()
    print("  Violation number distribution (top 15):")
    num_counts = (
        df_out['violation_number']
        .value_counts()
        .head(15)
    )
    for num, count in num_counts.items():
        print(f"    #{str(num):5s} {count:>6,}")
    print("=" * 65)

    return df_out


if __name__ == '__main__':
    parse_chicago_violations()
