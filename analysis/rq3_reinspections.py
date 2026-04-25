"""
rq3_reinspections.py — Task 3.2: Re-inspection Analysis (RQ3)
DataBridge Project | Data Engineering Spring 2026

RQ3: When restaurants fail an inspection, how often do they get
re-inspected, and what fraction recover (Pass) on the follow-up?
How does this differ across cities?

Methodology:
  1. From fact_inspections, find every initial Fail/Conditional inspection.
  2. For each, find the next inspection of the same restaurant within
     90 days. If found, treat that as the re-inspection event.
  3. Compute per city:
       - re-inspection rate         = pairs / failed-initial inspections
       - recovery rate              = pairs that re-Passed / total pairs
       - median days to re-inspect

Output:
  outputs/rq3_pairs.csv             (one row per fail -> followup pair)
  outputs/rq3_summary.csv           (per-city aggregate metrics)
  outputs/rq3_recovery_by_city.png  (grouped bar: re-inspection % + recovery %)
  outputs/rq3_days_distribution.png (box plot: days between fail + followup)

Usage:
  cd databridge/
  python analysis/rq3_reinspections.py

  Options:
    --db PATH        DuckDB path (default: data/integrated/databridge.duckdb)
    --out-dir DIR    Where to write outputs (default: outputs)
    --window DAYS    Max days between fail and re-inspection (default: 90)
"""

import argparse
import os
import sys
import time
from pathlib import Path

import duckdb
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for headless runs
import matplotlib.pyplot as plt


# ── Configuration ────────────────────────────────────────────────────────────

DB_PATH = Path("data") / "integrated" / "databridge.duckdb"
OUT_DIR = Path("outputs")
DEFAULT_WINDOW_DAYS = 90

CITY_ORDER = ["nyc", "chicago", "boston"]
CITY_COLORS = {"nyc": "#1f77b4", "chicago": "#ff7f0e", "boston": "#2ca02c"}


# ── SQL: build fail -> followup pairs ────────────────────────────────────────

PAIRS_SQL = """
WITH base AS (
    -- Use only initial standard inspections with a finalized outcome.
    SELECT
        inspection_id,
        restaurant_id,
        city,
        inspection_date,
        outcome_tier,
        is_reinspection
    FROM fact_inspections
    WHERE outcome_tier IS NOT NULL
      AND is_standard_inspection = TRUE
),
initial_fails AS (
    -- Every initial inspection where outcome != Pass.
    SELECT *
    FROM base
    WHERE is_reinspection = FALSE
      AND outcome_tier IN ('Fail', 'Conditional')
),
candidate_followups AS (
    -- Every later inspection of the same restaurant within the window.
    SELECT
        f.inspection_id          AS initial_id,
        f.city                   AS city,
        f.restaurant_id          AS restaurant_id,
        f.inspection_date        AS initial_date,
        f.outcome_tier           AS initial_outcome,
        n.inspection_id          AS followup_id,
        n.inspection_date        AS followup_date,
        n.outcome_tier           AS followup_outcome,
        n.is_reinspection        AS followup_flagged_as_reinspection,
        DATE_DIFF('day', f.inspection_date, n.inspection_date)
                                 AS days_between
    FROM initial_fails f
    JOIN base n
      ON  n.restaurant_id  = f.restaurant_id
      AND n.inspection_date > f.inspection_date
      AND DATE_DIFF('day', f.inspection_date, n.inspection_date) <= {window}
),
ranked AS (
    -- Keep only the FIRST followup per initial fail.
    SELECT *,
           ROW_NUMBER() OVER (
               PARTITION BY initial_id
               ORDER BY followup_date ASC
           ) AS rn
    FROM candidate_followups
)
SELECT
    initial_id,
    city,
    restaurant_id,
    initial_date,
    initial_outcome,
    followup_id,
    followup_date,
    followup_outcome,
    followup_flagged_as_reinspection,
    days_between
FROM ranked
WHERE rn = 1
"""


# ── Aggregation ──────────────────────────────────────────────────────────────

def compute_summary(pairs: pd.DataFrame, totals_failed: dict) -> pd.DataFrame:
    """Per-city aggregate metrics from the pairs table."""
    rows = []
    for city in CITY_ORDER:
        sub = pairs[pairs["city"] == city]
        n_failed = totals_failed.get(city, 0)
        n_pairs = len(sub)
        n_recover = (sub["followup_outcome"] == "Pass").sum()
        n_still_fail = (sub["followup_outcome"] == "Fail").sum()
        n_conditional = (sub["followup_outcome"] == "Conditional").sum()
        median_days = sub["days_between"].median() if n_pairs else np.nan
        mean_days = sub["days_between"].mean() if n_pairs else np.nan

        rows.append({
            "city":                  city,
            "initial_failed":        n_failed,
            "followup_pairs":        n_pairs,
            "reinspection_rate":     n_pairs / n_failed if n_failed else np.nan,
            "recovery_pass_count":   int(n_recover),
            "recovery_rate":         n_recover / n_pairs if n_pairs else np.nan,
            "still_fail_count":      int(n_still_fail),
            "conditional_count":     int(n_conditional),
            "median_days_between":   float(median_days) if pd.notna(median_days) else np.nan,
            "mean_days_between":     float(mean_days) if pd.notna(mean_days) else np.nan,
        })

    return pd.DataFrame(rows)


# ── Charts ───────────────────────────────────────────────────────────────────

def chart_rates(summary: pd.DataFrame, out_path: Path) -> None:
    """Grouped bar chart: re-inspection rate vs. recovery rate per city."""
    fig, ax = plt.subplots(figsize=(8, 5))

    cities = summary["city"].tolist()
    x = np.arange(len(cities))
    w = 0.38

    reinsp_pct = (summary["reinspection_rate"] * 100).fillna(0)
    recov_pct = (summary["recovery_rate"] * 100).fillna(0)

    bars1 = ax.bar(x - w/2, reinsp_pct, w,
                   label="Re-inspection rate (% of fails followed up)",
                   color="#4c72b0")
    bars2 = ax.bar(x + w/2, recov_pct, w,
                   label="Recovery rate (% of follow-ups passing)",
                   color="#55a868")

    for bars in (bars1, bars2):
        for b in bars:
            h = b.get_height()
            ax.text(b.get_x() + b.get_width()/2, h + 1.5,
                    f"{h:.1f}%", ha="center", va="bottom", fontsize=10)

    ax.set_xticks(x)
    ax.set_xticklabels([c.upper() for c in cities])
    ax.set_ylabel("Percent")
    ax.set_ylim(0, max(110, recov_pct.max() + 15))
    ax.set_title("RQ3 — Re-inspection follow-up and recovery, by city")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=1, frameon=False)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Wrote {out_path}")


def chart_days(pairs: pd.DataFrame, out_path: Path) -> None:
    """Box plot: distribution of days between initial fail and follow-up."""
    fig, ax = plt.subplots(figsize=(8, 5))

    data = []
    labels = []
    for city in CITY_ORDER:
        sub = pairs.loc[pairs["city"] == city, "days_between"].dropna()
        if len(sub) > 0:
            data.append(sub.values)
            labels.append(f"{city.upper()}\n(n={len(sub):,})")

    bp = ax.boxplot(data, labels=labels, patch_artist=True, showfliers=False)
    for patch, city in zip(bp["boxes"], CITY_ORDER[:len(bp["boxes"])]):
        patch.set_facecolor(CITY_COLORS[city])
        patch.set_alpha(0.6)

    ax.set_ylabel("Days between initial fail and follow-up")
    ax.set_title("RQ3 — Time to re-inspection, by city")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Wrote {out_path}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="RQ3: re-inspection rates and recovery rates by city."
    )
    parser.add_argument("--db", default=str(DB_PATH))
    parser.add_argument("--out-dir", default=str(OUT_DIR))
    parser.add_argument("--window", type=int, default=DEFAULT_WINDOW_DAYS,
                        help="Max days between fail and re-inspection")
    args = parser.parse_args()

    db_path = Path(args.db)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not db_path.exists():
        print(f"ERROR: {db_path} not found. Run load_duckdb.py first.")
        sys.exit(1)

    print("=" * 65)
    print("DataBridge — RQ3 Re-inspection Analysis")
    print("=" * 65)
    start = time.time()

    con = duckdb.connect(str(db_path), read_only=True)

    # ── 1. Per-city counts of initial Fail/Conditional inspections ─────────
    print(f"\n[1/3] Counting initial fail/conditional inspections...")
    totals = con.execute("""
        SELECT city, COUNT(*) AS n
        FROM fact_inspections
        WHERE is_standard_inspection = TRUE
          AND is_reinspection = FALSE
          AND outcome_tier IN ('Fail', 'Conditional')
        GROUP BY city ORDER BY city
    """).fetchall()
    totals_failed = {row[0]: row[1] for row in totals}
    for city in CITY_ORDER:
        print(f"    {city:10s} {totals_failed.get(city, 0):>8,} initial fails/conditionals")

    # ── 2. Build pairs ─────────────────────────────────────────────────────
    print(f"\n[2/3] Building fail -> followup pairs (window={args.window}d)...")
    pairs = con.execute(PAIRS_SQL.format(window=args.window)).df()
    pairs["initial_date"] = pd.to_datetime(pairs["initial_date"])
    pairs["followup_date"] = pd.to_datetime(pairs["followup_date"])

    print(f"    Total pairs: {len(pairs):,}")
    for city in CITY_ORDER:
        n = (pairs["city"] == city).sum()
        print(f"    {city:10s} {n:>8,} pairs")

    # ── 3. Summary + outputs ───────────────────────────────────────────────
    print(f"\n[3/3] Aggregating + writing outputs...")
    summary = compute_summary(pairs, totals_failed)

    pairs.to_csv(out_dir / "rq3_pairs.csv", index=False)
    print(f"  Wrote {out_dir / 'rq3_pairs.csv'}")
    summary.to_csv(out_dir / "rq3_summary.csv", index=False)
    print(f"  Wrote {out_dir / 'rq3_summary.csv'}")

    chart_rates(summary, out_dir / "rq3_recovery_by_city.png")
    chart_days(pairs, out_dir / "rq3_days_distribution.png")

    con.close()

    # ── Console summary ────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print("RQ3 SUMMARY")
    print(f"{'='*65}")
    print(f"  {'city':<10} {'failed':>8} {'pairs':>8} {'reinsp%':>8}"
          f" {'recover%':>9} {'med_days':>9}")
    for _, row in summary.iterrows():
        rr = (row["reinspection_rate"] * 100) if pd.notna(row["reinspection_rate"]) else 0
        rec = (row["recovery_rate"] * 100) if pd.notna(row["recovery_rate"]) else 0
        md = row["median_days_between"] if pd.notna(row["median_days_between"]) else 0
        print(f"  {row['city']:<10} {int(row['initial_failed']):>8,}"
              f" {int(row['followup_pairs']):>8,} {rr:>7.1f}%"
              f" {rec:>8.1f}% {md:>9.1f}")
    print(f"{'='*65}")

    elapsed = time.time() - start
    print(f"\nTotal time: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
