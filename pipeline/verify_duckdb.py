"""
verify_duckdb.py — Quick verification of the DataBridge DuckDB warehouse
DataBridge Project | Data Engineering Spring 2026

Run after load_duckdb.py to verify schema integrity, FK relationships,
and run a few sample analytical queries.

Usage:
  cd databridge/
  python pipeline/verify_duckdb.py
"""

import os
import sys
import duckdb

DB_PATH = os.path.join("data", "integrated", "databridge.duckdb")


def main():
    if not os.path.exists(DB_PATH):
        print(f"ERROR: {DB_PATH} not found. Run load_duckdb.py first.")
        sys.exit(1)

    con = duckdb.connect(DB_PATH, read_only=True)
    all_pass = True

    print("=" * 65)
    print("DataBridge — DuckDB Verification")
    print("=" * 65)

    # ── 1. Table row counts ────────────────────────────────────────────────
    print("\n[1] TABLE ROW COUNTS")
    expected = {
        "dim_geography": 30,
        "dim_restaurants": 40916,
        "fact_inspections": 148807,
        "dim_violations": 544641,
    }
    for table, exp in expected.items():
        actual = con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        status = "PASS" if actual == exp else f"FAIL (expected {exp})"
        if actual != exp:
            all_pass = False
        print(f"  {table:30s} {actual:>10,}  {status}")

    # ── 2. FK integrity: every inspection has a valid restaurant ───────────
    print("\n[2] FK INTEGRITY")
    orphan_insp = con.execute("""
        SELECT COUNT(*) FROM fact_inspections fi
        LEFT JOIN dim_restaurants dr ON fi.restaurant_id = dr.restaurant_id
        WHERE dr.restaurant_id IS NULL
    """).fetchone()[0]
    status = "PASS" if orphan_insp == 0 else f"FAIL ({orphan_insp} orphans)"
    if orphan_insp != 0:
        all_pass = False
    print(f"  fact_inspections → dim_restaurants orphans: {orphan_insp}  {status}")

    orphan_viol = con.execute("""
        SELECT COUNT(*) FROM dim_violations dv
        LEFT JOIN fact_inspections fi ON dv.inspection_id = fi.inspection_id
        WHERE fi.inspection_id IS NULL
    """).fetchone()[0]
    status = "PASS" if orphan_viol == 0 else f"FAIL ({orphan_viol} orphans)"
    if orphan_viol != 0:
        all_pass = False
    print(f"  dim_violations → fact_inspections orphans:  {orphan_viol}  {status}")

    # ── 3. Violation count consistency ─────────────────────────────────────
    print("\n[3] VIOLATION COUNT CONSISTENCY")
    print("  (fact_inspections.violation_count vs actual dim_violations rows)")
    for row in con.execute("""
        SELECT f.city, f.fact_sum, v.dim_count
        FROM (
            SELECT city, SUM(violation_count) AS fact_sum
            FROM fact_inspections GROUP BY city
        ) f
        JOIN (
            SELECT city, COUNT(*) AS dim_count
            FROM dim_violations GROUP BY city
        ) v ON f.city = v.city
        ORDER BY f.city
    """).fetchall():
        match = "PASS" if row[1] == row[2] else "MISMATCH"
        if row[1] != row[2]:
            all_pass = False
        print(f"  {row[0]:15s} fact_sum={row[1]:>10,}  dim_count={row[2]:>10,}  {match}")

    # ── 4. Sample analytical queries ───────────────────────────────────────
    print("\n[4] SAMPLE QUERIES")

    # RQ1: Compliance rates (initial, standard inspections)
    print("\n  RQ1 — Compliance rates (initial, standard):")
    results = con.execute("""
        SELECT city, outcome_tier,
               COUNT(*) AS n,
               ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (PARTITION BY city), 1) AS pct
        FROM fact_inspections
        WHERE is_reinspection = FALSE
          AND is_standard_inspection = TRUE
          AND outcome_tier IS NOT NULL
        GROUP BY city, outcome_tier
        ORDER BY city, outcome_tier
    """).fetchall()
    current = None
    for r in results:
        if r[0] != current:
            current = r[0]
            print(f"    {current}:")
        print(f"      {r[1]:15s} {r[2]:>8,}  ({r[3]}%)")

    # Top 5 restaurants by violation count per city
    print("\n  Top 3 most-violated restaurants per city:")
    for row in con.execute("""
        WITH ranked AS (
            SELECT dr.name, dr.city,
                   SUM(fi.violation_count) AS total_viols,
                   COUNT(fi.inspection_id) AS num_inspections,
                   ROW_NUMBER() OVER (PARTITION BY dr.city ORDER BY SUM(fi.violation_count) DESC) AS rn
            FROM fact_inspections fi
            JOIN dim_restaurants dr ON fi.restaurant_id = dr.restaurant_id
            GROUP BY dr.name, dr.city
        )
        SELECT city, name, total_viols, num_inspections
        FROM ranked WHERE rn <= 3
        ORDER BY city, total_viols DESC
    """).fetchall():
        name_trunc = row[1][:40] if row[1] else "N/A"
        print(f"    {row[0]:10s} {name_trunc:42s} {row[2]:>5} viols / {row[3]:>3} inspections")

    # Re-inspection recovery rates (RQ3 preview)
    print("\n  RQ3 preview — Re-inspection pass rates:")
    for row in con.execute("""
        SELECT city,
               COUNT(*) AS reinsp_count,
               SUM(CASE WHEN outcome_tier = 'Pass' THEN 1 ELSE 0 END) AS passed,
               ROUND(SUM(CASE WHEN outcome_tier = 'Pass' THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 1) AS pass_rate
        FROM fact_inspections
        WHERE is_reinspection = TRUE AND outcome_tier IS NOT NULL
        GROUP BY city ORDER BY city
    """).fetchall():
        print(f"    {row[0]:15s} {row[1]:>6,} re-inspections → "
              f"{row[2]:>5,} passed ({row[3]}%)")

    # ── Summary ────────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    if all_pass:
        print("ALL CHECKS PASSED — DuckDB warehouse is ready for analysis.")
    else:
        print("SOME CHECKS FAILED — review output above.")
    print("=" * 65)

    con.close()


if __name__ == "__main__":
    main()
