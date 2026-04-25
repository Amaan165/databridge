"""
run_all.py — Task 3.1: Master Pipeline Orchestrator
DataBridge Project | Data Engineering Spring 2026

Runs the full DataBridge pipeline end-to-end with logging, timing,
and error handling. LLM-dependent steps (taxonomy relabeling, crosswalk
validation) are skipped automatically when OPENAI_API_KEY is not set.

Pipeline stages:
  1. Clean         clean_nyc.py, clean_chicago.py, parse_chicago_violations.py,
                   clean_boston.py
  2. Summarize     inspection_summaries.py
  3. Taxonomy      taxonomy.py (HDBSCAN + optional LLM labels)
  4. Crosswalk     crosswalk.py (cosine sim + optional LLM validation)
  5. Warehouse     load_duckdb.py
  6. Taxonomy load load_taxonomy.py (taxonomy + crosswalk -> DuckDB)
  7. Verify        verify_duckdb.py
  8. Analysis      rq3_reinspections.py
                   (RQ1/RQ2 live in cross_city_analysis.ipynb;
                    RQ4 lives in rq4_crosswalk_eval.py — both run separately)

Usage:
  cd databridge/
  python pipeline/run_all.py                  # full pipeline
  python pipeline/run_all.py --skip clean     # skip a stage
  python pipeline/run_all.py --only warehouse # run one stage
  python pipeline/run_all.py --dry-run        # print plan only

Environment:
  OPENAI_API_KEY    Optional. Without it, taxonomy uses Chicago-anchored
                    labels (no LLM relabeling) and crosswalk skips
                    validation in the 0.70-0.85 similarity band.
"""

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


# ── Stage definitions ────────────────────────────────────────────────────────
# Each stage is a list of (label, command) pairs. Commands run in order
# within a stage; stages run sequentially.

STAGES = {
    "clean": [
        ("clean NYC",                   ["python", "pipeline/clean_nyc.py"]),
        ("clean Chicago",               ["python", "pipeline/clean_chicago.py"]),
        ("parse Chicago violations",    ["python", "pipeline/parse_chicago_violations.py"]),
        ("clean Boston",                ["python", "pipeline/clean_boston.py"]),
    ],
    "summarize": [
        ("inspection summaries",        ["python", "pipeline/inspection_summaries.py"]),
    ],
    "taxonomy": [
        # LLM flag added dynamically based on OPENAI_API_KEY presence
        ("violation taxonomy",          ["python", "pipeline/taxonomy.py"]),
    ],
    "crosswalk": [
        ("semantic crosswalk",          ["python", "pipeline/crosswalk.py"]),
    ],
    "warehouse": [
        ("load DuckDB warehouse",       ["python", "pipeline/load_duckdb.py"]),
    ],
    "taxonomy_load": [
        ("load taxonomy + crosswalk",   ["python", "pipeline/load_taxonomy.py"]),
    ],
    "verify": [
        ("verify DuckDB",               ["python", "pipeline/verify_duckdb.py"]),
    ],
    "analysis": [
        ("RQ3 re-inspections",          ["python", "analysis/rq3_reinspections.py"]),
    ],
}

# Logical execution order
STAGE_ORDER = [
    "clean", "summarize", "taxonomy", "crosswalk",
    "warehouse", "taxonomy_load", "verify", "analysis",
]


# ── Helpers ──────────────────────────────────────────────────────────────────

def banner(text: str, char: str = "=") -> None:
    print()
    print(char * 70)
    print(f"  {text}")
    print(char * 70)


def have_openai_key() -> bool:
    return bool(os.environ.get("OPENAI_API_KEY"))


def have_python_module(name: str) -> bool:
    """Check whether a Python module is importable in the current env."""
    try:
        __import__(name)
        return True
    except ImportError:
        return False


def have_command(cmd: str) -> bool:
    return shutil.which(cmd) is not None


def maybe_add_llm_flag(cmd: list[str]) -> list[str]:
    """Add --use-llm flag to taxonomy/crosswalk commands when OPENAI_API_KEY is set."""
    if have_openai_key() and ("taxonomy.py" in cmd[1] or "crosswalk.py" in cmd[1]):
        return cmd + ["--use-llm"]
    return cmd


def run_command(label: str, cmd: list[str]) -> tuple[bool, float]:
    """Run a single command, capturing exit status + duration."""
    print(f"\n  >>> {label}")
    print(f"      $ {' '.join(cmd)}")
    start = time.time()
    try:
        result = subprocess.run(cmd, check=False)
        ok = result.returncode == 0
    except FileNotFoundError as e:
        print(f"      MISSING: {e}")
        ok = False
    elapsed = time.time() - start
    status = "OK" if ok else "FAILED"
    print(f"      [{status}] {elapsed:.1f}s")
    return ok, elapsed


def preflight() -> list[str]:
    """Quick environment check. Returns list of warnings (empty == OK)."""
    warnings = []

    # Required Python modules
    required = ["pandas", "numpy", "duckdb", "sklearn"]
    for mod in required:
        if not have_python_module(mod):
            warnings.append(f"Required Python module not installed: {mod}")

    # Optional modules — warn only
    optional = {
        "sentence_transformers": "needed for taxonomy.py",
        "hdbscan":               "needed for taxonomy.py",
        "openai":                "needed for --use-llm in taxonomy/crosswalk",
        "matplotlib":            "needed for analysis charts",
    }
    for mod, why in optional.items():
        if not have_python_module(mod):
            warnings.append(f"Optional module missing ({mod}) — {why}")

    if not have_openai_key():
        warnings.append(
            "OPENAI_API_KEY not set — taxonomy + crosswalk will skip LLM steps"
        )

    # Required raw inputs
    raw_files = [
        Path("data/raw/NYC.csv"),
        Path("data/raw/Chicago.csv"),
        Path("data/raw/Boston.csv"),
    ]
    for p in raw_files:
        if not p.exists():
            warnings.append(f"Raw input missing: {p}")

    return warnings


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Run the full DataBridge pipeline end-to-end."
    )
    parser.add_argument("--skip", action="append", default=[],
                        choices=STAGE_ORDER,
                        help="Skip a stage (repeatable)")
    parser.add_argument("--only", action="append", default=[],
                        choices=STAGE_ORDER,
                        help="Run only this stage (repeatable). "
                             "Mutually exclusive with --skip.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print plan without executing")
    parser.add_argument("--continue-on-error", action="store_true",
                        help="Don't abort on first failure")
    args = parser.parse_args()

    if args.skip and args.only:
        parser.error("--skip and --only are mutually exclusive")

    # Resolve the stages to run
    if args.only:
        stages_to_run = [s for s in STAGE_ORDER if s in args.only]
    else:
        stages_to_run = [s for s in STAGE_ORDER if s not in args.skip]

    # Pre-flight
    banner("DATABRIDGE PIPELINE — RUN_ALL")
    warnings = preflight()
    if warnings:
        print("\n  Pre-flight warnings:")
        for w in warnings:
            print(f"    - {w}")
    else:
        print("\n  Pre-flight: OK")

    print(f"\n  Stages to run: {stages_to_run}")
    print(f"  LLM steps:     {'enabled' if have_openai_key() else 'skipped (no API key)'}")
    if args.dry_run:
        print("\n  --dry-run: not executing.")
        return

    # Run
    overall_start = time.time()
    results = []  # list of (stage, label, ok, seconds)
    aborted = False

    for stage in stages_to_run:
        banner(f"STAGE: {stage}", "-")
        for label, cmd in STAGES[stage]:
            cmd = maybe_add_llm_flag(cmd)
            ok, secs = run_command(label, cmd)
            results.append((stage, label, ok, secs))
            if not ok and not args.continue_on_error:
                aborted = True
                break
        if aborted:
            break

    # Summary
    banner("RUN SUMMARY")
    total = time.time() - overall_start
    n_ok = sum(1 for _, _, ok, _ in results if ok)
    n_fail = sum(1 for _, _, ok, _ in results if not ok)
    print(f"\n  Stages completed: {n_ok} ok, {n_fail} failed")
    print(f"  Total time:       {total:.1f}s")
    print()
    print(f"  {'stage':<18}{'task':<32}{'status':<8}{'time':>8}")
    for stage, label, ok, secs in results:
        status = "OK" if ok else "FAIL"
        print(f"  {stage:<18}{label:<32}{status:<8}{secs:>7.1f}s")
    if aborted:
        print("\n  Pipeline aborted on first failure (re-run with "
              "--continue-on-error to attempt all stages).")
        sys.exit(1)
    if n_fail:
        sys.exit(1)


if __name__ == "__main__":
    main()
