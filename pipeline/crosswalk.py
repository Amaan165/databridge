"""
crosswalk.py — Task 2.4: Semantic Crosswalk (Embedding-Based Matching)
DataBridge Project | Data Engineering Spring 2026

Builds cross-city violation mappings by computing pairwise cosine
similarity between violation embeddings and filtering pairs above
a similarity threshold.

Pipeline:
  1. Load violation_taxonomy.csv + embeddings
  2. For each city pair (NYC-Chicago, NYC-Boston, Chicago-Boston),
     compute cosine similarity between all violation pairs
  3. Filter pairs with similarity ≥ 0.70
  4. Auto-validate pairs with similarity ≥ 0.85
  5. Link to taxonomy_category_id where both violations share a cluster

Input:
  data/integrated/violation_taxonomy.csv       (from taxonomy.py)
  violation_taxonomy/violation_embeddings.npy  (from Week 1)

Output:
  data/integrated/violation_crosswalk.csv      (load_taxonomy.py format)

Usage:
  cd databridge/
  export OPENAI_API_KEY=sk-...    # only needed if using --use-llm
  python pipeline/crosswalk.py --use-llm

  Options:
    --taxonomy PATH      Taxonomy CSV (default: data/integrated/violation_taxonomy.csv)
    --embeddings PATH    Embeddings NPY (default: violation_taxonomy/violation_embeddings.npy)
    --output PATH        Crosswalk CSV (default: data/integrated/violation_crosswalk.csv)
    --threshold FLOAT    Min cosine similarity (default: 0.70)
    --auto-validate FLOAT  Auto-validate above this threshold (default: 0.85)
    --use-llm            Enable GPT-4o validation for 0.70-0.85 pairs
    --llm-model          OpenAI model (default: gpt-4o)
    --keep-rejected      Keep GPT-4o rejected pairs in output (default: drop them)

LLM validation (GPT-4o):
  After generating candidate pairs, GPT-4o validates each pair:
    - sim ≥ 0.85: auto-validated (match_validated = true)
    - sim 0.70–0.85: validated by GPT-4o (yes/no)
  By default, rejected pairs (GPT-4o said "no") are dropped from the output.
  Without --use-llm, pairs in 0.70-0.85 range are kept with match_validated=null.
"""

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

TAXONOMY_PATH = os.path.join("data", "integrated", "violation_taxonomy.csv")
EMBEDDINGS_PATH = os.path.join("violation_taxonomy", "violation_embeddings.npy")
OUTPUT_PATH = os.path.join("data", "integrated", "violation_crosswalk.csv")

DEFAULT_THRESHOLD = 0.70
DEFAULT_AUTO_VALIDATE = 0.85

CITY_PAIRS = [('nyc', 'chicago'), ('nyc', 'boston'), ('chicago', 'boston')]


# ─────────────────────────────────────────────────────────────────────────────
# Build Crosswalk
# ─────────────────────────────────────────────────────────────────────────────

def build_crosswalk(inv: pd.DataFrame, emb: np.ndarray,
                    threshold: float, auto_validate: float) -> pd.DataFrame:
    """Compute pairwise similarities and extract matches above threshold."""

    # Index by city
    city_indices = {
        c: inv[inv['city'] == c].index.tolist()
        for c in ['nyc', 'chicago', 'boston']
    }

    # Taxonomy lookup for shared-category detection
    tax_lookup = dict(zip(
        zip(inv['violation_text'], inv['city']),
        inv['taxonomy_category_id']
    ))

    rows = []
    for city_a, city_b in CITY_PAIRS:
        idx_a = city_indices[city_a]
        idx_b = city_indices[city_b]

        sim_matrix = cosine_similarity(emb[idx_a], emb[idx_b])

        pairs = 0
        for i, ia in enumerate(idx_a):
            for j, ib in enumerate(idx_b):
                sim = sim_matrix[i][j]
                if sim < threshold:
                    continue

                text_a = inv.iloc[ia]['violation_text']
                text_b = inv.iloc[ib]['violation_text']
                tax_a = tax_lookup.get((text_a, city_a))
                tax_b = tax_lookup.get((text_b, city_b))

                rows.append({
                    'violation_desc_city_a': text_a,
                    'city_a': city_a,
                    'violation_desc_city_b': text_b,
                    'city_b': city_b,
                    'cosine_similarity': round(float(sim), 4),
                    'match_validated': True if sim >= auto_validate else None,
                    'taxonomy_category_id': tax_a if tax_a == tax_b else None,
                })
                pairs += 1

        print(f"  {city_a:>8s} ↔ {city_b:<8s}: {pairs:>5,} pairs")

    df = pd.DataFrame(rows)
    df = df.sort_values('cosine_similarity', ascending=False).reset_index(drop=True)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# LLM Validation (Optional)
# ─────────────────────────────────────────────────────────────────────────────

def validate_with_llm(crosswalk: pd.DataFrame, model: str = "gpt-4o") -> pd.DataFrame:
    """Validate candidate crosswalk pairs (match_validated is null) with GPT-4o.

    Sends each pair as 'are these the same violation?' and sets match_validated
    to True (yes) or False (no). Falls back gracefully on API failure.

    Requires OPENAI_API_KEY in environment.
    """
    try:
        from openai import OpenAI
    except ImportError:
        print("  ERROR: openai package not installed. Run: pip install openai")
        print("  Skipping LLM validation — pairs in 0.70-0.85 range stay null.")
        return crosswalk

    if not os.environ.get("OPENAI_API_KEY"):
        print("  ERROR: OPENAI_API_KEY not set in environment.")
        print("  Skipping LLM validation — pairs in 0.70-0.85 range stay null.")
        return crosswalk

    client = OpenAI()
    needs_review = crosswalk[crosswalk['match_validated'].isna()]
    print(f"  Validating {len(needs_review)} pairs with {model}...")

    validated = 0
    rejected = 0
    failures = 0
    for idx in needs_review.index:
        row = crosswalk.loc[idx]
        prompt = (
            "Are these two food safety violations describing the same issue?\n\n"
            f"Violation A ({row['city_a']}): {row['violation_desc_city_a']}\n"
            f"Violation B ({row['city_b']}): {row['violation_desc_city_b']}\n\n"
            f"Cosine similarity: {row['cosine_similarity']}\n\n"
            "Answer only \"yes\" or \"no\"."
        )

        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=5,
                temperature=0.0,
            )
            answer = response.choices[0].message.content.strip().lower()
            is_match = answer.startswith("yes")
            crosswalk.at[idx, 'match_validated'] = is_match
            if is_match:
                validated += 1
            else:
                rejected += 1
        except Exception as e:
            failures += 1
            print(f"    row {idx} FAILED: {e}")
        time.sleep(0.3)

    print(f"  Confirmed: {validated}, Rejected: {rejected}, Failures: {failures}")
    return crosswalk


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Build cross-city violation crosswalk via embedding similarity."
    )
    parser.add_argument("--taxonomy", default=TAXONOMY_PATH)
    parser.add_argument("--embeddings", default=EMBEDDINGS_PATH)
    parser.add_argument("--output", default=OUTPUT_PATH)
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--auto-validate", type=float, default=DEFAULT_AUTO_VALIDATE)
    parser.add_argument("--use-llm", action="store_true",
                        help="Enable GPT-4o validation (requires OPENAI_API_KEY)")
    parser.add_argument("--llm-model", default="gpt-4o",
                        help="OpenAI model for validation")
    parser.add_argument("--keep-rejected", action="store_true",
                        help="Keep GPT-4o rejected pairs (default: drop)")
    args = parser.parse_args()

    print("=" * 65)
    print("DataBridge — Semantic Crosswalk Builder")
    print("=" * 65)
    start = time.time()

    # Load
    print("\n[1/2] Loading inputs...")
    inv = pd.read_csv(args.taxonomy)
    emb = np.load(args.embeddings)
    print(f"  Taxonomy: {len(inv)} violations")
    print(f"  Embeddings: {emb.shape}")

    # Build crosswalk
    print(f"\n[2/3] Building crosswalk (threshold={args.threshold})...")
    crosswalk = build_crosswalk(inv, emb, args.threshold, args.auto_validate)

    # Optional: LLM validation
    if args.use_llm:
        print(f"\n[3/3] Validating candidate pairs with LLM ({args.llm_model})...")
        crosswalk = validate_with_llm(crosswalk, model=args.llm_model)

        # Drop rejected pairs unless --keep-rejected
        if not args.keep_rejected:
            before = len(crosswalk)
            crosswalk = crosswalk[crosswalk['match_validated'] != False].reset_index(drop=True)
            print(f"  Dropped {before - len(crosswalk)} GPT-4o-rejected pairs")
    else:
        print(f"\n[3/3] Skipping LLM validation (use --use-llm to enable)")

    # Save
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    crosswalk.to_csv(args.output, index=False)
    print(f"\n  Saved {args.output} ({len(crosswalk)} rows)")

    # Summary
    validated = (crosswalk['match_validated'] == True).sum() if len(crosswalk) > 0 else 0
    needs_review = crosswalk['match_validated'].isna().sum()
    shared = crosswalk['taxonomy_category_id'].notna().sum()

    print(f"\n{'='*65}")
    print(f"CROSSWALK SUMMARY")
    print(f"{'='*65}")
    print(f"  Total pairs:             {len(crosswalk):,}")
    print(f"  Auto-validated (≥{args.auto_validate}):  {validated:,}")
    print(f"  Needs LLM review:        {needs_review:,}")
    print(f"  Shared taxonomy cluster: {shared:,}")
    print(f"  Mean similarity:         {crosswalk['cosine_similarity'].mean():.3f}")

    # Distribution
    print(f"\n  Similarity distribution:")
    bins = [(0.70, 0.75), (0.75, 0.80), (0.80, 0.85),
            (0.85, 0.90), (0.90, 0.95), (0.95, 1.01)]
    for lo, hi in bins:
        n = ((crosswalk['cosine_similarity'] >= lo) &
             (crosswalk['cosine_similarity'] < hi)).sum()
        bar = '█' * (n // 2) if n > 0 else ''
        print(f"    {lo:.2f}–{hi:.2f}: {n:>5,}  {bar}")

    print(f"{'='*65}")

    elapsed = time.time() - start
    print(f"\nTotal time: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
