"""
rq4_crosswalk_eval.py — Task 3.3: Crosswalk Evaluation (RQ4)
DataBridge Project | Data Engineering Spring 2026

RQ4: How well does the LLM-assisted semantic crosswalk align with
human-labeled cross-city pairs?

Methodology:
  1. Build ground truth from violation_taxonomy/manual_crosscity_matches.csv
     - 24 manual concepts × 3 city pairs (NYC↔CHI, NYC↔BOS, CHI↔BOS)
     - = 72 bilateral ground-truth pairs, with difficulty rating
  2. Compute cosine similarity for every cross-city bilateral pair in
     the 545-violation inventory using the existing embeddings.
  3. Recall against ground truth: how many of the 72 manual pairs are
     in the system's final crosswalk? Why are the others missed?
       - below the 0.70 candidate threshold? (embedding miss)
       - above 0.70 but rejected by GPT-4o? (LLM false negative)
  4. Precision-recall curve: sweep cosine threshold from 0.30 → 0.95
     using all 83K cross-city bilateral pairs, with manual matches as
     positives and the remaining pairs as (presumed) negatives.
  5. Embedding-only vs Embedding+LLM comparison: at the operating
     point of the production pipeline (≥0.85 auto, 0.70-0.85 LLM-gated),
     show how many positives survive each filter and how many
     negatives are blocked.
  6. Difficulty breakdown: easy/medium/hard recall at multiple
     thresholds, validating that human-labeled difficulty correlates
     with embedding similarity.

Output:
  outputs/rq4_summary.csv             headline metrics (recall, precision, F1)
  outputs/rq4_pr_curve.csv            full precision-recall sweep
  outputs/rq4_pr_curve.png            PR curve + similarity histogram
  outputs/rq4_difficulty_breakdown.csv recall by easy/medium/hard
  outputs/rq4_missed_pairs.csv        manual pairs not in final crosswalk + reason
  outputs/rq4_llm_rejections.csv      LLM false negatives (manual pairs the LLM rejected)
  outputs/rq4_similarity_distribution.png  histogram: positives vs negatives

Usage:
  cd databridge/
  python analysis/rq4_crosswalk_eval.py

  Options:
    --inventory PATH    Inventory CSV (default: violation_taxonomy/violation_inventory.csv)
    --embeddings PATH   Embeddings NPY (default: violation_taxonomy/violation_embeddings.npy)
    --manual PATH       Manual matches CSV (default: violation_taxonomy/manual_crosscity_matches.csv)
    --crosswalk PATH    Crosswalk CSV (default: data/integrated/violation_crosswalk.csv)
    --out-dir DIR       Output directory (default: outputs)

Note on the precision metric:
  Manual ground truth covers 24 concepts. Many crosswalk pairs outside
  this list are still valid (e.g., the four "Food-Contact Surface"
  Chicago↔Boston variants the system finds at sim≥0.85). Treating those
  as false positives understates the system's true precision. The PR
  curve here uses the manual-only definition, which is conservative —
  the report should note that real precision is higher than reported.
"""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ── Configuration ────────────────────────────────────────────────────────────

INVENTORY_PATH = Path("violation_taxonomy") / "violation_inventory.csv"
EMBEDDINGS_PATH = Path("violation_taxonomy") / "violation_embeddings.npy"
MANUAL_PATH = Path("violation_taxonomy") / "manual_crosscity_matches.csv"
CROSSWALK_PATH = Path("data") / "integrated" / "violation_crosswalk.csv"
OUT_DIR = Path("outputs")

CITY_PAIRS = [("nyc", "chicago"), ("nyc", "boston"), ("chicago", "boston")]

# Display labels keyed by the sorted-alphabetical form returned by
# city_pair_label(). Tuple is (display_string, color).
CITY_PAIR_DISPLAY = {
    "chicago-nyc":    ("NYC ↔ Chicago",    "#1f77b4"),
    "boston-nyc":     ("NYC ↔ Boston",     "#ff7f0e"),
    "boston-chicago": ("Chicago ↔ Boston", "#2ca02c"),
}

# Operating point of the production pipeline (matches crosswalk.py defaults)
THRESHOLD_CANDIDATE = 0.70   # below this, pair is dropped
THRESHOLD_AUTO = 0.85        # above this, auto-validated (no LLM call)

# Threshold sweep for PR curve
PR_THRESHOLDS = np.round(np.arange(0.30, 0.96, 0.05), 2)


# ── Helpers ──────────────────────────────────────────────────────────────────

def pair_key(text_a: str, city_a: str, text_b: str, city_b: str) -> frozenset:
    """Order-invariant key for a bilateral cross-city pair."""
    return frozenset([(text_a, city_a), (text_b, city_b)])


def city_pair_label(city_a: str, city_b: str) -> str:
    return "-".join(sorted([city_a, city_b]))


# ── Step 1: Build ground truth bilateral pairs ───────────────────────────────

def build_ground_truth(manual_df: pd.DataFrame) -> pd.DataFrame:
    """Expand the 24-row manual file into bilateral pairs.

    Each manual row may have NYC, Chicago, and/or Boston entries.
    Each non-null pair becomes one ground-truth row, tagged with
    category and difficulty.
    """
    rows = []
    for _, m in manual_df.iterrows():
        cells = {"nyc": m.get("nyc"), "chicago": m.get("chicago"),
                 "boston": m.get("boston")}
        cities = [c for c, v in cells.items() if pd.notna(v)]
        for i, ca in enumerate(cities):
            for cb in cities[i+1:]:
                rows.append({
                    "category":   m["category"],
                    "difficulty": m["difficulty"],
                    "city_a":     ca,
                    "text_a":     cells[ca],
                    "city_b":     cb,
                    "text_b":     cells[cb],
                    "city_pair":  city_pair_label(ca, cb),
                })
    return pd.DataFrame(rows)


# ── Step 2: Compute similarities for every cross-city bilateral pair ─────────

def compute_all_pair_similarities(inv: pd.DataFrame,
                                   emb: np.ndarray) -> pd.DataFrame:
    """For every pair (text_a in city A, text_b in city B) for each
    of the three city pairs, compute cosine similarity.

    Returns a DataFrame with one row per bilateral pair and columns:
      city_a, text_a, city_b, text_b, city_pair, sim
    """
    by_city = {c: inv[inv["city"] == c].reset_index().rename(
                   columns={"index": "inv_idx"})
               for c in ["nyc", "chicago", "boston"]}

    rows = []
    for ca, cb in CITY_PAIRS:
        a, b = by_city[ca], by_city[cb]
        sim_mat = cosine_similarity(emb[a["inv_idx"].values],
                                    emb[b["inv_idx"].values])
        for i in range(len(a)):
            for j in range(len(b)):
                rows.append({
                    "city_a":    ca,
                    "text_a":    a.iloc[i]["violation_text"],
                    "city_b":    cb,
                    "text_b":    b.iloc[j]["violation_text"],
                    "city_pair": city_pair_label(ca, cb),
                    "sim":       float(sim_mat[i, j]),
                })
    return pd.DataFrame(rows)


# ── Step 3: Label every pair as positive / negative using GT ─────────────────

def label_pairs(all_pairs: pd.DataFrame, gt: pd.DataFrame) -> pd.DataFrame:
    """Add an `is_positive` column. A pair is positive iff it appears in GT."""
    gt_keys = {pair_key(r.text_a, r.city_a, r.text_b, r.city_b)
               for r in gt.itertuples()}
    all_pairs = all_pairs.copy()
    all_pairs["is_positive"] = all_pairs.apply(
        lambda r: pair_key(r["text_a"], r["city_a"],
                           r["text_b"], r["city_b"]) in gt_keys,
        axis=1,
    )
    # Also tag GT pairs with their difficulty
    diff_lookup = {pair_key(r.text_a, r.city_a, r.text_b, r.city_b): r.difficulty
                   for r in gt.itertuples()}
    all_pairs["difficulty"] = all_pairs.apply(
        lambda r: diff_lookup.get(
            pair_key(r["text_a"], r["city_a"], r["text_b"], r["city_b"])),
        axis=1,
    )
    return all_pairs


# ── Step 4: Crosswalk recall (current production system) ─────────────────────

def evaluate_crosswalk(gt: pd.DataFrame, all_pairs: pd.DataFrame,
                       crosswalk: pd.DataFrame) -> dict:
    """Compare GT pairs against the system's final crosswalk."""
    cw_keys = {pair_key(r.violation_desc_city_a, r.city_a,
                        r.violation_desc_city_b, r.city_b)
               for r in crosswalk.itertuples()}

    sim_lookup = {pair_key(r.text_a, r.city_a, r.text_b, r.city_b): r.sim
                  for r in all_pairs.itertuples()}

    found = []
    missed_low_sim = []        # below 0.70 threshold
    llm_rejected = []          # 0.70 ≤ sim, but not in final crosswalk
    for r in gt.itertuples():
        k = pair_key(r.text_a, r.city_a, r.text_b, r.city_b)
        sim = sim_lookup.get(k, np.nan)
        in_cw = k in cw_keys
        record = {
            "category":   r.category,
            "difficulty": r.difficulty,
            "city_a":     r.city_a, "text_a": r.text_a,
            "city_b":     r.city_b, "text_b": r.text_b,
            "sim":        sim,
            "in_crosswalk": in_cw,
        }
        if in_cw:
            found.append(record)
        elif pd.notna(sim) and sim >= THRESHOLD_CANDIDATE:
            llm_rejected.append(record)
        else:
            missed_low_sim.append(record)

    return {
        "found":         pd.DataFrame(found),
        "llm_rejected":  pd.DataFrame(llm_rejected),
        "missed_low_sim":pd.DataFrame(missed_low_sim),
    }


# ── Step 5: Precision-recall curve ───────────────────────────────────────────

def compute_pr_curve(labeled: pd.DataFrame,
                     thresholds: np.ndarray) -> pd.DataFrame:
    """Sweep cosine-similarity thresholds and compute P/R/F1 at each."""
    n_pos = labeled["is_positive"].sum()
    rows = []
    for t in thresholds:
        predicted = labeled["sim"] >= t
        tp = ((predicted) & (labeled["is_positive"])).sum()
        fp = ((predicted) & (~labeled["is_positive"])).sum()
        fn = ((~predicted) & (labeled["is_positive"])).sum()
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / n_pos if n_pos > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)
              if (precision + recall) > 0 else 0.0)
        rows.append({
            "threshold": t,
            "tp": int(tp), "fp": int(fp), "fn": int(fn),
            "precision": precision, "recall": recall, "f1": f1,
            "n_predicted": int(predicted.sum()),
        })
    return pd.DataFrame(rows)


# ── Step 6: Difficulty breakdown ─────────────────────────────────────────────

def compute_difficulty_breakdown(gt: pd.DataFrame,
                                  all_pairs: pd.DataFrame,
                                  thresholds=(0.50, 0.60, 0.70, 0.85)) -> pd.DataFrame:
    sim_lookup = {pair_key(r.text_a, r.city_a, r.text_b, r.city_b): r.sim
                  for r in all_pairs.itertuples()}
    rows = []
    for diff in ["easy", "medium", "hard"]:
        sub = gt[gt["difficulty"] == diff]
        if len(sub) == 0:
            continue
        sims = [sim_lookup.get(pair_key(r.text_a, r.city_a,
                                         r.text_b, r.city_b), np.nan)
                for r in sub.itertuples()]
        sims = pd.Series(sims).dropna()
        row = {
            "difficulty":     diff,
            "n_pairs":        len(sub),
            "mean_similarity":round(sims.mean(), 3),
            "min_similarity": round(sims.min(), 3),
            "max_similarity": round(sims.max(), 3),
        }
        for t in thresholds:
            row[f"recall@{t:.2f}"] = round((sims >= t).sum() / len(sub), 3)
        rows.append(row)
    return pd.DataFrame(rows)


# ── Step 7: Embedding-only vs Embedding+LLM comparison ───────────────────────

def compare_filters(labeled: pd.DataFrame, crosswalk: pd.DataFrame) -> pd.DataFrame:
    """At the production operating point, compare three filter regimes:

      (a) embedding only, threshold 0.70 (no LLM)
      (b) embedding only, threshold 0.85 (strict, no LLM)
      (c) embedding + LLM (production: ≥0.85 auto, 0.70-0.85 LLM-gated)

    The final crosswalk file represents (c).
    """
    cw_keys = {pair_key(r.violation_desc_city_a, r.city_a,
                        r.violation_desc_city_b, r.city_b)
               for r in crosswalk.itertuples()}

    n_pos = labeled["is_positive"].sum()
    n_neg = (~labeled["is_positive"]).sum()

    rows = []

    # (a) embedding-only @ 0.70
    pred = labeled["sim"] >= 0.70
    tp = ((pred) & labeled["is_positive"]).sum()
    fp = ((pred) & ~labeled["is_positive"]).sum()
    rows.append({"regime": "embedding-only @ 0.70", "n_predicted": int(pred.sum()),
                 "tp": int(tp), "fp": int(fp),
                 "precision": tp / (tp + fp) if (tp + fp) else 0.0,
                 "recall": tp / n_pos})

    # (b) embedding-only @ 0.85
    pred = labeled["sim"] >= 0.85
    tp = ((pred) & labeled["is_positive"]).sum()
    fp = ((pred) & ~labeled["is_positive"]).sum()
    rows.append({"regime": "embedding-only @ 0.85", "n_predicted": int(pred.sum()),
                 "tp": int(tp), "fp": int(fp),
                 "precision": tp / (tp + fp) if (tp + fp) else 0.0,
                 "recall": tp / n_pos})

    # (c) embedding + LLM (production crosswalk)
    in_cw = labeled.apply(
        lambda r: pair_key(r["text_a"], r["city_a"],
                           r["text_b"], r["city_b"]) in cw_keys,
        axis=1,
    )
    tp = (in_cw & labeled["is_positive"]).sum()
    fp = (in_cw & ~labeled["is_positive"]).sum()
    rows.append({"regime": "embedding + LLM (production)", "n_predicted": int(in_cw.sum()),
                 "tp": int(tp), "fp": int(fp),
                 "precision": tp / (tp + fp) if (tp + fp) else 0.0,
                 "recall": tp / n_pos})

    df = pd.DataFrame(rows)
    df["precision"] = df["precision"].round(3)
    df["recall"] = df["recall"].round(3)
    return df


# ── Charts ───────────────────────────────────────────────────────────────────

def chart_pr_curve(pr: pd.DataFrame, out: Path) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # Left: precision and recall vs threshold
    ax1.plot(pr["threshold"], pr["precision"], "o-", label="Precision",
             color="#d62728", linewidth=2)
    ax1.plot(pr["threshold"], pr["recall"], "s-", label="Recall",
             color="#2ca02c", linewidth=2)
    ax1.plot(pr["threshold"], pr["f1"], "^-", label="F1",
             color="#1f77b4", linewidth=2)
    ax1.axvline(THRESHOLD_CANDIDATE, color="grey", linestyle="--", alpha=0.6,
                label=f"production cutoff ({THRESHOLD_CANDIDATE})")
    ax1.set_xlabel("Cosine similarity threshold")
    ax1.set_ylabel("Score")
    ax1.set_title("Precision / Recall / F1 vs threshold")
    ax1.legend(loc="best")
    ax1.set_ylim(-0.02, 1.02)
    ax1.grid(alpha=0.3)

    # Right: classic precision-recall curve
    ax2.plot(pr["recall"], pr["precision"], "o-", color="#1f77b4", linewidth=2)
    for _, r in pr.iterrows():
        if r["threshold"] in (0.50, 0.70, 0.85):
            ax2.annotate(f"t={r['threshold']:.2f}",
                         xy=(r["recall"], r["precision"]),
                         xytext=(5, 5), textcoords="offset points",
                         fontsize=9)
    ax2.set_xlabel("Recall")
    ax2.set_ylabel("Precision")
    ax2.set_title("Precision-Recall curve")
    ax2.set_xlim(-0.02, 1.02)
    ax2.set_ylim(-0.02, 1.02)
    ax2.grid(alpha=0.3)

    plt.suptitle("RQ4: Crosswalk evaluation against 72 manual ground-truth pairs",
                 fontsize=12, y=1.00)
    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()


def chart_similarity_distribution(labeled: pd.DataFrame, out: Path) -> None:
    """Plot similarity histograms for matches vs non-matches per city pair.

    Cosine similarities range slightly below 0 in our data (down to
    about -0.21) for unrelated pairs, so bins are extended to -0.25 to
    avoid silent drops that would distort the density normalization.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
    # Order: NYC↔Chicago, NYC↔Boston, Chicago↔Boston
    pair_order = ["chicago-nyc", "boston-nyc", "boston-chicago"]
    bins = np.arange(-0.25, 1.05, 0.05)

    for ax, cp in zip(axes, pair_order):
        sub = labeled[labeled["city_pair"] == cp]
        pos = sub[sub["is_positive"]]["sim"].values
        neg = sub[~sub["is_positive"]]["sim"].values
        display_name, color = CITY_PAIR_DISPLAY[cp]
        if len(neg) > 0:
            ax.hist(neg, bins=bins, alpha=0.5, color="grey",
                    label=f"non-matches (n={len(neg):,})", density=True)
        if len(pos) > 0:
            ax.hist(pos, bins=bins, alpha=0.7, color=color,
                    label=f"manual matches (n={len(pos)})", density=True)
        ax.axvline(THRESHOLD_CANDIDATE, color="red", linestyle="--", alpha=0.7,
                   label=f"threshold ({THRESHOLD_CANDIDATE})")
        ax.set_title(display_name)
        ax.set_xlabel("Cosine similarity")
        ax.set_xlim(-0.25, 1.0)
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Density")
    plt.suptitle("Similarity distributions: manual matches vs other pairs",
                 fontsize=12, y=1.02)
    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="RQ4: Evaluate semantic crosswalk against manual ground truth."
    )
    parser.add_argument("--inventory", default=str(INVENTORY_PATH))
    parser.add_argument("--embeddings", default=str(EMBEDDINGS_PATH))
    parser.add_argument("--manual", default=str(MANUAL_PATH))
    parser.add_argument("--crosswalk", default=str(CROSSWALK_PATH))
    parser.add_argument("--out-dir", default=str(OUT_DIR))
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("DataBridge — RQ4: Crosswalk Evaluation")
    print("=" * 70)
    start = time.time()

    # ── Load ────────────────────────────────────────────────────────────────
    print("\n[1/6] Loading inputs...")
    inv = pd.read_csv(args.inventory)
    emb = np.load(args.embeddings)
    manual = pd.read_csv(args.manual)
    crosswalk = pd.read_csv(args.crosswalk)
    print(f"  Inventory:  {len(inv)} violations across {inv['city'].nunique()} cities")
    print(f"  Embeddings: {emb.shape}")
    print(f"  Manual GT:  {len(manual)} concept rows")
    print(f"  Crosswalk:  {len(crosswalk)} validated pairs")

    if len(inv) != emb.shape[0]:
        print(f"  ERROR: inventory rows ({len(inv)}) ≠ embedding rows ({emb.shape[0]})")
        sys.exit(1)

    # ── Build GT ────────────────────────────────────────────────────────────
    print("\n[2/6] Expanding manual matches into bilateral GT pairs...")
    gt = build_ground_truth(manual)
    print(f"  Ground truth pairs: {len(gt)}")
    print(f"  By city pair:")
    for cp in ["chicago-nyc", "boston-nyc", "boston-chicago"]:
        sub = gt[gt["city_pair"] == cp]
        display = CITY_PAIR_DISPLAY[cp][0]
        print(f"    {display:20s} {len(sub):>3d} pairs")
    print(f"  By difficulty:")
    for diff in ["easy", "medium", "hard"]:
        sub = gt[gt["difficulty"] == diff]
        print(f"    {diff:20s} {len(sub):>3d} pairs")

    # ── Compute all pair similarities ──────────────────────────────────────
    print("\n[3/6] Computing similarities for all cross-city bilateral pairs...")
    all_pairs = compute_all_pair_similarities(inv, emb)
    print(f"  Total bilateral pairs: {len(all_pairs):,}")
    for cp in ["chicago-nyc", "boston-nyc", "boston-chicago"]:
        sub = all_pairs[all_pairs["city_pair"] == cp]
        display = CITY_PAIR_DISPLAY[cp][0]
        print(f"    {display:20s} {len(sub):>7,} pairs   "
              f"(mean sim {sub['sim'].mean():.3f})")

    # ── Label and evaluate ──────────────────────────────────────────────────
    print("\n[4/6] Labeling pairs and running crosswalk recall analysis...")
    labeled = label_pairs(all_pairs, gt)

    eval_results = evaluate_crosswalk(gt, all_pairs, crosswalk)
    n_found = len(eval_results["found"])
    n_llm = len(eval_results["llm_rejected"])
    n_low = len(eval_results["missed_low_sim"])
    print(f"  Of {len(gt)} GT pairs:")
    print(f"    in final crosswalk:                {n_found:>3} "
          f"({n_found/len(gt)*100:>4.1f}%)")
    print(f"    above 0.70 but rejected by LLM:    {n_llm:>3} "
          f"({n_llm/len(gt)*100:>4.1f}%)")
    print(f"    below 0.70 candidate threshold:    {n_low:>3} "
          f"({n_low/len(gt)*100:>4.1f}%)")

    # ── PR curve ────────────────────────────────────────────────────────────
    print(f"\n[5/6] Computing precision-recall curve over {len(PR_THRESHOLDS)} thresholds...")
    pr = compute_pr_curve(labeled, PR_THRESHOLDS)
    best = pr.loc[pr["f1"].idxmax()]
    print(f"  Best F1 at threshold {best['threshold']:.2f}: "
          f"P={best['precision']:.3f}  R={best['recall']:.3f}  F1={best['f1']:.3f}")
    print(f"  At production threshold 0.70:")
    prod = pr[pr["threshold"] == 0.70].iloc[0]
    print(f"    P={prod['precision']:.3f}  R={prod['recall']:.3f}  F1={prod['f1']:.3f}")

    diff_breakdown = compute_difficulty_breakdown(gt, all_pairs)
    filter_cmp = compare_filters(labeled, crosswalk)

    print(f"\n  Filter regime comparison:")
    for _, r in filter_cmp.iterrows():
        print(f"    {r['regime']:32s} predicted={r['n_predicted']:>3} "
              f"P={r['precision']:.3f}  R={r['recall']:.3f}")

    # ── Write outputs ───────────────────────────────────────────────────────
    print(f"\n[6/6] Writing outputs to {out_dir}/...")

    # Headline summary
    summary_rows = [
        {"metric": "n_ground_truth_pairs",        "value": len(gt)},
        {"metric": "n_crosswalk_pairs",           "value": len(crosswalk)},
        {"metric": "gt_in_crosswalk",             "value": n_found},
        {"metric": "gt_recall",                   "value": round(n_found / len(gt), 3)},
        {"metric": "gt_rejected_by_llm",          "value": n_llm},
        {"metric": "gt_below_threshold",          "value": n_low},
        {"metric": "best_f1_threshold",           "value": float(best["threshold"])},
        {"metric": "best_f1",                     "value": round(float(best["f1"]), 3)},
        {"metric": "best_f1_precision",           "value": round(float(best["precision"]), 3)},
        {"metric": "best_f1_recall",              "value": round(float(best["recall"]), 3)},
        {"metric": "production_threshold",        "value": THRESHOLD_CANDIDATE},
        {"metric": "production_precision",        "value": round(float(prod["precision"]), 3)},
        {"metric": "production_recall",           "value": round(float(prod["recall"]), 3)},
    ]
    pd.DataFrame(summary_rows).to_csv(out_dir / "rq4_summary.csv", index=False)
    print(f"  wrote rq4_summary.csv")

    pr.to_csv(out_dir / "rq4_pr_curve.csv", index=False)
    print(f"  wrote rq4_pr_curve.csv")

    diff_breakdown.to_csv(out_dir / "rq4_difficulty_breakdown.csv", index=False)
    print(f"  wrote rq4_difficulty_breakdown.csv")

    filter_cmp.to_csv(out_dir / "rq4_filter_comparison.csv", index=False)
    print(f"  wrote rq4_filter_comparison.csv")

    # Missed pairs (everything not in final crosswalk)
    missed = pd.concat([
        eval_results["llm_rejected"].assign(reason="rejected_by_llm"),
        eval_results["missed_low_sim"].assign(reason="below_threshold"),
    ], ignore_index=True).sort_values(["reason", "sim"], ascending=[True, False])
    missed.to_csv(out_dir / "rq4_missed_pairs.csv", index=False)
    print(f"  wrote rq4_missed_pairs.csv ({len(missed)} rows)")

    if len(eval_results["llm_rejected"]) > 0:
        eval_results["llm_rejected"].sort_values("sim", ascending=False)\
            .to_csv(out_dir / "rq4_llm_rejections.csv", index=False)
        print(f"  wrote rq4_llm_rejections.csv "
              f"({len(eval_results['llm_rejected'])} rows)")

    # Charts
    chart_pr_curve(pr, out_dir / "rq4_pr_curve.png")
    print(f"  wrote rq4_pr_curve.png")

    chart_similarity_distribution(labeled, out_dir / "rq4_similarity_distribution.png")
    print(f"  wrote rq4_similarity_distribution.png")

    # ── Console summary ────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("RQ4 SUMMARY")
    print(f"{'='*70}")
    print(f"  Ground truth:  {len(gt)} bilateral pairs from "
          f"{len(manual)} manual concepts")
    print(f"  Crosswalk:     {len(crosswalk)} validated pairs (production)")
    print()
    print(f"  Recall against manual ground truth: "
          f"{n_found}/{len(gt)} = {n_found/len(gt)*100:.1f}%")
    print(f"    + {n_llm} GT pairs were above 0.70 but rejected by GPT-4o")
    print(f"    + {n_low} GT pairs fell below the 0.70 candidate threshold")
    print()
    print(f"  Recall by difficulty (at threshold 0.70 / 0.50):")
    for _, r in diff_breakdown.iterrows():
        print(f"    {r['difficulty']:7s} (n={r['n_pairs']:>2}): "
              f"@0.70 {r['recall@0.70']*100:>5.1f}%, "
              f"@0.50 {r['recall@0.50']*100:>5.1f}%, "
              f"@0.85 {r['recall@0.85']*100:>5.1f}%")
    print()
    print(f"  Best F1 occurs at threshold {best['threshold']:.2f} "
          f"(P={best['precision']:.3f}, R={best['recall']:.3f}, F1={best['f1']:.3f})")
    print()
    print(f"  Caveat: precision is computed treating non-GT pairs as negatives.")
    print(f"          Many crosswalk pairs outside GT are still valid matches,")
    print(f"          so true precision is higher than reported here.")
    print(f"{'='*70}")

    elapsed = time.time() - start
    print(f"\nTotal time: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
