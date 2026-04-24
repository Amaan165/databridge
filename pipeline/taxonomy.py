"""
taxonomy.py — Task 2.3: Violation Taxonomy (Embed + Cluster + Label)
DataBridge Project | Data Engineering Spring 2026

Builds a unified violation taxonomy from the 545-violation inventory
using HDBSCAN clustering and GPT-4o labeling.

Pipeline:
  1. Load violation_inventory.csv + violation_embeddings.npy (from Week 1)
  2. HDBSCAN clustering (min_cluster_size=3, min_samples=1, eom)
  3. Assign noise points to nearest cluster centroid
  4. Merge clusters sharing the same Chicago anchor (similarity > 0.65)
  5. Label each cluster using nearest Chicago category as initial anchor
  6. (Optional) GPT-4o relabeling: send top-5 violations per cluster,
     get concise 2-5 word name
  7. Resolve duplicate category names with distinguishing suffixes

Input:
  violation_taxonomy/violation_inventory.csv   (545 rows)
  violation_taxonomy/violation_embeddings.npy  (545 × 384)

Output:
  data/integrated/violation_taxonomy.csv       (545 rows, load_taxonomy.py format)
  violation_taxonomy/taxonomy_categories.csv   (64 categories with metadata)

Usage:
  cd databridge/
  export OPENAI_API_KEY=sk-...    # only needed if using --use-llm
  python pipeline/taxonomy.py --use-llm

  Options:
    --inventory PATH     Violation inventory CSV (default: violation_taxonomy/violation_inventory.csv)
    --embeddings PATH    Embeddings NPY file (default: violation_taxonomy/violation_embeddings.npy)
    --output PATH        Output taxonomy CSV (default: data/integrated/violation_taxonomy.csv)
    --min-cluster-size   HDBSCAN min_cluster_size (default: 3)
    --min-samples        HDBSCAN min_samples (default: 1)
    --use-llm            Enable GPT-4o relabeling (requires OPENAI_API_KEY)
    --llm-model          OpenAI model for labeling (default: gpt-4o)
"""

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import hdbscan


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

INVENTORY_PATH = os.path.join("violation_taxonomy", "violation_inventory.csv")
EMBEDDINGS_PATH = os.path.join("violation_taxonomy", "violation_embeddings.npy")
OUTPUT_PATH = os.path.join("data", "integrated", "violation_taxonomy.csv")
CATEGORIES_PATH = os.path.join("violation_taxonomy", "taxonomy_categories.csv")

# HDBSCAN defaults — tuned for 545 violations in 384-dim embedding space
DEFAULT_MIN_CLUSTER_SIZE = 3
DEFAULT_MIN_SAMPLES = 1
MERGE_THRESHOLD = 0.65  # merge clusters sharing same anchor if centroids > this


# ─────────────────────────────────────────────────────────────────────────────
# Step 1: HDBSCAN Clustering
# ─────────────────────────────────────────────────────────────────────────────

def run_clustering(emb: np.ndarray, min_cluster_size: int,
                   min_samples: int) -> np.ndarray:
    """Run HDBSCAN and return cluster labels."""
    print(f"  HDBSCAN params: min_cluster_size={min_cluster_size}, "
          f"min_samples={min_samples}, method=eom")

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric='euclidean',
        cluster_selection_method='eom',
    )
    labels = clusterer.fit_predict(emb)

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = (labels == -1).sum()
    print(f"  Clusters: {n_clusters}, Noise: {n_noise} ({n_noise/len(labels)*100:.1f}%)")

    return labels


# ─────────────────────────────────────────────────────────────────────────────
# Step 2: Assign Noise to Nearest Cluster
# ─────────────────────────────────────────────────────────────────────────────

def assign_noise(emb: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Assign noise points (-1) to nearest cluster centroid."""
    cluster_ids = sorted(set(labels) - {-1})
    centroids = np.array([emb[labels == cid].mean(axis=0) for cid in cluster_ids])

    noise_mask = labels == -1
    if noise_mask.sum() == 0:
        return labels.copy()

    noise_sim = cosine_similarity(emb[noise_mask], centroids)
    nearest = np.array([cluster_ids[i] for i in noise_sim.argmax(axis=1)])

    labels_full = labels.copy()
    labels_full[noise_mask] = nearest
    print(f"  Assigned {noise_mask.sum()} noise points to nearest clusters")

    return labels_full


# ─────────────────────────────────────────────────────────────────────────────
# Step 3: Merge Clusters with Same Chicago Anchor
# ─────────────────────────────────────────────────────────────────────────────

def merge_similar_clusters(emb: np.ndarray, labels: np.ndarray,
                           inv: pd.DataFrame) -> np.ndarray:
    """Merge clusters that map to the same Chicago category and are similar."""
    cluster_ids = sorted(set(labels))
    centroids = np.array([emb[labels == cid].mean(axis=0) for cid in cluster_ids])

    # Get Chicago embeddings
    chi_mask = inv['city'] == 'chicago'
    chi_texts = inv.loc[chi_mask, 'violation_text'].tolist()
    chi_emb = emb[chi_mask.values]

    # Map each cluster to nearest Chicago category
    sim_to_chi = cosine_similarity(centroids, chi_emb)
    cluster_anchors = {}
    for i, cid in enumerate(cluster_ids):
        best_idx = sim_to_chi[i].argmax()
        cluster_anchors[cid] = chi_texts[best_idx]

    # Group clusters by anchor
    anchor_groups = {}
    for cid, anchor in cluster_anchors.items():
        anchor_groups.setdefault(anchor, []).append(cid)

    # Merge small clusters into the largest one sharing the same anchor
    merge_count = 0
    for anchor, cids in anchor_groups.items():
        if len(cids) <= 1:
            continue
        sizes = {cid: (labels == cid).sum() for cid in cids}
        target = max(cids, key=lambda c: sizes[c])

        for cid in cids:
            if cid == target:
                continue
            idx_t = cluster_ids.index(target)
            idx_c = cluster_ids.index(cid)
            sim = cosine_similarity(
                centroids[idx_t:idx_t+1], centroids[idx_c:idx_c+1]
            )[0][0]
            if sim > MERGE_THRESHOLD:
                labels[labels == cid] = target
                merge_count += 1

    print(f"  Merged {merge_count} cluster pairs")
    return labels


# ─────────────────────────────────────────────────────────────────────────────
# Step 4: Label Clusters Using Chicago Categories
# ─────────────────────────────────────────────────────────────────────────────

def clean_chicago_label(text: str) -> str:
    """Convert ALL-CAPS Chicago category to Title Case."""
    label = text.replace('&', 'and').title().strip()
    for word in ['And', 'Or', 'For', 'In', 'Of', 'Not', 'The', 'A', 'To',
                 'As', 'W/', 'Rte', 'At']:
        label = label.replace(f' {word} ', f' {word.lower()} ')
    return label


def label_clusters(emb: np.ndarray, labels: np.ndarray,
                   inv: pd.DataFrame) -> dict:
    """Assign human-readable category names using Chicago anchors."""
    cluster_ids = sorted(set(labels))
    centroids = np.array([emb[labels == cid].mean(axis=0) for cid in cluster_ids])

    chi_mask = inv['city'] == 'chicago'
    chi_texts = inv.loc[chi_mask, 'violation_text'].tolist()
    chi_emb = emb[chi_mask.values]

    sim_to_chi = cosine_similarity(centroids, chi_emb)

    # First pass: assign labels
    category_info = {}
    for i, cid in enumerate(cluster_ids):
        tax_id = i + 1
        best_idx = sim_to_chi[i].argmax()
        best_sim = float(sim_to_chi[i][best_idx])
        chi_anchor = chi_texts[best_idx]
        label = clean_chicago_label(chi_anchor)

        category_info[cid] = {
            'taxonomy_category_id': tax_id,
            'category_name': label,
            'chicago_anchor': chi_anchor,
            'anchor_similarity': round(best_sim, 3),
        }

    # Second pass: resolve duplicate names
    name_cids = {}
    for cid, info in category_info.items():
        name_cids.setdefault(info['category_name'], []).append(cid)

    for name, cids in name_cids.items():
        if len(cids) <= 1:
            continue
        for cid in cids[1:]:
            members = inv.loc[labels == cid]
            non_chi = members[members['city'] != 'chicago']
            if len(non_chi) > 0:
                top = non_chi.nlargest(1, 'frequency')['violation_text'].values[0]
            else:
                top = members.nlargest(1, 'frequency')['violation_text'].values[0]
            suffix = ' '.join(top.split()[:4])
            category_info[cid]['category_name'] = f"{name} - {suffix}"

    return category_info


# ─────────────────────────────────────────────────────────────────────────────
# Step 5: Optional GPT-4o Relabeling
# ─────────────────────────────────────────────────────────────────────────────

def relabel_with_llm(category_info: dict, inv: pd.DataFrame,
                     labels: np.ndarray, model: str = "gpt-4o") -> dict:
    """Use GPT-4o to generate concise category names from cluster contents.

    For each cluster, sends the top-5 most frequent violations to the LLM
    and asks for a 2-5 word category name. Falls back to original label
    on API failure.

    Requires OPENAI_API_KEY in environment.
    """
    try:
        from openai import OpenAI
    except ImportError:
        print("  ERROR: openai package not installed. Run: pip install openai")
        print("  Skipping LLM relabeling — keeping Chicago-anchor labels.")
        return category_info

    if not os.environ.get("OPENAI_API_KEY"):
        print("  ERROR: OPENAI_API_KEY not set in environment.")
        print("  Skipping LLM relabeling — keeping Chicago-anchor labels.")
        return category_info

    client = OpenAI()
    inv = inv.copy()
    inv['_cluster_id'] = labels

    print(f"  Relabeling {len(category_info)} clusters with {model}...")
    failures = 0
    for cid, info in category_info.items():
        members = inv[inv['_cluster_id'] == cid]
        if 'frequency' in members.columns:
            top5 = members.nlargest(5, 'frequency')['violation_text'].tolist()
        else:
            top5 = members['violation_text'].head(5).tolist()

        violations_list = "\n".join(f"  - {v}" for v in top5)
        prompt = (
            "These food safety violations belong to the same category:\n\n"
            f"{violations_list}\n\n"
            "Give this category a short, clear name (2-5 words). "
            "Just the name, nothing else. No quotes, no explanation."
        )

        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=20,
                temperature=0.0,
            )
            label = response.choices[0].message.content.strip().strip('"').strip("'")
            info['category_name_original'] = info['category_name']
            info['category_name'] = label
        except Exception as e:
            failures += 1
            print(f"    [{info['taxonomy_category_id']}] FAILED: {e}")
        time.sleep(0.3)  # avoid rate limiting

    print(f"  Relabeled {len(category_info) - failures}/{len(category_info)} "
          f"clusters via {model} ({failures} failures)")

    # Resolve any duplicate LLM names by appending a distinguishing suffix
    name_cids = {}
    for cid, info in category_info.items():
        name_cids.setdefault(info['category_name'], []).append(cid)

    dedup_count = 0
    for name, cids in name_cids.items():
        if len(cids) <= 1:
            continue
        # Keep the first, distinguish the rest
        for cid in cids[1:]:
            members = inv[inv['_cluster_id'] == cid]
            if 'frequency' in members.columns:
                top = members.nlargest(1, 'frequency')['violation_text'].values[0]
            else:
                top = members['violation_text'].iloc[0]
            suffix = ' '.join(top.split()[:3])
            category_info[cid]['category_name'] = f"{name} ({suffix})"
            dedup_count += 1

    if dedup_count > 0:
        print(f"  Resolved {dedup_count} duplicate LLM labels with suffixes")
    return category_info


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Build violation taxonomy via HDBSCAN + Chicago-anchored labeling."
    )
    parser.add_argument("--inventory", default=INVENTORY_PATH)
    parser.add_argument("--embeddings", default=EMBEDDINGS_PATH)
    parser.add_argument("--output", default=OUTPUT_PATH)
    parser.add_argument("--min-cluster-size", type=int,
                        default=DEFAULT_MIN_CLUSTER_SIZE)
    parser.add_argument("--min-samples", type=int,
                        default=DEFAULT_MIN_SAMPLES)
    parser.add_argument("--use-llm", action="store_true",
                        help="Enable GPT-4o relabeling (requires OPENAI_API_KEY)")
    parser.add_argument("--llm-model", default="gpt-4o",
                        help="OpenAI model for labeling")
    args = parser.parse_args()

    print("=" * 65)
    print("DataBridge — Violation Taxonomy Builder")
    print("=" * 65)
    start = time.time()

    # Load inputs
    print("\n[1/5] Loading inputs...")
    inv = pd.read_csv(args.inventory)
    emb = np.load(args.embeddings)
    print(f"  Inventory: {len(inv)} violations")
    print(f"  Embeddings: {emb.shape}")

    # Cluster
    print("\n[2/5] Running HDBSCAN clustering...")
    labels = run_clustering(emb, args.min_cluster_size, args.min_samples)

    # Assign noise
    print("\n[3/5] Assigning noise points...")
    labels = assign_noise(emb, labels)

    # Merge similar clusters
    print("\n[4/5] Merging similar clusters...")
    labels = merge_similar_clusters(emb, labels, inv)

    # Label clusters
    print("\n[5/5] Labeling clusters (Chicago-anchored)...")
    category_info = label_clusters(emb, labels, inv)

    # Optional: GPT-4o relabeling
    if args.use_llm:
        print(f"\n[5b] Relabeling clusters with LLM ({args.llm_model})...")
        category_info = relabel_with_llm(category_info, inv, labels,
                                          model=args.llm_model)

    # Build output DataFrame
    inv['cluster_id'] = labels
    cid_to_tax = {cid: info['taxonomy_category_id']
                  for cid, info in category_info.items()}
    cid_to_name = {cid: info['category_name']
                   for cid, info in category_info.items()}

    inv['taxonomy_category_id'] = inv['cluster_id'].map(cid_to_tax)
    inv['category_name'] = inv['cluster_id'].map(cid_to_name)

    # Save taxonomy
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    output_cols = ['violation_text', 'city', 'cluster_id',
                   'taxonomy_category_id', 'category_name']
    inv[output_cols].to_csv(args.output, index=False)
    print(f"\n  Saved {args.output} ({len(inv)} rows)")

    # Save category lookup
    cat_df = pd.DataFrame([
        info for info in category_info.values()
    ]).sort_values('taxonomy_category_id')
    cat_df.to_csv(CATEGORIES_PATH, index=False)
    print(f"  Saved {CATEGORIES_PATH} ({len(cat_df)} categories)")

    # Summary
    n_cats = len(category_info)
    print(f"\n{'='*65}")
    print(f"TAXONOMY SUMMARY")
    print(f"{'='*65}")
    print(f"  Total violations:  {len(inv)}")
    print(f"  Total categories:  {n_cats}")
    for city in ['nyc', 'chicago', 'boston']:
        n = (inv['city'] == city).sum()
        cats = inv[inv['city'] == city]['taxonomy_category_id'].nunique()
        print(f"    {city:15s} {n:>4} violations across {cats} categories")

    # Multi-city categories
    multi = 0
    for cid in set(labels):
        if inv.loc[labels == cid, 'city'].nunique() > 1:
            multi += 1
    print(f"  Multi-city categories: {multi}/{n_cats} ({multi/n_cats*100:.0f}%)")
    print(f"{'='*65}")

    elapsed = time.time() - start
    print(f"\nTotal time: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
