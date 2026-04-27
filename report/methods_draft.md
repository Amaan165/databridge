# Methods + Evaluation (Sections 5 + 6)

**DataBridge — NYU DS-GA 1019, Spring 2026**
**Author of this draft:** Rithujaa Rajendrakumar (taxonomy + crosswalk
+ RQ4 evaluation)

> This file is the working draft for §5 (Methods) and §6 (Evaluation)
> of `report/report_draft.md`. It is meant to be folded into the main
> report at assembly time. Numbers in `[BRACKETS]` are placeholders
> that will be populated from `outputs/rq4_*.csv` when the report is
> finalized; the values shown after the bracket are the current
> measured values from the most recent pipeline run.

---

## 5. Methods

The unified violation taxonomy and cross-city crosswalk that connect
the three cities' violation records are built in three steps:
(i) construct a single inventory of every distinct violation
description, (ii) embed each description with a sentence-transformer
and group them with density-based clustering, (iii) compute pairwise
similarities across cities and validate candidate matches with an LLM.
This section documents each step, the parameter choices, and the
practical constraints of the data that motivated those choices.

### 5.1 Violation inventory

The cleaning stage (§3) produces three per-violation CSVs. Each city
contributes a different number of distinct violation descriptions:

| City    | Distinct violation strings | Style                                       |
|---------|---------------------------:|---------------------------------------------|
| NYC     | 168                        | Prose-style, median 104 chars (max 952)     |
| Chicago | 64                         | Short ALL-CAPS, regulatory-code style       |
| Boston  | 313                        | Medium, dash-separated phrases              |
| **Total inventory** | **545**          |                                             |

The inventory (`violation_taxonomy/violation_inventory.csv`) records,
for each (text, city) pair, the number of times that description
appears in the cleaned data. This frequency signal is used later to
prioritize representative members for cluster labeling.

The 4.9× imbalance between Chicago (64 categories) and Boston (313)
is a property of the source data, not a cleaning artifact: Chicago
publishes a curated list of broad categories, while Boston's data is
tied directly to the underlying regulation IDs and therefore
fragments each broad concept into many sibling rows (e.g., 14
distinct rows for variants of "Food-Contact Surfaces"). The taxonomy
step is what re-aggregates Boston's fragments into Chicago-style
broad categories.

### 5.2 Embeddings

Each of the 545 violation strings is embedded using
`sentence-transformers/all-MiniLM-L6-v2`, producing a 384-dimensional
vector per description. Three considerations led to this choice:

1. **Semantic similarity, not lexical overlap.** Bag-of-words and
   TF-IDF would group Boston's "Food-Contact Surfaces-Cleanability"
   with "Food-Contact Surfaces-Construction" purely on shared tokens
   while leaving NYC's prose-style equivalent in a different cluster.
   Sentence-level transformer embeddings are designed for paraphrase
   detection and capture the semantic equivalence directly.
2. **Size.** all-MiniLM-L6-v2 is small enough to embed all 545
   violations in under five seconds on a laptop CPU, making the
   step part of the routine pipeline rather than a one-off.
3. **Public, reproducible, no key required.** Unlike OpenAI
   embeddings, the model is downloadable from HuggingFace and runs
   entirely offline. The teammates can rerun the pipeline without
   any API credentials.

The output is a `(545, 384)` float32 matrix saved as
`violation_taxonomy/violation_embeddings.npy`.

### 5.3 Taxonomy: HDBSCAN clustering

The 545 embeddings are clustered with HDBSCAN, with cluster labels
constituting the violation taxonomy. We run HDBSCAN with parameters
`min_cluster_size=3`, `min_samples=1`, and `cluster_selection_method='eom'`.

These values were selected by running a small grid (10 configurations
varying `min_cluster_size` ∈ {3, 5, 7, 10} × `min_samples` ∈ {1, 3, 5}
and the selection method `eom` vs `leaf`) and choosing the
configuration that produced (a) the largest number of clusters
without (b) excessive single-violation singletons or (c) implausibly
broad clusters mixing unrelated concepts. The chosen configuration
yields 66 raw clusters and a 40.9% noise rate, which after the
post-processing steps below collapses to the final 64 categories.

**Why density-based, not k-means.** k-means requires choosing k in
advance, which is exactly the question we are trying to answer. It
also assumes spherical clusters of similar size, but the violation
space is decidedly not spherical: a cluster like "Hot/Cold Holding
Temperatures" has dozens of close neighbors across all three cities,
while "Sodium Warning Icon Posted" has only NYC members and lives in
a sparse region. HDBSCAN's variable-density semantics handle both
ends of this spectrum without forcing a single bandwidth.

**Noise re-assignment.** HDBSCAN labels 223 violations (40.9%) as
noise. Leaving these unassigned would defeat the purpose of the
taxonomy, so each noise point is re-assigned to its nearest cluster
centroid by cosine similarity. This converts the soft "no confident
cluster" signal from HDBSCAN into a hard assignment without changing
the underlying cluster shapes — a noise point may sit at the edge of
its assigned cluster, but it is closer to that cluster's centroid
than to any other.

**Cluster merging (Chicago-anchor coalescing).** After noise
re-assignment, two clusters that both map to the same closest
Chicago category and whose centroids are themselves cosine-similar
(>0.65) are merged into one. This collapses near-duplicates that
HDBSCAN's density semantics had spuriously split, e.g., two clusters
that both anchor onto Chicago's `PHYSICAL FACILITIES INSTALLED,
MAINTAINED & CLEAN`. The merge step removes 2 redundant clusters,
yielding 64 final categories.

### 5.4 Cluster labeling

The final 64 clusters are given human-readable category names in two
passes.

**Pass 1: Chicago-anchored labels.** For each cluster, we find the
single Chicago violation closest to the cluster centroid and use its
text (after cleanup: title-case, expand `&` to `and`, strip
regulatory citation suffixes) as a baseline label. Chicago's 64
ALL-CAPS labels are essentially a curated regulatory taxonomy, so
anchoring to them gives a defensible default label even without an
LLM.

**Pass 2: GPT-4o relabeling (optional).** Chicago labels are
sometimes too long or framed in regulatory rather than analytical
language (e.g., `PHYSICAL FACILITIES INSTALLED, MAINTAINED & CLEAN`).
When `OPENAI_API_KEY` is present, the pipeline issues one
`gpt-4o` call per cluster, sending the cluster's top-5
most-frequent violations and prompting for a 2-5-word category name.
The 64 calls cost about $0.05 in total and complete in roughly 35
seconds. The GPT-4o labels are noticeably more readable
("Pest Control Management" instead of `INSECTS, RODENTS, AND ANIMALS
NOT PRESENT - CONTROLLING PESTS`) while remaining faithful to the
cluster contents — none of the LLM-generated labels mislabel the
cluster's dominant theme on manual inspection.

**Duplicate-name resolution.** GPT-4o occasionally emits the same
short name for two different clusters (e.g., assigning
"Single-Use Item Violations" to both cluster 25 and cluster 50, or
"Temperature Control Violations" to both 61 and 62). After labeling,
any duplicate name is resolved by inspecting the underlying clusters
and assigning distinguishing descriptors derived from the dominant
non-Chicago violation, yielding e.g.,
"Disposable Items and Use Restrictions" vs.
"Single-Service Article Compliance" and
"Time-Temperature Control Procedures" vs.
"Holding and Cooking Temperatures".

The taxonomy is written to
`data/integrated/violation_taxonomy.csv` (one row per inventory
violation) and a category lookup to
`violation_taxonomy/taxonomy_categories.csv` (one row per category,
with the originating Chicago anchor preserved for traceability).

### 5.5 Cross-city crosswalk

The taxonomy answers "what category does each violation belong to?"
The crosswalk answers the complementary question: "given a specific
violation in city A, what is the equivalent specific violation in
city B?" The two are related — two violations in the same taxonomy
cluster are almost certainly cross-city equivalents — but the
crosswalk is finer-grained and city-pair-aware, which is the format
that downstream cross-city queries actually need.

**Candidate generation.** For each of the three city pairs
(NYC-Chicago, NYC-Boston, Chicago-Boston) we compute the full
pairwise cosine-similarity matrix between embeddings of the two
cities' violations. We retain every pair with cosine similarity
≥ 0.70.

**Threshold tiers.**

- **`sim ≥ 0.85` — auto-validated.** The similarity is high enough
  that the pair is treated as a confirmed match without further
  validation. In practice these are pairs that share substantial
  surface text (e.g., Chicago's `WASHING FRUITS & VEGETABLES` and
  Boston's `Washing Fruits and Vegetables` at 0.997).
- **`0.70 ≤ sim < 0.85` — LLM-validated.** These are candidate
  matches whose embedding similarity is suggestive but not
  conclusive. Each is sent to `gpt-4o` with the prompt
  *"Are these two food safety violations describing the same issue?
  Answer only yes or no."* and the result is recorded in
  `match_validated`.

The 0.70 threshold was chosen empirically as the lowest similarity
at which manual ground-truth pairs (§6.1) were still substantially
represented in the candidate pool. The 0.85 auto-validation cutoff
was chosen to match the empirical observation that pairs above it
have essentially zero false-positive rate on the manual sample.

**Final crosswalk.** Pairs with `match_validated = True` (either
auto or LLM-confirmed) are kept; LLM-rejected pairs are removed.
The current pipeline run produced 126 candidates → 90 validated
matches (15 auto-validated, 75 LLM-confirmed; 36 rejected by GPT-4o).
The 90 validated pairs span all three city pairs:

| City pair | Pairs | Mean similarity |
|-----------|------:|----------------:|
| Chicago ↔ Boston | 50 | 0.807 |
| NYC ↔ Boston     | 23 | 0.743 |
| NYC ↔ Chicago    | 17 | 0.778 |

**Skew toward Chicago-Boston.** That 56% of validated pairs are
Chicago-Boston and only 19% are NYC-Chicago is a direct artifact of
the text-style mismatch noted in §5.1: NYC's prose-style descriptions
embed differently from Chicago's ALL-CAPS labels and Boston's
phrase-style entries, so even semantically equivalent NYC pairs
land at lower cosine similarity than the corresponding Chicago-Boston
pair would. §6 quantifies the impact of this skew on recall.

---

## 6. Evaluation (RQ4)

> *RQ4: How well does the LLM-assisted semantic crosswalk align with
> human-labeled cross-city pairs?*

The crosswalk is produced by an automated pipeline; we need a
ground-truth check. Section 6 reports the result of that check.

### 6.1 Manual ground truth

During Week 1 a hand-curated ground-truth file
(`violation_taxonomy/manual_crosscity_matches.csv`) was assembled of
24 cross-city violation concepts that should be equivalent across all
three cities. For each concept the file records the exact NYC,
Chicago, and Boston violation text that the analyst judged to be the
correct match, plus a difficulty rating:

| Difficulty | n  | Description                                                  |
|------------|---:|--------------------------------------------------------------|
| easy       | 13 | Concept is named almost identically in all three cities      |
| medium     | 7  | Concept is present in all three but the wording diverges     |
| hard       | 4  | Genuine taxonomy gap; one city frames the concept differently |

Because each concept is a 3-tuple, expanding each row into all
unordered city pairs gives **72 bilateral ground-truth pairs**
(24 × 3): 24 NYC-Chicago, 24 NYC-Boston, 24 Chicago-Boston.

All 72 ground-truth pairs reference inventory texts that exist
verbatim in `violation_inventory.csv`, so embeddings for every
ground-truth pair are directly available — there is no
text-normalization gap between manual GT and the production
embeddings.

### 6.2 Recall against ground truth

Of the 72 ground-truth bilateral pairs, the production crosswalk
contains **`[GT_RECALL=8]`/72 (`[GT_RECALL_PCT=11.1]`%)**. The other 64
pairs are missed for one of two reasons:

| Reason                                   | Count | % of GT |
|------------------------------------------|------:|--------:|
| Embedding similarity below 0.70 cutoff   | 60    | 83.3%   |
| Above 0.70 but rejected by GPT-4o        | 4     | 5.6%    |
| **In final crosswalk**                   | **8** | **11.1%** |

The dominant failure mode is that embedding similarity itself does
not separate the ground-truth pairs from non-pairs sharply enough at
0.70. The four LLM rejections are listed in
`outputs/rq4_llm_rejections.csv`. By definition all four are
human-judged cross-city matches (they are in the manual file), but
on close inspection two of them are unambiguous semantic equivalents
that the LLM should arguably have accepted — NYC's "Thawing procedure
improper" vs. Chicago's `APPROVED THAWING METHODS USED` (sim 0.78),
and NYC's "Toxic chemical improperly labeled, stored or used..."
vs. Chicago's `TOXIC SUBSTANCES PROPERLY IDENTIFIED, STORED, & USED`
(sim 0.76). The other two pair concepts that the analyst grouped
together but that describe different specific aspects (food-contact
surface *cleaning* vs. surface *cleanability*; handwashing-sink
*supply* vs. handwashing-sink *placement*); the LLM's rejection of
those is defensible. The pattern suggests the LLM sets a higher
specificity bar than our manual reviewer for borderline cases.

### 6.3 Recall by difficulty

The human-labeled difficulty rating predicts embedding similarity
almost perfectly, which validates both the manual labels and the
embedding model:

| Difficulty | n  | Mean cosine sim | Recall @ 0.50 | Recall @ 0.70 | Recall @ 0.85 |
|------------|---:|----------------:|--------------:|--------------:|--------------:|
| easy       | 39 | 0.634           | 87.2%         | 28.2%         | 5.1%          |
| medium     | 21 | 0.452           | 38.1%         | 4.8%          | 0.0%          |
| hard       | 12 | 0.399           | 25.0%         | 0.0%          | 0.0%          |

Two takeaways:

1. **The 0.70 threshold is too strict for the easy band.** Even on
   pairs the analyst judged unambiguous, embedding similarity sits
   below 0.70 on 71.8% of them — primarily NYC-anything pairs whose
   text-length asymmetry depresses cosine similarity.
2. **Hard pairs may be unrecoverable from text alone.** The "hard"
   category includes genuine taxonomy gaps (e.g., Boston has no
   direct allergen-awareness equivalent; Boston frames smoking as
   employee behavior rather than signage). No embedding threshold
   recovers these without false positives.

### 6.4 Precision-recall curve

Sweeping cosine similarity from 0.30 to 0.95 and treating every
non-GT bilateral pair as a presumed negative gives the following
operating characteristic (`outputs/rq4_pr_curve.csv`,
`outputs/rq4_pr_curve.png`):

| Threshold | Predictions | TP | FP    | Precision | Recall | F1    |
|-----------|------------:|---:|------:|----------:|-------:|------:|
| 0.50      | 1,226       | 45 | 1,181 | 0.037     | 0.625  | 0.069 |
| 0.65      | 222         | 20 | 202   | 0.090     | 0.278  | 0.136 |
| **0.70**  | **126**     | **12** | **114** | **0.095** | **0.167** | **0.121** |
| 0.85      | 15          |  2 |    13 | 0.133     | 0.028  | 0.046 |
| 0.90      |  9          |  1 |     8 | 0.111     | 0.014  | 0.025 |

The best F1 occurs at threshold **0.65**, which would recover ~28% of
the manual ground truth at the cost of ~10% precision. At the
production threshold (0.70) the system sits near the F1 peak but
slightly biased toward precision.

**Caveat on absolute precision.** The figures above treat every
bilateral pair *outside* the 24-concept manual file as a negative.
This is severely conservative: many of the crosswalk's 90 pairs are
valid matches that simply happen to fall outside our 24 manually
chosen concepts (e.g., the six "Food-Contact Surface" Chicago-Boston
variants the system identifies at sim ≥ 0.85 are all genuine
restatements of the same handful of underlying concepts, but only
one of them is represented in the manual file because the manual
file used a single representative per concept). The reported
precision is therefore a lower bound; an unrestricted manual audit
of all 90 crosswalk pairs would put true precision substantially
higher.

### 6.5 Embedding-only vs. LLM-validated

The LLM validation step in the 0.70-0.85 band is the central design
choice that separates this pipeline from pure embedding-based
approaches. We compare the three operating regimes against the
manual ground truth (`outputs/rq4_filter_comparison.csv`):

| Regime                      | Predictions | TP | FP  | Precision | Recall |
|-----------------------------|------------:|---:|----:|----------:|-------:|
| Embedding-only @ 0.70       | 126         | 12 | 114 | 0.095     | 0.167  |
| Embedding-only @ 0.85       |  15         |  2 |  13 | 0.133     | 0.028  |
| Embedding + LLM (production)|  90         |  8 |  82 | 0.089     | 0.111  |

The LLM step removes 36 candidates from the 0.70-cutoff pool. Of
those 36, four were ground-truth pairs (LLM false negatives, §6.2),
which is why the LLM regime's recall is 11% instead of 17%. The
remaining 32 LLM rejections are not in our ground-truth set and we
do not have manual labels for them; manual inspection of a sample
suggests most are genuine non-matches (e.g.,
"Food contact surface not properly *washed*" vs.
"Non-food contact surfaces *clean*", which share embedding-relevant
tokens but describe different surfaces and different actions).

In short: the LLM trades a small amount of recall on borderline
ground-truth pairs for a meaningful reduction in plausible-looking
false candidates that embedding similarity alone cannot rule out.
Without a substantially larger labeled negative set we cannot
quantify the precision improvement, only the recall cost.

### 6.6 Limitations

- **Ground-truth size.** 24 concepts is enough to characterize
  recall but too small to estimate absolute precision. A larger
  labeled sample (200-300 pairs sampled across the cosine-similarity
  range) would tighten the bounds in §6.4.
- **Single embedding model.** We did not benchmark alternative
  embedders (e.g., `mxbai-embed-large`, OpenAI `text-embedding-3-small`).
  The text-length asymmetry between NYC and the other cities suggests
  a model with stronger long-vs-short text matching might lift the
  NYC-anything recall meaningfully.
- **Single LLM.** GPT-4o was used both for cluster labeling and pair
  validation. We did not compare against an open-weights model
  (e.g., Llama-3.1-70B-Instruct) which would make the validation
  step zero-cost and reproducible without API credentials.
- **Cosine threshold is global.** A separate threshold per city pair
  would likely improve recall — NYC-Chicago and NYC-Boston pairs
  cluster at lower similarities than Chicago-Boston pairs do (§6.3).
  The current pipeline applies a single 0.70 cutoff to all three.
- **Operating-point lock-in.** Lowering the threshold to 0.65 would
  raise recall but also raise the LLM call volume from 111 calls to
  ~200, and we have not budgeted for that. The operating-point
  choice is driven by API cost, not pure F1.

### 6.7 Takeaways for the conclusion

1. **The crosswalk reliably finds the easy cross-city matches that
   share substantial surface text.** Above similarity 0.85 the
   precision is essentially 1.0 on manual inspection of all 15
   auto-validated pairs (the §6.4 figure of 0.133 reflects only the
   pessimistic GT-only metric, not actual match quality).
2. **It misses most NYC-X matches.** NYC's prose-style violation
   descriptions consistently embed at lower cosine similarity than
   the corresponding Chicago/Boston pair, even when the two are
   semantically equivalent.
3. **GPT-4o is conservative.** Of the four LLM rejections in the
   ground-truth band, three are arguably valid; the LLM is biased
   toward "not the same" when the two strings differ in surface
   wording.
4. **Difficulty ratings are predictive.** The "easy / medium / hard"
   labels assigned during Week 1 correlate strongly with embedding
   similarity, which is independent evidence that the embeddings
   capture the same notion of semantic distance the analyst was
   using.
