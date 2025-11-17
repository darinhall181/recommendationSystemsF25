
# Recommendation System Comparison

This project implements and evaluates **three recommendation setups**—a **Two-Tower model**, a **Pairwise BPR model**, and a lightweight **Context Distance Regularization** enhancement—across three diverse real-world review datasets: **Goodreads Spoilers**, **BeerAdvocate**, and **Steam Reviews**.

The goal is to analyze how each approach performs under different data conditions (sparsity, long-tail behavior, user consistency, multi-domain preferences), while keeping computation modest and the modeling interpretable.

---

## Systems Implementation

### Two-Tower Model (User–Item Embedding Model)

In the **two-tower** architecture, we learn:

* A **user embedding tower**:
  [
  u_i \in \mathbb{R}^d
  ]
* An **item embedding tower**:
  [
  v_j \in \mathbb{R}^d
  ]

The relevance score is computed via a dot product:
[
s(i,j) = u_i^\top v_j
]

**Key Features**

* Scales well to large datasets
* Easy to train and evaluate
* Embeddings are global and *not conditioned* on the specific user–item interaction context

**Drawbacks**

* **Pair-agnostic**: the same item vector is used for *all* users
* Prone to **popularity bias**
* Struggles with sparse users who have few interactions

---

### Pairwise BPR (Bayesian Personalized Ranking)

Pairwise models optimize the *relative order* of positive vs. negative items:

[
L = -\log \sigma\left(s(i,j^+) - s(i,j^-)\right)
]

This encourages:
[
s(i,j^+) > s(i,j^-)
]

**Key Features**

* Optimizes *ranking*, not rating prediction
* Handles implicit feedback (clicks, purchases, reviews) effectively
* Learns finer-grained user preference signals

**Drawbacks**

* Sensitive to the **quality of negative sampling**
* Still uses the same global item embeddings → limited contextualization
* Can overfit if users have small histories

---

### Context Distance Regularization (Our Twist)

To incorporate **lightweight contextual awareness** without a full graph model, we add a penalty encouraging positive-item embeddings to remain close to the user’s historical preference context.

Compute the **user context vector**:
[
\mu_i = \frac{1}{|H_i|} \sum_{k \in H_i} v_k
]

Define context distance:
[
d(i,j) = 1 - \text{cosine}(v_j, \mu_i)
]

Final loss:
[
L = L_{\text{BPR}} + \gamma \cdot d(i,j)
]

**Key Benefits**

* “Mini” contextual signal similar to ContextGNN principles
* No graph processing → still lightweight
* Helps with:

  * Sparse user histories
  * Domains with strong thematic clusters (books, beers, games)

**Drawbacks**

* Context vector quality depends on the size & consistency of the user’s past interactions
* Not as expressive as a full GNN

---

## Dataset Outline

We evaluate on three very different review platforms to guarantee diversity in:

* User behavior
* Item types
* Contextual consistency
* Sparsity levels
* Domain-specific rating tendencies

We intentionally extract **bare-minimum user–item–rating triples** to create consistent recommendation frames across all datasets.

---

### 1. Goodreads Spoilers Dataset

*(User → Book reviews + spoiler tags)*

**Portion used**

* User ID
* Book ID
* Rating (1–5)
* Optional: spoiler indicator (ignored for modeling but useful for analysis)

**Why it’s relevant**

* Highly **textual** domain – user tastes cluster strongly (genres, authors)
* Ratings usually stable: people develop niche reading preferences
* Long-tail: many books with very few interactions → tests robustness to item sparsity

**Challenges**

* Users may review sparsely (cold-start users)
* Spoiler presence has no numeric meaning, so we ignore it for training
* Item popularity distribution is extremely skewed

---

### 2. BeerAdvocate Reviews Dataset

*(User → Beer review scores)*

**Portion used**

* User ID
* Beer/Product ID
* Overall review rating

**Why it’s relevant**

* Dense per-user histories (beer enthusiasts leave many reviews)
* Strong **style-based** similarities (IPA, stout, lager) → good for context distance regularization
* Natural testing ground for preference clustering

**Challenges**

* Users tend to have **coherent but narrow** tastes
* Item graph is highly clustered → global two-tower embeddings struggle
* Some beers have very few ratings

---

### 3. Steam Reviews Dataset

*(User → Game reviews + playtime)*

**Portion used**

* User ID
* Game ID
* Review score (positive/negative or mapped numeric)
* Optional: playtime (ignored for modeling unless explicitly used)

**Why it’s relevant**

* Larger ecosystem with very **diverse preferences**
* Items (games) vary widely in genre, price, complexity
* Excellent domain for pairwise ranking because binary/implicit feedback works well

**Challenges**

* High item popularity skew (AAA games dominate)
* Many casual users → extremely sparse user vectors
* Review polarity varies widely by game genre

---

## Why These Three Together?

| Dataset      | Domain | Density     | Long-Tail | User Consistency | Why It’s Useful                                                      |
| ------------ | ------ | ----------- | --------- | ---------------- | -------------------------------------------------------------------- |
| Goodreads    | Books  | Medium      | **High**  | High             | Tests performance on thematic preference clustering                  |
| BeerAdvocate | Beers  | Medium–High | Medium    | Very High        | Shows how models behave when users have tight, stylistic preferences |
| Steam        | Games  | Low–Medium  | **High**  | Low–Medium       | Tests robustness to sparsity and implicit-feedback dynamics          |

This combination gives us:

* A **themed**, preference-clustered domain (Books)
* A **taste-consistent** domain (Beers)
* A **sparse, wide-domain** gaming dataset (Steam)

Together, they provide a **broad stress test** for pairwise ranking and two-tower models.

---

## Things to Consider When Working With These Datasets

### Data Preprocessing

* Ensure user and item IDs are **reindexed to consecutive integers**
* Consider discarding:

  * Users with < 3 reviews
  * Items with < 3 reviews
* Normalize ratings across datasets if needed
* Convert Steam’s binary reviews → numeric if required

### Evaluation

Use consistent metrics across datasets:

* HR@k
* NDCG@k
* MAP@k
* Recall@k

### Negative Sampling

* Uniform negatives often work but may underperform on long-tail books/games
* Popularity-weighted negatives can improve stability

### Sparsity Issues

* Goodreads & Steam especially benefit from context distance regularization
* BeerAdvocate might favor pairwise ranking due to user consistency

---

## Summary

This experiment compares:

* **Two-Tower (global embeddings)**
* **Pairwise BPR (relative ranking)**
* **Context Distance Regularization (our contextual twist)**

across **Goodreads Spoilers**, **BeerAdvocate**, and **Steam Reviews**, each chosen to represent a distinct behavioral domain.

The trio of datasets helps reveal:

* How models generalize from dense → sparse domains
* How popularity and context interact
* How lightweight contextual signals improve ranking quality



