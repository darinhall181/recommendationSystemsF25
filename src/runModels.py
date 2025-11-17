import json
import os
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse import csr_matrix
from implicit.als import AlternatingLeastSquares
from implicit.bpr import BayesianPersonalizedRanking
from implicit.nearest_neighbours import bm25_weight

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(THIS_DIR)
SAMPLE_DIR = os.path.join(ROOT, "sampleSets")


# Load triples the way dataCleaning.py saved them
def load_cleaned_triples(fname):
    path = os.path.join(SAMPLE_DIR, fname)
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    with open(path, "r") as f:
        return json.load(f)


# Slightly cleaned up user-wise split
def train_test_split_userwise(mat, random_state=42):
    rng = np.random.default_rng(random_state)

    mat = mat.tocsr()
    n_users, n_items = mat.shape

    tr_u, tr_i, tr_v = [], [], []
    te_u, te_i, te_v = [], [], []

    indptr = mat.indptr
    idx = mat.indices
    vals = mat.data

    for u in range(n_users):
        start, end = indptr[u], indptr[u+1]
        items = idx[start:end]
        scores = vals[start:end]

        if len(items) == 0:
            continue
        if len(items) == 1:
            tr_u.append(u)
            tr_i.append(items[0])
            tr_v.append(scores[0])
            continue

        # pick one test item
        t_idx = rng.integers(len(items))
        for j in range(len(items)):
            if j == t_idx:
                te_u.append(u)
                te_i.append(items[j])
                te_v.append(scores[j])
            else:
                tr_u.append(u)
                tr_i.append(items[j])
                tr_v.append(scores[j])

    train = csr_matrix((tr_v, (tr_u, tr_i)), shape=(n_users, n_items))
    test = csr_matrix((te_v, (te_u, te_i)), shape=(n_users, n_items))
    return train, test

# NDCG helper
def _ndcg_at_k(rec_list, relevant, k):
    if not relevant:
        return 0.0
    dcg = 0.0
    for rank, item in enumerate(rec_list[:k], start=1):
        if item in relevant:
            dcg += 1.0 / np.log2(rank + 1)
    ideal = min(len(relevant), k)
    idcg = sum(1.0 / np.log2(i + 1) for i in range(1, ideal + 1))
    return dcg / idcg if idcg > 0 else 0.0

# Interaction matrix helper
def make_interaction_matrix(df):
    """Build user–item CSR matrix."""
    users = df["user_id"].values
    items = df["item_id"].values
    ratings = df["rating"].astype(float).values

    mat = sp.coo_matrix(
        (ratings, (users, items)),
        shape=(df["user_id"].max() + 1, df["item_id"].max() + 1)
    ).tocsr()
    return mat


# Manual evaluation for implicit models
def evaluate(model, train_m, test_m, k=10):
    hr_list = []
    ndcg_list = []
    map_list = []
    recall_list = []

    for u in range(train_m.shape[0]):

        # implicit sometimes compresses users
        if u >= model.user_factors.shape[0]:
            continue

        if train_m[u].nnz == 0:
            continue

        test_items = test_m[u].indices
        if len(test_items) == 0:
            continue

        # get top-k
        recs = model.recommend(
            u,
            user_items=train_m[u],
            N=k,
            filter_already_liked_items=False,
        )
        rec_items = [r[0] for r in recs]

        rec_set = set(rec_items)
        rel_set = set(test_items)

        # HR
        hr_list.append(1.0 if rec_set & rel_set else 0.0)

        # Recall/MAP
        hits = 0
        ap_sum = 0.0
        for rank, item in enumerate(rec_items, start=1):
            if item in rel_set:
                hits += 1
                ap_sum += hits / rank

        recall_list.append(hits / len(rel_set))
        map_list.append(ap_sum / len(rel_set))
        ndcg_list.append(_ndcg_at_k(rec_items, rel_set, k))

    if not hr_list:
        return 0, 0, 0, 0

    return (
        float(np.mean(hr_list)),
        float(np.mean(ndcg_list)),
        float(np.mean(map_list)),
        float(np.mean(recall_list)),
    )


# Run ALS + ALS-BM25 + BPR on one dataset
def run_one(name, cleaned_file, k=10):
    print(f"=== Processing {name} ===")

    triples = load_cleaned_triples(cleaned_file)
    df_triples = pd.DataFrame(triples)  # convert list-of-dicts -> DataFrame
    ui = make_interaction_matrix(df_triples)

    train, test = train_test_split_userwise(ui, random_state=42)


    train_u = train.tocsr()
    test_u = test.tocsr()

    # BM25 confidence
    bm = bm25_weight(train_u).tocsr().astype(np.float32)
    bm.data = 1.0 + 10.0 * bm.data

    item_user_plain = train_u.T.tocsr()
    item_user_conf = bm.T.tocsr()

    results = []

    # ---- ALS baseline ----
    als = AlternatingLeastSquares(
        factors=50, iterations=15, regularization=0.1, random_state=42
    )
    als.fit(item_user_plain)
    results.append(["ALS", *evaluate(als, train_u, test_u, k)])

    # ---- ALS + BM25 ----
    als_bm = AlternatingLeastSquares(
        factors=50, iterations=15, regularization=0.1, random_state=42
    )
    als_bm.fit(item_user_conf)
    results.append(["ALS+BM25+CONF", *evaluate(als_bm, train_u, test_u, k)])

    # ---- BPR ----
    bpr = BayesianPersonalizedRanking(
        factors=50, iterations=30, regularization=0.01, random_state=42
    )
    bpr.fit(item_user_plain)
    results.append(["BPR", *evaluate(bpr, train_u, test_u, k)])

    df = pd.DataFrame(
        results,
        columns=["model", f"HR@{k}", f"NDCG@{k}", f"MAP@{k}", f"Recall@{k}"],
    )
    out = os.path.join(THIS_DIR, f"{name}_results.csv")
    df.to_csv(out, index=False)

    print(f"Done: {name} (saved to {out})")
    return df


# Run all three datasets
if __name__ == "__main__":
    outputs = []
    outputs.append(run_one("beeradvocate", "beeradvocate_cleaned.json"))
    outputs.append(run_one("goodreads", "goodreads_cleaned.json"))
    outputs.append(run_one("steam", "steam_cleaned.json"))

    final = pd.concat(outputs, ignore_index=True)
    final.to_csv(os.path.join(THIS_DIR, "all_results_summary.csv"), index=False)
    print("\nAll datasets finished.")
