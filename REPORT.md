INTRODUCTION 

This report provides an analysis of three different recommendation system algorithms applied to three diverse real-world review datasets. The goal of this project is to evaluate and compare the performance of three recommendation approaches: a Two-Tower model (user-item embedding model), a Pairwise BPR (Bayesian Personalized Ranking) model, and a Context Distance Regularization enhancement. The evaluation employs consistent metrics across datasets: Hit Rate at k (HR@k), Normalized Discounted Cumulative Gain at k (NDCG@k), Mean Average Precision at k (MAP@k), and Recall at k (Recall@k). The datasets present different characteristics: Goodreads Spoilers contains book reviews with thematic clustering, BeerAdvocate contains beer reviews with style-based similarities, and Steam Reviews contains game reviews with diverse preferences and high sparsity. These differences allow a comprehensive evaluation of algorithm performance under varying data conditions including sparsity levels, long-tail behavior, user consistency, and multi-domain preferences. 

DATA PREPROCESSING 

Dataset Loading and Sampling 

The three datasets were loaded from compressed JSON files stored in the sourceData directory. Due to the large size of the original datasets (BeerAdvocate: 1,586,615 records, Goodreads: 1,378,033 records, Steam: 25,799 records), we implemented a consistent sampling strategy to ensure fair comparison while maintaining computational efficiency. A random sample of 50,000 records was extracted from each dataset using a fixed random seed (seed=42) for reproducibility. For the Steam dataset, which contained fewer than 50,000 records, all available records were used. This sampling approach ensures that all three datasets are processed with the same methodology while preserving the original data distributions.

Dataset-Specific Extraction 

Each dataset required custom extraction logic to convert raw records into standardized user-item-rating triples:

BeerAdvocate Extraction: The extraction process identified user IDs from the 'review/profileName' field, item IDs from the 'beer/beerId' field, and ratings from the 'review/overall' field. Ratings were validated to ensure they fell within the valid range of 0-5, with half-point increments allowed (e.g., 1.0, 1.5, 2.0, etc.).

Goodreads Extraction: User IDs were extracted from the 'user_id' field, item IDs from the 'book_id' field, and ratings from the 'rating' field. Ratings were validated to ensure they fell within the 1-5 integer range.

Steam Extraction: The Steam dataset required special handling as it contains nested review structures. User IDs were extracted from the 'user_id' field at the top level, while each user's reviews were stored in a 'reviews' array. For each review, item IDs were extracted from 'item_id' and the boolean 'recommend' field was converted to a numeric rating: True → 5.0, False → 1.0. This binary-to-numeric conversion allows Steam reviews to be evaluated using the same rating-based metrics as the other datasets.

Data Cleaning Pipeline 

The extracted triples underwent a comprehensive cleaning process to ensure data quality and consistency:

Duplicate Removal: Duplicate user-item pairs were identified and removed, keeping only the first occurrence. This step eliminated cases where the same user reviewed the same item multiple times.

Minimum Review Filtering: To ensure sufficient data for reliable recommendation learning, users with fewer than 3 reviews and items with fewer than 3 reviews were filtered out. This threshold was chosen based on the README requirements and ensures that each user and item has enough interaction history for meaningful embedding learning.

ID Reindexing: All user and item IDs were reindexed to consecutive integers starting from 0. This transformation is essential for efficient embedding layer implementation in neural network architectures, as it allows direct indexing into embedding matrices without requiring lookup dictionaries during training.

Final Cleaned Datasets 

After the complete preprocessing pipeline, the following cleaned datasets were generated and saved to the sampleSets directory:

BeerAdvocate Cleaned: 33,172 interactions from 3,798 users and 3,963 items. Average rating: 3.84 (range: 1.0-5.0 with half-point increments). Average reviews per user: 8.73. Average reviews per item: 8.37. Sparsity: 99.78%.

Goodreads Cleaned: 27,674 interactions from 5,933 users and 5,253 items. Average rating: 3.84 (range: 1.0-5.0 integer ratings). Average reviews per user: 4.66. Average reviews per item: 5.27. Sparsity: 99.91%.

Steam Cleaned: 33,802 interactions from 7,292 users and 1,637 items. Average rating: 4.52 (range: 1.0-5.0 binary ratings). Average reviews per user: 4.64. Average reviews per item: 20.65. Sparsity: 99.72%.

All cleaned datasets maintain consistent structure with three fields per record: user_id (integer, 0-indexed), item_id (integer, 0-indexed), and rating (float, 1.0-5.0 scale). The high sparsity values (all above 99%) are expected for recommendation systems and reflect the inherent challenge of predicting user preferences in sparse interaction matrices.

ALGORITHM DESCRIPTION 

Two-Tower Model (User-Item Embedding Model) 

The Two-Tower architecture implements a dual-embedding approach where separate neural network towers learn user and item representations. In this model, we learn:

A user embedding tower: u_i ∈ ℝ^d

An item embedding tower: v_j ∈ ℝ^d

The relevance score between user i and item j is computed via a dot product:

s(i,j) = u_i^T · v_j

Key Features:

The Two-Tower model scales well to large datasets due to its efficient dot product computation and parallelizable architecture. It is easy to train and evaluate, making it a popular baseline for recommendation systems. The embeddings are global and not conditioned on specific user-item interaction context, which allows for efficient serving in production environments.

Drawbacks:

The model is pair-agnostic, meaning the same item vector is used for all users, limiting personalization. It is prone to popularity bias, as popular items tend to receive higher scores across all users. The model struggles with sparse users who have few interactions, as there is insufficient signal to learn meaningful user embeddings.

Pairwise BPR (Bayesian Personalized Ranking) 

The Pairwise BPR model optimizes the relative order of positive versus negative items rather than predicting absolute ratings. The model uses a pairwise ranking loss that encourages positive items to be ranked higher than negative items for each user.

The BPR loss function is defined as:

L_BPR = -log σ(s(i,j^+) - s(i,j^-))

where σ is the sigmoid function, j^+ is a positive item (one the user has interacted with), and j^- is a negative item (one the user has not interacted with, sampled uniformly or using popularity weighting).

This formulation encourages:

s(i,j^+) > s(i,j^-)

Key Features:

The BPR model optimizes ranking rather than rating prediction, which aligns better with recommendation system objectives where relative ordering matters more than absolute scores. It handles implicit feedback (clicks, purchases, reviews) effectively by treating all observed interactions as positive signals. The model learns finer-grained user preference signals by explicitly modeling the relative preference between item pairs.

Drawbacks:

The model's performance is sensitive to the quality of negative sampling. Uniform negative sampling may underperform on long-tail items, while popularity-weighted sampling can improve stability. The model still uses the same global item embeddings, providing limited contextualization. It can overfit if users have small interaction histories.

Context Distance Regularization (Enhanced Approach) 

To incorporate lightweight contextual awareness without the computational overhead of a full graph neural network, we introduce a Context Distance Regularization term that encourages positive-item embeddings to remain close to the user's historical preference context.

The user context vector is computed as the average of all item embeddings in the user's interaction history:

μ_i = (1/|H_i|) · Σ(k∈H_i) v_k

where H_i represents the set of items that user i has previously interacted with, and |H_i| is the number of such interactions.

The context distance is defined as:

d(i,j) = 1 - cosine(v_j, μ_i)

where cosine(v_j, μ_i) is the cosine similarity between the item embedding v_j and the user context vector μ_i.

The final loss function combines the BPR loss with the context distance regularization:

L = L_BPR + γ · d(i,j)

where γ is a hyperparameter controlling the strength of the regularization term.

Key Benefits:

The Context Distance Regularization provides a "mini" contextual signal similar to ContextGNN principles but without requiring graph processing, keeping the approach lightweight. It helps with sparse user histories by leveraging the user's past interaction patterns. The regularization is particularly effective in domains with strong thematic clusters (books, beers, games) where items within a cluster should have similar embeddings.

Drawbacks:

The context vector quality depends on the size and consistency of the user's past interactions. Users with very few or highly diverse interactions may not benefit as much from this approach. The method is not as expressive as a full graph neural network, which can model more complex relationships between users and items.

DATASET DESCRIPTION 

Goodreads Spoilers Dataset 

The Goodreads dataset contains book reviews with spoiler tags, representing a highly textual domain where user tastes cluster strongly around genres and authors. The dataset includes user IDs, book IDs, and ratings on a 1-5 integer scale. The spoiler indicator is available but ignored for modeling purposes as it has no direct numeric meaning for recommendation learning.

Why it's relevant:

The dataset exhibits strong thematic preference clustering, making it an excellent test case for context distance regularization. Users typically develop stable, niche reading preferences over time, creating coherent user profiles. The dataset has a high long-tail distribution, with many books receiving very few interactions, testing model robustness to item sparsity.

Challenges:

Users may review sparsely, creating cold-start problems for new users. The item popularity distribution is extremely skewed, with a small number of popular books receiving most interactions while many books remain largely unrated.

BeerAdvocate Reviews Dataset 

The BeerAdvocate dataset contains beer review scores from enthusiasts, representing a domain with dense per-user histories and strong style-based similarities (IPA, stout, lager, etc.). The dataset includes user IDs, beer/product IDs, and overall review ratings on a 0-5 scale with half-point increments.

Why it's relevant:

The dataset features dense per-user histories, as beer enthusiasts tend to leave many reviews, providing rich interaction signals. There are strong style-based similarities between beers, making this an ideal testing ground for context distance regularization. The dataset serves as a natural testing ground for preference clustering, as users often develop coherent but narrow taste profiles.

Challenges:

Users tend to have coherent but narrow tastes, which can limit the diversity of recommendations. The item graph is highly clustered, meaning that global two-tower embeddings may struggle to capture local similarity patterns. Some beers have very few ratings, creating long-tail challenges.

Steam Reviews Dataset 

The Steam dataset contains game reviews with binary recommendations (positive/negative), representing a larger ecosystem with very diverse preferences. The dataset includes user IDs, game IDs, and binary review scores that are converted to numeric ratings (positive → 5.0, negative → 1.0).

Why it's relevant:

The dataset represents a larger ecosystem with very diverse preferences, testing model generalization across varied user behaviors. Items (games) vary widely in genre, price, and complexity, creating a challenging recommendation environment. The binary nature of reviews makes this an excellent domain for pairwise ranking approaches, as the model only needs to distinguish between positive and negative feedback.

Challenges:

High item popularity skew, with AAA games dominating the interaction distribution. Many casual users create extremely sparse user vectors, making it difficult to learn meaningful embeddings. Review polarity varies widely by game genre, requiring models to capture domain-specific patterns.

Dataset Comparison Summary 

| Dataset      | Domain | Density     | Long-Tail | User Consistency | Why It's Useful                                                      |
| ------------ | ------ | ----------- | --------- | ---------------- | -------------------------------------------------------------------- |
| Goodreads    | Books  | Medium      | **High**  | High             | Tests performance on thematic preference clustering                  |
| BeerAdvocate | Beers  | Medium–High | Medium    | Very High        | Shows how models behave when users have tight, stylistic preferences |
| Steam        | Games  | Low–Medium  | **High**  | Low–Medium       | Tests robustness to sparsity and implicit-feedback dynamics          |

This combination provides a broad stress test for recommendation algorithms, covering themed preference-clustered domains (Books), taste-consistent domains (Beers), and sparse wide-domain datasets (Games).

EVALUATION METRICS 

To ensure fair and comprehensive comparison across all three recommendation models and datasets, we employ four standard ranking metrics:

Hit Rate at k (HR@k): Measures the fraction of users for whom at least one relevant item appears in the top-k recommendations. HR@k = (Number of users with at least one hit) / (Total number of users)

Normalized Discounted Cumulative Gain at k (NDCG@k): Measures the quality of ranking by considering both the relevance of items and their positions in the recommendation list. NDCG@k accounts for the fact that items ranked higher should contribute more to the score, with a discount factor applied to lower-ranked items.

Mean Average Precision at k (MAP@k): Computes the average precision across all users, where precision at each position is calculated and averaged. MAP@k provides a single-figure measure of quality across recall levels.

Recall at k (Recall@k): Measures the fraction of relevant items that are successfully retrieved in the top-k recommendations. Recall@k = (Number of relevant items in top-k) / (Total number of relevant items)

These metrics are computed consistently across all three datasets and all three models to enable direct comparison. The choice of k values (typically k=5, 10, 20) allows evaluation at different recommendation list lengths, reflecting different use cases from focused top recommendations to broader discovery lists.

RESULTS & EVALUATION 

[To be completed after model implementation and training]

PERFORMANCE COMPARISON 

[To be completed after model implementation and training]

CONCLUSIONS 

[To be completed after model implementation and training]

ACKNOWLEDGMENTS 

[To be completed]

REFERENCES 

[To be completed]

