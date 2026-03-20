from typing import Iterable
from scipy.sparse import csr_matrix, vstack
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import mean_absolute_error, mean_squared_error


def preprocess_text_column(df: pd.DataFrame, column_name: str, copy_df: bool = True) -> pd.DataFrame:
    """Normalize a text column.

    By default this works on a copy so notebook cells do not accidentally mutate
    shared dataframes.
    """
    # input validation
    if column_name not in df.columns:
        raise KeyError(f"Column '{column_name}' not found in dataframe")

    out_df = df.copy(deep=True) if copy_df else df
    out_df[column_name] = (
        out_df[column_name]
        .fillna("")
        .astype(str)
        .str.lower()
        .str.replace(r"[^\w\s]", "", regex=True)
        .str.replace("\n", " ", regex=False)
        .str.replace("\r", " ", regex=False)
    )
    return out_df


def tfidf_vectorizing(df: pd.DataFrame, column_name: str, stop_words: str | None = "english", **vectorizer_kwargs):
    """Fit TF-IDF on a dataframe text column without mutating input data."""
    if column_name not in df.columns:
        raise KeyError(f"Column '{column_name}' not found in dataframe")

    text_series = df[column_name].fillna("").astype(str)
    tfidf_vectorizer = TfidfVectorizer(stop_words=stop_words, **vectorizer_kwargs)
    tfidf_matrix = tfidf_vectorizer.fit_transform(text_series)
    return tfidf_vectorizer, tfidf_matrix

# can we change this to weighted mean?
def build_user_profiles(train_interactions: pd.DataFrame, tfidf_matrix, item_ids: Iterable, user_col: str = "user_id", item_col: str = "item_id", rating_col: str = "rating"):
    """Build user profiles by rating-weighted averaging of item TF-IDF vectors."""
    if user_col not in train_interactions.columns or item_col not in train_interactions.columns or rating_col not in train_interactions.columns:
        raise KeyError(f"Expected columns '{user_col}', '{item_col}', and '{rating_col}' in train_interactions")

    item_ids_list = [str(x) for x in item_ids]
    if tfidf_matrix.shape[0] != len(item_ids_list):
        raise ValueError("Length of item_ids must match number of rows in tfidf_matrix")

    item_to_idx = {item_id: idx for idx, item_id in enumerate(item_ids_list)}

    interactions = train_interactions[[user_col, item_col, rating_col]].copy()
    interactions[item_col] = interactions[item_col].astype(str)
    interactions[rating_col] = pd.to_numeric(interactions[rating_col], errors="coerce").fillna(0.0)
    interactions = interactions[interactions[item_col].isin(item_to_idx)].copy()

    users = interactions[user_col].drop_duplicates().tolist()
    if not users:
        return csr_matrix((0, tfidf_matrix.shape[1]))

    user_vectors = []
    for user_id in users:
        user_rows = interactions.loc[interactions[user_col] == user_id, [item_col, rating_col]]
        user_item_ids = user_rows[item_col].tolist()
        item_indices = [item_to_idx[item_id] for item_id in user_item_ids]
        weights = np.maximum(user_rows[rating_col].to_numpy(dtype=float), 0.0)

        if not item_indices:
            user_vec = csr_matrix((1, tfidf_matrix.shape[1]))
        else:
            user_item_matrix = tfidf_matrix[item_indices]
            weight_sum = weights.sum()
            if weight_sum <= 0:
                user_vec = csr_matrix(user_item_matrix.mean(axis=0))
            else:
                weighted_sum = user_item_matrix.multiply(weights.reshape(-1, 1)).sum(axis=0)
                user_vec = csr_matrix(weighted_sum / weight_sum)
        user_vectors.append(user_vec)

    return vstack(user_vectors).tocsr()

def calculate_rmse(test_df: pd.DataFrame, predictions_df: pd.DataFrame, prediction_col: str, user_col: str = "user_id", item_col: str = "item_id", rating_col: str = "rating") -> float:
    merged_df = pd.merge(
        test_df[[user_col, item_col, rating_col]],
        predictions_df[[user_col, item_col, prediction_col]],
        on=[user_col, item_col],
        how="inner",
    )
    return mean_squared_error(merged_df[rating_col], merged_df[prediction_col], squared=False)


def calculate_mae(test_df: pd.DataFrame, predictions_df: pd.DataFrame, prediction_col: str, user_col: str = "user_id", item_col: str = "item_id", rating_col: str = "rating") -> float:
    merged_df = pd.merge(
        test_df[[user_col, item_col, rating_col]],
        predictions_df[[user_col, item_col, prediction_col]],
        on=[user_col, item_col],
        how="inner",
    )
    return mean_absolute_error(merged_df[rating_col], merged_df[prediction_col])


def average_precision_at_k(ranked_items: Iterable, relevant_items: set, k: int = 10) -> float:
    ranked_items = list(ranked_items)[:k]
    total_relevant = len(relevant_items)
    if total_relevant == 0:
        return 0.0

    hits = 0
    precision_sum = 0.0
    for rank, item_id in enumerate(ranked_items, start=1):
        if item_id in relevant_items:
            hits += 1
            precision_sum += hits / rank

    return precision_sum / total_relevant


def _relevant_items_by_user(test_df: pd.DataFrame, relevance_threshold: float, user_col: str, item_col: str, rating_col: str) -> dict:
    return (
        test_df[test_df[rating_col] >= relevance_threshold]
        .groupby(user_col)[item_col]
        .apply(set)
        .to_dict()
    )


def _ranked_recommendations_by_user(recommendations_df: pd.DataFrame, prediction_col: str, k: int, user_col: str, item_col: str) -> dict:
    top_k = (
        recommendations_df.sort_values([user_col, prediction_col], ascending=[True, False])
        .groupby(user_col, as_index=False, group_keys=False)
        .head(k)
    )
    return top_k.groupby(user_col)[item_col].apply(list).to_dict()


def calculate_hit_rate(test_df: pd.DataFrame, recommendations_df: pd.DataFrame, prediction_col: str, k: int = 10, relevance_threshold: float = 3, user_col: str = "user_id", item_col: str = "item_id", rating_col: str = "rating") -> float:
    users = test_df[user_col].unique()
    relevant_by_user = _relevant_items_by_user(test_df, relevance_threshold, user_col, item_col, rating_col)
    ranked_by_user = _ranked_recommendations_by_user(recommendations_df, prediction_col, k, user_col, item_col)

    hits = 0
    for user_id in users:
        if set(ranked_by_user.get(user_id, [])).intersection(relevant_by_user.get(user_id, set())):
            hits += 1
    return hits / len(users) if len(users) > 0 else 0.0


def calculate_precision_at_k(test_df: pd.DataFrame, recommendations_df: pd.DataFrame, prediction_col: str, k: int = 10, relevance_threshold: float = 3, user_col: str = "user_id", item_col: str = "item_id", rating_col: str = "rating") -> float:
    users = test_df[user_col].unique()
    relevant_by_user = _relevant_items_by_user(test_df, relevance_threshold, user_col, item_col, rating_col)
    ranked_by_user = _ranked_recommendations_by_user(recommendations_df, prediction_col, k, user_col, item_col)

    precision_scores = []
    for user_id in users:
        ranked_items = ranked_by_user.get(user_id, [])
        if not ranked_items:
            precision_scores.append(0.0)
            continue
        relevant_items = relevant_by_user.get(user_id, set())
        n_relevant = sum(item in relevant_items for item in ranked_items)
        precision_scores.append(n_relevant / k if k > 0 else 0.0)

    return sum(precision_scores) / len(precision_scores) if precision_scores else 0.0


def calculate_map_at_k(test_df: pd.DataFrame, recommendations_df: pd.DataFrame, prediction_col: str, k: int = 10, relevance_threshold: float = 3, user_col: str = "user_id", item_col: str = "item_id", rating_col: str = "rating"):
    users = test_df[user_col].unique()
    relevant_by_user = _relevant_items_by_user(test_df, relevance_threshold, user_col, item_col, rating_col)
    ranked_by_user = _ranked_recommendations_by_user(recommendations_df, prediction_col, k, user_col, item_col)

    ap_rows = []
    for user_id in users:
        ap = average_precision_at_k(
            ranked_by_user.get(user_id, []),
            relevant_by_user.get(user_id, set()),
            k=k,
        )
        ap_rows.append({user_col: user_id, "ap": ap})

    ap_df = pd.DataFrame(ap_rows)
    return ap_df, (ap_df["ap"].mean() if not ap_df.empty else 0.0)


def calculate_mrr_at_k(test_df: pd.DataFrame, recommendations_df: pd.DataFrame, prediction_col: str, k: int = 10, relevance_threshold: float = 3, user_col: str = "user_id", item_col: str = "item_id", rating_col: str = "rating") -> float:
    users = test_df[user_col].unique()
    relevant_by_user = _relevant_items_by_user(test_df, relevance_threshold, user_col, item_col, rating_col)
    ranked_by_user = _ranked_recommendations_by_user(recommendations_df, prediction_col, k, user_col, item_col)

    reciprocal_ranks = []
    for user_id in users:
        relevant_items = relevant_by_user.get(user_id, set())
        rr = 0.0
        for rank, item_id in enumerate(ranked_by_user.get(user_id, []), start=1):
            if item_id in relevant_items:
                rr = 1.0 / rank
                break
        reciprocal_ranks.append(rr)

    return sum(reciprocal_ranks) / len(reciprocal_ranks) if reciprocal_ranks else 0.0


def calculate_coverage_at_k(test_df: pd.DataFrame, recommendations_df: pd.DataFrame, prediction_col: str, k: int = 10, catalog_items: Iterable | None = None, user_col: str = "user_id", item_col: str = "item_id") -> float:
    users = test_df[user_col].unique()
    ranked_by_user = _ranked_recommendations_by_user(recommendations_df, prediction_col, k, user_col, item_col)

    recommended_items = set()
    for user_id in users:
        recommended_items.update(ranked_by_user.get(user_id, []))

    if catalog_items is None:
        catalog_size = test_df[item_col].nunique()
    else:
        catalog_size = len(set(catalog_items))

    return len(recommended_items) / catalog_size if catalog_size > 0 else 0.0


if __name__ == "__main__":
    # Lightweight smoke test for safer preprocessing behavior.
    test_data = {"text": ["Hello, World!", "This is a test.\nNew line.", "Another test.\rCarriage return."]}
    df_test = pd.DataFrame(test_data)

    print("Before preprocessing:")
    print(df_test)

    df_processed = preprocess_text_column(df_test, "text", copy_df=True)
    print("\nAfter preprocessing:")
    print(df_processed)
