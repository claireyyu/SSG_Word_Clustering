import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from clustering.utils import export_word_list, filter_english_only
import os
import json
from sklearn.preprocessing import StandardScaler
import itertools

def run_kmeans_grid_search(df, min_score=0.5):
    """
    Perform grid search over combinations of submission, real-world, and spelling weights.
    Returns the best weights and corresponding silhouette score.
    """
    weight_range = np.arange(0.1, 1.1, 0.1)
    combinations = list(itertools.product(weight_range, repeat=3))
    best_score = -1
    best_weights = (None, None, None)

    for w_sub, w_real, w_spell in combinations:
        if w_real <= w_sub or w_real <= w_spell:
            continue  # Enforce real-world frequency has the highest weight

        # Create a new DataFrame copy for this iteration
        df_iter = df.copy()
        df_iter.loc[:, "W_Sub"] = df_iter["Standardized_Frequency"] * w_sub
        df_iter.loc[:, "W_Real"] = df_iter["Standardized_RealWorld_Log"] * w_real
        df_iter.loc[:, "W_Spell"] = df_iter["Standardized_Spelling_Easiness"] * w_spell

        X = df_iter[["W_Sub", "W_Real", "W_Spell"]]
        kmeans = KMeans(n_clusters=3, random_state=42)
        labels = kmeans.fit_predict(X)
        score = silhouette_score(X, labels)

        if score > min_score and score > best_score:
            best_score = score
            best_weights = (w_sub, w_real, w_spell)

    return best_weights, best_score

def run_final_kmeans(df, weights):
    """
    Run final KMeans clustering using the given weights and add cluster labels.
    """
    w_sub, w_real, w_spell = weights
    df.loc[:, "W_Sub"] = df["Standardized_Frequency"] * w_sub
    df.loc[:, "W_Real"] = df["Standardized_RealWorld_Log"] * w_real
    df.loc[:, "W_Spell"] = df["Standardized_Spelling_Easiness"] * w_spell

    X = df[["W_Sub", "W_Real", "W_Spell"]]
    kmeans = KMeans(n_clusters=3, random_state=42)
    df["KMeans_Cluster"] = kmeans.fit_predict(X)

    # Rank clusters by average real-world frequency
    cluster_order = df.groupby("KMeans_Cluster")["Real_World_Frequency_Log"].mean().sort_values(ascending=False).index
    cluster_label_map = {old: new for new, old in enumerate(cluster_order)}
    df["KMeans_Label_Ranked"] = df["KMeans_Cluster"].map(cluster_label_map)

    return df

def summarize_and_export(df, output_dir, summary_path):
    """
    Clean non-English words, export cleaned word lists and summary report.
    """
    os.makedirs(output_dir, exist_ok=True)
    label_column = "KMeans_Label_Ranked"
    word_column = "Word"

    # Clean English only
    df_clean = filter_english_only(df, word_column=word_column)

    # Export cleaned clusters
    export_word_list(df_clean, label_column, word_column, output_dir)

    # Export summary
    summary = {
        "num_clusters": 3,
        "cluster_counts": df_clean[label_column].value_counts().to_dict()
    }

    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    return df_clean

def save_clustering_summary(df, weights, score, summary_path):
    """
    Save clustering summary to JSON file.
    
    Parameters:
    df (pd.DataFrame): DataFrame with cluster labels
    weights (tuple): Best weights found (w_sub, w_real, w_spell)
    score (float): Best silhouette score
    summary_path (str): Path to save summary JSON
    """
    w_sub, w_real, w_spell = weights
    summary = {
        "num_clusters": 3,
        "cluster_counts": df["KMeans_Label_Ranked"].value_counts().to_dict(),
        "weights": {
            "submission": float(w_sub),
            "real_world": float(w_real),
            "spelling": float(w_spell)
        },
        "silhouette_score": float(score)
    }

    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

def run_clustering_pipeline(df, output_dir=None, summary_path=None):
    """
    Run the complete clustering pipeline:
    1. Grid search for optimal weights
    2. Apply K-means clustering
    3. Export results if output_dir is provided
    
    Parameters:
    df (pd.DataFrame): Input DataFrame with features
    output_dir (str, optional): Directory to save results
    summary_path (str, optional): Path to save clustering summary
    
    Returns:
    pd.DataFrame: DataFrame with cluster labels
    float: Best silhouette score
    """
    # Grid search for optimal weights
    best_weights, best_score = run_kmeans_grid_search(df)
    w_sub, w_real, w_spell = best_weights
    print(f"Best weights: Submission={w_sub:.1f}, RealWorld={w_real:.1f}, Spelling={w_spell:.1f}")
    print(f"Silhouette Score: {best_score:.4f}")

    # Apply final clustering with best weights
    df_clustered = run_final_kmeans(df, best_weights)
    
    # Save summary if path provided
    if summary_path:
        save_clustering_summary(df_clustered, best_weights, best_score, summary_path)
    
    return df_clustered, best_score

