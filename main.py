import argparse
import yaml
import os
from clustering.preprocessing import run_preprocessing
from clustering.feature_engineering import run_feature_engineering
from clustering.clustering_core import run_clustering_pipeline
from clustering.utils import (
    filter_inappropriate_content,
    get_cluster_sizes,
    get_removed_words
)

def load_config(path):
    """
    Load YAML configuration file.
    """
    with open(path, "r") as f:
        return yaml.safe_load(f)

def main(config_path):
    """
    Main entry point: run the full clustering pipeline using the config file.
    """
    config = load_config(config_path)

    input_file = config["input_file"]
    output_dir = config["output_dir"]
    temp_dir = config.get("temp_dir", os.path.join(output_dir, "temp"))
    summary_path = os.path.join(output_dir, "summary.json")

    # Step 1: Preprocessing
    print("Step 1: Preprocessing input file...")
    df = run_preprocessing(input_file)

    # Optional: save preprocessed temp file
    # os.makedirs(temp_dir, exist_ok=True)
    # df.to_csv(os.path.join(temp_dir, "preprocessed.csv"), index=False)

    # Step 2: Feature Engineering
    print("Step 2: Extracting features (word frequency, spelling)...")
    df = run_feature_engineering(df)

    # Optional: save feature-enhanced temp file
    # df.to_csv(os.path.join(temp_dir, "features.csv"), index=False)

    # Step 3: Clustering
    print("Step 3: Running clustering...")
    df_clustered, score = run_clustering_pipeline(
        df,
        summary_path=summary_path
    )

    # Get cluster sizes before filtering
    cluster_sizes_before = get_cluster_sizes(df_clustered)
    print("\nCluster sizes before content filtering:")
    for cluster, size in sorted(cluster_sizes_before.items()):
        print(f"Cluster {cluster}: {size} words")

    # Step 4: Content Filtering
    print("\nStep 4: Filtering inappropriate content...")
    content_filtering = config.get("content_filtering", {})
    
    # Get profanity settings
    profanity_config = content_filtering.get("profanity", {})
    profanity_enabled = profanity_config.get("enabled", True)
    custom_words = profanity_config.get("custom_words", []) if profanity_enabled else None
    
    # Get custom list settings
    custom_list_config = content_filtering.get("custom_list", {})
    custom_list_enabled = custom_list_config.get("enabled", False)
    custom_list_path = custom_list_config.get("path") if custom_list_enabled else None
    
    # Apply filtering
    df_filtered, rows_removed = filter_inappropriate_content(
        df_clustered,
        custom_words=custom_words,
        custom_list_path=custom_list_path
    )
    print(f"Removed {rows_removed} rows containing inappropriate content")

    # Get cluster sizes after filtering
    cluster_sizes_after = get_cluster_sizes(df_filtered)
    print("\nCluster sizes after content filtering:")
    for cluster, size in sorted(cluster_sizes_after.items()):
        print(f"Cluster {cluster}: {size} words")
        if cluster in cluster_sizes_before:
            diff = size - cluster_sizes_before[cluster]
            if diff != 0:
                print(f"  Change: {diff:+d} words")

    # Export removed words if any were removed
    if rows_removed > 0:
        print("\nExporting removed words...")
        removed_words_df = get_removed_words(df_clustered, df_filtered)
        removed_words_path = os.path.join(output_dir, "removed_words.csv")
        removed_words_df.to_csv(removed_words_path, index=False)
        print(f"Removed words saved to: {removed_words_path}")

    # Step 5: Export filtered results
    print("\nStep 5: Exporting filtered results...")
    clusters_dir = os.path.join(output_dir, "clusters_by_label")
    os.makedirs(clusters_dir, exist_ok=True)
    
    # Export filtered results by cluster
    for cluster in sorted(df_filtered["KMeans_Label_Ranked"].unique()):
        cluster_df = df_filtered[df_filtered["KMeans_Label_Ranked"] == cluster]
        output_path = os.path.join(clusters_dir, f"cluster{cluster}_words.csv")
        cluster_df[["Word"]].to_csv(output_path, index=False)
    
    print(f"Filtered clustered words saved to: {clusters_dir}")
    
    # Export final filtered results
    output_path = os.path.join(output_dir, "final_cleaned_clusters.csv")
    df_filtered.to_csv(output_path, index=False)
    print(f"Final filtered results saved to: {output_path}")
    
    print("\nPipeline completed successfully.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run Word Difficulty Clustering Pipeline"
    )
    parser.add_argument(
        "--config", 
        required=True, 
        help="Path to YAML config file"
    )
    args = parser.parse_args()
    main(args.config)
