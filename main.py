import argparse
import yaml
import os
from clustering.preprocessing import run_preprocessing
from clustering.feature_engineering import run_feature_engineering
from clustering.clustering_core import run_clustering_pipeline

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
    df = run_preprocessing(input_file)

    # Optional: save preprocessed temp file
    os.makedirs(temp_dir, exist_ok=True)
    df.to_csv(os.path.join(temp_dir, "preprocessed.csv"), index=False)

    # Step 2: Feature Engineering
    df = run_feature_engineering(df)

    # Optional: save feature-enhanced temp file
    df.to_csv(os.path.join(temp_dir, "features.csv"), index=False)

    # Step 3: Clustering and Export
    df_cleaned, score = run_clustering_pipeline(
        df,
        output_dir=os.path.join(output_dir, "clusters_by_label"),
        summary_path=summary_path
    )

    # Final export
    df_cleaned.to_csv(os.path.join(output_dir, "final_cleaned_clusters.csv"), index=False)
    print("Pipeline completed successfully.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Word Difficulty Clustering Pipeline")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    args = parser.parse_args()
    main(args.config)
