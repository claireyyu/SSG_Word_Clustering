# Word Difficulty Clustering Pipeline

A Python pipeline for clustering words based on their difficulty levels using K-means clustering. The pipeline processes word data, extracts features, and clusters words into three difficulty levels.

## Features

- Preprocessing of word data
- Feature engineering (frequency, spelling complexity)
- K-means clustering with grid search for optimal weights
- Content filtering to remove inappropriate words
- Export of results by cluster and summary statistics

## Requirements

- Python 3.8+
- pandas
- numpy
- scikit-learn
- better_profanity (for content filtering)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/claireyyu/SSG_Word_Clustering.git
cd SSG_Word_Clustering
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Prepare your input CSV file with the following columns:
   - Word
   - Frequency
   - Real_World_Frequency
   - Spelling_Easiness

2. Configure the pipeline in `config.yaml`:
```yaml
input_file: data/your_input.csv
output_dir: output/
temp_dir: output/temp/
content_filtering:
  # Option 1: Use profanity library
  profanity:
    enabled: false
    custom_words: []  # Add custom words to filter here
  # Option 2: Use custom word list
  custom_list:
    enabled: true
    path: data/custom_filter_words.csv  # Path to CSV file with custom word list
```

3. Run the pipeline:
```bash
python main.py --config config.yaml
```

## Output

The pipeline generates the following outputs:

1. `output/clusters_by_label/`: Directory containing CSV files for each cluster
   - `cluster0_words.csv`: Easy words
   - `cluster1_words.csv`: Medium difficulty words
   - `cluster2_words.csv`: Hard words

2. `output/removed_words.csv`: List of words removed during content filtering

3. `output/summary.json`: Summary of clustering results including:
   - Number of clusters
   - Words per cluster
   - Best weights found
   - Silhouette score

4. `output/final_cleaned_clusters.csv`: Complete filtered dataset with cluster labels

## Content Filtering

The pipeline supports two methods for content filtering:

1. Using the `better_profanity` library:
   - Enable in config: `content_filtering.profanity.enabled: true`
   - Add custom words: `content_filtering.profanity.custom_words: ["word1", "word2"]`

2. Using a custom word list:
   - Enable in config: `content_filtering.custom_list.enabled: true`
   - Specify CSV path: `content_filtering.custom_list.path: data/custom_filter_words.csv`
   - CSV format: `word,reason` (reason is optional)
