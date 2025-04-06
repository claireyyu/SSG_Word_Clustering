# Word Difficulty Clustering: Technical Workflow

This document outlines the full processing pipeline, including the logic behind each step, customization options, and comparisons of different clustering approaches considered during development.

## Overview

This pipeline performs unsupervised clustering of English words into difficulty levels based on three feature groups:

- Submission frequency from an external source
- Real-world word frequency using the `wordfreq` library
- Spelling complexity derived from structural patterns

The pipeline uses a modular Python implementation and is fully configurable via `config.yaml`.

## Feature Dimensions Used in Clustering

The word difficulty clustering pipeline relies on three core feature dimensions to represent the relative accessibility of words. Each feature contributes a different perspective on what makes a word easy or difficult for learners, particularly in game-based ESL contexts.

1. **Submission Frequency (Game-Specific Usage)**  
   This feature reflects how frequently a word appears in player submissions within the context of the game *Criss Cross Castle*, developed by Simply Sweet Games. The goal of the clustering model is to improve accessibility for ESL (English as a Second Language) learners, and submission frequency provides insight into which words are more likely to be familiar or preferred by players. This game-centric dataset enables fine-tuning difficulty labeling specifically for in-game vocabulary use.

2. **Real-World Word Frequency (Web-Based English Usage)**  
   Real-world frequency data is sourced from the [`wordfreq`](https://pypi.org/project/wordfreq/) Python library. Unlike more traditional corpora such as Google Ngram (which is skewed toward formal and literary texts), `wordfreq` includes data from subtitles, blogs, Wikipedia, and other web-based sources. This better reflects modern, everyday language use. Given that the target application involves casual gameplay, `wordfreq` was selected as a more appropriate reference for capturing real-world accessibility.

3. **Spelling Complexity (Structural Patterns in Word Formation)**  
   Spelling complexity features are derived from observable structural patterns that impact how easy a word is to read, process, or reproduce. These features are informed by linguistic findings in:

   > Wei, C., Kuang, D., & Xu, K. (2023). A Word Difficulty Classification Research Based on K-Means Method. *Highlights in Science, Engineering and Technology*, 70, 215–222. https://doi.org/10.54097/hset.v70i.12188

   The implemented spelling features include:
   - **Repeated characters**: Words with repeated letters tend to be harder (e.g., *squirrel*).
   - **Vowel count**: More vowels generally correlate with easier pronunciation.
   - **Character patterns**: The presence of common English letter patterns such as `"TH"` and `"ER"` can aid readability.
   - **Initial and terminal letters**: Words beginning with `S`, `C`, `A`, or `T` and ending with `E`, `Y`, `R`, or `T` are statistically more likely to be familiar and easy for learners.

Together, these features enable a multidimensional clustering model that accounts for both user behavior and linguistic accessibility.

## 1. Preprocessing Pipeline (`clustering/preprocessing.py`)

### 1.1 Load & Normalize Dataset
- Input file: `accepted_words.csv`
- Standardizes column names to `Word` and `Original_Frequency`
- Converts all words to uppercase to prevent case-sensitive duplication

### 1.2 Filter Tutorial and Invalid Words
- Removes predefined tutorial words: `SPELL`, `WAND`
- Applies `is_valid_word()` to keep only:
  - Alphabetic words (A-Z)
  - Length >= 2

### 1.3 Lemmatization
- Uses NLTK's WordNet Lemmatizer
- Converts cleaned words to their base dictionary forms

> Output: `preprocessed DataFrame`, passed into feature engineering

## 2. Feature Engineering (`clustering/feature_engineering.py`)

### 2.1 Submission Frequency
- Raw values are standardized using `StandardScaler`

### 2.2 Real-World Frequency
- Retrieved using `wordfreq.word_frequency()`
- Log-transformed using `log(frequency + 1e-9)` to reduce skew
- Filled with fallback value (-12) where unavailable
- Standardized with `StandardScaler`

### 2.3 Spelling Features
- Includes:
  - Repeated character detection
  - Vowel and consonant counts
  - Presence of TH, ER
  - Starting with S/C/A/T
  - Ending with E/Y/R/T
- Binary features are inverted to reflect easiness (1 = easier)
- Aggregated spelling easiness score = mean of all spelling-related features
- Also standardized with `StandardScaler`

> Output: `DataFrame` with 3 core features ready for clustering

## 3. Clustering (`clustering/clustering_core.py`)

### 3.1 Grid Search Over Feature Weights
- Performs grid search over 3 weights (0.1 to 1.0)
  - Submission frequency
  - Real-world frequency
  - Spelling easiness
- Constraints:
  - Real-world frequency weight must be highest
  - Silhouette Score > 0.5
- Uses `KMeans(n_clusters=3)` for all tests
- Selects best weights with highest silhouette score

### 3.2 Final KMeans Clustering
- Applies clustering using best weight combination
- Renames clusters based on average real-world frequency:
  - 0 = Easy
  - 1 = Medium
  - 2 = Hard

> Output: final clustered DataFrame with ranked difficulty labels

## 4. Output & Post-Processing

### 4.1 Output Files
- `output/final_cleaned_clusters.csv`: full clustered words
- `output/clusters_by_label/cluster0_words.csv` etc.: per-cluster exports
- `output/summary.json`: metadata on clustering results
- `output/removed_words.csv`: words removed due to inappropriate content

### 4.2 Cleaning Non-English Words
- Filters out:
  - Words not in NLTK's English corpus
  - Words with invalid formatting or length
- Uses `filter_english_only()`

### 4.3 Content Filtering
- Removes inappropriate content using two methods:
  1. [`better_profanity` library:](https://github.com/snguyenthanh/better_profanity/tree/master)
     - Built-in profanity detection
     - Custom word list support
  2. Custom word list from CSV:
     - Flexible format with optional reason column
     - Case-insensitive matching
- Provides detailed statistics about removed words:
  - Total number of words removed
  - Per-cluster impact (words removed from each cluster)
  - Exports removed words with their original cluster labels
- Customizable through `config.yaml`:
  ```yaml
  content_filtering:
    # Option 1: Use profanity library
    profanity:
      enabled: true
      custom_words: []  # Add custom words to filter here
    # Option 2: Use custom word list
    custom_list:
      enabled: true
      path: data/custom_filter_words.csv  # Path to CSV file with custom word list
  ```
- Custom word list CSV format:
  ```csv
  word,reason
  word1,reason1
  word2,reason2
  ```

## 5. Customization Options

| Feature                     | Location                        | How to Customize                   |
|----------------------------|----------------------------------|------------------------------------|
| Input file path            | `config.yaml`                    | Change `input_file`                |
| Tutorial words             | `preprocessing.py`               | Modify `tutorial_words` set        |
| Valid word constraints     | `utils.py:is_valid_word()`       | Adjust regex, length rules         |
| Spelling features          | `feature_engineering.py`         | Add/remove feature functions       |
| Weight search range        | `clustering_core.py`             | Edit `np.arange()` in grid search  |
| Clustering algorithm       | `clustering_core.py`             | Replace or extend from KMeans      |
| Content filtering method   | `config.yaml`                    | Choose between profanity/custom list |
| Content filtering words    | `config.yaml` or CSV file        | Add words to filter                |
| Content filtering logic    | `utils.py:filter_inappropriate_content()` | Modify filtering rules |

## Summary

The final pipeline performs multi-dimensional word difficulty clustering using a blend of corpus-based, usage-based, and structure-based features. All major steps are modular, customizable, and easily extensible. The pipeline includes comprehensive content filtering to ensure appropriate vocabulary for educational contexts.

