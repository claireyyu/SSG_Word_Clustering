import re
from nltk.corpus import words
from nltk import download
import pandas as pd
from better_profanity import profanity
import os

# Download English word list on first run
download('words')
ENGLISH_WORDS = set(w.upper() for w in words.words())

def is_valid_word(word):
    """
    Check if a word is valid:
    - Must be a string
    - Must contain only uppercase letters A-Z
    - Must be at least 2 characters long
    """
    if not isinstance(word, str):
        return False
    word = word.strip().upper()
    return bool(re.fullmatch(r'[A-Z]{2,}', word))

def is_clean_english_word(word):
    """
    Further checks if a word is a valid English word
    using the NLTK word list
    """
    if not is_valid_word(word):
        return False
    return word.upper() in ENGLISH_WORDS

def filter_english_only(df, word_column="Word", min_len=3, max_len=12):
    """
    Filter a DataFrame to include only clean English words.
    - Applies length constraint
    - Filters out words with non-alphabetic characters
    - Filters out words not in the NLTK English lexicon
    Returns a filtered DataFrame.
    """
    df_filtered = df.copy()
    df_filtered = df_filtered[df_filtered[word_column].apply(lambda w: (
        isinstance(w, str) and
        min_len <= len(w.strip()) <= max_len and
        re.fullmatch(r"[A-Z]+", w.strip().upper()) and
        w.strip().upper() in ENGLISH_WORDS
    ))]
    return df_filtered.reset_index(drop=True)

def export_word_list(df, label_column, word_column, output_dir):
    """
    Export words grouped by cluster label into separate CSV files
    """
    os.makedirs(output_dir, exist_ok=True)

    unique_labels = sorted(df[label_column].unique())
    for label in unique_labels:
        words = df[df[label_column] == label][word_column]
        out_path = os.path.join(output_dir, f"cluster{label}_words.csv")
        words.to_csv(out_path, index=False, header=False)

def get_cluster_sizes(df, label_column="KMeans_Label_Ranked"):
    """
    Get the size of each cluster in the DataFrame.
    Returns a dictionary mapping cluster labels to their sizes.
    """
    return df[label_column].value_counts().to_dict()

def get_removed_words(original_df, filtered_df, label_column="KMeans_Label_Ranked"):
    """
    Get words that were removed during filtering, along with their cluster
    information if available. Returns a DataFrame containing removed words.
    """
    # Find removed words by comparing the two DataFrames
    removed_mask = ~original_df["Word"].isin(filtered_df["Word"])
    removed_words = original_df[removed_mask].copy()
    
    # Select columns based on what's available
    columns_to_keep = ["Word"]
    if label_column in removed_words.columns:
        columns_to_keep.append(label_column)
        # Rename the label column if it exists
        removed_words = removed_words[columns_to_keep]
        removed_words.columns = ["Word", "Original_Cluster"]
    else:
        removed_words = removed_words[columns_to_keep]
    
    return removed_words

def load_custom_word_list(csv_path):
    """
    Load custom word list from a CSV file.
    Expected CSV format: word,reason (optional)
    Returns a set of words to filter.
    """
    if not csv_path or not os.path.exists(csv_path):
        return set()
    
    df = pd.read_csv(csv_path)
    # Convert all words to uppercase for case-insensitive comparison
    return set(df['word'].str.upper().tolist())

def filter_inappropriate_content(df, custom_words=None, custom_list_path=None):
    """
    Filter a DataFrame to remove rows containing inappropriate words.
    Supports both profanity library and custom word lists.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame to filter
    custom_words (list, optional): Additional words to add to profanity filter
    custom_list_path (str, optional): Path to CSV file with custom word list
    
    Returns:
    pd.DataFrame: Filtered DataFrame
    int: Number of rows removed
    """
    # Initialize the profanity filter if custom words are provided
    if custom_words is not None:
        profanity.add_censor_words(custom_words)
    
    # Load custom word list if provided
    custom_word_set = load_custom_word_list(custom_list_path)
    
    # Function to check if any cell in a row contains profanity or custom words
    def contains_inappropriate(row):
        for value in row:
            if not isinstance(value, str):
                continue
            value_upper = value.upper()
            # Check against profanity library only if enabled
            if custom_words is not None and profanity.contains_profanity(value):
                return True
            # Check against custom word list
            if value_upper in custom_word_set:
                return True
        return False
    
    # Store original length
    original_len = len(df)
    
    # Filter out rows with inappropriate content
    filtered_df = df[~df.apply(contains_inappropriate, axis=1)]
    
    # Calculate number of rows removed
    rows_removed = original_len - len(filtered_df)
    
    return filtered_df, rows_removed

