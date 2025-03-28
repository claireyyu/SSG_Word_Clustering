import re
from nltk.corpus import words
from nltk import download
import pandas as pd

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
    import os
    os.makedirs(output_dir, exist_ok=True)

    unique_labels = sorted(df[label_column].unique())
    for label in unique_labels:
        words = df[df[label_column] == label][word_column]
        out_path = os.path.join(output_dir, f"cluster{label}_words.csv")
        words.to_csv(out_path, index=False, header=False)

