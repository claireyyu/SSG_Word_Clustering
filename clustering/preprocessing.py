import pandas as pd
import os
import nltk
from nltk.stem import WordNetLemmatizer
from clustering.utils import is_valid_word

# Download necessary NLTK resources
nltk.download("wordnet")
nltk.download("omw-1.4")

def run_preprocessing(input_file):
    """
    Load and clean the dataset:
    - Standardize column names and casing
    - Remove tutorial words
    - Filter invalid words
    - Apply lemmatization
    Returns a cleaned DataFrame
    """
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"File not found: {input_file}")

    df = pd.read_csv(input_file)
    df.columns = ["Word", "Original_Frequency"]
    df["Word"] = df["Word"].str.upper()

    # Remove tutorial words
    tutorial_words = {"SPELL", "WAND"}
    df = df[~df["Word"].isin(tutorial_words)]

    # Keep only valid alphabetic words (A-Z, length >= 2)
    df["Cleaned_Word"] = df["Word"].apply(lambda w: w if is_valid_word(w) else None)
    df = df.dropna(subset=["Cleaned_Word"])

    # Apply lemmatization
    lemmatizer = WordNetLemmatizer()
    df["Lemmatized_Word"] = df["Cleaned_Word"].apply(lambda w: lemmatizer.lemmatize(w))

    return df.reset_index(drop=True)

