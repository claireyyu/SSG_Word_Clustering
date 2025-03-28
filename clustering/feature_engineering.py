import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from wordfreq import word_frequency

def add_real_world_frequency(df):
    """
    Add real-world word frequency from the wordfreq library,
    including a log-transformed version.
    """
    df["Real_World_Frequency"] = df["Lemmatized_Word"].apply(
        lambda x: word_frequency(str(x).lower(), 'en')
    )
    df["Real_World_Frequency_Log"] = df["Real_World_Frequency"].apply(
        lambda x: np.log(x + 1e-9)
    )
    df["Real_World_Frequency_Log"] = df["Real_World_Frequency_Log"].fillna(-12)
    return df

def add_spelling_features(df):
    """
    Add features related to spelling complexity and easiness.
    """
    def has_repeated(word):
        return int(len(set(word)) < len(word))

    def vowel_count(word):
        return sum(1 for ch in str(word).upper() if ch in 'AEIOU')

    def consonant_count(word):
        return sum(1 for ch in str(word).upper() if ch.isalpha() and ch not in 'AEIOU')

    def has_th(word):
        return int('TH' in str(word).upper())

    def has_er(word):
        return int('ER' in str(word).upper())

    def first_scat(word):
        return int(str(word)[0].upper() in 'SCAT') if word else 0

    def last_eyrt(word):
        return int(str(word)[-1].upper() in 'EYRT') if word else 0

    df["Has_Repeated"] = df["Lemmatized_Word"].apply(has_repeated)
    df["Vowel_Count"] = df["Lemmatized_Word"].apply(vowel_count)
    df["Consonant_Count"] = df["Lemmatized_Word"].apply(consonant_count)
    df["Has_th"] = df["Lemmatized_Word"].apply(has_th)
    df["Has_er"] = df["Lemmatized_Word"].apply(has_er)
    df["First_scat"] = df["Lemmatized_Word"].apply(first_scat)
    df["Last_eyrt"] = df["Lemmatized_Word"].apply(last_eyrt)

    return df

def standardize_features_and_compute_spelling_score(df):
    """
    Standardize all features and compute a composite spelling easiness score.
    Binary features like Has_Repeated are inverted to represent easiness.
    """
    scaler = StandardScaler()

    df["Standardized_Frequency"] = scaler.fit_transform(df[["Original_Frequency"]])
    df["Standardized_RealWorld_Log"] = scaler.fit_transform(df[["Real_World_Frequency_Log"]])

    # Invert binary difficulty features
    df["Has_Repeated"] = 1 - df["Has_Repeated"]

    spelling_cols = [
        "Has_Repeated", "Vowel_Count", "Consonant_Count",
        "Has_th", "Has_er", "First_scat", "Last_eyrt"
    ]

    df_spelling_scaled = scaler.fit_transform(df[spelling_cols])
    df["Standardized_Spelling_Easiness"] = df_spelling_scaled.mean(axis=1)

    return df

def run_feature_engineering(df):
    """
    Apply all feature engineering steps in order.
    Returns a DataFrame ready for clustering.
    """
    df = add_real_world_frequency(df)
    df = add_spelling_features(df)
    df = standardize_features_and_compute_spelling_score(df)
    return df

