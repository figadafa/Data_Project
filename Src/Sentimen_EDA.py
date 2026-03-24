# eda.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from collections import Counter
import string
import os

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

OUTPUT_DIR = "Output"  # sesuaikan dengan folder kamu


# ──────────────────────────────────────────────
# Load Data
# ──────────────────────────────────────────────
def load_data(filepath):
    df = pd.read_csv(filepath)
    return df


# ──────────────────────────────────────────────
# Basic Inspection
# ──────────────────────────────────────────────
def basic_inspection(df):
    print("\n=== BASIC DATA INFO ===")
    print(f"Shape: {df.shape}\n")

    print("Missing values:")
    print(df.isnull().sum(), "\n")

    print("Label distribution:")
    print(df["label"].value_counts(), "\n")


# ──────────────────────────────────────────────
# Preprocessing (EDA only)
# ──────────────────────────────────────────────
def preprocess_text(text):
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    tokens = text.split()
    
    # remove stopwords
    tokens = [word for word in tokens if word not in ENGLISH_STOP_WORDS]
    
    return tokens


# ──────────────────────────────────────────────
# Label Distribution
# ──────────────────────────────────────────────
def plot_label_distribution(df):
    plt.figure()
    sns.countplot(x="label", data=df)
    plt.title("Label Distribution")

    plt.savefig(f"{OUTPUT_DIR}/label_distribution.png")
    plt.show()


# ──────────────────────────────────────────────
# Text Length Distribution
# ──────────────────────────────────────────────
def plot_text_length(df):
    df["text_length"] = df["Text"].apply(lambda x: len(str(x).split()))

    plt.figure()
    sns.histplot(data=df, x="text_length", hue="label", bins=30)
    plt.title("Text Length Distribution by Label")

    plt.savefig(f"{OUTPUT_DIR}/text_length_distribution.png")
    plt.show()

    return df


# ──────────────────────────────────────────────
# Top Words (Overall)
# ──────────────────────────────────────────────
def plot_top_words(df):
    all_words = []

    for text in df["Text"]:
        tokens = preprocess_text(str(text))
        all_words.extend(tokens)

    counter = Counter(all_words)
    most_common = counter.most_common(20)

    words = [w[0] for w in most_common]
    counts = [w[1] for w in most_common]

    plt.figure()
    plt.bar(words, counts)
    plt.title("Top 20 Words (Cleaned)")
    plt.xticks(rotation=45)

    plt.savefig(f"{OUTPUT_DIR}/top_words.png")
    plt.show()

    return most_common


# ──────────────────────────────────────────────
# Top Words per Label (IMPORTANT)
# ──────────────────────────────────────────────
def plot_top_words_per_label(df):
    for label in [0, 1]:
        words = []

        subset = df[df["label"] == label]

        for text in subset["Text"]:
            tokens = preprocess_text(str(text))
            words.extend(tokens)

        counter = Counter(words)
        most_common = counter.most_common(20)

        w = [x[0] for x in most_common]
        c = [x[1] for x in most_common]

        plt.figure()
        plt.bar(w, c)
        plt.title(f"Top Words - {'Negative' if label == 0 else 'Positive'}")
        plt.xticks(rotation=45)

        plt.savefig(f"{OUTPUT_DIR}/top_words_label_{label}.png")
        plt.show()


# ──────────────────────────────────────────────
# Insights
# ──────────────────────────────────────────────
def print_insights(df, most_common):
    print("\n=== INSIGHTS ===")

    # imbalance
    label_ratio = df["label"].value_counts(normalize=True)
    print("\nLabel Ratio:")
    print(label_ratio)

    if max(label_ratio) > 0.7:
        print("→ Dataset is imbalanced")
    else:
        print("→ Dataset is relatively balanced")

    # text length
    avg_length = df["text_length"].mean()
    print(f"\nAverage text length: {avg_length:.2f} words")

    # common words
    print("\nTop 10 words (cleaned):")
    for word, count in most_common[:10]:
        print(f"{word}: {count}")


# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────
def main():
    filepath = "Data/amazon.csv"

    print("[1] Loading dataset...")
    df = load_data(filepath)

    print("[2] Basic inspection...")
    basic_inspection(df)

    print("[3] Label distribution...")
    plot_label_distribution(df)

    print("[4] Text length...")
    df = plot_text_length(df)

    print("[5] Top words overall...")
    most_common = plot_top_words(df)

    print("[6] Top words per label...")
    plot_top_words_per_label(df)

    print("[7] Insights...")
    print_insights(df, most_common)


if __name__ == "__main__":
    main()