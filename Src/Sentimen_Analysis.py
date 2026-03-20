
import os
import re
import string
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

warnings.filterwarnings("ignore")

# =============================================================================
# SECTION 1 ── DATA LOADING
# =============================================================================

def load_data(filepath: str) -> pd.DataFrame:
    """
    Load CSV dataset and validate required columns.
    Drops rows with missing values and casts label to int.
    """
    df = pd.read_csv(filepath)
    assert "Text" in df.columns and "label" in df.columns, (
        "CSV must contain 'Text' and 'label' columns."
    )
    df = df.dropna(subset=["Text", "label"])
    df["label"] = df["label"].astype(int)
    return df


# =============================================================================
# SECTION 2 ── TEXT PREPROCESSING
# =============================================================================

def preprocess_text(text: str) -> str:
    """
    Clean a single text string:
      - Convert to lowercase
      - Remove all digits
      - Remove punctuation
      - Strip leading/trailing whitespace
    """
    text = text.lower()
    text = re.sub(r"\d+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = text.strip()
    return text


def preprocess_column(series: pd.Series) -> pd.Series:
    """Apply preprocess_text to every row in a Series."""
    return series.apply(preprocess_text)


# =============================================================================
# SECTION 3 ── TRAIN / TEST SPLIT  (80 : 20, stratified)
# =============================================================================

def split_data(df: pd.DataFrame):
    """
    Stratified 80/20 split — preserves class ratio in both sets.
    Returns X_train, X_test, y_train, y_test.
    """
    X = df["Text"]
    y = df["label"]
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


# =============================================================================
# SECTION 4 ── TF-IDF VECTORISATION  (two configurations)
# =============================================================================

TFIDF_CONFIGS = {
    "Unigram (1,1)      ": dict(max_features=10_000, ngram_range=(1, 1)),
    "Bigram  (1,2)      ": dict(max_features=10_000, ngram_range=(1, 2)),
}


def vectorize(X_train, X_test, ngram_range=(1, 2)):
    """
    Fit a TF-IDF vectoriser on the training set and transform both sets.
    Returns (X_train_vec, X_test_vec, fitted_tfidf).
    """
    tfidf = TfidfVectorizer(max_features=10_000, ngram_range=ngram_range)
    X_train_vec = tfidf.fit_transform(X_train)
    X_test_vec  = tfidf.transform(X_test)
    return X_train_vec, X_test_vec, tfidf


# =============================================================================
# SECTION 5 ── MODEL DEFINITIONS
# =============================================================================

def get_models() -> dict:
    """
    Return all models in a name → estimator dictionary.

    Improvements over previous version:
      - class_weight='balanced' on SVM and Logistic Regression
        handles datasets where one class outnumbers the other.
      - DummyClassifier added as a sanity-check baseline.
        Any real model must beat this; otherwise the signal is too weak.
    """
    return {
        "Baseline (Dummy)   ": DummyClassifier(strategy="most_frequent", random_state=42),
        "SVM (LinearSVC)    ": LinearSVC(
            class_weight="balanced",   # NEW ── penalise majority class less
            random_state=42,
            max_iter=1_000,
        ),
        "Naive Bayes        ": MultinomialNB(),
        "Logistic Regression": LogisticRegression(
            class_weight="balanced",   # NEW ── same imbalance fix
            random_state=42,
            max_iter=1_000,
        ),
    }


# =============================================================================
# SECTION 6 ── TRAINING & EVALUATION
# =============================================================================

def train_and_evaluate(
    models: dict,
    X_train_vec,
    X_test_vec,
    y_train,
    y_test,
    tfidf_label: str = "",
) -> list:
    """
    Train every model on the vectorised training set and collect metrics.
    Returns a list of result dicts (one per model).
    """
    results = []

    for name, model in models.items():
        model.fit(X_train_vec, y_train)
        y_pred = model.predict(X_test_vec)

        results.append({
            "TF-IDF Config" : tfidf_label.strip(),
            "Model"         : name.strip(),
            "Accuracy"      : accuracy_score(y_test, y_pred),
            "Precision"     : precision_score(y_test, y_pred, average="weighted", zero_division=0),
            "Recall"        : recall_score(y_test, y_pred, average="weighted", zero_division=0),
            "F1-Score"      : f1_score(y_test, y_pred, average="weighted", zero_division=0),
            "_y_pred"       : y_pred,    # internal — used for confusion matrix
            "_model"        : model,     # internal — used for best-model saving
            "_tfidf_label"  : tfidf_label.strip(),
        })

    return results


# =============================================================================
# SECTION 7 ── DISPLAY: COMPARISON TABLE
# =============================================================================

def print_comparison_table(results: list, title: str = "MODEL COMPARISON TABLE"):
    """Print a formatted side-by-side metric comparison for all models."""
    col_w    = 24
    tfidf_w  = 20
    metric_w = 11

    header = (
        f"{'TF-IDF':<{tfidf_w}}"
        f"{'Model':<{col_w}}"
        f"{'Accuracy':>{metric_w}}"
        f"{'Precision':>{metric_w}}"
        f"{'Recall':>{metric_w}}"
        f"{'F1-Score':>{metric_w}}"
    )
    divider = "─" * len(header)

    print("\n" + "=" * len(header))
    print(f"{title:^{len(header)}}")
    print("=" * len(header))
    print(header)
    print(divider)

    for r in results:
        print(
            f"{r['TF-IDF Config']:<{tfidf_w}}"
            f"{r['Model']:<{col_w}}"
            f"{r['Accuracy']:>{metric_w}.4f}"
            f"{r['Precision']:>{metric_w}.4f}"
            f"{r['Recall']:>{metric_w}.4f}"
            f"{r['F1-Score']:>{metric_w}.4f}"
        )

    print("=" * len(header))


# =============================================================================
# SECTION 8 ── DISPLAY: CONFUSION MATRICES
# =============================================================================

def print_confusion_matrices(results: list, y_test):
    """Print a labelled confusion matrix for every model / TF-IDF combo."""
    print("\n" + "=" * 56)
    print(f"{'CONFUSION MATRICES':^56}")
    print("=" * 56)

    for r in results:
        cm = confusion_matrix(y_test, r["_y_pred"])
        label = f"{r['TF-IDF Config']}  │  {r['Model']}"
        print(f"\n  ▸ {label}")
        print(f"  {'':22} {'Pred Negative':>14}  {'Pred Positive':>13}")
        print(f"  {'Actual Negative':<22} {cm[0][0]:>14}  {cm[0][1]:>13}")
        print(f"  {'Actual Positive':<22} {cm[1][0]:>14}  {cm[1][1]:>13}")
        print("  " + "─" * 52)


# =============================================================================
# SECTION 9 ── DISPLAY: FULL CLASSIFICATION REPORTS
# =============================================================================

def print_classification_reports(results: list, y_test):
    """Print sklearn's full classification report for each model."""
    print("\n" + "=" * 56)
    print(f"{'FULL CLASSIFICATION REPORTS':^56}")
    print("=" * 56)

    for r in results:
        label = f"{r['TF-IDF Config']}  │  {r['Model']}"
        print(f"\n  ── {label} ──")
        print(
            classification_report(
                y_test,
                r["_y_pred"],
                target_names=["Negative", "Positive"],
            )
        )


# =============================================================================
# SECTION 10 ── FEATURE INTERPRETABILITY  (NEW)
# =============================================================================

def print_top_features(model, tfidf: TfidfVectorizer, model_name: str, top_n: int = 10):
    """
    For linear models (SVM / Logistic Regression), extract the top N words
    most associated with each class using the model's learned coefficients.

    Positive coefficients → words pushing toward class 1 (Positive sentiment)
    Negative coefficients → words pushing toward class 0 (Negative sentiment)
    """
    if not hasattr(model, "coef_"):
        return

    feature_names = np.array(tfidf.get_feature_names_out())
    coef          = model.coef_

    coefficients = coef[0] if coef.shape[0] == 1 else coef[1] - coef[0]

    top_pos_idx = coefficients.argsort()[-top_n:][::-1]  
    top_neg_idx = coefficients.argsort()[:top_n]         

    print(f"\n  ── Feature Importance │ {model_name} ──")
    print(f"  {'Top ' + str(top_n) + ' → Positive sentiment':<35} {'Top ' + str(top_n) + ' → Negative sentiment'}")
    print("  " + "─" * 70)

    for p_idx, n_idx in zip(top_pos_idx, top_neg_idx):
        pos_word = feature_names[p_idx]
        neg_word = feature_names[n_idx]
        pos_val  = coefficients[p_idx]
        neg_val  = coefficients[n_idx]
        print(f"  {pos_word:<20} ({pos_val:+.4f})      {neg_word:<20} ({neg_val:+.4f})")


# =============================================================================
# SECTION 11 ── BEST MODEL SAVING  (NEW)
# =============================================================================

def save_best_model(results: list, tfidf_map: dict, output_dir: str = "."):
    """
    Identify the model with the highest weighted F1-Score across all
    TF-IDF configs and save it (together with its vectoriser) using joblib.

    Files saved:
      best_model.joblib   ← trained sklearn estimator
      best_tfidf.joblib   ← fitted TfidfVectorizer for that config
    """
    real_results = [r for r in results if "Dummy" not in r["Model"]]
    best         = max(real_results, key=lambda r: r["F1-Score"])

    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, "best_model.joblib")
    tfidf_path = os.path.join(output_dir, "best_tfidf.joblib")

    joblib.dump(best["_model"], model_path)
    joblib.dump(tfidf_map[best["_tfidf_label"]], tfidf_path)

    print("\n" + "=" * 56)
    print(f"{'BEST MODEL SAVED':^56}")
    print("=" * 56)
    print(f"  Model      : {best['Model']}")
    print(f"  TF-IDF     : {best['TF-IDF Config']}")
    print(f"  F1-Score   : {best['F1-Score']:.4f}")
    print(f"  Saved to   : {model_path}")
    print(f"             : {tfidf_path}")
    print("=" * 56)

    return best


# =============================================================================
# SECTION 12 ── EXPORT RESULTS TO CSV  (NEW)
# =============================================================================

def export_results_csv(results: list, filepath: str = "results.csv"):
    """
    Save the comparison metrics for every model + TF-IDF config to a CSV.
    Internal columns (prefixed with '_') are excluded from the export.
    """
    export_cols = ["TF-IDF Config", "Model", "Accuracy", "Precision", "Recall", "F1-Score"]
    df_results  = pd.DataFrame(results)[export_cols]
    df_results  = df_results.sort_values("F1-Score", ascending=False).reset_index(drop=True)
    df_results.to_csv(filepath, index=False, float_format="%.4f")
    print(f"\n  ✔  Results exported → {filepath}")


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def main():
    filepath   = r"C:\GitHub\SiLuPa\Data_Project\Data\amazon.csv"
    output_dir =  r"C:\GitHub\SiLuPa\Data_Project\Output" 
    print("\n[1/7] Loading dataset ...")
    df = load_data(filepath)
    print(f"      Total samples : {len(df)}")
    print(f"      Label balance : {df['label'].value_counts().to_dict()}")
    print("\n[2/7] Preprocessing text ...")
    df["Text"] = preprocess_column(df["Text"])
    print("\n[3/7] Splitting dataset 80/20 (stratified) ...")
    X_train, X_test, y_train, y_test = split_data(df)
    print(f"      Train : {len(X_train)} samples | Test : {len(X_test)} samples")
    print("\n[4/7] Running experiments across TF-IDF configurations ...")

    all_results = []     
    tfidf_map   = {}  

    for tfidf_label, tfidf_kwargs in TFIDF_CONFIGS.items():
        ngram = tfidf_kwargs["ngram_range"]
        print(f"\n      ▸ TF-IDF config : {tfidf_label.strip()}")

        X_train_vec, X_test_vec, tfidf = vectorize(X_train, X_test, ngram_range=ngram)
        tfidf_map[tfidf_label.strip()] = tfidf
        print(f"        Vocabulary size : {len(tfidf.vocabulary_)}")

        models  = get_models()
        results = train_and_evaluate(
            models, X_train_vec, X_test_vec, y_train, y_test,
            tfidf_label=tfidf_label,
        )
        all_results.extend(results)
    print("\n[5/7] Displaying evaluation results ...")
    print_comparison_table(all_results)
    print_confusion_matrices(all_results, y_test)
    print_classification_reports(all_results, y_test)

    print("\n[6/7] Feature interpretability (linear models, Bigram TF-IDF) ...")
    bigram_label = "Bigram  (1,2)"
    bigram_tfidf = tfidf_map.get(bigram_label)

  
    bigram_results = [r for r in all_results if r["TF-IDF Config"] == bigram_label]

    print("\n" + "=" * 56)
    print(f"{'FEATURE IMPORTANCE (top 10 per class)':^56}")
    print("=" * 56)
    for r in bigram_results:
        print_top_features(r["_model"], bigram_tfidf, r["Model"], top_n=10)
    print("\n[7/7] Saving best model and exporting results ...")
    os.makedirs(output_dir, exist_ok=True)
    save_best_model(all_results, tfidf_map, output_dir=output_dir)
    export_results_csv(all_results, filepath=os.path.join(output_dir, "results.csv"))


if __name__ == "__main__":
    main()