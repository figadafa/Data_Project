## 🚀 Sentiment Analysis Pipeline (TF-IDF + Classical ML)

A portfolio of Natural Language Processing (NLP) project that demonstrates an end to end machine learning pipeline for sentiment classification using traditional models without deep learning.

## 📌 Overview

In real world applications such as e-commerce platforms, thousands of user reviews are generated daily.
Manually analyzing sentiment is inefficient, subjective, and not scalable.

This project addresses that problem by building a fully automated sentiment classification pipeline that can classify reviews as:

* Positive (1)

* Negative (0)

The focus is not just model accuracy, but also:

> reproducibility

> interpretability

> structured experimentation

 ## ✨ Key Features

🔄 End to End Pipeline (data → preprocessing → modeling → evaluation → deployment-ready output)

⚖️ Class Imbalance Handling using class_weight="balanced"

🧪 Experiment Design with multiple TF-IDF configurations:

Unigram (1,1)

Bigram (1,2)

🤖 Multi-Model Comparison:

Dummy Baseline

SVM (LinearSVC)

Naive Bayes

Logistic Regression

📊 Comprehensive Evaluation Metrics

🔍 Feature Interpretability (top positive & negative words)

💾 Automatic Best Model Selection & Saving using joblib

📁 Results Export for further analysis

## 📁 Project Structure

```
Data_Project/
│
├── sentiment_analysis_pipeline.py   # Main pipeline script
├── Data/
    ├──  amazon.csv                  # Dataset
├── README.md                        # This file
│
└── output/                          # Auto-created on first run
    ├── best_model.joblib            # Best trained estimator
    ├── best_tfidf.joblib            # Matching fitted TF-IDF vectoriser
    └── results.csv                  # All model metrics, sorted by F1-Score
```

---

## 📊 Dataset Format

The pipeline expects a **CSV file** with exactly these two columns:

| Column  | Description                |
| ------- | -------------------------- |
| `Text`  | Raw input text             |
| `label` | 0 = Negative, 1 = Positive |

**Example:**

```csv
Text,label
"This product is absolutely amazing!",1
"Terrible experience, would not recommend.",0
"Works as expected, happy with the purchase.",1
"Broke after one day. Complete waste of money.",0
```


## ⚙️ Installation

**1. Clone the repository**

```bash
git clone https://github.com/your-username/sentiment-analysis-pipeline.git
cd sentiment-analysis-pipeline
```

**2. (Recommended) Create a virtual environment**

```bash
python -m venv venv
source venv/bin/activate        # macOS / Linux
venv\Scripts\activate           # Windows
```

**3. Install dependencies**

```bash
pip install -r requirements.txt
```

---

## 🚀 Usage

**1. Place your dataset** in the project root and name it `dataset.csv`
   (or update the `filepath` variable in `main()`)

**2. Run the pipeline**

```bash
python sentiment_analysis_pipeline.py
```

**3. Check the `output/` folder** for saved model files and `results.csv`

---

## 🔄 Pipeline Walkthrough

```
Load Data
↓
Text Preprocessing
↓
Train-Test Split (Stratified)
↓
TF-IDF Feature Extraction
↓
Model Training (Multiple Models)
↓
Evaluation & Comparison
↓
Feature Interpretation
↓
Best Model Selection
↓
Model Saving + Results Export
```

---

## 🤖 Models

| Model               | Purpose                            |
| ------------------- | ---------------------------------- |
| DummyClassifier     | Baseline benchmark                 |
| LinearSVC           | High-performance linear classifier |
| MultinomialNB       | Fast probabilistic model           |
| Logistic Regression | Interpretable linear model         |


---

## 📈 Evaluation Metrics

1. Accuracy

2. Precision

3. Recall

4. F1-Score (primary ranking metric)

5. Confusion Matrix

6. Classification Report

>All metrics use weighted averaging to  account for class imbalance.
---

## 📤📤 Output Files
1. best_model.joblib
    
    Best-performing model based on F1-score.

2. best_tfidf.joblib
    
   TF-IDF vectorizer used for the best model.

3. results.csv
     
     Full comparison of all experiments.

## 🔍 Example Usage
```
import joblib

model = joblib.load("output/best_model.joblib")
tfidf = joblib.load("output/best_tfidf.joblib")

text = ["The product quality is excellent"]
X = tfidf.transform(text)

print(model.predict(X))  # → [1]
---
```
## ⚠️ Limitations

1. Dataset imbalance may bias predictions toward the majority class

2. TF-IDF ignores deeper semantic meaning beyond n-grams

3. Cannot capture contextual relationships like transformer-based models (e.g., BERT)

4. Performance depends heavily on data quality and preprocessing


## 🚀 Future Improvements

> Implement deep learning models (LSTM, BERT)

>Apply hyperparameter tuning (GridSearchCV / RandomSearch)

>Build REST API using FastAPI

>Deploy as a web application for real-time predictions

>Add cross-validation for more robust evaluation

## 🎯 Project Positioning

This project is designed as an entry-level NLP portfolio project demonstrating:

* structured ML workflow

* model comparison

* practical problem solving

* reproducible experimentation


## 📦 Requirements
pandas>=1.5.0
scikit-learn>=1.2.0
numpy>=1.23.0
joblib>=1.2.0
```

Install all at once:

```bash
pip install pandas scikit-learn numpy joblib
```

Or generate a `requirements.txt` from your environment:

```bash
pip freeze > requirements.txt
```

---

## 📄 License

This project is licensed under the **MIT License** — feel free to use, modify, and distribute it.

---

## 🙋 Author

**Your Name**
- GitHub: [@figadafa](https://github.com/figadafa)
- LinkedIn: [Figa Brilliant Daffa](https://www.linkedin.com/in/figabrilliantdaffa/)

---