import pandas as pd
import re
import os
import pickle

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# -------------------------------
# STEP 1: LOAD DATASETS
# NOTE: Using only True.csv / Fake.csv — Kaggle_True/Fake are duplicates.
# -------------------------------

true_df = pd.read_csv("dataset/True.csv", low_memory=False)
fake_df = pd.read_csv("dataset/Fake.csv", low_memory=False)

# -------------------------------
# STEP 2: ADD LABELS  (1 = real, 0 = fake)
# -------------------------------

true_df["label"] = 1
fake_df["label"] = 0

print(f"True samples: {len(true_df)}")
print(f"Fake samples: {len(fake_df)}")

# -------------------------------
# STEP 3: KEEP REQUIRED COLUMNS — combine title + text for richer signal
# -------------------------------

def make_combined(df):
    """Combine title and text columns if both exist."""
    if "title" in df.columns and "text" in df.columns:
        df = df.copy()
        df["text"] = df["title"].fillna("") + " " + df["text"].fillna("")
    elif "title" in df.columns:
        df = df.copy()
        df["text"] = df["title"].fillna("")
    return df[["text", "label"]]

true_df = make_combined(true_df)
fake_df = make_combined(fake_df)

# -------------------------------
# STEP 4: MERGE DATASETS
# -------------------------------

df = pd.concat([true_df, fake_df], ignore_index=True)

print(f"Total rows before cleaning: {len(df)}")

# -------------------------------
# STEP 5: CLEAN DATA
# -------------------------------

df.dropna(subset=["text"], inplace=True)
df["text"] = df["text"].astype(str)
df.drop_duplicates(subset="text", inplace=True)

print(f"Rows after cleaning: {len(df)}")
print(f"Class distribution:\n{df['label'].value_counts()}")

# -------------------------------
# STEP 6: CLEAN TEXT
# -------------------------------

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z ]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

df["text"] = df["text"].apply(clean_text)

# Remove empty texts after cleaning
df = df[df["text"].str.len() > 10]

# Shuffle dataset
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# -------------------------------
# STEP 7: SPLIT DATA
# -------------------------------

train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df["label"])

# Remove overlap
test_df = test_df[~test_df["text"].isin(train_df["text"])]

X_train = train_df["text"]
y_train = train_df["label"]

X_test = test_df["text"]
y_test = test_df["label"]

print(f"Training samples: {len(X_train)}")
print(f"Testing samples:  {len(X_test)}")
print(f"Train class dist:\n{y_train.value_counts()}")

# -------------------------------
# STEP 8: TF-IDF
# -------------------------------

vectorizer = TfidfVectorizer(
    max_features=10000,
    ngram_range=(1, 2),
    sublinear_tf=True,   # dampen high-frequency terms
    min_df=2,            # ignore very rare terms
)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec  = vectorizer.transform(X_test)

# -------------------------------
# STEP 9: TRAIN MODEL — balanced weights to fix prediction bias
# -------------------------------

model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced",   # KEY FIX: prevents bias toward majority class
    C=1.0,
    solver="lbfgs",
)
model.fit(X_train_vec, y_train)

print(f"\nModel intercept: {model.intercept_}")

# -------------------------------
# STEP 10: EVALUATE
# -------------------------------

y_pred = model.predict(X_test_vec)

print(f"\nAccuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"\nConfusion Matrix:\n{confusion_matrix(y_test, y_pred)}")
print(f"\nClassification Report:\n{classification_report(y_test, y_pred, target_names=['Fake', 'Real'])}")

# -------------------------------
# STEP 11: SAVE MODEL — saved before any print that could fail
# -------------------------------

os.makedirs("model", exist_ok=True)
pickle.dump(model, open("model/model.pkl", "wb"))
pickle.dump(vectorizer, open("model/vectorizer.pkl", "wb"))

print("\nModel and vectorizer saved successfully!")

# Quick sanity checks (ASCII only to avoid encoding issues on Windows)
test_cases = [
    ("The Federal Reserve raised interest rates by 25 basis points in its latest meeting.", "real"),
    ("SHOCKING vaccines cause mind control chips in 90 percent of patients", "fake"),
    ("NASA confirms moon landing was faked in Hollywood studios.", "fake"),
    ("The president signed the infrastructure bill into law at the White House today.", "real"),
]
print("\n--- Sanity Check Predictions ---")
for text, expected in test_cases:
    cleaned = clean_text(text)
    vec = vectorizer.transform([cleaned])
    pred = model.predict(vec)[0]
    proba = model.predict_proba(vec)[0]
    label = "REAL" if pred == 1 else "FAKE"
    conf = round(float(max(proba)) * 100, 2)
    ok = (pred == 1) == (expected == "real")
    status = "[OK]  " if ok else "[FAIL]"
    print("  " + status + " [" + label + " " + str(conf) + "%] exp=" + expected.upper() + " -- " + text[:65])