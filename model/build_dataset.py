import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
from shared.features import extract_features
import random

PHISH_FILE = "model/phishtank_urls.csv"
LEGIT_FILE = "model/legit_urls.csv"

def load_urls(file, label, limit=None):
    df = pd.read_csv(file)
    df = df[['url']].dropna()
    if limit:
        df = df.sample(n=min(limit, len(df)), random_state=42)
    df['label'] = label
    return df

def build_dataset():
    print("Loading datasets...")

    # Balance dataset (IMPORTANT)
    phish = load_urls(PHISH_FILE, 1, limit=5000)
    legit = load_urls(LEGIT_FILE, 0, limit=5000)

    df = pd.concat([phish, legit], ignore_index=True)

    # Shuffle dataset
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    print(f"Total samples: {len(df)}")
    print(f"Phishing: {(df['label']==1).sum()} | Legit: {(df['label']==0).sum()}")

    print("Extracting features...")

    feature_rows = []

    for i, row in df.iterrows():
        try:
            feats = extract_features(row['url'])
            feats['label'] = row['label']
            feature_rows.append(feats)
        except Exception as e:
            continue  # skip bad URLs

        if i % 1000 == 0:
            print(f"Processed {i}...")

    final_df = pd.DataFrame(feature_rows)

    print("Saving dataset...")
    final_df.to_csv("model/final_dataset.csv", index=False)

    print("Done.")

if __name__ == "__main__":
    build_dataset()