import os
import json
import warnings

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split, cross_val_score

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "final_dataset.csv")
MODEL_PATH = os.path.join(BASE_DIR, "phishguard_model.pkl")
META_PATH = os.path.join(BASE_DIR, "model_metadata.json")


def validate_dataset(df: pd.DataFrame) -> None:
    if "label" not in df.columns:
        raise ValueError("Dataset must contain a 'label' column.")

    if df.empty:
        raise ValueError("Dataset is empty.")

    class_counts = df["label"].value_counts().to_dict()
    if not {0, 1}.issubset(set(class_counts.keys())):
        raise ValueError(f"Expected binary labels 0 and 1, got: {class_counts}")

    if len(df.columns) <= 1:
        raise ValueError("Dataset has no feature columns.")


def print_top_features(model, feature_names, top_k=10):
    if not hasattr(model, "feature_importances_"):
        return {}

    importances = model.feature_importances_
    pairs = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)[:top_k]

    print("\n🔍 Top 10 Important Features:")
    top_dict = {}
    max_imp = max(v for _, v in pairs) if pairs else 1.0

    for name, value in pairs:
        bar_len = int((value / max_imp) * 14) if max_imp > 0 else 0
        bar = "█" * bar_len
        print(f"   {name:<35} {bar} {value:.4f}")
        top_dict[name] = float(value)

    return top_dict


def main():
    print("=" * 60)
    print("  PhishGuard AI - Model Training (REAL DATA)")
    print("=" * 60)

    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}\n"
            f"Run build_dataset.py first."
        )

    print("\n📥 Loading dataset...")
    df = pd.read_csv(DATA_PATH)

    validate_dataset(df)

    print(f"\n📊 Dataset: {len(df)} samples")
    label_counts = df["label"].value_counts().sort_index()
    legit_count = int(label_counts.get(0, 0))
    phish_count = int(label_counts.get(1, 0))
    print(f"   Legitimate: {legit_count} ({legit_count / len(df) * 100:.1f}%)")
    print(f"   Phishing:   {phish_count} ({phish_count / len(df) * 100:.1f}%)")

    feature_names = [c for c in df.columns if c != "label"]
    print(f"   Features: {len(feature_names)}")

    X = df[feature_names]
    y = df["label"]

    print("\n🔀 Train/test split...")
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    print(f"   Train: {len(X_train)} | Test: {len(X_test)}")

    print("\n🧠 Training Random Forest model...")
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=20,
        min_samples_split=4,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced_subsample",
    )
    model.fit(X_train, y_train)

    print("\n📈 Model Performance:")
    print("-" * 40)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    print(classification_report(
        y_test,
        y_pred,
        target_names=["Legitimate", "Phishing"]
    ))

    auc = roc_auc_score(y_test, y_prob)
    print(f"   ROC-AUC Score: {auc:.4f}")

    print("\n🧪 Running 5-fold cross-validation...")
    cv_scores = cross_val_score(
        model,
        X,
        y,
        cv=5,
        scoring="roc_auc",
        n_jobs=-1
    )
    print(f"   5-Fold CV AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    cm = confusion_matrix(y_test, y_pred)
    print("\n🧱 Confusion Matrix:")
    print(cm)

    top_features = print_top_features(model, feature_names, top_k=10)

    print("\n💾 Saving model and metadata...")
    joblib.dump(model, MODEL_PATH)

    metadata = {
        "feature_names": feature_names,
        "n_features": len(feature_names),
        "training_samples": int(len(X_train)),
        "test_samples": int(len(X_test)),
        "test_auc": float(auc),
        "cv_auc_mean": float(cv_scores.mean()),
        "cv_auc_std": float(cv_scores.std()),
        "top_features": top_features,
        "model_type": "RandomForestClassifier",
        "dataset_path": DATA_PATH,
    }

    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"✅ Model saved to: {MODEL_PATH}")
    print(f"✅ Metadata saved to: {META_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    main()