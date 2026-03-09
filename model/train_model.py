"""
PhishGuard AI - Model Training Script
======================================
This script trains a Random Forest classifier to detect phishing URLs.

SETUP:
1. pip install scikit-learn pandas numpy tldextract joblib requests
2. Download dataset (instructions below)
3. Run: python train_model.py

DATASET OPTIONS (free):
- PhiUSIIL Phishing URL Dataset: https://archive.ics.uci.edu/dataset/967
- Kaggle Phishing URLs: https://www.kaggle.com/datasets/shashwatwork/web-page-phishing-detection-dataset
- PhishTank: https://www.phishtank.com/developer_info.php

The script can also generate synthetic training data for testing.
"""

import re
import math
import json
import joblib
import numpy as np
import pandas as pd
import os
from urllib.parse import urlparse
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────
# FEATURE EXTRACTION ENGINE
# ─────────────────────────────────────────────────────────────

SUSPICIOUS_KEYWORDS = [
    'login', 'signin', 'verify', 'account', 'update', 'secure', 'banking',
    'paypal', 'amazon', 'apple', 'google', 'microsoft', 'netflix', 'ebay',
    'password', 'credential', 'confirm', 'suspend', 'alert', 'urgent',
    'free', 'winner', 'prize', 'click', 'limited', 'expire', 'warning',
    'blocked', 'unusual', 'activity', 'support', 'helpdesk', 'official'
]

TRUSTED_TLDS = {'.com', '.org', '.net', '.edu', '.gov', '.io', '.co'}
SUSPICIOUS_TLDS = {
    '.tk', '.ml', '.ga', '.cf', '.gq', '.pw', '.top', '.xyz', '.club',
    '.online', '.site', '.website', '.info', '.biz', '.link', '.click'
}

def extract_features(url: str) -> dict:
    """Extract 30+ features from a URL for ML classification."""
    features = {}

    # ── Normalize URL ──
    if not url.startswith(('http://', 'https://')):
        url = 'http://' + url

    try:
        parsed = urlparse(url)
        hostname = parsed.hostname or ''
        path = parsed.path or ''
        query = parsed.query or ''
        full = url.lower()
    except Exception:
        return {k: 0 for k in get_feature_names()}

    # ── Length-based features ──
    features['url_length'] = len(url)
    features['hostname_length'] = len(hostname)
    features['path_length'] = len(path)
    features['query_length'] = len(query)

    # ── Character count features ──
    features['num_dots'] = url.count('.')
    features['num_hyphens'] = url.count('-')
    features['num_underscores'] = url.count('_')
    features['num_slashes'] = url.count('/')
    features['num_at_signs'] = url.count('@')
    features['num_question_marks'] = url.count('?')
    features['num_equals'] = url.count('=')
    features['num_ampersands'] = url.count('&')
    features['num_percent'] = url.count('%')
    features['num_digits_in_url'] = sum(c.isdigit() for c in url)

    # ── Digit/letter ratio ──
    letters = sum(c.isalpha() for c in url)
    digits = features['num_digits_in_url']
    features['digit_letter_ratio'] = digits / max(letters, 1)

    # ── IP address as hostname ──
    ip_pattern = r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$'
    features['has_ip_address'] = int(bool(re.match(ip_pattern, hostname)))

    # ── Protocol ──
    features['is_https'] = int(parsed.scheme == 'https')

    # ── Suspicious keywords ──
    features['suspicious_keyword_count'] = sum(1 for kw in SUSPICIOUS_KEYWORDS if kw in full)
    features['has_login_keyword'] = int(any(kw in full for kw in ['login', 'signin', 'logon']))
    features['has_banking_keyword'] = int(any(kw in full for kw in ['bank', 'paypal', 'payment', 'transfer']))
    features['has_security_keyword'] = int(any(kw in full for kw in ['secure', 'verify', 'confirm', 'account']))

    # ── TLD analysis ──
    tld = ''
    parts = hostname.split('.')
    if len(parts) > 1:
        tld = '.' + parts[-1]
    features['is_suspicious_tld'] = int(tld in SUSPICIOUS_TLDS)
    features['is_trusted_tld'] = int(tld in TRUSTED_TLDS)
    features['subdomain_count'] = max(0, len(parts) - 2)

    # ── Entropy (randomness of domain) ──
    def shannon_entropy(s):
        if not s:
            return 0
        prob = [s.count(c) / len(s) for c in set(s)]
        return -sum(p * math.log2(p) for p in prob if p > 0)

    domain = parts[0] if parts else ''
    features['domain_entropy'] = shannon_entropy(domain)
    features['hostname_entropy'] = shannon_entropy(hostname)

    # ── Brand impersonation ──
    brands = ['paypal', 'amazon', 'apple', 'google', 'microsoft', 'netflix',
              'facebook', 'instagram', 'twitter', 'ebay', 'chase', 'wellsfargo']
    features['brand_impersonation'] = int(any(b in full and b not in hostname.split('.')[0] for b in brands))

    # ── Structural red flags ──
    features['has_double_slash_redirect'] = int('//' in path)
    features['has_hex_encoding'] = int('%' in url and any(url[i:i+3].startswith('%') for i in range(len(url)-2)))
    features['has_port'] = int(bool(parsed.port))
    features['path_depth'] = path.count('/')
    features['has_fragment'] = int(bool(parsed.fragment))

    # ── Punycode/internationalized ──
    features['has_punycode'] = int('xn--' in hostname)

    # ── Number of subdomains (deep nesting = suspicious) ──
    features['excessive_subdomains'] = int(features['subdomain_count'] > 3)

    return features


def get_feature_names():
    """Return consistent list of feature names."""
    sample = extract_features('http://example.com')
    return list(sample.keys())


# ─────────────────────────────────────────────────────────────
# SYNTHETIC DATA GENERATOR (for testing without real dataset)
# ─────────────────────────────────────────────────────────────

def generate_synthetic_dataset(n_samples=5000):
    """
    Generate synthetic training data.
    Replace this with real dataset for production use!
    """
    print("⚠️  Generating synthetic data. Use a real dataset for production!")
    print("   Download from: https://archive.ics.uci.edu/dataset/967")

    legitimate_urls = [
        'https://www.google.com', 'https://www.amazon.com',
        'https://github.com/login', 'https://stackoverflow.com',
        'https://www.wikipedia.org', 'https://docs.python.org',
        'https://www.youtube.com', 'https://www.reddit.com',
        'https://www.linkedin.com', 'https://www.twitter.com',
        'https://www.bbc.com/news', 'https://www.nytimes.com',
        'https://stripe.com/payments', 'https://aws.amazon.com',
        'https://cloud.google.com', 'https://azure.microsoft.com',
    ]

    phishing_urls = [
        'http://paypal-secure-login.tk/verify/account',
        'http://192.168.1.1/banking/login.php',
        'http://amazon-security-update.ml/confirm',
        'http://apple-id-verify.gq/suspended/account',
        'http://google-account-alert.cf/signin/verify',
        'http://secure-paypal.login.xyz/update',
        'http://microsoft-helpdesk.online/password/reset',
        'http://netflix-billing.site/payment/update',
        'http://ebay-account.club/verify/identity',
        'http://chase-bank-secure.pw/login',
    ]

    rng = np.random.RandomState(42)
    rows = []

    for _ in range(n_samples // 2):
        url = legitimate_urls[rng.randint(0, len(legitimate_urls))]
        # Add slight variations
        suffix = '/' + ''.join(rng.choice(list('abcdefghijklmnop'), rng.randint(3, 15)))
        feats = extract_features(url + suffix)
        feats['label'] = 0  # Legitimate
        rows.append(feats)

    for _ in range(n_samples // 2):
        url = phishing_urls[rng.randint(0, len(phishing_urls))]
        suffix = '/' + ''.join(rng.choice(list('0123456789abcdef'), rng.randint(5, 20)))
        feats = extract_features(url + suffix)
        feats['label'] = 1  # Phishing
        rows.append(feats)

    df = pd.DataFrame(rows)
    return df


def load_real_dataset(filepath: str):
    """
    Load a real phishing dataset CSV.

    Expected columns (PhiUSIIL format):
    - 'URL' or 'url': the URL string
    - 'label' or 'status': 1=phishing, 0=legitimate (or 'phishing'/'legitimate')

    Adjust column names as needed for your dataset.
    """
    print(f"📂 Loading dataset from: {filepath}")
    df = pd.read_csv(filepath)
    print(f"   Loaded {len(df)} rows, columns: {list(df.columns)}")

    # Auto-detect URL column
    url_col = None
    for col in ['URL', 'url', 'Url', 'address']:
        if col in df.columns:
            url_col = col
            break

    # Auto-detect label column
    label_col = None
    for col in ['label', 'Label', 'status', 'Status', 'phishing']:
        if col in df.columns:
            label_col = col
            break

    if not url_col or not label_col:
        raise ValueError(f"Cannot find URL or label columns. Available: {list(df.columns)}")

    # Normalize labels
    df['label'] = df[label_col].apply(
        lambda x: 1 if str(x).lower() in ['1', 'phishing', 'bad'] else 0
    )

    print(f"   Extracting features from {len(df)} URLs... (this may take a few minutes)")
    features = []
    for i, url in enumerate(df[url_col]):
        if i % 1000 == 0:
            print(f"   Progress: {i}/{len(df)}")
        feats = extract_features(str(url))
        feats['label'] = df['label'].iloc[i]
        features.append(feats)

    return pd.DataFrame(features)


# ─────────────────────────────────────────────────────────────
# TRAINING PIPELINE
# ─────────────────────────────────────────────────────────────

def train(dataset_path: str = None):
    print("=" * 60)
    print("  PhishGuard AI - Model Training")
    print("=" * 60)

    # Load data
    if dataset_path:
        df = load_real_dataset(dataset_path)
    else:
        df = generate_synthetic_dataset(n_samples=6000)

    print(f"\n📊 Dataset: {len(df)} samples")
    print(f"   Phishing: {df['label'].sum()} ({df['label'].mean()*100:.1f}%)")
    print(f"   Legitimate: {(df['label']==0).sum()} ({(df['label']==0).mean()*100:.1f}%)")
    print(f"   Features: {len(df.columns)-1}")

    # Prepare features
    feature_cols = [c for c in df.columns if c != 'label']
    X = df[feature_cols].values
    y = df['label'].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"\n🔀 Train: {len(X_train)} | Test: {len(X_test)}")

    # Build model pipeline
    model = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        ))
    ])

    print("\n🧠 Training Random Forest model...")
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    print("\n📈 Model Performance:")
    print("-" * 40)
    print(classification_report(y_test, y_pred, target_names=['Legitimate', 'Phishing']))

    auc = roc_auc_score(y_test, y_prob)
    print(f"   ROC-AUC Score: {auc:.4f}")

    # Cross-validation
    cv_scores = cross_val_score(model, X, y, cv=5, scoring='roc_auc')
    print(f"   5-Fold CV AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # Feature importance
    rf = model.named_steps['clf']
    importances = pd.Series(rf.feature_importances_, index=feature_cols)
    top10 = importances.nlargest(10)
    print("\n🔍 Top 10 Important Features:")
    for feat, imp in top10.items():
        bar = '█' * int(imp * 100)
        print(f"   {feat:<35} {bar} {imp:.4f}")

    # Save model and metadata
    model_path = os.path.join(os.path.dirname(__file__), 'phishguard_model.pkl')
    joblib.dump(model, model_path)

    metadata = {
        'feature_names': feature_cols,
        'n_features': len(feature_cols),
        'training_samples': len(X_train),
        'test_auc': float(auc),
        'cv_auc_mean': float(cv_scores.mean()),
        'cv_auc_std': float(cv_scores.std()),
        'top_features': top10.to_dict()
    }

    with open(os.path.join(os.path.dirname(__file__), 'model_metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"\n✅ Model saved to: {model_path}")
    print(f"✅ Metadata saved to: model_metadata.json")
    print("=" * 60)

    return model, feature_cols


if __name__ == '__main__':
    import sys
    dataset = sys.argv[1] if len(sys.argv) > 1 else None
    train(dataset_path=dataset)
