"""
PhishTank + PhiUSIIL Combined Retraining Script
================================================
Downloads fresh PhishTank data and merges it with your existing
PhiUSIIL dataset, then retrains the model.

Usage:
    python add_phishtank.py [path_to_PhiUSIIL_dataset.csv]

PhishTank API is free but requires a free account for higher rate limits.
Get your app key at: https://www.phishtank.com/api_info.php
Set it as env var:   PHISHTANK_APP_KEY=your_key_here
Or just run without a key (slower, 100 req/hr limit).
"""

import os, sys, re, math, json, time, csv, hashlib, logging, requests
import pandas as pd
import numpy as np
from urllib.parse import urlparse
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, roc_auc_score
import joblib

logging.basicConfig(level=logging.INFO, format='%(message)s')
log = logging.getLogger(__name__)

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'phishguard_model.pkl')
META_PATH  = os.path.join(BASE_DIR, 'model_metadata.json')
CACHE_PATH = os.path.join(BASE_DIR, 'phishtank_cache.csv')

PHISHTANK_APP_KEY = os.getenv('PHISHTANK_APP_KEY', '')

# ── Same feature extraction as app.py ──────────────────────────

SUSPICIOUS_KEYWORDS = [
    'login','signin','verify','account','update','secure','banking',
    'paypal','amazon','apple','google','microsoft','netflix','ebay',
    'password','credential','confirm','suspend','alert','urgent',
    'free','winner','prize','click','limited','expire','warning',
    'blocked','unusual','activity','support','helpdesk','official'
]
SUSPICIOUS_TLDS = {'.tk','.ml','.ga','.cf','.gq','.pw','.top','.xyz','.club',
                   '.online','.site','.website','.info','.biz','.link','.click'}
TRUSTED_TLDS    = {'.com','.org','.net','.edu','.gov','.io','.co'}

def shannon_entropy(s):
    if not s: return 0
    prob = [s.count(c)/len(s) for c in set(s)]
    return -sum(p * math.log2(p) for p in prob if p > 0)

def extract_features(url: str) -> dict:
    if not url.startswith(('http://','https://')):
        url = 'http://' + url
    try:
        parsed   = urlparse(url)
        hostname = parsed.hostname or ''
        path     = parsed.path or ''
        query    = parsed.query or ''
        full     = url.lower()
    except Exception:
        return {}
    parts = hostname.split('.')
    tld   = ('.' + parts[-1]) if len(parts) > 1 else ''
    f = {}
    f['url_length']               = len(url)
    f['hostname_length']          = len(hostname)
    f['path_length']              = len(path)
    f['query_length']             = len(query)
    f['num_dots']                 = url.count('.')
    f['num_hyphens']              = url.count('-')
    f['num_underscores']          = url.count('_')
    f['num_slashes']              = url.count('/')
    f['num_at_signs']             = url.count('@')
    f['num_question_marks']       = url.count('?')
    f['num_equals']               = url.count('=')
    f['num_ampersands']           = url.count('&')
    f['num_percent']              = url.count('%')
    f['num_digits_in_url']        = sum(c.isdigit() for c in url)
    letters = sum(c.isalpha() for c in url)
    f['digit_letter_ratio']       = f['num_digits_in_url'] / max(letters, 1)
    f['has_ip_address']           = int(bool(re.match(r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$', hostname)))
    f['is_https']                 = int(parsed.scheme == 'https')
    f['suspicious_keyword_count'] = sum(1 for kw in SUSPICIOUS_KEYWORDS if kw in full)
    f['has_login_keyword']        = int(any(kw in full for kw in ['login','signin','logon']))
    f['has_banking_keyword']      = int(any(kw in full for kw in ['bank','paypal','payment','transfer']))
    f['has_security_keyword']     = int(any(kw in full for kw in ['secure','verify','confirm','account']))
    f['is_suspicious_tld']        = int(tld in SUSPICIOUS_TLDS)
    f['is_trusted_tld']           = int(tld in TRUSTED_TLDS)
    f['subdomain_count']          = max(0, len(parts) - 2)
    domain = parts[0] if parts else ''
    f['domain_entropy']           = shannon_entropy(domain)
    f['hostname_entropy']         = shannon_entropy(hostname)
    brands = ['paypal','amazon','apple','google','microsoft','netflix','facebook',
              'instagram','twitter','ebay','chase','wellsfargo','roblox','steam',
              'discord','spotify','tiktok','youtube','linkedin','dropbox','adobe']
    f['brand_impersonation']      = int(any(b in full and b not in domain for b in brands))
    f['has_double_slash_redirect']= int('//' in path)
    f['has_hex_encoding']         = int('%' in url)
    f['has_port']                 = int(bool(parsed.port))
    f['path_depth']               = path.count('/')
    f['has_fragment']             = int(bool(parsed.fragment))
    f['has_punycode']             = int('xn--' in hostname)
    f['excessive_subdomains']     = int(f['subdomain_count'] > 3)
    # New features for PhishTank-style domains
    letters_only = re.sub(r'[^a-z]', '', domain.lower())
    vowel_ratio  = sum(1 for c in letters_only if c in 'aeiou') / max(len(letters_only), 1)
    consonant_run = max((len(m.group()) for m in re.finditer(r'[^aeiou]+', domain, re.I)), default=0)
    f['domain_vowel_ratio']       = round(vowel_ratio, 4)
    f['max_consonant_run']        = consonant_run
    f['domain_length']            = len(domain)
    f['domain_hyphen_count']      = domain.count('-')
    return f

def get_feature_names():
    return list(extract_features('http://example.com').keys())


# ── PhishTank downloader ────────────────────────────────────────

def download_phishtank(max_urls: int = 10000) -> pd.DataFrame:
    """Download verified phishing URLs from PhishTank."""

    # Check cache first (refresh if older than 24 hours)
    if os.path.exists(CACHE_PATH):
        age_hours = (time.time() - os.path.getmtime(CACHE_PATH)) / 3600
        if age_hours < 24:
            log.info(f"Using cached PhishTank data ({age_hours:.1f}h old)")
            return pd.read_csv(CACHE_PATH)
        log.info("Cache is stale, re-downloading...")

    log.info("Downloading PhishTank verified phishing URLs...")
    log.info("(This requires a free PhishTank account for the JSON API)")

    headers = {'User-Agent': 'PhishGuard-AI/1.0 (phishing research tool)'}
    params  = {'format': 'json'}
    if PHISHTANK_APP_KEY:
        params['app_key'] = PHISHTANK_APP_KEY

    try:
        resp = requests.post(
            'http://data.phishtank.com/data/online-valid.json',
            data=params, headers=headers, timeout=60, stream=True
        )
        resp.raise_for_status()
        data = resp.json()

        urls = []
        for entry in data[:max_urls]:
            if entry.get('verified') == 'yes' and entry.get('online') == 'yes':
                urls.append(entry['url'])

        log.info(f"Downloaded {len(urls)} verified+online phishing URLs")
        df = pd.DataFrame({'url': urls, 'label': 1})
        df.to_csv(CACHE_PATH, index=False)
        return df

    except Exception as e:
        log.error(f"PhishTank download failed: {e}")
        log.info("Falling back to manual URL list...")
        return pd.DataFrame({'url': [], 'label': []})


def load_manual_phishing_urls() -> pd.DataFrame:
    """
    Manually curated PhishTank-style URLs that the model misses.
    Add any URLs here that you find on PhishTank or encounter yourself.
    """
    manual_urls = [
        # Gibberish domains (high entropy, unpronounceable)
        "https://encyclopeid-annem.com",
        "http://xkqzjpmn.com/login",
        "http://secure-verify-accountupdate.com",
        "http://update-account-informations.com/signin",
        "http://verification-required-account.com",
        "http://account-suspended-verify-now.com",
        # Subdomain impersonation
        "http://paypal.com.secure-payment.net",
        "http://apple.com.id-verify.xyz",
        "http://amazon.com.account-alert.ru",
        "https://roblox.com.ge/verify",
        "https://microsoft.com.login-verify.tk",
        # Lookalike domains
        "http://paypa1.com/login",
        "http://arnazon.com/deals",
        "http://micosoft.net/support",
        "http://gooogle.com/account",
        "http://faceb00k.com/login",
        # Random + trust words
        "http://secure-banking-update-required.com",
        "http://urgent-account-verification.net",
        "http://login-confirm-identity.com",
        "http://your-account-has-been-limited.com",
    ]
    return pd.DataFrame({'url': manual_urls, 'label': 1})


# ── Legit URL sources ───────────────────────────────────────────

def load_legit_urls() -> pd.DataFrame:
    """Return a set of known-good URLs for balance."""
    legit = [
        "https://www.google.com", "https://www.youtube.com", "https://www.facebook.com",
        "https://www.amazon.com", "https://www.wikipedia.org", "https://www.twitter.com",
        "https://www.instagram.com", "https://www.linkedin.com", "https://www.reddit.com",
        "https://www.netflix.com", "https://www.microsoft.com", "https://www.apple.com",
        "https://www.github.com", "https://stackoverflow.com", "https://www.bbc.com",
        "https://www.nytimes.com", "https://www.cnn.com", "https://www.bbc.co.uk",
        "https://www.paypal.com", "https://www.ebay.com", "https://www.spotify.com",
        "https://www.discord.com", "https://www.twitch.tv", "https://www.dropbox.com",
        "https://www.adobe.com", "https://www.salesforce.com", "https://www.zoom.us",
        "https://www.slack.com", "https://www.shopify.com", "https://www.stripe.com",
        "https://www.cloudflare.com", "https://www.digitalocean.com", "https://www.heroku.com",
        "https://docs.python.org", "https://www.django-rest-framework.org",
        "https://fastapi.tiangolo.com", "https://scikit-learn.org", "https://numpy.org",
    ]
    return pd.DataFrame({'url': legit, 'label': 0})


# ── Main retraining pipeline ────────────────────────────────────

def retrain(phiusill_path: str = None):
    log.info("=" * 60)
    log.info("PhishGuard AI — Retraining with PhishTank Data")
    log.info("=" * 60)

    all_dfs = []

    # 1. Load PhiUSIIL base dataset
    if phiusill_path and os.path.exists(phiusill_path):
        log.info(f"\n📂 Loading PhiUSIIL dataset: {phiusill_path}")
        base_df = pd.read_csv(phiusill_path)

        # Detect URL and label columns
        url_col   = next((c for c in base_df.columns if 'url' in c.lower()), None)
        label_col = next((c for c in base_df.columns if 'label' in c.lower() or 'phish' in c.lower() or 'class' in c.lower()), None)

        if url_col and label_col:
            base_df = base_df[[url_col, label_col]].rename(columns={url_col: 'url', label_col: 'label'})
            # Normalize labels to 0/1
            if base_df['label'].dtype == object:
                base_df['label'] = base_df['label'].map(lambda x: 1 if str(x).lower() in ['1','phishing','phish'] else 0)
            log.info(f"   PhiUSIIL: {len(base_df):,} samples")
            all_dfs.append(base_df)
        else:
            log.warning(f"   Could not find URL/label columns in {phiusill_path}")
    else:
        log.info("\n⚠️  No PhiUSIIL dataset path provided — training on PhishTank + manual URLs only")

    # 2. Download PhishTank data
    log.info("\n🌐 Fetching PhishTank data...")
    pt_df = download_phishtank(max_urls=15000)
    if not pt_df.empty:
        log.info(f"   PhishTank: {len(pt_df):,} phishing URLs")
        all_dfs.append(pt_df)

    # 3. Add manually curated URLs
    manual_df = load_manual_phishing_urls()
    log.info(f"   Manual: {len(manual_df)} curated phishing URLs")
    all_dfs.append(manual_df)

    # 4. Add legit URLs
    legit_df = load_legit_urls()
    log.info(f"   Legit: {len(legit_df)} known-safe URLs")
    all_dfs.append(legit_df)

    # 5. Combine and deduplicate
    df = pd.concat(all_dfs, ignore_index=True).drop_duplicates(subset='url')
    log.info(f"\n📊 Combined dataset: {len(df):,} samples")
    log.info(f"   Phishing: {df['label'].sum():,} ({df['label'].mean()*100:.1f}%)")
    log.info(f"   Legitimate: {(df['label']==0).sum():,} ({(1-df['label'].mean())*100:.1f}%)")

    # 6. Extract features
    log.info(f"\n⚙️  Extracting {len(get_feature_names())} features...")
    feature_names = get_feature_names()
    rows, labels, failed = [], [], 0
    for i, (_, row) in enumerate(df.iterrows()):
        if i % 10000 == 0 and i > 0:
            log.info(f"   Progress: {i:,}/{len(df):,}")
        try:
            f = extract_features(str(row['url']))
            if f:
                rows.append([f.get(k, 0) for k in feature_names])
                labels.append(int(row['label']))
            else:
                failed += 1
        except Exception:
            failed += 1

    if failed:
        log.warning(f"   Skipped {failed} URLs (parse errors)")

    X = np.array(rows)
    y = np.array(labels)
    log.info(f"   Feature matrix: {X.shape}")

    # 7. Train
    log.info("\n🧠 Training Random Forest...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    model = Pipeline([
        ('scaler', StandardScaler()),
        ('clf',    RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            min_samples_split=4,
            min_samples_leaf=2,
            class_weight='balanced',   # Handles class imbalance
            random_state=42,
            n_jobs=-1
        ))
    ])
    model.fit(X_train, y_train)

    # 8. Evaluate
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    auc     = roc_auc_score(y_test, y_proba)

    log.info("\n📈 Model Performance:")
    log.info("-" * 40)
    log.info(classification_report(y_test, y_pred, target_names=['Legitimate','Phishing']))
    log.info(f"   ROC-AUC Score: {auc:.4f}")

    # Quick sanity check on known URLs
    log.info("\n🔍 Sanity check on known URLs:")
    test_cases = [
        ("https://encyclopeid-annem.com",                               "PHISHING"),
        ("https://roblox.com.ge/games/test",                            "PHISHING"),
        ("http://paypal-secure-login.tk/verify/account/confirm",        "PHISHING"),
        ("https://www.google.com",                                      "SAFE"),
        ("https://www.paypal.com",                                      "SAFE"),
        ("https://github.com",                                          "SAFE"),
    ]
    for url, expected in test_cases:
        f   = extract_features(url)
        vec = np.array([[f.get(k, 0) for k in feature_names]])
        score = model.predict_proba(vec)[0][1]
        status = "✅" if (score > 0.5) == (expected == "PHISHING") else "❌"
        log.info(f"   {status} {url[:55]:<55} {score*100:5.1f}% [{expected}]")

    # 9. Save
    joblib.dump(model, MODEL_PATH)
    meta = {
        'test_auc':      auc,
        'feature_names': feature_names,
        'n_features':    len(feature_names),
        'train_samples': len(X_train),
        'test_samples':  len(X_test),
        'trained_at':    time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'datasets':      ['PhiUSIIL', 'PhishTank', 'manual'] if phiusill_path else ['PhishTank', 'manual'],
    }
    with open(META_PATH, 'w') as f_meta:
        json.dump(meta, f_meta, indent=2)

    log.info(f"\n✅ Model saved to: {MODEL_PATH}")
    log.info(f"✅ Metadata saved to: {META_PATH}")
    log.info(f"\n🔄 Restart your backend to use the new model:")
    log.info(f"   python -m uvicorn app:app --reload --port 8000")
    log.info("=" * 60)


if __name__ == '__main__':
    phiusill_csv = sys.argv[1] if len(sys.argv) > 1 else None
    retrain(phiusill_csv)
