"""
PhishGuard AI - Backend API
"""

import os, re, json, math, time, hashlib, logging
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Optional, List
from urllib.parse import urlparse

import joblib
import numpy as np
import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

try:
    import whois as whois_lib
    WHOIS_AVAILABLE = True
except ImportError:
    WHOIS_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("phishguard")

app = FastAPI(title="PhishGuard AI", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

SCREENSHOT_API_KEY = os.getenv("SCREENSHOT_API_KEY", "")

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, '..', 'model', 'phishguard_model.pkl')
META_PATH  = os.path.join(BASE_DIR, '..', 'model', 'model_metadata.json')

try:
    ml_model = joblib.load(MODEL_PATH)
    with open(META_PATH) as f:
        metadata = json.load(f)
    log.info(f"Model loaded | AUC: {metadata.get('test_auc','?')}")
except Exception as e:
    ml_model = None
    metadata = {}
    log.warning(f"Model not loaded: {e}")

scan_history: list = []

SUSPICIOUS_KEYWORDS = [
    'login','signin','verify','account','update','secure','banking',
    'paypal','amazon','apple','google','microsoft','netflix','ebay',
    'password','credential','confirm','suspend','alert','urgent',
    'free','winner','prize','click','limited','expire','warning',
    'blocked','unusual','activity','support','helpdesk','official'
]
SUSPICIOUS_TLDS = {'.tk','.ml','.ga','.cf','.gq','.pw','.top','.xyz','.club','.online','.site','.website','.info','.biz','.link','.click'}
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
        return {k: 0 for k in get_feature_names()}
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
    brands = ['paypal','amazon','apple','google','microsoft','netflix','facebook','instagram','twitter','ebay','chase','wellsfargo']
    f['brand_impersonation']      = int(any(b in full and b not in domain for b in brands))
    f['has_double_slash_redirect']= int('//' in path)
    f['has_hex_encoding']         = int('%' in url)
    f['has_port']                 = int(bool(parsed.port))
    f['path_depth']               = path.count('/')
    f['has_fragment']             = int(bool(parsed.fragment))
    f['has_punycode']             = int('xn--' in hostname)
    f['excessive_subdomains']     = int(f['subdomain_count'] > 3)
    return f


def get_feature_names():
    return list(extract_features('http://example.com').keys())


def get_risk_info(score: float) -> dict:
    if score >= 0.85: return {"level":"CRITICAL","color":"#ff2d55","emoji":"🚨","label":"Critical Risk"}
    if score >= 0.65: return {"level":"HIGH",    "color":"#ff6b35","emoji":"⚠️", "label":"High Risk"}
    if score >= 0.40: return {"level":"MEDIUM",  "color":"#ffd60a","emoji":"⚡", "label":"Medium Risk"}
    if score >= 0.20: return {"level":"LOW",     "color":"#30d158","emoji":"✅", "label":"Low Risk"}
    return                   {"level":"SAFE",    "color":"#34c759","emoji":"🛡️","label":"Safe"}


def analyze_details(url: str, features: dict, score: float) -> dict:
    if not url.startswith(('http://','https://')): url = 'http://' + url
    parsed   = urlparse(url)
    hostname = parsed.hostname or ''
    parts    = hostname.split('.')
    tld      = ('.' + parts[-1]) if len(parts) > 1 else ''
    red, green = [], []
    if features.get('is_https'):
        green.append('Uses HTTPS encryption')
    else:
        red.append('No HTTPS — connection is unencrypted')
    if features.get('has_ip_address'):
        red.append('URL uses a raw IP address instead of a domain — very suspicious')
    if features.get('is_suspicious_tld'):
        red.append(f'Free/abused domain extension ({tld}) commonly used for phishing')
    if features.get('is_trusted_tld') and not features.get('is_suspicious_tld'):
        green.append('Standard trusted domain extension')
    kw = features.get('suspicious_keyword_count', 0)
    if kw >= 3:
        red.append(f'Contains {kw} suspicious keywords (login, verify, secure...)')
    elif kw >= 1:
        green.append(f'Only {kw} sensitive keyword(s) found in URL')
    if features.get('brand_impersonation'):
        red.append('Appears to impersonate a well-known brand')
    if features.get('url_length', 0) > 100:
        red.append(f'Very long URL ({features["url_length"]} chars) used to hide real destination')
    else:
        green.append('URL length looks normal')
    if features.get('subdomain_count', 0) > 3:
        red.append(f'Excessive subdomain nesting ({features["subdomain_count"]} levels)')
    elif features.get('subdomain_count', 0) == 0:
        green.append('Clean domain with no suspicious subdomains')
    if features.get('num_at_signs', 0) > 0:
        red.append("URL contains '@' — can trick browsers to a different site")
    if features.get('has_punycode'):
        red.append('Punycode encoding (xn--) — disguises fake domains as real ones')
    if features.get('domain_entropy', 0) > 3.8:
        red.append('Domain name looks randomly generated — typical of phishing domains')
    else:
        green.append('Domain name looks natural and readable')
    if score >= 0.75:
        verdict = 'This URL shows multiple signs of being a phishing website. We strongly recommend NOT visiting this site.'
    elif score >= 0.45:
        verdict = 'This URL has suspicious characteristics. Proceed with extreme caution and do not enter any personal information.'
    elif score >= 0.2:
        verdict = 'This URL appears mostly safe but has minor concerns. Verify before entering any sensitive data.'
    else:
        verdict = 'This URL appears safe and legitimate. No signs of phishing detected.'
    return {
        'verdict': verdict,
        'red_flags': red,
        'green_flags': green,
        'domain': hostname,
        'protocol': parsed.scheme,
        'path': parsed.path,
        'has_query': bool(parsed.query)
    }


def fetch_screenshot(url: str) -> dict:
    if not SCREENSHOT_API_KEY:
        return {
            'available': False,
            'image_url': None,
            'error': 'No API key set. Add SCREENSHOT_API_KEY to your .env file.',
            'setup_url': 'https://screenshotone.com'
        }
    try:
        # Ensure URL has a scheme — ScreenshotOne requires a full valid URL
        if not url.startswith(('http://', 'https://')):
            url = 'https://' + url
        params = {
            'access_key': SCREENSHOT_API_KEY,
            'url': url,
            'viewport_width': 1280,
            'viewport_height': 800,
            'format': 'jpg',
            'image_quality': 80,
            'block_ads': 'true',
            'block_cookie_banners': 'true',
            'timeout': 15,
            'delay': 1
        }
        base      = 'https://api.screenshotone.com/take'
        query_str = '&'.join(f"{k}={requests.utils.quote(str(v))}" for k, v in params.items())
        image_url = f"{base}?{query_str}"
        # ScreenshotOne does not support HEAD requests — use GET with stream=True
        r = requests.get(image_url, timeout=20, stream=True)
        r.close()
        if r.status_code == 200:
            return {'available': True, 'image_url': image_url, 'error': None}
        return {'available': False, 'image_url': None, 'error': f'Screenshot API returned {r.status_code}. Check your API key.'}
    except Exception as e:
        return {'available': False, 'image_url': None, 'error': str(e)}


def get_domain_age(url: str) -> dict:
    if not WHOIS_AVAILABLE:
        return {'age_days': None, 'created': None, 'label': 'Install python-whois'}
    try:
        if not url.startswith(('http://', 'https://')):
            url = 'http://' + url
        hostname = urlparse(url).hostname or ''
        if hostname.startswith('www.'):
            hostname = hostname[4:]
        w = whois_lib.whois(hostname)
        created = w.creation_date
        if isinstance(created, list):
            created = created[0]
        if not created:
            return {'age_days': None, 'created': None, 'label': 'No WHOIS data'}
        age_days = (datetime.now() - created.replace(tzinfo=None)).days
        if age_days < 30:
            label = f'{age_days}d old - Very new'
        elif age_days < 180:
            label = f'{age_days // 30}mo old - New'
        elif age_days < 365:
            label = f'{age_days // 30}mo old'
        else:
            label = f'{age_days // 365}yr {(age_days % 365) // 30}mo old'
        return {
            'age_days': age_days,
            'created': created.strftime('%Y-%m-%d'),
            'label': label
        }
    except Exception:
        return {'age_days': None, 'created': None, 'label': 'Lookup failed'}


def run_scan(url: str) -> dict:
    url = url.strip()
    if not url: raise ValueError('URL is required')
    t0            = time.time()
    features      = extract_features(url)
    feature_names = get_feature_names()
    vec           = np.array([[features.get(k, 0) for k in feature_names]])
    log.info(f"Scanning: {url} | is_https={features.get('is_https')} susp_tld={features.get('is_suspicious_tld')} kw={features.get('suspicious_keyword_count')} brand={features.get('brand_impersonation')}")
    if ml_model:
        risk_score = float(ml_model.predict_proba(vec)[0][1])
        log.info(f"AI score: {risk_score:.4f} ({round(risk_score*100,1)}%)")
    else:
        risk_score = min(1.0,
            features.get('suspicious_keyword_count',0)*0.12 +
            features.get('is_suspicious_tld',0)*0.30 +
            (1-features.get('is_https',0))*0.22 +
            features.get('has_ip_address',0)*0.35 +
            features.get('brand_impersonation',0)*0.25)
        log.info(f"Heuristic score: {risk_score:.4f}")
    risk_info = get_risk_info(risk_score)
    details   = analyze_details(url, features, risk_score)
    # Run screenshot and WHOIS in parallel to save time
    with ThreadPoolExecutor(max_workers=2) as ex:
        f_screenshot = ex.submit(fetch_screenshot, url)
        f_age        = ex.submit(get_domain_age, url)
        screenshot   = f_screenshot.result()
        domain_age   = f_age.result()
    result = {
        'scan_id':           hashlib.md5(f"{url}{time.time()}".encode()).hexdigest()[:12],
        'url':               url,
        'scanned_at':        datetime.now(timezone.utc).isoformat(),
        'scan_time_seconds': round(time.time()-t0, 3),
        'risk_score':        round(risk_score*100, 1),
        'risk_raw':          risk_score,
        'is_phishing':       risk_score >= 0.5,
        'risk_level':        risk_info['level'],
        'risk_label':        risk_info['label'],
        'risk_color':        risk_info['color'],
        'risk_emoji':        risk_info['emoji'],
        'details':           details,
        'screenshot':        screenshot,
        'domain_age':        domain_age,
        'blocked':           risk_score >= 0.5,
    }
    scan_history.insert(0, result)
    if len(scan_history) > 500: scan_history.pop()
    return result


class ScanRequest(BaseModel):
    url: str
    note: Optional[str] = None

class BulkScanRequest(BaseModel):
    urls: List[str]

@app.get('/')
def root(): return {'status': 'PhishGuard running', 'model_loaded': ml_model is not None}

@app.get('/health')
def health(): return {'status': 'ok', 'model_loaded': ml_model is not None, 'screenshot_enabled': bool(SCREENSHOT_API_KEY), 'total_scans': len(scan_history)}

@app.post('/scan')
def scan_url(req: ScanRequest):
    try:
        result = run_scan(req.url)
        if req.note: result['note'] = req.note
        return result
    except ValueError as e:
        raise HTTPException(400, str(e))

@app.post('/scan/bulk')
def bulk_scan(req: BulkScanRequest):
    urls = [u.strip() for u in req.urls if u.strip()]
    if not urls:       raise HTTPException(400, 'No URLs provided')
    if len(urls) > 50: raise HTTPException(400, 'Maximum 50 URLs per bulk scan')
    results = []
    for url in urls:
        try:    results.append(run_scan(url))
        except Exception as e: results.append({'url': url, 'error': str(e), 'risk_score': 0, 'is_phishing': False})
    phishing = [r for r in results if r.get('is_phishing')]
    return {'summary': {'total': len(results), 'phishing': len(phishing), 'safe': len(results)-len(phishing)}, 'results': results}

@app.get('/history')
def get_history(limit: int = 50, offset: int = 0): return {'total': len(scan_history), 'items': scan_history[offset:offset+limit]}

@app.get('/history/{scan_id}')
def get_scan(scan_id: str):
    for s in scan_history:
        if s.get('scan_id') == scan_id: return s
    raise HTTPException(404, 'Scan not found')

@app.delete('/history')
def clear_history():
    scan_history.clear()
    return {'message': 'History cleared'}

@app.patch('/history/{scan_id}/block')
def toggle_block(scan_id: str):
    for s in scan_history:
        if s.get('scan_id') == scan_id:
            s['blocked'] = not s.get('blocked', False)
            return {'scan_id': scan_id, 'blocked': s['blocked']}
    raise HTTPException(404, 'Scan not found')

@app.get('/stats')
def get_stats():
    if not scan_history: return {'message': 'No scans yet'}
    total    = len(scan_history)
    phishing = sum(1 for s in scan_history if s.get('is_phishing'))
    dist     = {'SAFE': 0, 'LOW': 0, 'MEDIUM': 0, 'HIGH': 0, 'CRITICAL': 0}
    for s in scan_history: dist[s.get('risk_level', 'SAFE')] = dist.get(s.get('risk_level', 'SAFE'), 0) + 1
    return {
        'total_scans':       total,
        'phishing_detected': phishing,
        'safe_sites':        total-phishing,
        'phishing_rate':     round(phishing/total*100, 1),
        'avg_risk_score':    round(sum(s.get('risk_score',0) for s in scan_history)/total, 1),
        'risk_distribution': dist
    }

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8000)
