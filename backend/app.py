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
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
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

from shared.features import (
    normalize_url,
    extract_features,
    extract_domain_parts
)

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
    # Use feature names from metadata so they always match the trained model
    MODEL_FEATURE_NAMES = metadata.get('feature_names', None)
    log.info(f"Model loaded | AUC: {metadata.get('test_auc','?')} | Features: {len(MODEL_FEATURE_NAMES) if MODEL_FEATURE_NAMES else '?'}")
except Exception as e:
    ml_model = None
    metadata = {}
    MODEL_FEATURE_NAMES = None
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


def extract_features_OLD(url: str) -> dict:
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
    # Extended features (added in PhishTank retraining)
    import re as _re
    _letters_only = _re.sub(r'[^a-z]', '', domain.lower())
    _vowel_ratio  = sum(1 for c in _letters_only if c in 'aeiou') / max(len(_letters_only), 1)
    _cons_run     = max((len(m.group()) for m in _re.finditer(r'[^aeiou]+', domain, _re.I)), default=0)
    f['domain_vowel_ratio']       = round(_vowel_ratio, 4)
    f['max_consonant_run']        = _cons_run
    f['domain_length']            = len(domain)
    f['domain_hyphen_count']      = domain.count('-')
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


def _build_screenshot_api_url(target_url: str) -> str:
    """Build the ScreenshotOne API URL for a given target."""
    if not target_url.startswith(('http://', 'https://')):
        target_url = 'https://' + target_url
    params = {
        'access_key':          SCREENSHOT_API_KEY,
        'url':                 target_url,
        'viewport_width':      1280,
        'viewport_height':     800,
        'format':              'jpg',
        'image_quality':       80,
        'block_ads':           'true',
        'block_cookie_banners':'true',
        'timeout':             20,
        'delay':               1,
        'full_page':           'false',
    }
    qs = '&'.join(f"{k}={requests.utils.quote(str(v))}" for k, v in params.items())
    return f"https://api.screenshotone.com/take?{qs}"


def fetch_screenshot(url: str) -> dict:
    """
    Do NOT call ScreenshotOne here.
    Just return the proxy path so only /screenshot makes the real API request.
    """
    if not SCREENSHOT_API_KEY:
        return {
            'available': False,
            'image_url': None,
            'error': 'No API key configured. Add SCREENSHOT_API_KEY to your .env file.',
            'setup_url': 'https://screenshotone.com'
        }

    if not url.startswith(('http://', 'https://')):
        url = 'https://' + url

    proxy_url = f"/screenshot?url={requests.utils.quote(url, safe='')}"
    return {
        'available': True,
        'image_url': proxy_url,
        'error': None
    }

def get_domain_age(url: str) -> dict:
    if not WHOIS_AVAILABLE:
        return {'age_days': None, 'created': None, 'label': 'Install python-whois'}
    try:
        if not url.startswith(('http://', 'https://')):
            url = 'http://' + url
        hostname = urlparse(url).hostname or ''
        if hostname.startswith('www.'):
            hostname = hostname[4:]
        import socket
        old_timeout = socket.getdefaulttimeout()
        socket.setdefaulttimeout(5)  # 5 second max for WHOIS
        try:
            w = whois_lib.whois(hostname)
        finally:
            socket.setdefaulttimeout(old_timeout)
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
    # Auto-add scheme if missing so bare domains like "example.com" work
    if not url.startswith(('http://','https://')):
        url = 'https://' + url
    t0            = time.time()
    features      = extract_features(url)
    # Use feature names from saved model metadata (handles any number of features)
    feature_names = MODEL_FEATURE_NAMES if MODEL_FEATURE_NAMES else get_feature_names()
    vec           = np.array([[features.get(k, 0) for k in feature_names]])

    # Trusted domain whitelist — model can over-score well-known legitimate sites
    _hostname_check = (urlparse(url).hostname or '').lower().lstrip('www.')
    TRUSTED_DOMAINS = {
        'google.com','youtube.com','facebook.com','amazon.com','wikipedia.org',
        'twitter.com','instagram.com','linkedin.com','reddit.com','netflix.com',
        'microsoft.com','apple.com','github.com','stackoverflow.com','bbc.com',
        'bbc.co.uk','nytimes.com','cnn.com','paypal.com','ebay.com','spotify.com',
        'discord.com','twitch.tv','dropbox.com','adobe.com','zoom.us','slack.com',
        'shopify.com','stripe.com','cloudflare.com','roblox.com','steam.com',
        'steamcommunity.com','steampowered.com','tiktok.com','whatsapp.com',
    }
    
    log.info(f"Scanning: {url} | is_https={features.get('is_https')} susp_tld={features.get('is_suspicious_tld')} kw={features.get('suspicious_keyword_count')} brand={features.get('brand_impersonation')}")
    if ml_model:
        risk_score = float(ml_model.predict_proba(vec)[0][1])
        # Soft trusted-domain adjustment (NOT forced safe)
        _hostname_check = (urlparse(url).hostname or '').lower()

        if any(_hostname_check == d or _hostname_check.endswith('.' + d) for d in TRUSTED_DOMAINS):
            log.info(f"Trusted domain detected: {_hostname_check} → reducing risk")
            risk_score *= 0.3  # reduce but do NOT zero out
        log.info(f"AI score: {risk_score:.4f} ({round(risk_score*100,1)}%)")
    else:
        risk_score = min(1.0,
            features.get('suspicious_keyword_count',0)*0.12 +
            features.get('is_suspicious_tld',0)*0.30 +
            (1-features.get('is_https',0))*0.22 +
            features.get('has_ip_address',0)*0.35 +
            features.get('brand_impersonation',0)*0.25)
        log.info(f"Heuristic score: {risk_score:.4f}")

    # ── Heuristic override layer ──────────────────────────────────
    # The ML model misses certain attack patterns not well-represented
    # in training data. These rules catch them and floor the score.
    heuristic_boost = 0.0
    heuristic_flags = []

    if not url.startswith(('http://','https://')):
        _url_check = 'http://' + url
    else:
        _url_check = url
    _parsed   = urlparse(_url_check)
    _hostname = (_parsed.hostname or '').lower()
    _parts    = _hostname.split('.')
    _tld      = ('.' + _parts[-1]) if len(_parts) > 1 else ''
    _full     = _url_check.lower()

    # 1. Subdomain brand impersonation — e.g. roblox.com.ge, paypal.com.phishing.net
    #    Real domain is the last 2 parts; if a brand appears in subdomains only → phishing
    BRANDS = ['paypal','amazon','apple','google','microsoft','netflix','facebook',
              'instagram','twitter','ebay','chase','wellsfargo','roblox','steam',
              'discord','spotify','tiktok','youtube','linkedin','dropbox','adobe']
    real_domain = '.'.join(_parts[-2:]) if len(_parts) >= 2 else _hostname
    brand_in_subdomain = any(b in _hostname and b not in real_domain for b in BRANDS)
    if brand_in_subdomain:
        heuristic_boost = max(heuristic_boost, 0.82)
        heuristic_flags.append(f'Brand name found in subdomain but NOT in real domain ({real_domain}) — classic subdomain impersonation attack')

    # 2. Brand + non-trusted TLD (e.g. paypal.net.ru, google.com.br.tk)
    if any(b in _hostname for b in BRANDS) and _tld not in {'.com','.org','.net','.gov','.edu','.io'}:
        heuristic_boost = max(heuristic_boost, 0.70)
        if not any(f'non-trusted TLD' in flag for flag in heuristic_flags):
            heuristic_flags.append(f'Brand name combined with non-standard TLD ({_tld}) — suspicious')

    # 3. Multiple dots in hostname with brand name (e.g. www.paypal.com.secure.tk)
    if len(_parts) > 3 and any(b in _hostname for b in BRANDS):
        heuristic_boost = max(heuristic_boost, 0.75)

    # 4. Typosquatting — common brand misspellings
    TYPOS = {
        'paypa1':'paypal','micosoft':'microsoft','micros0ft':'microsoft',
        'arnazon':'amazon','arnaz0n':'amazon','g00gle':'google','gooogle':'google',
        'faceb00k':'facebook','yotube':'youtube','discrod':'discord',
        'rob1ox':'roblox','robl0x':'roblox','steamn':'steam',
    }
    for typo in TYPOS:
        if typo in _hostname:
            heuristic_boost = max(heuristic_boost, 0.88)
            heuristic_flags.append(f'Typosquatting detected: "{typo}" mimics a real brand — common phishing trick')
            break

    # 5. Homograph / confusable characters in domain
    CONFUSABLES = {'0':'o','1':'l','rn':'m','vv':'w'}
    for fake, real in CONFUSABLES.items():
        if fake in (_parts[0] if _parts else '') and real not in (_parts[0] if _parts else ''):
            heuristic_boost = max(heuristic_boost, 0.72)
            heuristic_flags.append(f'Homograph attack: "{fake}" used to mimic "{real}" in domain name')

    # 6. Misleading "secure" / "official" combined with brand
    TRUST_WORDS = ['secure','official','verify','update','login','signin','support','helpdesk']
    if any(w in _full for w in TRUST_WORDS) and any(b in _full for b in BRANDS):
        heuristic_boost = max(heuristic_boost, 0.65)
        heuristic_flags.append('Combines a brand name with trust-inducing words (secure, official, verify…) — common social engineering tactic')

    # 7. Country-code TLD masquerading (brand.com.XX)
    CC_TLDS = {'.ge','.ru','.cn','.pw','.cc','.su','.ws','.to','.ly','.gg'}
    if _tld in CC_TLDS and any(b in _hostname for b in BRANDS):
        heuristic_boost = max(heuristic_boost, 0.80)
        heuristic_flags.append(f'Brand name combined with country-code TLD ({_tld}) — frequently used in impersonation attacks')

    # 8. Gibberish / unpronounceable domain (encyclopeid-annem, xkqzjp, etc.)
    _domain_part = _parts[0] if _parts else ''
    _letters_only = re.sub(r'[^a-z]', '', _domain_part.lower())
    _vowel_ratio  = sum(1 for c in _letters_only if c in 'aeiou') / max(len(_letters_only), 1)
    _consonant_run = max((len(m.group()) for m in re.finditer(r'[^aeiou]+', _domain_part, re.I)), default=0)
    _dom_entropy  = shannon_entropy(_domain_part)
    _is_gibberish = (
        (_consonant_run >= 5 and len(_domain_part) > 8) or
        (_dom_entropy > 3.5 and len(_domain_part) > 10) or
        (_vowel_ratio < 0.18 and len(_letters_only) > 6)
    )
    if _is_gibberish:
        heuristic_boost = max(heuristic_boost, 0.62)
        heuristic_flags.append(f'Domain name appears randomly generated or unpronounceable (entropy={_dom_entropy:.2f}) — commonly seen in auto-generated phishing domains')

    # 9. Long hyphenated domain with no brand (e.g. encyclopeid-annem, secure-account-verify-now)
    _hyphen_parts = _domain_part.split('-')
    if len(_hyphen_parts) >= 2 and len(_domain_part) > 14 and not any(b in _domain_part for b in BRANDS):
        heuristic_boost = max(heuristic_boost, 0.55)
        heuristic_flags.append(f'Long hyphenated domain ({_domain_part}) with no recognisable brand — pattern common in phishing registrations')

    # 10. PhishTank-style: random-looking domain + .com pretending to be legit
    #     Legitimate .com sites rarely have entropy > 3.4 AND length > 14
    if _tld == '.com' and _dom_entropy > 3.4 and len(_domain_part) > 14:
        heuristic_boost = max(heuristic_boost, 0.60)
        if not any('randomly generated' in f for f in heuristic_flags):
            heuristic_flags.append(f'High-entropy .com domain ({_domain_part}) — legitimate sites rarely have this pattern')

    # Apply boost: take the max of ML score and heuristic floor
    if heuristic_boost > risk_score:
        log.info(f"Heuristic override: {risk_score:.4f} → {heuristic_boost:.4f} | Flags: {heuristic_flags}")
        risk_score = heuristic_boost

    log.info(f"Final score: {risk_score:.4f} ({round(risk_score*100,1)}%)")
    risk_info = get_risk_info(risk_score)
    details   = analyze_details(url, features, risk_score)
    # Inject heuristic flags into red flags so they appear in the report
    if heuristic_flags:
        details['red_flags'] = heuristic_flags + details['red_flags']
        # Override verdict if heuristic boosted score significantly
        if heuristic_boost >= 0.75:
            details['verdict'] = 'This URL shows signs of an impersonation attack — it uses a well-known brand name to trick you into thinking it is legitimate, but the real domain is different. We strongly recommend NOT visiting this site.'
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


from fastapi.responses import StreamingResponse
import urllib.parse as _urlparse

import ipaddress
import socket
from urllib.parse import urlparse


def is_private_host(hostname: str) -> bool:
    try:
        ip = socket.gethostbyname(hostname)
        ip_obj = ipaddress.ip_address(ip)

        return (
            ip_obj.is_private
            or ip_obj.is_loopback
            or ip_obj.is_reserved
            or ip_obj.is_link_local
        )
    except:
        return True

@app.get('/screenshot')
def proxy_screenshot(url: str):
    """
    Proxy endpoint: fetches the screenshot from ScreenshotOne server-side
    and streams the image bytes to the browser.
    This avoids CORS issues and keeps the API key hidden from the client.
    """
    parsed = urlparse(url)
    hostname = parsed.hostname

    if not hostname or is_private_host(hostname):
        return {
            "available": False,
            "image_url": None,
            "error": "Blocked for security reasons"
        }
    
    if not SCREENSHOT_API_KEY:
        raise HTTPException(503, 'Screenshot service not configured')
    try:
        target = _urlparse.unquote(url)
        if not target.startswith(('http://', 'https://')):
            target = 'https://' + target
        api_url = _build_screenshot_api_url(target)
        r = requests.get(api_url, timeout=30, stream=True)
        if r.status_code != 200:
            raise HTTPException(502, f'ScreenshotOne returned {r.status_code}')
        content_type = r.headers.get('content-type', 'image/jpeg')
        def iter_content():
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    yield chunk
        return StreamingResponse(iter_content(), media_type=content_type,
                                 headers={'Cache-Control': 'public, max-age=300'})
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(502, str(e))


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
    except Exception as e:
        log.error(f"Scan error for {req.url}: {e}", exc_info=True)
        raise HTTPException(500, f"Scan failed: {str(e)}")

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

# Serve the frontend folder as static files
FRONTEND_DIR = os.path.join(BASE_DIR, '..', 'frontend')
if os.path.exists(FRONTEND_DIR):
    app.mount('/static', StaticFiles(directory=FRONTEND_DIR), name='static')

    @app.get('/app', include_in_schema=False)
    def serve_frontend():
        return FileResponse(os.path.join(FRONTEND_DIR, 'index.html'))

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8000)
