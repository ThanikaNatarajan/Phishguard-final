import math
import re
from collections import Counter
from urllib.parse import urlparse

try:
    import tldextract
except ImportError:
    tldextract = None


SUSPICIOUS_WORDS = {
    "login", "verify", "secure", "update", "account", "banking",
    "confirm", "signin", "reset", "payment", "wallet", "support",
    "password", "alert", "suspended", "recovery", "unlock"
}

BRANDS = {
    "paypal", "google", "microsoft", "apple", "amazon", "facebook",
    "instagram", "netflix", "bankofamerica", "chase", "roblox",
    "discord", "steam", "dropbox", "outlook", "office365"
}


def normalize_url(url: str) -> str:
    url = (url or "").strip()
    if not url:
        return url
    if not re.match(r"^https?://", url, re.IGNORECASE):
        url = "https://" + url
    return url


def _fallback_domain_parts(hostname: str):
    parts = (hostname or "").lower().split(".")
    if len(parts) >= 2:
        subdomain = ".".join(parts[:-2])
        domain = parts[-2]
        suffix = parts[-1]
    elif len(parts) == 1:
        subdomain = ""
        domain = parts[0]
        suffix = ""
    else:
        subdomain = ""
        domain = ""
        suffix = ""
    return subdomain, domain, suffix


def extract_domain_parts(hostname: str):
    hostname = (hostname or "").strip().lower()
    if not hostname:
        return {
            "subdomain": "",
            "domain": "",
            "suffix": "",
            "registered_domain": "",
            "full_hostname": ""
        }

    if tldextract:
        ext = tldextract.extract(hostname)
        subdomain = ext.subdomain or ""
        domain = ext.domain or ""
        suffix = ext.suffix or ""
    else:
        subdomain, domain, suffix = _fallback_domain_parts(hostname)

    registered_domain = ".".join([p for p in [domain, suffix] if p])

    return {
        "subdomain": subdomain,
        "domain": domain,
        "suffix": suffix,
        "registered_domain": registered_domain,
        "full_hostname": hostname
    }


def shannon_entropy(text: str) -> float:
    if not text:
        return 0.0
    counts = Counter(text)
    total = len(text)
    return -sum((count / total) * math.log2(count / total) for count in counts.values())


def max_consonant_run(text: str) -> int:
    text = re.sub(r"[^a-z]", "", (text or "").lower())
    vowels = set("aeiou")
    best = 0
    cur = 0
    for ch in text:
        if ch not in vowels:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return best


def has_brand_impersonation(full_hostname: str, registered_domain: str) -> int:
    fh = (full_hostname or "").lower()
    rd = (registered_domain or "").lower()

    for brand in BRANDS:
        if brand in fh and brand not in rd:
            return 1
    return 0


def extract_features(url: str) -> dict:
    url = normalize_url(url)
    parsed = urlparse(url)

    hostname = (parsed.hostname or "").lower()
    path = parsed.path or ""
    query = parsed.query or ""
    full = url.lower()

    domain_info = extract_domain_parts(hostname)
    domain = domain_info["domain"]
    registered_domain = domain_info["registered_domain"]
    subdomain = domain_info["subdomain"]

    url_len = len(url)
    hostname_len = len(hostname)
    digit_count = sum(c.isdigit() for c in url)
    hyphen_count = url.count("-")
    dot_count = url.count(".")
    slash_count = url.count("/")
    at_count = url.count("@")
    question_count = url.count("?")
    equals_count = url.count("=")
    underscore_count = url.count("_")
    percent_count = url.count("%")

    suspicious_word_count = sum(word in full for word in SUSPICIOUS_WORDS)
    subdomain_count = len([p for p in subdomain.split(".") if p])

    letters_only_domain = re.sub(r"[^a-z]", "", domain)
    vowel_ratio = (
        sum(c in "aeiou" for c in letters_only_domain) / len(letters_only_domain)
        if letters_only_domain else 0.0
    )

    features = {
        "url_length": url_len,
        "hostname_length": hostname_len,
        "digit_count": digit_count,
        "hyphen_count": hyphen_count,
        "dot_count": dot_count,
        "slash_count": slash_count,
        "at_count": at_count,
        "question_count": question_count,
        "equals_count": equals_count,
        "underscore_count": underscore_count,
        "percent_count": percent_count,
        "has_ip": int(bool(re.fullmatch(r"\d{1,3}(\.\d{1,3}){3}", hostname))),
        "is_https": int(parsed.scheme == "https"),
        "subdomain_count": subdomain_count,
        "path_length": len(path),
        "query_length": len(query),
        "suspicious_word_count": suspicious_word_count,
        "entropy": shannon_entropy(hostname),
        "domain_entropy": shannon_entropy(domain),
        "brand_impersonation": has_brand_impersonation(hostname, registered_domain),
        "domain_length": len(domain),
        "domain_hyphen_count": domain.count("-"),
        "domain_digit_count": sum(c.isdigit() for c in domain),
        "domain_vowel_ratio": vowel_ratio,
        "max_consonant_run": max_consonant_run(domain),
    }

    return features