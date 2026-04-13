# 🛡️ PhishGuard AI

### Intelligent Phishing URL Detection & Risk Analysis Engine

PhishGuard AI is a real-world phishing detection system that combines **machine learning**, **URL intelligence**, and **security heuristics** to identify malicious links with high accuracy and clear explanations.

Built with a production-oriented mindset, it goes beyond simple classification by providing **interpretable risk scores**, **attack pattern detection**, and **secure backend architecture**.

---

## ⚡ Key Features

* 🔍 **Real-time URL Scanning**
  Analyze any URL instantly with a trained ML model + rule-based intelligence

* 🧠 **Hybrid Detection Engine**
  Combines:

  * Machine Learning (Random Forest)
  * Heuristic Security Rules
  * Domain Intelligence

* 🎯 **High Accuracy Detection**

  * Handles tricky phishing like:

    * `google.com.secure-login.xyz`
    * `paypal.com.login.verify.ru`

* 📊 **Explainable Results**

  * Risk score (0–100%)
  * Threat classification (Safe / Suspicious / Critical)
  * Human-readable explanations of WHY a URL is dangerous

* 🖼️ **Live Website Preview (Secure)**

  * Screenshot rendering via proxy
  * SSRF-protected backend

* 📦 **Bulk URL Scanning**

  * Scan multiple URLs simultaneously using real backend logic (no fake/demo fallback)

---

## 🧠 How It Works

### 1. Feature Extraction (Shared System)

All URLs pass through a centralized feature pipeline:

* URL length, entropy, digit ratios
* Subdomain patterns & depth
* Suspicious keywords (login, verify, secure, etc.)
* Domain structure & TLD analysis
* Character distribution patterns

> Defined in: `shared/features.py`

---

### 2. Machine Learning Model

* Model: **Random Forest Classifier**
* Trained on:

  * 🔴 Phishing URLs (PhishTank)
  * 🟢 Legitimate URLs (Tranco Top Domains)
* Balanced dataset with real-world examples

---

### 3. Risk Engine (Hybrid Logic)

Final score = **ML prediction + security heuristics**

Enhancements include:

* Brand impersonation detection
* Suspicious subdomain detection
* Trusted domain soft-adjustment (no blind trust)

---

### 4. Secure Backend Processing

* FastAPI backend
* SSRF protection for screenshot endpoint
* Safe URL handling and validation

---

## 📁 Project Structure

```
phishguard/
│
├── backend/        # FastAPI backend
├── frontend/       # Web UI
├── model/          # Training + dataset pipeline
├── shared/         # Feature extraction logic
│
├── requirements.txt
└── README.md
```

---

## 🚀 Getting Started

### 1. Clone Repo

```
git clone <your-repo-url>
cd phishguard
```

---

### 2. Install Dependencies

```
pip install -r requirements.txt
```

---

### 3. Prepare Dataset (Required)

You need:

* PhishTank URLs (phishing dataset)
* Tranco top domains (legitimate dataset)

Place them in:

```
model/phishtank_urls.csv
model/legit_urls.csv
```

---

### 4. Build Dataset

```
python model/build_dataset.py
```

---

### 5. Train Model

```
python model/train_model.py
```

---

### 6. Run Backend

```
uvicorn backend.app:app --reload
```

---

### 7. Open Frontend

Open:

```
frontend/index.html
```

---

## 🔒 Security Considerations

* SSRF protection implemented in screenshot endpoint
* No blind trust for known domains
* Backend validates all URLs before processing

---

## 📈 Example Results

| URL                         | Risk            |
| --------------------------- | --------------- |
| google.com                  | 0% (Safe)       |
| amazon.com                  | 0% (Safe)       |
| google.com.secure-login.xyz | 90%+ (Critical) |
| paypal.com.login.verify.ru  | 95%+ (Critical) |

---

## 🧠 Tech Stack

* **Backend:** FastAPI
* **Frontend:** Vanilla JS + HTML/CSS
* **ML:** Scikit-learn (Random Forest)
* **Data:** Pandas, real-world datasets

---

## 🎯 Future Improvements

* Model calibration (confidence tuning)
* Hard-negative dataset expansion
* Cloud deployment (API + UI)
* Browser extension integration

---

## 👨‍💻 Author

Built as a full-stack AI security project focusing on:

* real-world applicability
* system design
* ML + security integration

---

## ⭐ Final Note

This is not just a classifier — it's a **phishing intelligence system** designed to reflect how real detection pipelines work.

If you found this useful, consider starring ⭐ the repo.
