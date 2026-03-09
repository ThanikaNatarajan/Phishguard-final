# 🛡️ PhishGuard AI — Phishing Detection with Screenshot Preview

A full-stack AI-powered phishing URL detector with a safe website screenshot preview feature.

---

## 📁 Project Structure

```
phishguard/
├── frontend/
│   └── index.html          ← Full React app (open in browser, works standalone)
├── backend/
│   ├── app.py              ← FastAPI server
│   └── .env.example        ← Copy to .env and add your API key
└── model/
    ├── train_model.py      ← ML training script
    ├── phishguard_model.pkl← Trained model (pre-generated)
    └── model_metadata.json ← Training stats
```

---

## ⚡ Quick Start

### Option A — Frontend Only (Demo Mode)
Just open `frontend/index.html` in your browser. The app works immediately using a built-in client-side URL analyzer. Screenshot preview will show setup instructions until you configure the backend.

### Option B — Full Stack with Screenshots

**Step 1: Get a free ScreenshotOne API key**
- Go to https://screenshotone.com and sign up (free tier = 100 screenshots/month)
- Copy your Access Key

**Step 2: Configure the backend**
```bash
cd backend
cp .env.example .env
# Edit .env and paste your key:  SCREENSHOT_API_KEY=your_key_here
```

**Step 3: Install Python packages**
```bash
pip install fastapi uvicorn scikit-learn joblib requests python-dotenv python-multipart aiofiles
```

**Step 4: Train the AI model (already done, but re-run anytime)**
```bash
cd model
python train_model.py
# To use a real dataset: python train_model.py path/to/dataset.csv
# Download real data from: https://archive.ics.uci.edu/dataset/967
```

**Step 5: Start the backend**
```bash
cd backend
uvicorn app:app --reload --port 8000
```

**Step 6: Open the frontend**
Open `frontend/index.html` — it will auto-connect to the backend and screenshots will be live.

---

## 📸 How the Screenshot Feature Works

When you scan a URL, PhishGuard requests a screenshot of the page **from the server**, not your browser. This means:
- ✅ Your browser **never connects** to the suspicious site
- ✅ You can **see what the site looks like** (fake login pages, spoofed brands, etc.)
- ✅ The screenshot is rendered safely and embedded in the report
- ✅ Great for showing non-technical users visual proof of phishing

The screenshot is taken by [ScreenshotOne](https://screenshotone.com), a headless browser API.

---

## 🧠 AI Model Features (34 extracted signals)

| Category | Features |
|---|---|
| Structure | URL length, hostname length, subdomain count, path depth |
| Security | HTTPS, raw IP address, suspicious TLD, punycode encoding |
| Content | Keyword count (login/verify/secure), brand impersonation |
| Entropy | Domain randomness score (auto-generated phishing domains score high) |
| Tricks | @-sign redirect, double-slash redirect, hex encoding |

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| POST | `/scan` | Scan a single URL |
| POST | `/scan/bulk` | Scan up to 50 URLs |
| GET | `/history` | Get scan history |
| GET | `/stats` | Aggregate statistics |
| DELETE | `/history` | Clear all history |
| PATCH | `/history/{id}/block` | Toggle block status |

---

## 🧩 Chrome Extension Integration

PhishGuard integrates with the **Web Site Blocker** extension:
- Install: https://chromewebstore.google.com/detail/web-site-blocker/aoabjfoanlljmgnohepbkimcekolejjn
- When PhishGuard flags a URL as phishing, click **"🚫 Block with Extension"** in the report to add it to your blocklist instantly.

---

## 🏋️ Training with a Real Dataset

For production-level accuracy, download a real phishing dataset:
- **PhiUSIIL** (recommended): https://archive.ics.uci.edu/dataset/967 — 235,000 labeled URLs
- **Kaggle**: Search "phishing URL detection dataset"

Then run:
```bash
python model/train_model.py your_dataset.csv
```

The script auto-detects the URL and label columns.
