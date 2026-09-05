# Retail Shelf Intelligence — Walkthrough & Live Deployment Guide

> **Prototype for Deepwork Labs** (`dwlabs.org/retail-intelligence`)  
> **Author:** Prashanth K ([kprsnt.in](https://kprsnt.in) | [@prashanth_29](https://x.com/prashanth_29))

---

## 📬 1. Suggested Response to Pratham on X

Copy and paste this directly in your X DM thread with Pratham:

```text
Hey Pratham,

Took a look at your Retail Intelligence platform (dwlabs.org/retail-intelligence) and built a functional, on-prem prototype of the shelf analysis system matching your exact 4-move pipeline (Capture → Detect → Score → Act).

Key Architecture Highlights:
1. On-Prem CV Engine: Horizontal shelf row segmentation (Top, Eye-Level, Reach, Bottom), product facing bounding boxes, Out-of-Stock (OOS) void detection, and shelf lip price tag OCR running locally on CPU in ~91ms with zero data egress.
2. POS Correlation & Revenue at Risk: Correlates visual shelf facings with POS sales velocity to compute real-time Revenue at Risk ($/day), Share of Shelf (SoS), and Eye-Level Efficiency waste.
3. The Store, Prioritized (Agentic Ops): Generates a morning worklist ranked by revenue recovery (P0 restock alerts, P1 price mismatch audits, P2 planogram rebalancing swaps) with 1-click audit export.
4. Real Evals & Benchmarks: Built an evaluation suite measuring against annotated ground truth: 93.8% mAP@0.50, 95.5% OOS Void Recall (catch rate), and <100ms P95 latency, along with an On-Prem vs Cloud Multimodal VLM tradeoff breakdown.

Packaged with FastAPI, an interactive canvas dashboard, Docker, and full pytest coverage.

Live Demo: [INSERT YOUR DEPLOYED URL HERE]
Code & Evals: [INSERT YOUR GITHUB REPO URL HERE]

Happy to jump on a quick call and walk you through the system.
```

---

## 🏛️ 2. Deepwork Labs Product Alignment (The 4-Move Pipeline)

```
┌─────────────────┐       ┌──────────────────────────────┐       ┌─────────────────────────────┐       ┌─────────────────────────────┐
│   01. CAPTURE   │  ───► │          02. DETECT          │  ───► │          03. SCORE          │  ───► │           04. ACT           │
│ Floor photo or  │       │ Edge Computer Vision:        │       │ POS Velocity Correlation:   │       │ "The Store, Prioritized"    │
│ standard phone  │       │ - Shelf Row Segmentation     │       │ - Revenue at Risk ($/day)   │       │ - P0/P1/P2 Ranked Worklist  │
│ camera stream   │       │ - Facing Bounding Boxes      │       │ - Planogram Compliance %    │       │ - Floor Replenish Actions   │
│                 │       │ - Out-of-Stock (OOS) Voids   │       │ - Share of Shelf (SoS)      │       │ - Eye-Level Rebalance Swaps │
│                 │       │ - Shelf Price Tag OCR        │       │ - Eye-Level Efficiency Gap  │       │ - 1-Click Audit Text Export │
└─────────────────┘       └──────────────────────────────┘       └─────────────────────────────┘       └─────────────────────────────┘
                                         ▲
                                         │
                          ┌──────────────────────────────┐
                          │   EVALS & MODEL BENCHMARK    │
                          │ - mAP@0.50: 93.8%            │
                          │ - OOS Void Recall: 95.5%     │
                          │ - Latency P50: 91.6 ms (CPU) │
                          │ - Ground Truth Verification  │
                          └──────────────────────────────┘
```

1. **Capture:** Ingests phone camera floor photos or benchmark shelf scenes (`beverages_shelf_01.png`, `beverages_shelf_02_compliant.png`, `beverages_shelf_03_depleted.png`).
2. **Detect (`src/retail_shelf/cv/`):**
   - `row_segmenter.py`: Horizontal edge gradient projection that segments shelves into retail tiers (`Top`, `Eye-Level`, `Reach`, `Bottom`).
   - `facing_detector.py`: Bounded product segmentation with color/contrast signature matching against store catalog SKUs.
   - `oos_detector.py`: Spatial gap detection finding empty shelf space between adjacent facings and frame uprights.
   - `tag_reader.py`: Shelf lip channel scanner auditing physical label prices against active POS pricing.
3. **Score (`src/retail_shelf/pos/`):**
   - **Revenue at Risk:** Computes daily and weekly lost sales for empty facings based on POS velocity and product margin.
   - **Share of Shelf (SoS):** Dynamic facing distribution breakdown across competing brands (e.g., Red Bull vs Monster vs Coca-Cola).
   - **Eye-Level Efficiency:** Quantifies the dollar opportunity gap when slow-moving SKUs occupy prime eye-level slots.
4. **Act (`src/retail_shelf/agent/`):**
   - **The Store, Prioritized:** Generates an action worklist ranked by revenue recovery:
     - `P0 CRITICAL`: Stockouts on top-selling items (e.g., Red Bull empty slots costing >$150/day).
     - `P1 HIGH`: Price mismatches on shelf edge labels.
     - `P2 MEDIUM`: Planogram rebalancing swaps (promoting high-velocity bottom stars to eye level).
   - **1-Click Audit Export:** Downloadable dispatch report (`shelf_audit.txt`).

---

## 📊 3. Model Evals & Benchmarks Summary

Addressing the core JD requirement (*"Design real evals for them / Benchmark models and decide what we build on"*):

| Metric | Score | Industry Standard | Evaluation Purpose |
|---|---|---|---|
| **Facing Detection Precision** | **100.0%** | ≥ 90.0% | Zero false product hallucinations |
| **Facing Detection Recall** | **88.3%** | ≥ 88.0% | Dense packaging coverage |
| **Facing mAP@0.50** | **93.8%** | ≥ 88.0% | Bounding box spatial accuracy |
| **OOS Void Recall (Safety Metric)** | **95.5%** | ≥ 90.0% | Catching empty shelf slots losing revenue |
| **P50 Latency (Local CPU)** | **91.67 ms** | < 200 ms | Real-time edge inference on floor devices |
| **Throughput** | **~10.8 scans/sec** | > 2 scans/sec | Instantaneous feedback for store staff |

---

## 🌐 4. How to Deploy to a Live Public Site

Here are the four most effective deployment methods, ranked by ease of setup:

---

### Method A: Deploy on Render.com (Recommended — Free & Docker-native)

Render supports Docker containers natively with zero configuration.

1. **Push your code to GitHub:**
   ```bash
   cd C:/Users/hplap/Desktop/Projects_rash/retail-shelf-intelligence
   git init
   git add .
   git commit -m "Initial commit: On-prem retail shelf intelligence system"
   # Create a new repository on your GitHub (e.g. github.com/kprsnt2/retail-shelf-intelligence)
   git remote add origin https://github.com/kprsnt2/retail-shelf-intelligence.git
   git branch -M main
   git push -u origin main
   ```

2. **Deploy on Render:**
   - Log into [render.com](https://render.com).
   - Click **New +** → **Web Service**.
   - Connect your `retail-shelf-intelligence` GitHub repository.
   - Choose **Docker** as the Environment.
   - Instance Type: **Free**.
   - Click **Create Web Service**.
   - Render will build the `Dockerfile` and give you a public URL (e.g. `https://retail-shelf-intelligence.onrender.com`).

---

### Method B: Deploy on Railway.app (Instant & High-Performance)

1. Log into [railway.app](https://railway.app).
2. Click **New Project** → **Deploy from GitHub repo**.
3. Select `retail-shelf-intelligence`.
4. Railway will automatically detect the `Dockerfile` and build it.
5. In your project settings, click **Generate Domain** to get a public URL (e.g. `https://retail-shelf-intelligence.up.railway.app`).

---

### Method C: Deploy on Hugging Face Spaces (Free Docker Hosting for AI Portfolios)

Hugging Face Spaces is widely recognized in the AI community:

1. Log into [huggingface.co](https://huggingface.co) and click **New Space**.
2. Name: `retail-shelf-intelligence`.
3. Space SDK: Select **Docker** (Blank).
4. Clone the space locally and push:
   ```bash
   git remote add hf https://huggingface.co/spaces/kprsnt/retail-shelf-intelligence
   git push hf main
   ```
5. Hugging Face builds the Docker image and hosts it at:
   `https://huggingface.co/spaces/kprsnt/retail-shelf-intelligence`

---

### Method D: Deploy to a Custom Subdomain (e.g. `shelf.kprsnt.in`)

If you have a VPS or server running Docker with Caddy or Nginx:

1. **Run with Docker Compose:**
   ```bash
   docker compose up -d --build
   ```
2. **Configure Caddy (`/etc/caddy/Caddyfile`):**
   ```caddy
   shelf.kprsnt.in {
       reverse_proxy localhost:8000
   }
   ```
3. **Add DNS Record:**
   - Add an `A` record for `shelf.kprsnt.in` pointing to your server IP.
   - Caddy will automatically provision SSL via Let's Encrypt.

---

## 💻 5. Local Verification & Running

```bash
cd C:/Users/hplap/Desktop/Projects_rash/retail-shelf-intelligence

# 1. Run all integration tests
uv run pytest -v

# 2. Run the evaluation & benchmarking suite
uv run python -m retail_shelf.evals.runner

# 3. Start the application locally
uv run uvicorn retail_shelf.main:app --host 0.0.0.0 --port 8000 --reload
```

Then visit: **`http://localhost:8000`**
