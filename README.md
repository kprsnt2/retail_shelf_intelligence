---
title: Retail Shelf Intelligence
emoji: 🛒
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 6.26.0
app_file: app.py
short_description: Edge CV & Agentic Commerce for On-Prem Shelves
pinned: false
---

# Retail Shelf Intelligence — On-Prem Edge CV & Agentic Commerce

> **Built for Deepwork Labs** (`dwlabs.org/retail-intelligence`)  
> *Assignment Prototype: On-prem retail intelligence that reads any shelf photo into per-facing, per-row, per-product sales with zero cloud egress.*

---

## 🏛️ System Overview & Architecture

Retail floor execution has traditionally been a blind spot: Point of Sale (POS) data tells you what sold, but cannot identify when a planogram broke, when prime eye-level slots were wasted on slow movers, or when empty slots silently leaked thousands of dollars in lost revenue.

This system implements Deepwork Labs' exact **4-Move Pipeline** (`Capture` → `Detect` → `Score` → `Act`) running entirely **on local store hardware** with zero external cloud dependencies.

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

---

## 🚀 Key Modules & Capabilities

### 1. Computer Vision Pipeline (`src/retail_shelf/cv/`)
- **Row Segmenter (`row_segmenter.py`):** Horizontal contrast energy gradient projection that cuts shelf layers into standard retail merchandising tiers (`Top`, `Eye-Level`, `Reach`, `Bottom`).
- **Facing Detector (`facing_detector.py`):** Bounded product segmentation with multi-can subdivision and visual color/contrast feature extraction matched against catalog SKUs.
- **OOS Void Detector (`oos_detector.py`):** Spatial gap detection between adjacent product facings and boundary uprights. Pinpoints empty slots costing sales.
- **Price Tag Reader (`tag_reader.py`):** Shelf lip channel scanner auditing physical label prices against active POS pricing.

### 2. POS Correlation & Financial Scoring (`src/retail_shelf/pos/`)
- **Revenue at Risk ($/day & $/wk):** Calculates actual financial loss per missing facing:
  $$\text{Daily Loss} = \left(\frac{\text{Daily Velocity}}{\text{Target Facings}} \times \text{Missing Facings}\right) \times \text{POS Price} \times \text{Goodwill Factor}$$
- **Share of Shelf (SoS):** Dynamic facing distribution breakdown across competing brands (e.g. Red Bull vs Monster vs Coke).
- **Eye-Level Efficiency Rating:** Identifies **"Eye-Level Waste"**—quantifying the dollar gap when slow-moving SKUs occupy high-traffic eye-level tiers while fast movers are relegated to the bottom.

### 3. Agentic Store Operations ("The Store, Prioritized") (`src/retail_shelf/agent/`)
- **Action Prioritizer (`ops_agent.py`):** Translates raw telemetry into an action worklist ranked by revenue recovery:
  - `P0 CRITICAL`: Out-of-stock on high-velocity top sellers (e.g. Red Bull / Monster empty facings costing >$150/day).
  - `P1 HIGH`: Price tag discrepancies where shelf label is lower than POS database price.
  - `P2 MEDIUM`: Planogram rebalancing swaps (promote bottom-tier star to eye-level).
- **Executive Synthesis:** Autonomous morning brief providing store managers with immediate floor guidance and recoverable dollar totals.
- **1-Click Audit Export:** Downloadable operations dispatch file (`shelf_audit.txt`).

### 4. Real Model Evals & Benchmarks (`src/retail_shelf/evals/`)
Addressing the core JD requirement (*"Design real evals for them / Benchmark models and decide what we build on"*):
- **Safety Metric Priority:** In retail shelf monitoring, **OOS Void Recall** (catch rate) is the load-bearing metric—a false negative directly bleeds daily margin.
- **Automated Runner (`runner.py`):** Evaluates detections against pixel ground truth across multiple test scenes:
  - Facing Detection Precision: **100.0%**
  - Facing Detection Recall: **88.3%**
  - Facing mAP@0.50: **93.8%**
  - **OOS Void Recall (Stockout Catch Rate): 95.5%**
  - Latency: **P50: 91.67 ms**, **P95: 100.66 ms** on standard CPU.

---

## ⚡ Performance: On-Prem vs Cloud Multimodal VLM

| Architectural Dimension | Local On-Prem Pipeline (This System) | Cloud Multimodal VLM (GPT-4o / Gemini) |
|---|---|---|
| **Data Sovereignty** | **100% On-Premise** (Zero video or sales egress) | Egresses photos & financial records to cloud |
| **Inference Cost / 1,000 Scans** | **$0.00** (Runs on local electricity) | ~$25.00 – $40.00 in API token fees |
| **Inference Latency** | **< 100 ms** (Real-time edge execution) | 4,000 ms – 9,000 ms (Network + queue delay) |
| **Offline Resilience** | **Full resilience** during retail network drops | Completely non-operational if ISP fails |
| **Determinism & Evals** | Bounded IoU, exact bboxes, predictable behavior | Semantic hallucinations on dense packaging |

---

## 🛠️ Quickstart Guide

### Option 1: Run Locally with `uv` (Recommended)

```bash
# Clone and navigate to repository
cd retail-shelf-intelligence

# Run automated tests
uv run pytest -v

# Run the benchmark suite to generate BENCHMARK_REPORT.md
uv run python -m retail_shelf.evals.runner

# Launch the interactive on-prem web application
uv run uvicorn retail_shelf.main:app --host 0.0.0.0 --port 8000 --reload
```

Open your browser to: **`http://localhost:8000`**

### Option 2: Run with Docker / Docker Compose

```bash
docker compose up --build
```

---

## 📡 REST API Specification

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | System health, version, and on-prem engine verification |
| `GET` | `/api/shelf/samples` | List available pre-packaged benchmark scenes |
| `GET` | `/api/shelf/samples/{filename}` | Serve benchmark shelf image |
| `POST` | `/api/shelf/analyze` | Full end-to-end shelf scan (accepts file upload or `sample_id`) |
| `GET` | `/api/pos/catalog` | Active store SKU catalog with costs, prices, and velocities |
| `GET` | `/api/pos/planograms` | Target store planograms |
| `GET` | `/api/agent/report/export` | Download one-click store ops audit text report |
| `POST` | `/api/eval/run` | Execute on-demand benchmark evaluation runs |
| `GET` | `/api/eval/report` | Retrieve latest benchmark report markdown |

---

## 📂 Repository Layout

```
retail-shelf-intelligence/
├── src/
│   └── retail_shelf/
│       ├── main.py                 # FastAPI application & server entrypoint
│       ├── config.py               # Merchandising thresholds, costs, and paths
│       ├── models/                 # Pydantic schemas (Shelf, POS, Analytics, Agent)
│       ├── cv/                     # On-Prem Computer Vision Pipeline
│       │   ├── row_segmenter.py    # Shelf row layer segmentation
│       │   ├── facing_detector.py  # Product package facing detection
│       │   ├── oos_detector.py     # Out-of-Stock void gap finder
│       │   ├── tag_reader.py       # Shelf lip price tag auditor
│       │   └── vision_engine.py    # Unified CV execution engine
│       ├── pos/                    # Retail POS Integration & Scoring
│       │   ├── catalog.py          # Store SKU catalog manager
│       │   ├── planogram.py        # Target planogram compliance checker
│       │   └── pos_correlator.py   # Revenue at Risk & SoS analytics engine
│       ├── agent/                  # Agentic Store Operations
│       │   ├── rebalancing.py      # Eye-level waste & SKU relocation engine
│       │   └── ops_agent.py        # Prioritized morning worklist generator
│       ├── evals/                  # Model Benchmarking & Evals
│       │   ├── metrics.py          # IoU, mAP@0.50, and OOS Void Recall formulas
│       │   └── runner.py           # Automated evaluation & benchmark compiler
│       ├── api/                    # REST route controllers
│       └── static/                 # Deepwork-styled Interactive UI
│           ├── index.html          # Canvas bbox viewer & dashboard
│           ├── styles.css          # Deepwork Labs warm editorial theme
│           └── app.js              # Canvas hover tooltips, filter toggles, tabs
├── data/                           # Catalogs, planograms, and sample shelf scenes
├── tests/                          # End-to-end integration and evaluation tests
├── BENCHMARK_REPORT.md             # Compiled evaluation report
├── Dockerfile                      # On-prem container definition
├── docker-compose.yml              # Local container orchestration
└── pyproject.toml                  # Python package configuration
```

---

## 👤 Author
**Prashanth K**  
- Portfolio: [kprsnt.in](https://kprsnt.in)  
- GitHub: [github.com/kprsnt2](https://github.com/kprsnt2)
- X / Twitter: [@prashanth_29](https://x.com/prashanth_29)
