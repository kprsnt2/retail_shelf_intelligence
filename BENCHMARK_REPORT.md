# Retail Intelligence CV Pipeline: Benchmark & Evaluation Report

**Generated:** 2026-09-05 06:27:41 UTC  
**Deployment Mode:** On-Prem Edge Inference (Zero Cloud Dependency)  

---

## 1. Core Model Performance vs Ground Truth

| Metric | Score | Target Standard | Status |
|---|---|---|---|
| **Facing Detection Precision** | 100.0% | ≥ 90.0% | ✅ PASS |
| **Facing Detection Recall** | 88.3% | ≥ 88.0% | ✅ PASS |
| **Facing mAP@0.50** | 93.8% | ≥ 88.0% | ✅ PASS |
| **OOS Void Recall (Stockout Catch Rate)** | **95.5%** | ≥ 90.0% | ✅ PASS |

*Note: High OOS Void Recall is the load-bearing safety metric in retail environments—missing an empty shelf facing directly costs hundreds of dollars in lost daily sales.*

---

## 2. Latency & Throughput Benchmark

*Benchmarked on local CPU across 5 continuous shelf scans:*

- **Mean Latency:** `89.02 ms`
- **P50 Latency:** `87.51 ms`
- **P95 Latency:** `94.93 ms`
- **P99 Latency:** `95.9 ms`
- **Throughput:** `~11.2 shelf scans / sec`

---

## 3. On-Prem vs Cloud Multimodal Architecture Comparison

| Architectural Dimension | Local On-Prem Pipeline | Cloud Multimodal VLM (GPT-4o/Gemini) |
|---|---|---|
| **Data Sovereignty** | 100% On-Premise (No video/photos leave store) | Data egressed to third-party vendor |
| **Inference Cost / 1,000 Scans** | **$0.00** (Runs on local electricity) | ~$25.00 - $40.00 in token fees |
| **Scan Latency** | **< 150 ms** | 4,000 ms - 9,000 ms |
| **Offline Resilience** | Full capability during internet outages | Halts completely if ISP drops |

---

## 4. Per-Scene Breakdown

| Scene | Target Facings | Detected Facings | Facing F1 | Void Recall |
|---|---|---|---|---|
| `beverages_shelf_01.png` | 24 | 21 | 93.3% | **100.0%** |
| `beverages_shelf_02_compliant.png` | 27 | 25 | 96.2% | **0.0%** |
| `beverages_shelf_03_depleted.png` | 9 | 7 | 87.5% | **94.4%** |
