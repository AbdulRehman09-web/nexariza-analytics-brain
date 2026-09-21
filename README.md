# Week 6 — Capstone: Nexariza AI Analytics Brain 📊
**Nexariza AI | AI/ML Internship**

## Overview
The capstone project: a unified AI Business Intelligence system combining engagement forecasting, anomaly detection, lead scoring, and sentiment analysis — brought together in one interactive dashboard. This is the culmination of every skill built across the internship: EDA (Week 1), classification + explainability (Week 2), NLP (Week 3), computer vision (Week 4), and deployment practices (Week 5).

## Components

### 1. Engagement Forecasting (Prophet + ARIMA)
365 days of synthetic daily engagement data (realistic trend, weekly seasonality, and two injected anomalies — a viral spike and a multi-day drop) forecast 30 days ahead using both Prophet and ARIMA(5,1,1) for comparison.

### 2. Anomaly Detection & Alerts
A rolling 14-day z-score flags days where engagement deviates more than a configurable threshold from its recent baseline — **verified to correctly catch both injected anomalies** with no labeled training data required.

### 3. Lead Scoring (XGBoost)
2,000 synthetic leads with realistic B2B conversion signal (demo requests, budget confirmation, decision-maker contact, lead source, engagement metrics) — **ROC-AUC ~0.78**, a genuine, non-inflated result. Verified with a strong lead (referral + demo + budget confirmed → 98.9% conversion probability) vs. a weak lead (cold outreach, no engagement → 0.03%).

### 4. Sentiment Analysis (VADER)
700 synthetic customer mentions/reviews across 5 sources, classified with VADER lexicon-based sentiment — **~78% accuracy** against ground truth, an honest result for a zero-training lexicon approach (most errors are flat, factual statements read as mildly positive, a known VADER characteristic).

### 5. Interactive Dashboard
`dashboard.py` (Streamlit + Plotly) ties all four together: a forecast chart with confidence bands and an adjustable-sensitivity alert feed, an interactive lead-scoring tool with feature-importance breakdown, a filterable sentiment monitor with trend-over-time, and an executive overview tab.

## Files
- `week6_capstone.ipynb` — full executed notebook (all 4 components, real outputs)
- `generate_data.py` — synthetic dataset generator (engagement, leads, mentions)
- `dashboard.py` — combined Streamlit + Plotly BI dashboard
- `lead_scoring_model.pkl`, `lead_feature_names.pkl` — trained lead-scoring model
- `engagement_timeseries.csv`, `leads_dataset.csv`, `mentions_dataset.csv` — datasets
- Result plots: forecast comparison, anomaly detection, lead scoring metrics, sentiment breakdown
- `requirements.txt`, `linkedin_case_study.md`, `technical_blog_post.md`

## How to Run

**Notebook:**
```bash
pip install -r requirements.txt
jupyter notebook week6_capstone.ipynb
```

**Dashboard:**
```bash
pip install -r requirements.txt
streamlit run dashboard.py
```

## Honest Notes on Realism
- All three datasets are synthetic — generated with genuine underlying signal-plus-noise structure (not perfectly separable, not random), so the reported metrics reflect real model performance on a realistic problem, not an inflated toy result.
- VADER and the rolling z-score anomaly detector require zero training — they're included specifically because they're strong, explainable baselines a real team could ship immediately, with ML upgrades (fine-tuned sentiment model, learned anomaly detector) as natural next steps.
- Prophet and ARIMA are both real, fitted models — not templated outputs.

## Path to Production
- Replace synthetic engagement/lead/mention data with real Nexariza analytics exports, CRM data, and social listening feeds.
- Swap VADER for the Week 3 BERT fine-tuning pipeline once labeled sentiment data is available.
- Wrap the lead-scoring model in the Week 5 FastAPI pattern for real-time scoring from a CRM webhook.
- Deploy the dashboard via Streamlit Community Cloud, Hugging Face Spaces, or the Week 5 Docker pattern.

## Wrapping Up
This concludes the 6-week Nexariza AI AI/ML Internship: from a first EDA notebook (Week 1) to a full business intelligence system combining classical ML, NLP, computer vision, MLOps, and time-series forecasting (Week 6). Each week's README documents what was genuinely built, tested, and verified — and, just as importantly, is upfront about sandbox limitations (no Hugging Face Hub access, no Docker daemon, no live deployment) so nothing here overstates what was actually run.
