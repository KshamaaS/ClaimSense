---
title: ClaimSense
emoji: 🏥
colorFrom: purple
colorTo: indigo
sdk: streamlit
sdk_version: 1.38.0
app_file: app.py
pinned: false
---

# ClaimSense — Agentic Clinical Claims Intelligence

**Built for the Cotiviti AI/Healthcare Informatics Internship Assessment**
**Author:** Kshamaa Suresh | Columbia University MS Data Science

---

## What This Is

ClaimSense is a proof-of-concept agentic clinical claims intelligence platform that demonstrates how AI can close the gap between anomaly detection and clinical action — not just flagging suspicious claims, but reasoning about *why* they are suspicious and recommending what to do next.

This mirrors the core challenge in healthcare payment integrity: existing tools surface signals. What they rarely do is explain those signals in actionable terms, or recommend the specific next step a reviewer should take. ClaimSense demonstrates that gap being closed.

---

## The Three-Stage Pipeline

### Stage 1 — Synthetic Claims Generation
120 realistic medical claims are generated with injected anomalies across six categories:
- Diagnosis-procedure mismatch (e.g., URI diagnosis + knee replacement procedure)
- Duplicate claims (same patient, procedure, and date of service)
- High-frequency billing (provider billing high-cost procedures at unusual frequency)
- Amount outliers (billed amount > 2.5 standard deviations above procedure mean)
- Future dates of service (impossible claim dates)
- Unbundling patterns (separately billing components that should be bundled)

### Stage 2 — Rule-Based Anomaly Detection
A multi-rule detection engine flags claims and assigns:
- A confidence score (0–1)
- The specific rules triggered
- A risk level (Normal / Low / Medium / High / Critical)

### Stage 3 — Agentic Reasoning (Groq LLM)
For each flagged claim, a Groq-powered LLM agent provides:
- **Chain-of-reasoning explanation** grounded in the actual claim data
- **Recommended action** (auto-deny, escalate, request documentation, approve)
- **Risk assessment** summary
- **Faithfulness score** — a RAGAS-style check verifying the explanation is grounded in the claim data rather than hallucinated

---

## Setup

### Environment Variables
Add your Groq API key as a HuggingFace Space secret:
```
GROQ_API_KEY = your_key_here
```
Get a free key at [console.groq.com](https://console.groq.com)

### Local Development
```bash
pip install -r requirements.txt
export GROQ_API_KEY=your_key_here
streamlit run app.py
```

---

## Technical Stack

| Component | Technology |
|---|---|
| Synthetic data | Python (pandas, numpy) |
| Anomaly detection | Rule-based engine (6 detectors) |
| LLM reasoning | Groq API (llama3-70b-8192) |
| Faithfulness evaluation | RAGAS-style claim-level verification |
| Frontend | Streamlit + Plotly |
| Hosting | HuggingFace Spaces (CPU free tier) |

---

## Why This Matters for Cotiviti

The CMS estimated $31 billion in improper Medicare payments in FY2023. The bottleneck is rarely detection — it is the gap between a flagged signal and a reviewed, actioned decision. ClaimSense demonstrates how agentic AI can compress that gap from days to seconds, while maintaining a full audit trail of reasoning for every decision.

---

## Disclaimer
This is a research and engineering demonstration prototype. All patient data is fully synthetic. No real PHI is used or generated. This system is not intended for clinical or payment decision use without appropriate validation, regulatory review, and human oversight.
