# ClaimSense — Cotiviti AI Internship Assessment

**Candidate:** Kshamaa Suresh  
**Institution:** Columbia University, MS Data Science (December 2026)  
**Topic:** Clinical Decision Making and Pattern Recognition in Health Care  

---

## ⚠️ Disclaimer

This is a research and engineering demonstration prototype. All claims data is fully synthetic — no real patient data, PHI, or actual claims records are used anywhere in this project. This system is not validated for clinical or payment decision use and should not be deployed in production without appropriate clinical validation, regulatory review, and human oversight.

---

## What This Project Is

ClaimSense demonstrates how Agentic Generative AI can close the gap between anomaly detection and clinical action in healthcare payment integrity — not just flagging suspicious claims, but reasoning about why they are suspicious and recommending the right next step.

The core insight: existing tools surface signals. What they rarely do is explain those signals in actionable terms, or recommend the specific next step a reviewer should take. ClaimSense demonstrates that gap being closed.

**The CMS estimated $31 billion in improper Medicare payments in FY2023.** The bottleneck is rarely detection — it is the distance between a flagged signal and a reviewed, actioned decision.

---

## Live Demo

**→ [Open ClaimSense on HuggingFace Spaces](https://huggingface.co/spaces/kshamaasuresh/ClaimSense)**

## Video Presentation

The video presentation has also been uploaded on youtube for clear viewing if required.
**→ [Open ClaimSense recording on Youtube]([https://huggingface.co/spaces/kshamaasuresh/ClaimSense](https://youtu.be/Roi1XCxqhKQ?is=7VeslUN_WsPBcBt7))**

The app runs on HuggingFace's free CPU tier. All four pages are interactive:

- **📊 Claims Dashboard** — risk distribution, anomaly type breakdown, color-coded claims table
- **🔍 Anomaly Deep Dive** — select any flagged claim, run AI chain-of-reasoning analysis
- **⚡ Action Center** — batch analysis across all flagged claims with recommended actions
- **💬 Claims Q&A** — conversational assistant grounded in the analyzed claims dataset

---

## Deliverables

| Deliverable | Description |
|---|---|
| `Cotiviti_Report_KshamaaSuresh.docx` | Two-page Word report with APA citations + bibliography (Topic 2) |
| `ClaimSense_Cotiviti.pptx` | PowerPoint presentation — 10 slides covering problem, solution, live demo, architecture, and strategic recommendations |
| `Video_Presentation.mp4` | Video presentation with screenshare of working POC |
| `app.py` | Streamlit application — all four pages |
| `agent.py` | Groq LLM reasoning engine with RAGAS-style faithfulness evaluation |
| `data_generator.py` | Synthetic claims dataset generator — 120 claims, 6 injected anomaly types |
| `detector.py` | Rule-based anomaly detection engine — 6 detectors, confidence scoring, risk levels |
| `requirements.txt` | Python dependencies |

---

## The Three-Stage Pipeline

**Stage 1 — Anomaly Detection**

A rule-based Python engine flags claims across six anomaly categories:

- Diagnosis-procedure mismatch (e.g., upper respiratory infection billed with total knee arthroplasty)
- Duplicate claims (same patient, procedure, and date of service)
- High-frequency billing (provider billing high-cost procedures at unusual frequency)
- Amount outliers (billed amount more than 2.5 standard deviations above the procedure mean)
- Future dates of service (impossible claim dates)
- Unbundling patterns (separately billing components that should be bundled)

Each flagged claim receives a confidence score (0–1) and a risk level (Normal / Low / Medium / High / Critical).

**Stage 2 — AI Chain-of-Reasoning**

For each flagged claim, a Groq-powered LLM agent (llama-3.3-70b-versatile) generates:

- A chain-of-reasoning explanation grounded in the specific claim data
- A recommended action (auto-deny, escalate, request documentation, approve)
- A risk assessment summary
- A RAGAS-style faithfulness score verifying every claim in the explanation is grounded in actual claim data rather than hallucinated

**Stage 3 — Claims Q&A**

A multi-turn conversational assistant grounded in the analyzed claims dataset. Physicians, reviewers, and patient advocates can ask natural language questions and receive answers anchored in what the system has actually found.

---

## Technical Stack

| Component | Technology |
|---|---|
| Synthetic data generation | Python, pandas, numpy |
| Anomaly detection | Rule-based engine (6 detectors) |
| LLM reasoning | Groq API — llama-3.3-70b-versatile |
| Faithfulness evaluation | RAGAS-style claim-level verification |
| Frontend | Streamlit + Plotly |
| Hosting | HuggingFace Spaces (CPU free tier) |

---

## Strategic Recommendations for Cotiviti

The written report proposes three investments:

**1. Agentic Payment Integrity Platform** — Move beyond flag-and-review toward agents that reason, explain, and recommend action at the point of adjudication, with a complete audit trail for every decision.

**2. Faithfulness-First Evaluation Standard** — Systematic RAGAS-style verification that every AI-generated clinical or payment decision output is grounded in retrieved policy rather than hallucinated. Market-differentiating in a space where trust is the primary barrier to adoption.

**3. Clinical AI Governance Framework** — Model drift monitoring, bias auditing, human oversight thresholds, and regulatory compliance documentation before scaling agentic decision-making across client health plans.

---

## About the Candidate

Kshamaa Suresh is an MS Data Science student at Columbia University graduating December 2026, with two years of production AI engineering experience at Ecolab — building Claude API systems processing 10M+ records daily, ETL pipelines on Databricks and Snowflake, and automated evaluation frameworks. She also serves as a Graduate Research Assistant at Barnard's Neural Computation Lab with a published paper at CSHL Neuronal Circuits 2026 and an accepted poster at CCN 2026.

**Portfolio:** [kshamaas.github.io/Portfolio](https://kshamaas.github.io/Portfolio)  
**GitHub:** [github.com/KshamaaS](https://github.com/KshamaaS)  
**Email:** ks4423@columbia.edu
