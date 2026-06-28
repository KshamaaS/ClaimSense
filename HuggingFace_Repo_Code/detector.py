"""
ClaimSense — Anomaly Detection Engine
Rule-based + statistical detection across six anomaly categories.
Each flagged claim receives a confidence score and a list of triggered rules.
"""

import pandas as pd
import numpy as np
from datetime import datetime
from collections import Counter


def detect_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    """
    Run all anomaly detectors on the claims DataFrame.
    Returns the DataFrame with added columns:
      - flagged: bool
      - confidence: float (0-1)
      - triggered_rules: list of rule names
      - risk_level: str (Low / Medium / High / Critical)
    """
    df = df.copy()
    df["flagged"] = False
    df["confidence"] = 0.0
    df["triggered_rules"] = [[] for _ in range(len(df))]
    df["risk_level"] = "Normal"

    df = _detect_diagnosis_procedure_mismatch(df)
    df = _detect_duplicate_claims(df)
    df = _detect_high_frequency_billing(df)
    df = _detect_amount_outliers(df)
    df = _detect_impossible_dates(df)
    df = _detect_unbundling_patterns(df)

    # Assign risk level based on confidence
    def assign_risk(row):
        if not row["flagged"]:
            return "Normal"
        if row["confidence"] >= 0.85:
            return "Critical"
        elif row["confidence"] >= 0.65:
            return "High"
        elif row["confidence"] >= 0.45:
            return "Medium"
        else:
            return "Low"

    df["risk_level"] = df.apply(assign_risk, axis=1)
    return df


# ── Individual Detectors ──────────────────────────────────────────────────────

# Clinically incoherent dx/procedure pairs
_MISMATCH_PAIRS = {
    ("J06.9", "27447"),   # URI + knee replacement
    ("J06.9", "70553"),   # URI + brain MRI
    ("J18.9", "27447"),   # Pneumonia + knee replacement
    ("Z00.00", "27447"),  # Routine exam + knee replacement
    ("F32.9", "27447"),   # Depression + knee replacement
    ("M54.5", "43239"),   # Low back pain + GI endoscopy
    ("I10", "43239"),     # Hypertension + GI endoscopy
}

def _detect_diagnosis_procedure_mismatch(df):
    for idx, row in df.iterrows():
        pair = (row["diagnosis_code"], row["procedure_code"])
        if pair in _MISMATCH_PAIRS:
            df.at[idx, "flagged"] = True
            df.at[idx, "confidence"] = max(df.at[idx, "confidence"], 0.90)
            df.at[idx, "triggered_rules"].append("diagnosis_procedure_mismatch")
    return df


def _detect_duplicate_claims(df):
    # Same patient + procedure + date of service appearing more than once
    key = df.groupby(["patient_id", "procedure_code", "date_of_service"]).size()
    duplicates = key[key > 1].reset_index()
    dup_set = set(
        zip(duplicates["patient_id"], duplicates["procedure_code"], duplicates["date_of_service"])
    )
    for idx, row in df.iterrows():
        k = (row["patient_id"], row["procedure_code"], row["date_of_service"])
        if k in dup_set:
            df.at[idx, "flagged"] = True
            df.at[idx, "confidence"] = max(df.at[idx, "confidence"], 0.88)
            df.at[idx, "triggered_rules"].append("duplicate_claim")
    return df


def _detect_high_frequency_billing(df):
    # Provider billing the same high-cost procedure > 3 times in any 30-day window
    df["dos_dt"] = pd.to_datetime(df["date_of_service"])
    high_cost_procs = ["27447", "70553", "43239"]

    for provider_id in df["provider_id"].unique():
        sub = df[(df["provider_id"] == provider_id) & (df["procedure_code"].isin(high_cost_procs))].copy()
        if len(sub) < 3:
            continue
        sub = sub.sort_values("dos_dt")
        dates = sub["dos_dt"].tolist()
        for i, d in enumerate(dates):
            window = [x for x in dates if abs((x - d).days) <= 30]
            if len(window) >= 3:
                mask = (df["provider_id"] == provider_id) & (df["procedure_code"].isin(high_cost_procs))
                for idx2 in df[mask].index:
                    df.at[idx2, "flagged"] = True
                    df.at[idx2, "confidence"] = max(df.at[idx2, "confidence"], 0.80)
                    if "high_frequency_billing" not in df.at[idx2, "triggered_rules"]:
                        df.at[idx2, "triggered_rules"].append("high_frequency_billing")
                break
    return df


def _detect_amount_outliers(df):
    # Per-procedure statistical outlier: amount > mean + 2.5 * std
    for proc_code in df["procedure_code"].unique():
        sub = df[df["procedure_code"] == proc_code]["billed_amount"]
        if len(sub) < 5:
            continue
        mean, std = sub.mean(), sub.std()
        if std == 0:
            continue
        threshold = mean + 2.5 * std
        for idx in df[(df["procedure_code"] == proc_code) & (df["billed_amount"] > threshold)].index:
            df.at[idx, "flagged"] = True
            df.at[idx, "confidence"] = max(df.at[idx, "confidence"], 0.72)
            if "amount_outlier" not in df.at[idx, "triggered_rules"]:
                df.at[idx, "triggered_rules"].append("amount_outlier")
    return df


def _detect_impossible_dates(df):
    today = pd.Timestamp(datetime.now().date())
    for idx, row in df.iterrows():
        dos = pd.to_datetime(row["date_of_service"])
        if dos > today:
            df.at[idx, "flagged"] = True
            df.at[idx, "confidence"] = max(df.at[idx, "confidence"], 0.98)
            df.at[idx, "triggered_rules"].append("future_date_of_service")
    return df


def _detect_unbundling_patterns(df):
    # Flag claims where the procedure description contains "unbundled"
    for idx, row in df.iterrows():
        if "unbundl" in str(row.get("procedure_description", "")).lower():
            df.at[idx, "flagged"] = True
            df.at[idx, "confidence"] = max(df.at[idx, "confidence"], 0.85)
            if "unbundling" not in df.at[idx, "triggered_rules"]:
                df.at[idx, "triggered_rules"].append("unbundling")
    return df
