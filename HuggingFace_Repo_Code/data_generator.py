"""
ClaimSense — Synthetic Claims Data Generator
Generates realistic medical claims with injected anomalies for demonstration.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

random.seed(42)
np.random.seed(42)

# --- Reference Tables ---

DIAGNOSIS_CODES = {
    "Z00.00": "Routine general medical examination",
    "I10": "Essential (primary) hypertension",
    "E11.9": "Type 2 diabetes mellitus without complications",
    "J06.9": "Acute upper respiratory infection, unspecified",
    "M54.5": "Low back pain",
    "F32.9": "Major depressive disorder, single episode, unspecified",
    "J18.9": "Pneumonia, unspecified organism",
    "I25.10": "Atherosclerotic heart disease of native coronary artery",
    "N39.0": "Urinary tract infection, site not specified",
    "K21.0": "Gastro-esophageal reflux disease with esophagitis",
}

PROCEDURE_CODES = {
    "99213": ("Office visit, established patient, low complexity", 120, 180),
    "99214": ("Office visit, established patient, moderate complexity", 180, 280),
    "99215": ("Office visit, established patient, high complexity", 250, 380),
    "93000": ("Electrocardiogram, routine ECG", 80, 130),
    "71046": ("Chest X-ray, 2 views", 120, 200),
    "80053": ("Comprehensive metabolic panel", 40, 80),
    "99232": ("Subsequent hospital care, moderate complexity", 180, 260),
    "27447": ("Total knee arthroplasty", 8000, 14000),
    "43239": ("Upper GI endoscopy with biopsy", 900, 1600),
    "70553": ("MRI brain with contrast", 1200, 2200),
}

PROVIDERS = {
    "PRV-001": ("Dr. Sarah Mitchell", "Internal Medicine", "normal"),
    "PRV-002": ("Dr. James Chen", "Cardiology", "normal"),
    "PRV-003": ("Dr. Patricia Wong", "Family Medicine", "normal"),
    "PRV-004": ("Dr. Robert Garcia", "Orthopedics", "normal"),
    "PRV-005": ("Dr. Lisa Patel", "Gastroenterology", "normal"),
    "PRV-006": ("Dr. Michael Torres", "Family Medicine", "high_volume"),
    "PRV-007": ("Dr. Amanda Foster", "Internal Medicine", "normal"),
    "PRV-008": ("Dr. Kevin Park", "Neurology", "normal"),
}

PATIENTS = [f"PAT-{str(i).zfill(4)}" for i in range(1, 201)]
INSURERS = ["BlueCross BlueShield", "Aetna", "UnitedHealth", "Cigna", "Humana"]


def generate_normal_claim(claim_id, date):
    """Generate a single normal, clinically appropriate claim."""
    provider_id = random.choice(list(PROVIDERS.keys()))
    provider_name, specialty, _ = PROVIDERS[provider_id]
    patient_id = random.choice(PATIENTS)

    # Pick a clinically coherent dx + procedure pair
    dx_code = random.choice(list(DIAGNOSIS_CODES.keys()))
    proc_code = random.choice(list(PROCEDURE_CODES.keys()))
    proc_desc, min_amt, max_amt = PROCEDURE_CODES[proc_code]
    amount = round(random.uniform(min_amt, max_amt), 2)

    return {
        "claim_id": claim_id,
        "date_of_service": date.strftime("%Y-%m-%d"),
        "patient_id": patient_id,
        "provider_id": provider_id,
        "provider_name": provider_name,
        "specialty": specialty,
        "diagnosis_code": dx_code,
        "diagnosis_description": DIAGNOSIS_CODES[dx_code],
        "procedure_code": proc_code,
        "procedure_description": proc_desc,
        "billed_amount": amount,
        "insurer": random.choice(INSURERS),
        "anomaly_type": "none",
        "anomaly_label": 0,
    }


def inject_anomalies(claims_df):
    """Inject specific, labeled anomaly types into a subset of claims."""
    anomaly_indices = random.sample(range(len(claims_df)), k=18)
    anomaly_types = [
        "diagnosis_procedure_mismatch",
        "duplicate_claim",
        "high_frequency_billing",
        "upcoding",
        "unbundling",
        "impossible_date",
    ]

    for i, idx in enumerate(anomaly_indices):
        atype = anomaly_types[i % len(anomaly_types)]
        row = claims_df.iloc[idx].copy()

        if atype == "diagnosis_procedure_mismatch":
            # Assign a procedure inconsistent with the diagnosis
            row["procedure_code"] = "27447"
            row["procedure_description"] = PROCEDURE_CODES["27447"][0]
            row["billed_amount"] = round(random.uniform(8000, 14000), 2)
            row["diagnosis_code"] = "J06.9"
            row["diagnosis_description"] = DIAGNOSIS_CODES["J06.9"]

        elif atype == "duplicate_claim":
            # Same patient, same procedure, same date as a prior claim
            base_idx = random.choice([j for j in range(len(claims_df)) if j != idx])
            base = claims_df.iloc[base_idx]
            row["patient_id"] = base["patient_id"]
            row["procedure_code"] = base["procedure_code"]
            row["procedure_description"] = base["procedure_description"]
            row["date_of_service"] = base["date_of_service"]
            row["billed_amount"] = base["billed_amount"]

        elif atype == "high_frequency_billing":
            # High-volume provider billing expensive procedure repeatedly
            row["provider_id"] = "PRV-006"
            row["provider_name"] = PROVIDERS["PRV-006"][0]
            row["procedure_code"] = "70553"
            row["procedure_description"] = PROCEDURE_CODES["70553"][0]
            row["billed_amount"] = round(random.uniform(1200, 2200), 2)

        elif atype == "upcoding":
            # Bill a higher-complexity code than clinically warranted
            row["procedure_code"] = "99215"
            row["procedure_description"] = PROCEDURE_CODES["99215"][0]
            row["billed_amount"] = round(random.uniform(350, 500), 2)  # above normal max

        elif atype == "unbundling":
            # Separately bill components that should be bundled
            row["billed_amount"] = round(random.uniform(900, 1400), 2)
            row["procedure_code"] = "80053"
            row["procedure_description"] = "Unbundled lab panel — individual components billed separately"

        elif atype == "impossible_date":
            # Future date of service
            future_date = datetime.now() + timedelta(days=random.randint(30, 180))
            row["date_of_service"] = future_date.strftime("%Y-%m-%d")

        row["anomaly_type"] = atype
        row["anomaly_label"] = 1
        claims_df.iloc[idx] = row

    return claims_df


def generate_claims_dataset(n=120):
    """Generate a full synthetic claims dataset with injected anomalies."""
    start_date = datetime(2024, 1, 1)
    claims = []
    for i in range(n):
        date = start_date + timedelta(days=random.randint(0, 365))
        claims.append(generate_normal_claim(f"CLM-{str(i+1).zfill(4)}", date))

    df = pd.DataFrame(claims)
    df = inject_anomalies(df)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    return df


if __name__ == "__main__":
    df = generate_claims_dataset()
    print(f"Generated {len(df)} claims — {df['anomaly_label'].sum()} flagged anomalies")
    print(df[df["anomaly_label"] == 1][["claim_id", "anomaly_type", "billed_amount"]].to_string())
