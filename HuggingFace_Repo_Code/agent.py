"""
ClaimSense — Agentic Reasoning Engine
Uses Groq LLM to generate chain-of-reasoning explanations, action recommendations,
and a multi-turn chatbot for claims Q&A.
"""

import os
from groq import Groq


def get_client():
    api_key = os.environ.get("GROQ_API_KEY", "")
    if not api_key:
        raise ValueError("GROQ_API_KEY environment variable not set.")
    return Groq(api_key=api_key)


SYSTEM_PROMPT = """You are a senior clinical claims auditor at a healthcare analytics firm.
You analyze flagged medical claims and provide:
1. A clear chain-of-reasoning explanation of why the claim is suspicious
2. A specific recommended action for the reviewer
3. A risk assessment

Format your response EXACTLY as:

REASONING:
[2-4 sentences explaining exactly why this claim was flagged]

RECOMMENDED ACTION:
[One specific action]

RISK ASSESSMENT:
[One sentence: overall risk level and primary concern]"""


CHATBOT_SYSTEM = """You are ClaimSense, an intelligent clinical claims assistant.
You help healthcare reviewers, physicians, and administrators understand claim patterns,
anomalies, and payment integrity findings.
Be concise, factual, and professional. Ground answers in the claims data provided."""


def analyze_claim(row: dict) -> dict:
    try:
        client = get_client()
    except ValueError as e:
        return {
            "reasoning": f"API key not configured: {str(e)}",
            "action": "Manual review required — configure GROQ_API_KEY",
            "risk_assessment": "Unable to assess — API key missing",
            "faithfulness_score": 0.0,
            "raw_response": str(e),
        }

    rules_str = ", ".join(row.get("triggered_rules", [])) or "statistical outlier"

    user_message = f"""Analyze this flagged medical claim:

CLAIM ID: {row['claim_id']}
DATE OF SERVICE: {row['date_of_service']}
PATIENT ID: {row['patient_id']}
PROVIDER: {row['provider_name']} ({row['provider_id']}) — {row['specialty']}
DIAGNOSIS: {row['diagnosis_code']} — {row['diagnosis_description']}
PROCEDURE: {row['procedure_code']} — {row['procedure_description']}
BILLED AMOUNT: ${row['billed_amount']:,.2f}
INSURER: {row['insurer']}
CONFIDENCE: {row['confidence']:.0%}
TRIGGERED RULES: {rules_str}"""

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message}
            ],
            temperature=0.2,
            max_tokens=600,
        )
        raw = response.choices[0].message.content.strip()
        parsed = _parse_response(raw)
        parsed["faithfulness_score"] = _faithfulness_check(parsed["reasoning"], row)
        parsed["raw_response"] = raw
        return parsed
    except Exception as e:
        return {
            "reasoning": f"Analysis failed: {str(e)}",
            "action": "Manual review required",
            "risk_assessment": "Unable to assess — API error",
            "faithfulness_score": 0.0,
            "raw_response": str(e),
        }


def chat_with_claims(messages: list, claims_context: str) -> str:
    try:
        client = get_client()
    except ValueError:
        return "GROQ_API_KEY is not configured. Please add it to your Space secrets."

    system = CHATBOT_SYSTEM + f"\n\nCLAIMS DATA CONTEXT:\n{claims_context}"
    full_messages = [{"role": "system", "content": system}] + messages

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=full_messages,
            temperature=0.3,
            max_tokens=600,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error: {str(e)}"


def _parse_response(text: str) -> dict:
    sections = {"reasoning": "", "action": "", "risk_assessment": ""}
    current = None
    lines = []

    for line in text.split("\n"):
        stripped = line.strip()
        if stripped.startswith("REASONING:"):
            current = "reasoning"
            rest = stripped.replace("REASONING:", "").strip()
            lines = [rest] if rest else []
        elif stripped.startswith("RECOMMENDED ACTION:"):
            if current:
                sections[current] = " ".join(lines).strip()
            current = "action"
            rest = stripped.replace("RECOMMENDED ACTION:", "").strip()
            lines = [rest] if rest else []
        elif stripped.startswith("RISK ASSESSMENT:"):
            if current:
                sections[current] = " ".join(lines).strip()
            current = "risk_assessment"
            rest = stripped.replace("RISK ASSESSMENT:", "").strip()
            lines = [rest] if rest else []
        elif stripped and current:
            lines.append(stripped)

    if current and lines:
        sections[current] = " ".join(lines).strip()
    return sections


def _faithfulness_check(reasoning: str, row: dict) -> float:
    if not reasoning or len(reasoning) < 20:
        return 0.0
    r = reasoning.lower()
    checks = [
        row["claim_id"].lower() in r or row["procedure_code"].lower() in r,
        row["diagnosis_code"].lower() in r or row["diagnosis_description"].lower()[:15].lower() in r,
        row["provider_name"].split()[-1].lower() in r or row["specialty"].lower() in r,
        str(int(row["billed_amount"])) in r or row["procedure_description"].split()[0].lower() in r,
        any(rule.replace("_", " ").lower()[:8] in r for rule in row.get("triggered_rules", [])),
    ]
    score = sum(checks) / len(checks)
    if len(reasoning) > 200:
        score = min(1.0, score + 0.1)
    return round(score, 2)