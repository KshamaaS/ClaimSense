"""
ClaimSense — Agentic Clinical Claims Intelligence
"""

import streamlit as st
import pandas as pd
import plotly.express as px
from collections import Counter
from data_generator import generate_claims_dataset
from detector import detect_anomalies
from agent import analyze_claim, chat_with_claims

st.set_page_config(
    page_title="ClaimSense — Agentic Clinical Claims Intelligence",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .main-title { font-size:2.2rem; font-weight:800; color:#7B2D8B; margin-bottom:0; }
    .subtitle { font-size:1.05rem; color:#888888; margin-top:0; margin-bottom:1.5rem; }
    .reasoning-box { background:#2d2d2d; border-left:4px solid #7B2D8B; padding:1rem; border-radius:4px; font-size:0.95rem; line-height:1.6; color:#f0f0f0; }
    .action-box { background:#3a3000; border-left:4px solid #F39C12; padding:1rem; border-radius:4px; font-size:0.95rem; color:#f0f0f0; }
    .faith-box { background:#003a10; border-left:4px solid #27AE60; padding:1rem; border-radius:4px; font-size:0.95rem; color:#f0f0f0; }
    .risk-box { background:#3a0000; border-left:4px solid #C0392B; padding:1rem; border-radius:4px; font-size:0.95rem; color:#f0f0f0; }
    .chat-msg-user { background:#3d2a5a; border-radius:12px 12px 0 12px; padding:0.75rem 1rem; margin:0.4rem 0; color:#f0f0f0; }
    .chat-msg-ai { background:#2a2a2a; border-radius:12px 12px 12px 0; padding:0.75rem 1rem; margin:0.4rem 0; color:#f0f0f0; }
</style>
""", unsafe_allow_html=True)

# ── Session State ─────────────────────────────────────────────────────────────
if "claims_df" not in st.session_state:
    with st.spinner("Generating synthetic claims dataset..."):
        raw = generate_claims_dataset(n=120)
        st.session_state.claims_df = detect_anomalies(raw)

if "analyses" not in st.session_state:
    st.session_state.analyses = {}

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "pending_analysis" not in st.session_state:
    st.session_state.pending_analysis = None

df = st.session_state.claims_df

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown('<div class="main-title">🏥 ClaimSense</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Agentic Clinical Claims Intelligence — Anomaly Detection · Chain-of-Reasoning · Action Recommendations · Claims Q&A</div>', unsafe_allow_html=True)
st.warning("⚠️ **Proof of Concept — Research Demonstration Only.** All claims data is fully synthetic and generated for demonstration purposes. No real patient data, PHI, or actual claims records are used. This system is not validated for clinical or payment decision use.")

# ── Main Navigation ───────────────────────────────────────────────────────────
page = st.radio(
    "Navigate",
    ["📊 Claims Dashboard", "🔍 Anomaly Deep Dive", "⚡ Action Center", "💬 Claims Q&A"],
    key="nav_page",
    horizontal=True,
)

st.markdown("---")

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### About ClaimSense")
    st.markdown(
        "ClaimSense demonstrates how **Agentic Generative AI** closes the gap between "
        "anomaly detection and clinical action.\n\n"
        "Built for the Cotiviti AI/Healthcare Informatics Internship assessment."
    )
    st.markdown("---")
    total = len(df)
    flagged = df["flagged"].sum()
    flagged_billed = df[df["flagged"]]["billed_amount"].sum()
    st.markdown("### Dataset Summary")
    st.metric("Total Claims", total)
    st.metric("Flagged Anomalies", int(flagged))
    st.metric("Flag Rate", f"{flagged/total:.1%}")
    st.metric("Flagged Exposure", f"${flagged_billed:,.0f}")
    st.markdown("---")
    show_flagged_only = st.checkbox("Show flagged claims only", value=False)
    risk_filter = st.multiselect(
        "Risk Level",
        ["Critical", "High", "Medium", "Low", "Normal"],
        default=["Critical", "High", "Medium", "Low", "Normal"],
    )

view_df = df.copy()
if show_flagged_only:
    view_df = view_df[view_df["flagged"]]
view_df = view_df[view_df["risk_level"].isin(risk_filter)]



# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — CLAIMS DASHBOARD
# ═══════════════════════════════════════════════════════════════════════════════
if page == "📊 Claims Dashboard":
    st.subheader("Claims Overview")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Claims", len(df))
    with col2:
        critical = (df["risk_level"] == "Critical").sum()
        st.metric("Critical Risk", int(critical))
    with col3:
        high = (df["risk_level"] == "High").sum()
        st.metric("High Risk", int(high))
    with col4:
        total_billed = df["billed_amount"].sum()
        st.metric("Flagged Exposure", f"${flagged_billed:,.0f}", delta=f"of ${total_billed:,.0f} total")

    st.markdown("---")
    col_left, col_right = st.columns([1.2, 0.8])

    with col_left:
        st.markdown("#### Risk Distribution")
        risk_counts = df["risk_level"].value_counts().reset_index()
        risk_counts.columns = ["Risk Level", "Count"]
        risk_order = ["Critical", "High", "Medium", "Low", "Normal"]
        risk_colors = {"Critical": "#C0392B", "High": "#E67E22", "Medium": "#E6B800", "Low": "#3498DB", "Normal": "#27AE60"}
        risk_counts["Risk Level"] = pd.Categorical(risk_counts["Risk Level"], categories=risk_order, ordered=True)
        risk_counts = risk_counts.sort_values("Risk Level")
        fig_bar = px.bar(risk_counts, x="Risk Level", y="Count", color="Risk Level",
                         color_discrete_map=risk_colors, text="Count")
        fig_bar.update_layout(showlegend=False, margin=dict(t=20, b=20), height=300,
                              paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        fig_bar.update_traces(textposition="outside")
        st.plotly_chart(fig_bar, use_container_width=True)

    with col_right:
        st.markdown("#### Anomaly Types")
        flagged_df = df[df["flagged"]].copy()
        all_rules = []
        for rules in flagged_df["triggered_rules"]:
            all_rules.extend(rules)
        if all_rules:
            rule_counts = Counter(all_rules)
            rule_df = pd.DataFrame(rule_counts.items(), columns=["Rule", "Count"])
            rule_df["Rule"] = rule_df["Rule"].str.replace("_", " ").str.title()
            fig_pie = px.pie(rule_df, values="Count", names="Rule",
                             color_discrete_sequence=px.colors.qualitative.Set2)
            fig_pie.update_layout(margin=dict(t=20, b=20), height=300, paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_pie, use_container_width=True)

    st.markdown("---")
    st.markdown("#### Claims Table")
    display_cols = ["claim_id", "date_of_service", "patient_id", "provider_name",
                    "diagnosis_code", "procedure_code", "billed_amount", "risk_level", "confidence"]
    display_df = view_df[display_cols].copy()
    display_df["billed_amount"] = display_df["billed_amount"].apply(lambda x: f"${x:,.2f}")
    display_df["confidence"] = display_df["confidence"].apply(lambda x: f"{x:.0%}" if x > 0 else "—")

    def highlight_risk(row):
        styles = {
            "Critical": "background-color: #7a0000; color: #ffffff; font-weight: 600",
            "High":     "background-color: #7a3800; color: #ffffff; font-weight: 600",
            "Medium":   "background-color: #6b5c00; color: #ffffff",
            "Low":      "background-color: #003a6b; color: #ffffff",
            "Normal":   "",
        }
        return [styles.get(row["risk_level"], "")] * len(row)

    st.dataframe(display_df.style.apply(highlight_risk, axis=1), use_container_width=True, height=420)

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — ANOMALY DEEP DIVE
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🔍 Anomaly Deep Dive":
    st.subheader("Anomaly Deep Dive — AI Chain-of-Reasoning Analysis")

    flagged_claims = df[df["flagged"]].copy()
    if flagged_claims.empty:
        st.info("No flagged claims found.")
    else:
        claim_options = flagged_claims.apply(
            lambda r: f"{r['claim_id']} — {r['risk_level']} — {r['provider_name']} — ${r['billed_amount']:,.2f}", axis=1
        ).tolist()
        selected_option = st.selectbox("Select a flagged claim to analyze:", claim_options)
        selected_id = selected_option.split(" — ")[0]
        selected_row = flagged_claims[flagged_claims["claim_id"] == selected_id].iloc[0].to_dict()

        col_a, col_b = st.columns([1, 1])
        with col_a:
            st.markdown("##### Claim Details")
            for k, v in {
                "Claim ID": selected_row["claim_id"],
                "Date of Service": selected_row["date_of_service"],
                "Patient ID": selected_row["patient_id"],
                "Provider": f"{selected_row['provider_name']} ({selected_row['specialty']})",
                "Diagnosis": f"{selected_row['diagnosis_code']} — {selected_row['diagnosis_description']}",
                "Procedure": f"{selected_row['procedure_code']} — {selected_row['procedure_description']}",
                "Billed Amount": f"${selected_row['billed_amount']:,.2f}",
                "Insurer": selected_row["insurer"],
            }.items():
                st.markdown(f"**{k}:** {v}")

        with col_b:
            st.markdown("##### Detection Summary")
            risk = selected_row["risk_level"]
            risk_emoji = {"Critical": "🔴", "High": "🟠", "Medium": "🟡", "Low": "🔵", "Normal": "🟢"}
            st.markdown(f"**Risk Level:** {risk_emoji.get(risk, '')} {risk}")
            st.markdown(f"**Confidence Score:** {selected_row['confidence']:.0%}")
            rules = selected_row.get("triggered_rules", [])
            if rules:
                st.markdown("**Triggered Rules:**")
                for rule in rules:
                    st.markdown(f"  - {rule.replace('_', ' ').title()}")

        st.markdown("---")

        # Show cached result if available
        if selected_id in st.session_state.analyses:
            result = st.session_state.analyses[selected_id]
            st.success("✅ Cached analysis — click button below to re-run")
            st.markdown("##### 🧠 Chain-of-Reasoning")
            st.markdown(f'<div class="reasoning-box">{result["reasoning"]}</div>', unsafe_allow_html=True)
            col_act, col_faith = st.columns([1, 1])
            with col_act:
                st.markdown("##### ⚡ Recommended Action")
                st.markdown(f'<div class="action-box">{result["action"]}</div>', unsafe_allow_html=True)
            with col_faith:
                st.markdown("##### ✅ Faithfulness Score")
                faith_score = result["faithfulness_score"]
                faith_color = "#27AE60" if faith_score >= 0.7 else "#E67E22" if faith_score >= 0.4 else "#C0392B"
                faith_label = "Highly grounded" if faith_score >= 0.7 else "Partially grounded" if faith_score >= 0.4 else "Low grounding"
                st.markdown(f'<div class="faith-box">Score: <strong style="color:{faith_color}">{faith_score:.0%}</strong> — {faith_label}</div>', unsafe_allow_html=True)
            st.markdown("##### ⚠️ Risk Assessment")
            st.markdown(f'<div class="risk-box">{result["risk_assessment"]}</div>', unsafe_allow_html=True)
            st.markdown("---")

        if st.button("🤖 Run AI Analysis", type="primary", use_container_width=True):
            with st.spinner("AI agent analyzing claim..."):
                result = analyze_claim(selected_row)
                st.session_state.analyses[selected_id] = result
            st.markdown("##### 🧠 Chain-of-Reasoning")
            st.markdown(f'<div class="reasoning-box">{result["reasoning"]}</div>', unsafe_allow_html=True)
            col_act2, col_faith2 = st.columns([1, 1])
            with col_act2:
                st.markdown("##### ⚡ Recommended Action")
                st.markdown(f'<div class="action-box">{result["action"]}</div>', unsafe_allow_html=True)
            with col_faith2:
                st.markdown("##### ✅ Faithfulness Score")
                faith_score = result["faithfulness_score"]
                faith_color = "#27AE60" if faith_score >= 0.7 else "#E67E22" if faith_score >= 0.4 else "#C0392B"
                faith_label = "Highly grounded" if faith_score >= 0.7 else "Partially grounded" if faith_score >= 0.4 else "Low grounding"
                st.markdown(f'<div class="faith-box">Score: <strong style="color:{faith_color}">{faith_score:.0%}</strong> — {faith_label}</div>', unsafe_allow_html=True)
            st.markdown("##### ⚠️ Risk Assessment")
            st.markdown(f'<div class="risk-box">{result["risk_assessment"]}</div>', unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — ACTION CENTER
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "⚡ Action Center":
    st.subheader("Action Center — Priority Claims Queue")
    critical_high = df[df["flagged"]].copy()
    st.markdown(f"**{len(critical_high)} flagged claims require review**")

    if st.button("🚀 Analyze All Priority Claims", type="primary"):
        progress_bar = st.progress(0)
        status_text = st.empty()
        results_list = []
        for i, (_, row) in enumerate(critical_high.iterrows()):
            row_dict = row.to_dict()
            status_text.text(f"Analyzing {i+1}/{len(critical_high)}: {row_dict['claim_id']}...")
            if row_dict["claim_id"] not in st.session_state.analyses:
                result = analyze_claim(row_dict)
                st.session_state.analyses[row_dict["claim_id"]] = result
            else:
                result = st.session_state.analyses[row_dict["claim_id"]]
            results_list.append({
                "Claim ID": row_dict["claim_id"],
                "Risk": row_dict["risk_level"],
                "Provider": row_dict["provider_name"],
                "Amount": f"${row_dict['billed_amount']:,.2f}",
                "Recommended Action": result.get("action", "Manual review"),
                "Faithfulness": f"{result.get('faithfulness_score', 0):.0%}",
            })
            progress_bar.progress((i + 1) / len(critical_high))
        status_text.text("✅ Analysis complete.")
        results_df = pd.DataFrame(results_list)

        # HTML table with word wrap on Recommended Action
        risk_colors = {"Critical": "#7a0000", "High": "#7a3800", "Medium": "#6b5c00", "Low": "#003a6b"}
        rows_html = ""
        for _, r in results_df.iterrows():
            bg = risk_colors.get(r["Risk"], "#2d2d2d")
            rows_html += f"""
            <tr>
              <td style='padding:8px;border-bottom:1px solid #444;white-space:nowrap'>{r['Claim ID']}</td>
              <td style='padding:8px;border-bottom:1px solid #444;background:{bg};color:#fff;font-weight:bold;white-space:nowrap;border-radius:4px'>{r['Risk']}</td>
              <td style='padding:8px;border-bottom:1px solid #444;white-space:nowrap'>{r['Provider']}</td>
              <td style='padding:8px;border-bottom:1px solid #444;white-space:nowrap'>{r['Amount']}</td>
              <td style='padding:8px;border-bottom:1px solid #444;max-width:400px;word-wrap:break-word;white-space:normal;line-height:1.5'>{r['Recommended Action']}</td>
              <td style='padding:8px;border-bottom:1px solid #444;white-space:nowrap;color:#27AE60;font-weight:bold'>{r['Faithfulness']}</td>
            </tr>"""

        html_table = f"""
        <div style='overflow-x:auto;max-height:600px;overflow-y:auto'>
        <table style='width:100%;border-collapse:collapse;font-size:13px;font-family:Arial'>
          <thead>
            <tr style='background:#7B2D8B;color:#fff;position:sticky;top:0'>
              <th style='padding:10px;text-align:left'>Claim ID</th>
              <th style='padding:10px;text-align:left'>Risk</th>
              <th style='padding:10px;text-align:left'>Provider</th>
              <th style='padding:10px;text-align:left'>Amount</th>
              <th style='padding:10px;text-align:left;min-width:350px'>Recommended Action</th>
              <th style='padding:10px;text-align:left'>Faithfulness</th>
            </tr>
          </thead>
          <tbody>{rows_html}</tbody>
        </table>
        </div>"""

        st.markdown(html_table, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — CLAIMS Q&A
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "💬 Claims Q&A":
    st.subheader("💬 Claims Q&A — Ask Anything About Your Claims Data")

    flagged_summary = df[df["flagged"]].copy()
    context_lines = [
        f"Total claims: {len(df)}", f"Flagged: {df['flagged'].sum()}",
        f"Critical: {(df['risk_level']=='Critical').sum()}",
        f"High: {(df['risk_level']=='High').sum()}",
        f"Flagged exposure: ${df[df['flagged']]['billed_amount'].sum():,.2f}", "", "Flagged claims:"
    ]
    for _, row in flagged_summary.head(15).iterrows():
        rules = ", ".join(row["triggered_rules"]) if row["triggered_rules"] else "outlier"
        context_lines.append(f"- {row['claim_id']}: {row['provider_name']}, {row['diagnosis_code']}/{row['procedure_code']}, ${row['billed_amount']:,.2f}, Risk={row['risk_level']}, Rules=[{rules}]")
    claims_context = "\n".join(context_lines)

    st.markdown("**Suggested questions:**")
    col_q1, col_q2, col_q3 = st.columns(3)
    with col_q1:
        if st.button("Which provider has the most flags?"):
            st.session_state.chat_history.append({"role": "user", "content": "Which provider has the most flagged claims?"})
            with st.spinner("Thinking..."):
                resp = chat_with_claims(st.session_state.chat_history, claims_context)
                st.session_state.chat_history.append({"role": "assistant", "content": resp})
    with col_q2:
        if st.button("What is the total financial risk?"):
            st.session_state.chat_history.append({"role": "user", "content": "What is the total financial exposure from flagged claims?"})
            with st.spinner("Thinking..."):
                resp = chat_with_claims(st.session_state.chat_history, claims_context)
                st.session_state.chat_history.append({"role": "assistant", "content": resp})
    with col_q3:
        if st.button("Explain duplicate claims"):
            st.session_state.chat_history.append({"role": "user", "content": "Explain duplicate claims and why they are a problem."})
            with st.spinner("Thinking..."):
                resp = chat_with_claims(st.session_state.chat_history, claims_context)
                st.session_state.chat_history.append({"role": "assistant", "content": resp})

    st.markdown("---")
    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            st.markdown(f'<div class="chat-msg-user">👤 <strong>You:</strong> {msg["content"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="chat-msg-ai">🏥 <strong>ClaimSense:</strong> {msg["content"]}</div>', unsafe_allow_html=True)

    st.markdown("---")
    with st.form("chat_form", clear_on_submit=True):
        col_input, col_send = st.columns([5, 1])
        with col_input:
            user_input = st.text_input("Ask a question...", label_visibility="collapsed")
        with col_send:
            send_btn = st.form_submit_button("Send", use_container_width=True)
        if send_btn and user_input.strip():
            st.session_state.chat_history.append({"role": "user", "content": user_input.strip()})
            with st.spinner("Thinking..."):
                resp = chat_with_claims(st.session_state.chat_history, claims_context)
                st.session_state.chat_history.append({"role": "assistant", "content": resp})

    if st.button("🗑️ Clear Chat History"):
        st.session_state.chat_history = []