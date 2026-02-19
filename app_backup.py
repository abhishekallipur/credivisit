"""
CrediVist — Alternative Credit Scoring Engine
Main Streamlit Application
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Add project root to path
sys.path.insert(0, os.path.dirname(__file__))

from src.feature_engineering import engineer_features, extract_all_features
from src.scoring_engine import (
    compute_all_scores, compute_base_score, compute_final_score,
    get_score_breakdown, SUB_SCORE_WEIGHTS
)
from src.ml_model import CreditRiskModel, ML_FEATURES
from src.explainability import ScoreExplainer, FEATURE_LABELS
from src.transaction_parser import TransactionParser, generate_sample_statement
from src.alternative_profiles import (
    PERSONAS, compute_persona_score, get_persona_form_fields,
    get_improvement_tips
)
from src.document_analyzer import (
    analyze_documents, auto_detect_persona, SAMPLE_GENERATORS
)
from src.loan_engine import (
    get_transaction_loan_recommendations, get_persona_loan_recommendations,
    compare_loans, get_financial_tips, get_seasonal_recommendations,
    calculate_emi, generate_repayment_schedule, get_score_tier,
    analyze_repayment_capacity,
    search_loans, check_loan_eligibility, get_all_loans_catalog,
    get_loan_categories, TRANSACTION_LOANS, PERSONA_LOANS,
)

# ─── Page Config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CrediVist — Alternative Credit Scoring",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 1rem 0;
    }
    .main-header h1 {
        background: linear-gradient(135deg, #6366f1, #8b5cf6, #a855f7);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.6rem;
        font-weight: 800;
        margin-bottom: 0;
    }
    .main-header p {
        color: #94a3b8;
        font-size: 1.1rem;
    }
    .score-card {
        text-align: center;
        padding: 2rem;
        border-radius: 16px;
        background: linear-gradient(145deg, #1e1b4b, #312e81);
        border: 1px solid #4338ca;
    }
    .score-value {
        font-size: 4rem;
        font-weight: 900;
        margin: 0;
    }
    .score-label {
        font-size: 1.1rem;
        color: #a5b4fc;
        margin-top: 4px;
    }
    .metric-card {
        background: #1e293b;
        border-radius: 12px;
        padding: 1.2rem;
        border: 1px solid #334155;
        text-align: center;
    }
    .metric-card h3 {
        color: #94a3b8;
        font-size: 0.85rem;
        margin-bottom: 4px;
    }
    .metric-card .val {
        font-size: 1.6rem;
        font-weight: 700;
    }
    .subscore-bar {
        height: 8px;
        border-radius: 4px;
        background: #1e293b;
        margin-top: 4px;
    }
    .subscore-fill {
        height: 100%;
        border-radius: 4px;
    }
    div[data-testid="stSidebar"] {
        background: #0f172a;
    }
</style>
""", unsafe_allow_html=True)


# ─── Session State Init ────────────────────────────────────────────────────
@st.cache_data
def load_or_generate_data():
    """Load existing data or generate synthetic data."""
    data_path = os.path.join(os.path.dirname(__file__), "data", "credit_data.csv")
    if os.path.exists(data_path):
        df = pd.read_csv(data_path)
    else:
        from data.generate_synthetic_data import generate_dataset
        df = generate_dataset()
        df.to_csv(data_path, index=False)
    return df


@st.cache_resource
def train_model(df):
    """Train the ML model and return it along with processed data."""
    # Feature engineering
    df_feat = engineer_features(df)
    # Compute base scores
    df_scored = compute_all_scores(df_feat)
    # Train ML model
    model = CreditRiskModel()
    metrics = model.train(df_scored)
    # Predict risk for all users
    risk_probs = model.predict_risk(df_scored)
    df_scored["risk_probability"] = risk_probs
    # Compute final scores
    final_scores = []
    for idx, row in df_scored.iterrows():
        fs = compute_final_score(row["base_trust_score"], row["risk_probability"], row)
        final_scores.append(fs)
    final_df = pd.DataFrame(final_scores)
    # Drop overlapping columns from final_df before concat
    overlap_cols = [c for c in final_df.columns if c in df_scored.columns]
    final_df = final_df.drop(columns=overlap_cols, errors="ignore")
    df_scored = pd.concat([df_scored.reset_index(drop=True), final_df.reset_index(drop=True)], axis=1)
    return model, df_scored, metrics


# ─── Helper: Gauge Chart ───────────────────────────────────────────────────
def create_gauge(score, grade, color):
    """Create a Plotly gauge chart for the trust score."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        number={"font": {"size": 60, "color": color}},
        title={"text": f"<b>{grade}</b>", "font": {"size": 20, "color": color}},
        gauge={
            "axis": {"range": [300, 900], "tickwidth": 2, "tickcolor": "#475569",
                     "tickvals": [300, 400, 500, 650, 750, 900],
                     "ticktext": ["300", "400", "500", "650", "750", "900"]},
            "bar": {"color": color, "thickness": 0.3},
            "bgcolor": "#1e293b",
            "borderwidth": 0,
            "steps": [
                {"range": [300, 400], "color": "rgba(239,68,68,0.15)"},
                {"range": [400, 500], "color": "rgba(249,115,22,0.15)"},
                {"range": [500, 650], "color": "rgba(234,179,8,0.15)"},
                {"range": [650, 750], "color": "rgba(132,204,22,0.15)"},
                {"range": [750, 900], "color": "rgba(34,197,94,0.15)"},
            ],
            "threshold": {
                "line": {"color": "white", "width": 3},
                "thickness": 0.8,
                "value": score
            }
        }
    ))
    fig.update_layout(
        height=280,
        margin=dict(t=40, b=20, l=30, r=30),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#e2e8f0"}
    )
    return fig


def create_subscore_radar(breakdown):
    """Create radar chart showing sub-score breakdown."""
    categories = list(breakdown.keys())
    values = [breakdown[c]["score"] for c in categories]
    values.append(values[0])  # close the polygon
    categories.append(categories[0])

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill="toself",
        fillcolor="rgba(99,102,241,0.2)",
        line=dict(color="#6366f1", width=2),
        marker=dict(size=6, color="#818cf8"),
        name="Sub-Scores"
    ))
    fig.update_layout(
        polar=dict(
            bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(visible=True, range=[0, 100], tickfont=dict(size=10, color="#64748b")),
            angularaxis=dict(tickfont=dict(size=11, color="#cbd5e1")),
        ),
        showlegend=False,
        height=350,
        margin=dict(t=30, b=30, l=60, r=60),
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def create_component_bars(breakdown):
    """Create horizontal bar chart for detailed components."""
    all_components = []
    for category, data in breakdown.items():
        for comp_name, comp_val in data["components"].items():
            all_components.append({
                "Category": category,
                "Component": comp_name,
                "Score": comp_val
            })

    comp_df = pd.DataFrame(all_components)
    fig = px.bar(
        comp_df, y="Component", x="Score", color="Category",
        orientation="h",
        color_discrete_sequence=["#6366f1", "#8b5cf6", "#06b6d4", "#f59e0b"],
        height=400,
    )
    fig.update_layout(
        xaxis=dict(range=[0, 100], title="Score"),
        yaxis=dict(title=""),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e2e8f0"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
        margin=dict(t=40, b=20, l=10, r=10),
    )
    return fig


def create_income_chart(monthly_incomes):
    """Line chart of monthly income history."""
    months = [f"Month {i+1}" for i in range(len(monthly_incomes))]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=months, y=monthly_incomes,
        mode="lines+markers",
        line=dict(color="#6366f1", width=3),
        marker=dict(size=8, color="#818cf8"),
        fill="tozeroy",
        fillcolor="rgba(99,102,241,0.1)",
    ))
    fig.update_layout(
        height=250,
        xaxis=dict(title=""),
        yaxis=dict(title="Income (₹)"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e2e8f0"),
        margin=dict(t=10, b=30, l=10, r=10),
    )
    return fig


def create_score_distribution(df):
    """Histogram of all users' trust scores."""
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=df["final_trust_score"],
        nbinsx=30,
        marker_color="#6366f1",
        opacity=0.8,
    ))
    fig.update_layout(
        xaxis=dict(title="Trust Score", range=[300, 900]),
        yaxis=dict(title="Count"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e2e8f0"),
        height=300,
        margin=dict(t=10, b=30, l=10, r=10),
    )
    return fig


# ─── MAIN APP ──────────────────────────────────────────────────────────────
def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🏦 CrediVist</h1>
        <p>Alternative Credit Scoring Engine for the Underbanked</p>
    </div>
    """, unsafe_allow_html=True)

    # Load data & train model
    with st.spinner("Loading data & training AI model..."):
        raw_df = load_or_generate_data()
        model, df, metrics = train_model(raw_df)

    # ── Sidebar ─────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("### 🔍 Navigation")
        page = st.radio(
            "Select View",
            ["📊 Individual Score", "📤 Upload & Score", "🌍 Alternative Score", "� Loan Search", "�📈 Portfolio Analytics", "🤖 Model Performance", "🧪 Score Simulator", "🚀 Score Builder"],
            label_visibility="collapsed"
        )

        st.markdown("---")
        st.markdown("### 📋 About CrediVist")
        st.markdown("""
        CrediVist uses **alternative data signals** to build credit profiles
        for **anyone** — gig workers, farmers, students, street vendors,
        homemakers, and underbanked individuals — even **without a bank account**.

        **Scoring Range:** 300 – 900

        | Grade | Range |
        |-------|-------|
        | Excellent | 750+ |
        | Good | 650–749 |
        | Fair | 500–649 |
        | Poor | 400–499 |
        | Very Poor | <400 |
        """)

        st.markdown("---")
        st.markdown(
            "<div style='text-align:center; color:#64748b; font-size:0.8rem'>"
            "Built for hackathon by <b>Team CrediVist</b></div>",
            unsafe_allow_html=True
        )

    # ── Page: Individual Score ──────────────────────────────────────────
    if page == "📊 Individual Score":
        st.markdown("## 📊 Individual Credit Assessment")

        # User selector
        col_sel1, col_sel2 = st.columns([2, 1])
        with col_sel1:
            user_ids = df["user_id"].tolist()
            selected_user = st.selectbox("Select User", user_ids, index=0)
        with col_sel2:
            st.markdown(f"**Total Users:** {len(df)}")

        user_row = df[df["user_id"] == selected_user].iloc[0]

        # Score Card Row
        score = float(user_row["final_trust_score"])
        grade = user_row["grade"]
        color = user_row["grade_color"]
        risk = float(user_row["risk_probability"])
        confidence = float(user_row.get("confidence", 0.75))
        base = float(user_row["base_trust_score"])

        col1, col2 = st.columns([1, 1])

        with col1:
            st.plotly_chart(create_gauge(score, grade, color), use_container_width=True)

        with col2:
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Final Score", f"{score:.0f}")
            m2.metric("Base Score", f"{base:.0f}")
            m3.metric("Risk Prob", f"{risk:.1%}")
            m4.metric("Confidence", f"{confidence:.0%}")

            st.markdown("")
            # Monthly income chart
            incomes = json.loads(user_row["monthly_incomes"]) if isinstance(user_row["monthly_incomes"], str) else user_row["monthly_incomes"]
            st.plotly_chart(create_income_chart(incomes), use_container_width=True)

        # Sub-score breakdown
        st.markdown("### 📋 Score Breakdown")
        breakdown = get_score_breakdown(user_row)

        col_r, col_b = st.columns([1, 1])
        with col_r:
            st.plotly_chart(create_subscore_radar(breakdown), use_container_width=True)
        with col_b:
            st.plotly_chart(create_component_bars(breakdown), use_container_width=True)

        # Sub-score detail cards
        cols = st.columns(4)
        for i, (cat_name, cat_data) in enumerate(breakdown.items()):
            with cols[i]:
                score_val = cat_data["score"]
                color_val = "#22c55e" if score_val >= 70 else "#eab308" if score_val >= 40 else "#ef4444"
                st.markdown(f"""
                <div class="metric-card">
                    <h3>{cat_name}</h3>
                    <div class="val" style="color:{color_val}">{score_val:.1f}</div>
                    <div style="color:#64748b; font-size:0.75rem">Weight: {cat_data['weight']}</div>
                </div>
                """, unsafe_allow_html=True)

        # Explainability
        st.markdown("### 🧠 AI Explanation")
        try:
            explainer = ScoreExplainer(model)
            explainer.initialize(df)
            explanation = explainer.explain_single(user_row)

            col_e1, col_e2 = st.columns(2)
            with col_e1:
                st.markdown("#### ✅ Positive Factors")
                for f in explanation.get("top_positive_factors", [])[:5]:
                    val = f["feature_value"]
                    st.markdown(f"- **{f['feature']}**: {val:.2f}")

            with col_e2:
                st.markdown("#### ⚠️ Risk Factors")
                for f in explanation.get("top_risk_factors", [])[:5]:
                    val = f["feature_value"]
                    st.markdown(f"- **{f['feature']}**: {val:.2f}")

            st.markdown("---")
            st.markdown(explanation.get("explanation_text", ""))

            # SHAP waterfall
            with st.expander("📊 SHAP Waterfall Plot"):
                fig = explainer.plot_waterfall(user_row)
                st.pyplot(fig)

        except Exception as e:
            st.warning(f"Explainability module fallback: {e}")

        # Loan Recommendations (Individual Score page)
        st.markdown("### 💳 Loan Recommendations")
        try:
            ind_income = float(user_row.get("mean_income", 20000))
            ind_expenses = float(user_row.get("fixed_expenses", ind_income * 0.5))
            ind_emi = 0
            # Detect existing EMI from profile
            emi_count = int(user_row.get("recurring_payments_detected", 0))
            emi_consistency = float(user_row.get("emi_consistency_score", 0))
            if emi_count > 0 and emi_consistency > 0.5:
                ind_emi = ind_income * 0.15  # estimate ~15% of income as existing EMI

            ind_loan_recs = get_transaction_loan_recommendations(
                score=score,
                monthly_income=ind_income,
                monthly_expenses=ind_expenses,
                existing_emi=ind_emi,
            )

            # Pre-approval badge
            ind_tier = ind_loan_recs["tier"]
            pre_st = ind_loan_recs["pre_approval_status"]
            tc = ind_tier["color"]
            st.markdown(
                f'<div style="background:{tc}22; border:1px solid {tc}; '
                f'border-radius:8px; padding:10px 16px; margin-bottom:12px;">'
                f'<span style="font-size:1.2rem; font-weight:bold; color:{tc};">'
                f'{pre_st}</span> &nbsp;·&nbsp; '
                f'Max {ind_tier["max_simultaneous_loans"]} loans &nbsp;·&nbsp; '
                f'Exposure up to ₹{ind_loan_recs["max_total_exposure"]:,.0f}</div>',
                unsafe_allow_html=True,
            )

            if ind_loan_recs["eligible_loans"]:
                top_ind = compare_loans(ind_loan_recs["eligible_loans"])
                tcols_ind = st.columns(min(len(top_ind), 3))
                for ti, tl in enumerate(top_ind):
                    with tcols_ind[ti]:
                        st.markdown(
                            f'<div class="metric-card">'
                            f'<h3>{tl["icon"]} {tl["name"]}</h3>'
                            f'<div class="val" style="color:#22c55e">₹{tl["recommended_amount"]:,.0f}</div>'
                            f'<div style="color:#94a3b8; font-size:0.85rem;">'
                            f'{tl["effective_rate"]}% · {tl["suggested_tenure"]} months</div>'
                            f'<div style="color:#64748b; font-size:0.8rem; margin-top:4px;">'
                            f'EMI: ₹{tl["emi"]:,.0f}/month</div>'
                            f'</div>',
                            unsafe_allow_html=True,
                        )

                with st.expander(f"📋 All {ind_loan_recs['total_eligible']} Eligible Loans"):
                    for loan in ind_loan_recs["eligible_loans"]:
                        st.markdown(f"**{loan['icon']} {loan['name']}** — "
                                    f"Up to ₹{loan['max_loan_amount']:,.0f} @ {loan['effective_rate']}% "
                                    f"· EMI ₹{loan['emi']:,.0f}/mo · {loan['suggested_tenure']} months")
                        if loan.get("subsidy"):
                            st.caption(f"💰 {loan['subsidy']}")
            else:
                st.warning("No loans eligible at current score. See improvement path below.")

            # Improvement path
            if ind_loan_recs.get("improvement_path"):
                with st.expander("📈 Credit Improvement Path"):
                    for imp in ind_loan_recs["improvement_path"]:
                        if imp["type"] == "score_upgrade":
                            st.markdown(f"🎯 **{imp['title']}** (+{imp.get('gap', 0):.0f} pts)")
                            for action in imp.get("actions", []):
                                st.markdown(f"  - {action}")
                        elif imp["type"] == "maintenance":
                            st.success(f"✅ {imp['title']}")
        except Exception as e:
            st.caption(f"Loan recommendation engine: {e}")

        # User raw data
        with st.expander("📁 Raw User Data"):
            display_cols = [c for c in user_row.index if not c.startswith("detail_") and c != "monthly_incomes" and c != "platforms"]
            st.dataframe(pd.DataFrame([user_row[display_cols]]))

    # ── Page: Loan Search ──────────────────────────────────────────────
    elif page == "🔍 Loan Search":
        st.markdown("## 🔍 Loan Search & Eligibility Checker")
        st.markdown(
            "Browse **34+ loan products** across transaction-based and persona-specific schemes. "
            "Search for any loan, then check if **you** are eligible."
        )

        # ── Tab Layout ──
        search_tab, eligibility_tab = st.tabs(["🗂️ Browse & Search Loans", "✅ Check My Eligibility"])

        # ===================================================================
        # TAB 1: Browse & Search
        # ===================================================================
        with search_tab:
            st.markdown("### 🗂️ Loan Catalog")

            # --- Filters Row ---
            fc1, fc2, fc3, fc4 = st.columns(4)
            with fc1:
                search_query = st.text_input(
                    "🔎 Search by keyword",
                    placeholder="e.g. KCC, Mudra, education, gold...",
                    key="loan_search_query",
                )
            with fc2:
                all_categories = ["All"] + get_loan_categories()
                cat_filter = st.selectbox("📂 Category", all_categories, key="loan_cat_filter")
            with fc3:
                source_options = ["All", "Transaction-based (Bank History)", "Persona-based (No Bank)"]
                source_sel = st.selectbox("📋 Loan Type", source_options, key="loan_source_filter")
            with fc4:
                persona_options = ["All", "Farmer", "Student", "Street Vendor", "Homemaker", "General (No Bank)"]
                persona_sel = st.selectbox("👤 Persona", persona_options, key="loan_persona_filter")

            # --- Advanced Filters ---
            with st.expander("⚙️ Advanced Filters"):
                afc1, afc2, afc3 = st.columns(3)
                with afc1:
                    collateral_opts = ["Any", "No Collateral Only", "Collateral Required"]
                    collateral_sel = st.selectbox("🔒 Collateral", collateral_opts, key="loan_collateral_filter")
                with afc2:
                    subsidy_only = st.checkbox("💰 Subsidized loans only", key="loan_subsidy_filter")
                with afc3:
                    max_interest = st.slider("📉 Max interest rate (%)", 0.0, 42.0, 0.0, 0.5, key="loan_max_rate")

                min_loan_amount = st.number_input(
                    "💵 Minimum loan amount (₹)", min_value=0, value=0, step=5000,
                    key="loan_min_amount",
                )

            # --- Map filter selections ---
            source_map = {"All": "", "Transaction-based (Bank History)": "transaction", "Persona-based (No Bank)": "persona"}
            persona_map = {"All": "", "Farmer": "farmer", "Student": "student", "Street Vendor": "street_vendor", "Homemaker": "homemaker", "General (No Bank)": "general_no_bank"}
            collateral_map = {"Any": "", "No Collateral Only": "no", "Collateral Required": "yes"}

            filtered_loans = search_loans(
                query=search_query,
                category="" if cat_filter == "All" else cat_filter,
                source_filter=source_map.get(source_sel, ""),
                persona_filter=persona_map.get(persona_sel, ""),
                collateral_filter=collateral_map.get(collateral_sel, ""),
                subsidy_filter=subsidy_only,
                max_rate=max_interest,
                min_amount=float(min_loan_amount),
            )

            # --- Results Count ---
            total_catalog = len(get_all_loans_catalog())
            st.markdown(
                f"**Showing {len(filtered_loans)} of {total_catalog} loans**"
                + (f" matching \"{search_query}\"" if search_query else "")
            )

            if not filtered_loans:
                st.warning("No loans match your filters. Try broadening your search.")
            else:
                # --- Display loans as cards ---
                for i in range(0, len(filtered_loans), 3):
                    cols = st.columns(3)
                    for j, col in enumerate(cols):
                        if i + j < len(filtered_loans):
                            loan = filtered_loans[i + j]
                            with col:
                                # Source badge
                                if loan["source"] == "transaction":
                                    badge = "🏦 Transaction-based"
                                else:
                                    persona_label = (loan.get("persona", "") or "").replace("_", " ").title()
                                    badge = f"🌍 {persona_label}"

                                rate_low, rate_high = loan["interest_range"]
                                amt_low, amt_high = loan["amount_range"]
                                tenure_low, tenure_high = loan["tenure_range"]

                                st.markdown(f"""
<div style="border:1px solid #e2e8f0; border-radius:12px; padding:16px; margin-bottom:12px;
            background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);">
    <div style="font-size:0.75rem; color:#64748b; margin-bottom:4px;">{badge}</div>
    <div style="font-size:1.2rem; font-weight:700;">{loan['icon']} {loan['name']}</div>
    <div style="font-size:0.8rem; color:#475569; margin:6px 0;">{loan['description']}</div>
    <div style="display:flex; gap:12px; flex-wrap:wrap; margin-top:8px;">
        <span style="background:#dbeafe; color:#1e40af; padding:2px 8px; border-radius:6px; font-size:0.75rem;">
            {rate_low}% – {rate_high}%
        </span>
        <span style="background:#dcfce7; color:#166534; padding:2px 8px; border-radius:6px; font-size:0.75rem;">
            ₹{amt_low:,} – ₹{amt_high:,}
        </span>
        <span style="background:#fef3c7; color:#92400e; padding:2px 8px; border-radius:6px; font-size:0.75rem;">
            {tenure_low} – {tenure_high} months
        </span>
    </div>
    <div style="margin-top:8px; font-size:0.75rem;">
        {"🔓 No Collateral" if not loan.get("collateral") else "🔒 Collateral Required"}
        {"  •  💰 <b>Subsidized</b>" if loan.get("subsidy") else ""}
    </div>
    <div style="font-size:0.7rem; color:#64748b; margin-top:4px;">
        Min Score: {loan.get("min_score", "N/A")} &nbsp;|&nbsp;
        Category: {loan.get("category", "N/A")}
    </div>
</div>
""", unsafe_allow_html=True)

                                with st.expander(f"📋 Details — {loan['name']}", expanded=False):
                                    st.markdown(f"**Lenders:** {', '.join(loan.get('lenders', []))}")
                                    st.markdown(f"**Documents:** {', '.join(loan.get('documents', []))}")
                                    if loan.get("subsidy"):
                                        st.success(f"**Subsidy:** {loan['subsidy']}")
                                    if loan.get("eligibility_criteria"):
                                        st.info(
                                            "**Special Criteria:** "
                                            + ", ".join(c.replace("_", " ").title() for c in loan["eligibility_criteria"])
                                        )

        # ===================================================================
        # TAB 2: Check Eligibility
        # ===================================================================
        with eligibility_tab:
            st.markdown("### ✅ Check Your Eligibility for a Specific Loan")
            st.markdown(
                "Select any loan, enter your details, and get an **instant eligibility verdict** "
                "with gap analysis and improvement steps."
            )

            # --- Step 1: Select the loan ---
            st.markdown("#### Step 1: Choose a Loan")
            elig_c1, elig_c2 = st.columns(2)
            with elig_c1:
                elig_source = st.radio(
                    "Are you checking with bank history or without?",
                    ["🏦 With Bank History (Transaction-based)", "🌍 Without Bank History (Persona-based)"],
                    key="elig_source_radio",
                )
            with elig_c2:
                if "Persona" in elig_source:
                    elig_persona_opts = {
                        "Farmer": "farmer", "Student": "student",
                        "Street Vendor": "street_vendor", "Homemaker": "homemaker",
                        "General (No Bank Account)": "general_no_bank",
                    }
                    elig_persona_label = st.selectbox(
                        "Select your persona", list(elig_persona_opts.keys()),
                        key="elig_persona_sel",
                    )
                    elig_persona = elig_persona_opts[elig_persona_label]
                else:
                    elig_persona = ""
                    st.info("Transaction-based loans — no persona needed")

            # Build loan selection dropdown
            if "Transaction" in elig_source:
                loan_options = {v["name"]: k for k, v in TRANSACTION_LOANS.items()}
                elig_source_key = "transaction"
            else:
                persona_catalog = PERSONA_LOANS.get(elig_persona, {})
                loan_options = {v["name"]: k for k, v in persona_catalog.items()}
                elig_source_key = "persona"

            if not loan_options:
                st.warning("No loans available for this selection.")
            else:
                selected_loan_name = st.selectbox(
                    "🏷️ Select Loan",
                    list(loan_options.keys()),
                    key="elig_loan_sel",
                )
                selected_loan_key = loan_options[selected_loan_name]

                # Show selected loan quick info
                if elig_source_key == "transaction":
                    sel_loan = TRANSACTION_LOANS[selected_loan_key]
                else:
                    sel_loan = PERSONA_LOANS.get(elig_persona, {}).get(selected_loan_key, {})

                if sel_loan:
                    rate_l, rate_h = sel_loan["interest_range"]
                    amt_l, amt_h = sel_loan["amount_range"]
                    st.markdown(f"""
<div style="border-left:4px solid #3b82f6; padding:10px 16px; background:#eff6ff; border-radius:0 8px 8px 0; margin:8px 0;">
    <b>{sel_loan.get('icon','')} {sel_loan['name']}</b> — {sel_loan['description']}<br/>
    <span style="color:#1e40af;">Rate: {rate_l}%–{rate_h}%</span> &nbsp;|&nbsp;
    <span style="color:#166534;">Amount: ₹{amt_l:,}–₹{amt_h:,}</span> &nbsp;|&nbsp;
    Min Score: {sel_loan.get('min_score', 'N/A')}
    {f" &nbsp;|&nbsp; <b style='color:#b45309;'>Subsidy: {sel_loan['subsidy']}</b>" if sel_loan.get('subsidy') else ""}
</div>
""", unsafe_allow_html=True)

                # --- Step 2: Enter your details ---
                st.markdown("#### Step 2: Enter Your Details")
                det_c1, det_c2, det_c3 = st.columns(3)
                with det_c1:
                    elig_score = st.number_input(
                        "Your Trust Score (300–900)", min_value=300, max_value=900,
                        value=550, step=10, key="elig_score_input",
                    )
                with det_c2:
                    elig_income = st.number_input(
                        "Monthly Income (₹)", min_value=0, value=15000,
                        step=1000, key="elig_income_input",
                    )
                with det_c3:
                    elig_expenses = st.number_input(
                        "Monthly Expenses (₹)", min_value=0, value=5000,
                        step=500, key="elig_expenses_input",
                    )

                det_c4, det_c5, det_c6 = st.columns(3)
                with det_c4:
                    elig_existing_emi = st.number_input(
                        "Existing EMI (₹/month)", min_value=0, value=0,
                        step=500, key="elig_emi_input",
                    )
                with det_c5:
                    elig_desired_amount = st.number_input(
                        "Desired Loan Amount (₹, 0 = auto)",
                        min_value=0, value=0, step=5000,
                        key="elig_desired_amount",
                    )
                with det_c6:
                    elig_desired_tenure = st.number_input(
                        "Desired Tenure (months, 0 = auto)",
                        min_value=0, value=0, step=3,
                        key="elig_desired_tenure",
                    )

                # Persona-specific fields
                elig_persona_data = {}
                if elig_source_key == "persona" and elig_persona:
                    criteria = sel_loan.get("eligibility_criteria", []) if sel_loan else []
                    if criteria:
                        st.markdown("**Persona-Specific Details:**")
                        pc_cols = st.columns(min(len(criteria), 3))
                        for ci, criterion in enumerate(criteria):
                            with pc_cols[ci % len(pc_cols)]:
                                label = criterion.replace("_", " ").title()
                                if criterion in ("owns_land", "has_license", "is_shg_member",
                                                  "has_enterprise", "has_internship",
                                                  "is_group_member", "has_warehouse_receipt"):
                                    elig_persona_data[criterion] = st.checkbox(
                                        label, value=False, key=f"elig_pd_{criterion}"
                                    )
                                elif criterion in ("land_acres",):
                                    elig_persona_data[criterion] = st.number_input(
                                        f"{label} (acres)", min_value=0.0, value=2.0,
                                        step=0.5, key=f"elig_pd_{criterion}"
                                    )
                                elif criterion in ("crops_per_year",):
                                    elig_persona_data[criterion] = st.number_input(
                                        label, min_value=1, value=2, step=1,
                                        key=f"elig_pd_{criterion}"
                                    )
                                elif criterion in ("years_in_trade",):
                                    elig_persona_data[criterion] = st.number_input(
                                        label, min_value=0, value=2, step=1,
                                        key=f"elig_pd_{criterion}"
                                    )
                                elif criterion in ("score_value",):
                                    elig_persona_data[criterion] = elig_score
                                else:
                                    elig_persona_data[criterion] = st.text_input(
                                        label, key=f"elig_pd_{criterion}"
                                    )

                # --- Step 3: Check Eligibility Button ---
                st.markdown("---")
                if st.button("🔍 Check My Eligibility", type="primary", use_container_width=True, key="elig_check_btn"):
                    result = check_loan_eligibility(
                        loan_key=selected_loan_key,
                        source=elig_source_key,
                        persona=elig_persona,
                        score=float(elig_score),
                        monthly_income=float(elig_income),
                        monthly_expenses=float(elig_expenses),
                        existing_emi=float(elig_existing_emi),
                        persona_data=elig_persona_data,
                        desired_amount=float(elig_desired_amount),
                        desired_tenure=int(elig_desired_tenure),
                    )

                    st.markdown("---")

                    # --- Verdict Banner ---
                    verdict = result["verdict"]
                    verdict_config = {
                        "ELIGIBLE": ("✅ You Are Eligible!", "#22c55e", "#f0fdf4"),
                        "ELIGIBLE_WITH_CAUTION": ("⚠️ Eligible with Conditions", "#eab308", "#fefce8"),
                        "MICRO_ONLY": ("🔸 Eligible for Micro Amount Only", "#f97316", "#fff7ed"),
                        "NOT_ELIGIBLE": ("❌ Not Eligible Currently", "#ef4444", "#fef2f2"),
                        "LOAN_NOT_FOUND": ("❓ Loan Not Found", "#64748b", "#f8fafc"),
                    }
                    v_title, v_color, v_bg = verdict_config.get(
                        verdict, ("❓ Unknown", "#64748b", "#f8fafc")
                    )

                    st.markdown(f"""
<div style="background:{v_bg}; border:2px solid {v_color}; border-radius:12px;
            padding:20px; text-align:center; margin:12px 0;">
    <div style="font-size:1.8rem; font-weight:800; color:{v_color};">{v_title}</div>
    <div style="font-size:1.1rem; margin-top:4px;">
        {result['loan_icon']} <b>{result['loan_name']}</b> &nbsp;|&nbsp;
        Score: {result['score_used']:.0f} ({result['tier']}) &nbsp;|&nbsp;
        Income: ₹{elig_income:,}/mo
    </div>
</div>
""", unsafe_allow_html=True)

                    # --- Checks Passed / Failed ---
                    chk_c1, chk_c2 = st.columns(2)
                    with chk_c1:
                        st.markdown("##### ✅ Checks Passed")
                        if result["reasons_pass"]:
                            for reason in result["reasons_pass"]:
                                st.markdown(f"- ✅ {reason}")
                        else:
                            st.markdown("_No checks passed_")
                    with chk_c2:
                        st.markdown("##### ❌ Checks Failed")
                        if result["reasons_fail"]:
                            for reason in result["reasons_fail"]:
                                st.markdown(f"- ❌ {reason}")
                        else:
                            st.markdown("_All checks passed!_")

                    # --- Gap Analysis Table ---
                    if result["gap_analysis"]:
                        st.markdown("##### 📊 Gap Analysis")
                        gap_df = pd.DataFrame(result["gap_analysis"])
                        gap_df.columns = [c.replace("_", " ").title() for c in gap_df.columns]
                        st.dataframe(gap_df, use_container_width=True, hide_index=True)

                    # --- Loan Details (if eligible) ---
                    if result["loan_details"]:
                        st.markdown("##### 💳 Loan Details")
                        ld = result["loan_details"]
                        ld_c1, ld_c2, ld_c3, ld_c4 = st.columns(4)
                        ld_c1.metric("Effective Rate", f"{ld['effective_rate']}%")
                        ld_c2.metric("Max Eligible Amount", f"₹{ld['max_eligible_amount']:,.0f}")
                        ld_c3.metric("Monthly EMI", f"₹{ld['emi']:,.0f}")
                        ld_c4.metric("Total Payable", f"₹{ld['total_payable']:,.0f}")

                        ld2_c1, ld2_c2, ld2_c3, ld2_c4 = st.columns(4)
                        ld2_c1.metric("Loan Amount", f"₹{ld['actual_amount']:,.0f}")
                        ld2_c2.metric("Tenure", f"{ld['actual_tenure_months']} months")
                        ld2_c3.metric("Total Interest", f"₹{ld['total_interest']:,.0f}")
                        ld2_c4.metric("Processing Fee", ld["processing_fee"])

                        if ld.get("subsidy"):
                            st.success(f"💰 **Subsidy Available:** {ld['subsidy']}")
                        if ld.get("collateral_required"):
                            st.warning("🔒 Collateral required for this loan")
                        else:
                            st.info("🔓 No collateral needed")

                        # Documents & Lenders
                        doc_c1, doc_c2 = st.columns(2)
                        with doc_c1:
                            st.markdown("**📄 Documents Needed:**")
                            for doc in ld.get("documents_needed", []):
                                st.markdown(f"- {doc}")
                        with doc_c2:
                            st.markdown("**🏦 Available Lenders:**")
                            for lender in ld.get("lenders", []):
                                st.markdown(f"- {lender}")

                        # Repayment Schedule
                        with st.expander("📅 Repayment Schedule (first 12 months)"):
                            schedule = generate_repayment_schedule(
                                ld["actual_amount"], ld["effective_rate"],
                                ld["actual_tenure_months"]
                            )
                            if schedule:
                                show_months = min(12, len(schedule))
                                sched_df = pd.DataFrame(schedule[:show_months])
                                sched_df.columns = ["Month", "EMI (₹)", "Principal (₹)",
                                                     "Interest (₹)", "Balance (₹)"]
                                st.dataframe(sched_df, use_container_width=True, hide_index=True)

                                # Visual: principal vs interest over time
                                fig_sched = go.Figure()
                                fig_sched.add_trace(go.Bar(
                                    x=[s["month"] for s in schedule[:show_months]],
                                    y=[s["principal"] for s in schedule[:show_months]],
                                    name="Principal", marker_color="#22c55e",
                                ))
                                fig_sched.add_trace(go.Bar(
                                    x=[s["month"] for s in schedule[:show_months]],
                                    y=[s["interest"] for s in schedule[:show_months]],
                                    name="Interest", marker_color="#ef4444",
                                ))
                                fig_sched.update_layout(
                                    barmode="stack", height=300,
                                    title="Monthly EMI Breakdown",
                                    xaxis_title="Month", yaxis_title="Amount (₹)",
                                )
                                st.plotly_chart(fig_sched, use_container_width=True)

                    # --- Improvement Steps ---
                    if result["improvement_steps"]:
                        st.markdown("##### 🛤️ Next Steps")
                        for step in result["improvement_steps"]:
                            icon = "✅" if verdict == "ELIGIBLE" else "💡"
                            st.markdown(f"- {icon} {step}")

                    # --- Repayment Capacity Summary ---
                    with st.expander("📊 Repayment Capacity Details"):
                        rc = result["repayment_capacity"]
                        rc_c1, rc_c2, rc_c3, rc_c4 = st.columns(4)
                        rc_c1.metric("Monthly Income", f"₹{rc['monthly_income']:,.0f}")
                        rc_c2.metric("Current FOIR", f"{rc['current_foir']:.1%}")
                        rc_c3.metric("FOIR Limit", f"{rc['foir_limit']:.0%}")
                        rc_c4.metric("Max New EMI", f"₹{rc['max_new_emi']:,.0f}")

                        if rc.get("risk_flags"):
                            for flag in rc["risk_flags"]:
                                st.warning(f"⚠️ {flag}")

    # ── Page: Portfolio Analytics ───────────────────────────────────────
    elif page == "📈 Portfolio Analytics":
        st.markdown("## 📈 Portfolio Analytics")

        # Summary metrics
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Total Users", len(df))
        c2.metric("Avg Trust Score", f"{df['final_trust_score'].mean():.0f}")
        c3.metric("Default Rate", f"{df['default'].mean():.1%}")
        c4.metric("Avg Income", f"₹{df['mean_income'].mean():,.0f}")
        c5.metric("Avg Risk", f"{df['risk_probability'].mean():.1%}")

        st.markdown("---")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### Score Distribution")
            st.plotly_chart(create_score_distribution(df), use_container_width=True)

        with col2:
            st.markdown("#### Grade Breakdown")
            grade_counts = df["grade"].value_counts()
            fig = px.pie(
                values=grade_counts.values,
                names=grade_counts.index,
                color_discrete_sequence=["#22c55e", "#84cc16", "#eab308", "#f97316", "#ef4444"],
                hole=0.4,
            )
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#e2e8f0"),
                height=300,
                margin=dict(t=10, b=10, l=10, r=10),
            )
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")

        col3, col4 = st.columns(2)

        with col3:
            st.markdown("#### Risk vs Trust Score")
            fig = px.scatter(
                df, x="final_trust_score", y="risk_probability",
                color="grade",
                color_discrete_map={
                    "Excellent": "#22c55e", "Good": "#84cc16",
                    "Fair": "#eab308", "Poor": "#f97316", "Very Poor": "#ef4444"
                },
                hover_data=["user_id", "mean_income"],
                opacity=0.6,
            )
            fig.update_layout(
                xaxis_title="Trust Score", yaxis_title="Risk Probability",
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#e2e8f0"), height=350,
                margin=dict(t=10, b=30, l=10, r=10),
            )
            st.plotly_chart(fig, use_container_width=True)

        with col4:
            st.markdown("#### Sub-Score Averages")
            sub_cols = ["sub_financial_stability", "sub_payment_discipline",
                        "sub_digital_behavior", "sub_work_reliability"]
            sub_labels = ["Financial\nStability", "Payment\nDiscipline",
                          "Digital\nBehavior", "Work\nReliability"]
            avgs = [df[c].mean() for c in sub_cols]

            fig = go.Figure(go.Bar(
                x=sub_labels, y=avgs,
                marker_color=["#6366f1", "#8b5cf6", "#06b6d4", "#f59e0b"],
                text=[f"{v:.1f}" for v in avgs],
                textposition="outside",
            ))
            fig.update_layout(
                yaxis=dict(range=[0, 100], title="Average Score"),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#e2e8f0"), height=350,
                margin=dict(t=10, b=30, l=10, r=10),
            )
            st.plotly_chart(fig, use_container_width=True)

        # Feature importance
        st.markdown("#### 🔬 Global Feature Importance")
        try:
            explainer = ScoreExplainer(model)
            explainer.initialize(df)
            fig = explainer.plot_global_importance(df)
            st.pyplot(fig)
        except Exception:
            imp = model.get_feature_importance()
            if imp:
                imp_df = pd.DataFrame({
                    "Feature": [FEATURE_LABELS.get(k, k) for k in list(imp.keys())[:10]],
                    "Importance": list(imp.values())[:10]
                })
                fig = px.bar(imp_df, x="Importance", y="Feature", orientation="h",
                             color_discrete_sequence=["#6366f1"])
                fig.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#e2e8f0"), height=350,
                )
                st.plotly_chart(fig, use_container_width=True)

        # Full leaderboard
        st.markdown("#### 📋 User Leaderboard")
        leaderboard_cols = ["user_id", "final_trust_score", "grade", "risk_probability",
                            "mean_income", "profile", "sub_financial_stability",
                            "sub_payment_discipline", "sub_digital_behavior",
                            "sub_work_reliability"]
        lb_df = df[leaderboard_cols].sort_values("final_trust_score", ascending=False)
        st.dataframe(lb_df, use_container_width=True, height=400)

    # ── Page: Model Performance ─────────────────────────────────────────
    elif page == "🤖 Model Performance":
        st.markdown("## 🤖 AI Model Performance")

        st.markdown("### Architecture")
        st.code("""
Raw Financial Data
        ↓
Feature Engineering Engine (10 Alternative Criteria)
        ↓
4 Sub-Scores (Rule-Based Structured Scoring)
        ↓
ML Risk Adjustment Model (XGBoost + Logistic Regression)
        ↓
Final Alternative Trust Score (300 – 900)
        ↓
Explainability Layer (SHAP)
        """, language="text")

        st.markdown("---")

        # Model metrics
        st.markdown("### Model Metrics")

        for model_name, model_metrics in metrics.items():
            if model_name == "cross_val_auc":
                continue
            with st.expander(f"📊 {model_name.replace('_', ' ').title()}", expanded=True):
                mc1, mc2 = st.columns(2)
                mc1.metric("Accuracy", f"{model_metrics['accuracy']:.2%}")
                mc2.metric("ROC AUC", f"{model_metrics['roc_auc']:.4f}")

                # Confusion matrix
                cm = np.array(model_metrics["confusion_matrix"])
                fig = px.imshow(
                    cm,
                    labels=dict(x="Predicted", y="Actual", color="Count"),
                    x=["Non-Default", "Default"],
                    y=["Non-Default", "Default"],
                    color_continuous_scale="Blues",
                    text_auto=True,
                )
                fig.update_layout(
                    height=300,
                    paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#e2e8f0"),
                    margin=dict(t=10, b=10, l=10, r=10),
                )
                st.plotly_chart(fig, use_container_width=True)

                # Classification report
                report = model_metrics["classification_report"]
                report_df = pd.DataFrame(report).transpose()
                st.dataframe(report_df.round(3), use_container_width=True)

        if "cross_val_auc" in metrics:
            st.markdown("### Cross-Validation")
            cv = metrics["cross_val_auc"]
            st.info(f"5-Fold CV AUC: **{cv['mean']:.4f}** ± {cv['std']:.4f}")

        # Feature importance
        st.markdown("### Feature Importance (Model)")
        imp = model.get_feature_importance()
        if imp:
            imp_df = pd.DataFrame({
                "Feature": [FEATURE_LABELS.get(k, k) for k in imp.keys()],
                "Importance": list(imp.values())
            })
            fig = px.bar(imp_df, x="Importance", y="Feature", orientation="h",
                         color="Importance", color_continuous_scale="Viridis")
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#e2e8f0"), height=500,
                margin=dict(t=10, b=30, l=10, r=10),
                yaxis=dict(autorange="reversed"),
            )
            st.plotly_chart(fig, use_container_width=True)

    # ── Page: Score Simulator ───────────────────────────────────────────
    elif page == "🧪 Score Simulator":
        st.markdown("## 🧪 Score Simulator")
        st.markdown("Adjust parameters to see how they affect the trust score in real-time.")

        with st.form("simulator_form"):
            st.markdown("#### 💰 Income & Cash Flow")
            sim_c1, sim_c2, sim_c3 = st.columns(3)
            with sim_c1:
                mean_income = st.slider("Mean Monthly Income (₹)", 5000, 60000, 25000, 1000)
            with sim_c2:
                income_stability = st.slider("Income Stability", 0.0, 1.0, 0.75, 0.05)
            with sim_c3:
                cash_flow = st.slider("Cash Flow Health Ratio", 0.0, 1.0, 0.45, 0.05)

            st.markdown("#### 📱 Payment Behavior")
            sim_c4, sim_c5, sim_c6 = st.columns(3)
            with sim_c4:
                utility_score = st.slider("Utility Bill Timeliness", 0.0, 1.0, 0.70, 0.05)
            with sim_c5:
                emi_score = st.slider("EMI-like Behavior", 0.0, 1.0, 0.50, 0.05)
            with sim_c6:
                recharge_reg = st.slider("Recharge Regularity", 0.0, 1.0, 0.65, 0.05)

            st.markdown("#### 🔄 Digital Behavior")
            sim_c7, sim_c8 = st.columns(2)
            with sim_c7:
                txn_regularity = st.slider("Transaction Regularity", 0.0, 1.0, 0.60, 0.05)
            with sim_c8:
                expense_ratio = st.slider("Essential Expense Ratio", 0.0, 1.0, 0.65, 0.05)

            st.markdown("#### 🏢 Work & Resilience")
            sim_c9, sim_c10, sim_c11 = st.columns(3)
            with sim_c9:
                work_rel = st.slider("Work Reliability", 0.0, 1.0, 0.70, 0.05)
            with sim_c10:
                savings_score = st.slider("Savings Discipline", 0.0, 1.0, 0.40, 0.05)
            with sim_c11:
                shock_recovery = st.slider("Shock Recovery", 0.0, 1.0, 0.80, 0.05)

            sim_c12, sim_c13 = st.columns(2)
            with sim_c12:
                income_diversity = st.slider("Income Diversity", 0.0, 1.0, 0.40, 0.05)
            with sim_c13:
                income_trend = st.slider("Income Trend", -0.5, 0.5, 0.05, 0.01)

            submitted = st.form_submit_button("🔮 Calculate Score", use_container_width=True)

        if submitted:
            # Build a synthetic row
            sim_row = pd.Series({
                "feat_income_stability": income_stability,
                "feat_income_trend": income_trend,
                "feat_cash_flow_ratio": cash_flow,
                "feat_income_diversity": income_diversity,
                "feat_utility_score": utility_score,
                "feat_emi_score": emi_score,
                "feat_txn_regularity": txn_regularity,
                "feat_expense_score": expense_ratio,
                "feat_savings_score": savings_score,
                "feat_work_reliability": work_rel,
                "feat_shock_recovery": shock_recovery,
                "recharge_regularity": recharge_reg,
                "mean_income": mean_income,
                "income_std": mean_income * (1 - income_stability) * 0.3,
                "num_income_sources": max(1, int(income_diversity * 5)),
                "tenure_months": int(work_rel * 48),
                "platform_rating": 3.0 + work_rel * 2.0,
                "active_days_per_month": int(work_rel * 30),
                "avg_monthly_savings": int(savings_score * mean_income * 0.2),
                "total_transactions": int(txn_regularity * 150),
            })

            base_result = compute_base_score(sim_row)
            base_score = base_result["base_trust_score"]

            # ML risk prediction
            try:
                risk_prob = model.predict_single(sim_row)
            except Exception:
                risk_prob = 0.2

            final = compute_final_score(base_score, risk_prob, sim_row)

            # Display results
            st.markdown("---")
            r1, r2 = st.columns([1, 1])

            with r1:
                st.plotly_chart(
                    create_gauge(final["final_trust_score"], final["grade"], final["grade_color"]),
                    use_container_width=True
                )

            with r2:
                rm1, rm2, rm3 = st.columns(3)
                rm1.metric("Final Score", f"{final['final_trust_score']:.0f}")
                rm2.metric("Risk Probability", f"{risk_prob:.1%}")
                rm3.metric("Confidence", f"{final['confidence']:.0%}")

                st.markdown(f"**Grade:** {final['grade']}")
                st.markdown(f"**Base Score:** {base_score:.0f}")

                # Sub-score breakdown
                breakdown = get_score_breakdown(pd.Series(base_result))
                for cat_name, cat_data in breakdown.items():
                    score_pct = cat_data["score"]
                    bar_color = "#22c55e" if score_pct >= 70 else "#eab308" if score_pct >= 40 else "#ef4444"
                    st.markdown(f"**{cat_name}**: {score_pct:.1f}/100 ({cat_data['weight']})")
                    st.progress(int(min(score_pct, 100)))

            # Loan eligibility recommendation
            st.markdown("### 💳 Loan Eligibility Recommendation")
            score_val = final["final_trust_score"]
            if score_val >= 750:
                st.success(f"✅ **Eligible for Premium Loans** — Score: {score_val:.0f}\n\n"
                           f"Up to ₹{mean_income * 6:,.0f} | Interest: 10-12% | Tenure: 24 months")
            elif score_val >= 650:
                st.info(f"✅ **Eligible for Standard Loans** — Score: {score_val:.0f}\n\n"
                        f"Up to ₹{mean_income * 4:,.0f} | Interest: 14-16% | Tenure: 12 months")
            elif score_val >= 500:
                st.warning(f"⚠️ **Eligible for Micro Loans** — Score: {score_val:.0f}\n\n"
                           f"Up to ₹{mean_income * 2:,.0f} | Interest: 18-22% | Tenure: 6 months")
            else:
                st.error(f"❌ **Not Eligible Currently** — Score: {score_val:.0f}\n\n"
                         f"Recommendation: Build payment history for 3-6 months.")

    # ── Page: Upload & Score ────────────────────────────────────────────
    elif page == "📤 Upload & Score":
        st.markdown("## 📤 Upload Bank Statement & Get Your Score")
        st.markdown(
            "Upload your bank or UPI transaction history (CSV/Excel) to get "
            "an instant AI-powered alternative credit score."
        )

        # File upload + sample download
        col_up1, col_up2 = st.columns([2, 1])
        with col_up1:
            uploaded_file = st.file_uploader(
                "Upload your bank statement",
                type=["csv", "xlsx", "xls"],
                help="Your file should have columns for Date, Description, "
                     "and Debit/Credit amounts.",
            )
        with col_up2:
            st.markdown("**No file handy?**")
            sample_df = generate_sample_statement()
            csv_bytes = sample_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "📥 Download Sample Statement",
                csv_bytes,
                "sample_bank_statement.csv",
                "text/csv",
                use_container_width=True,
            )

        if uploaded_file is not None:
            try:
                parser = TransactionParser()
                file_ext = uploaded_file.name.rsplit(".", 1)[-1].lower()
                parsed_df = parser.parse_file(uploaded_file, file_ext)
                categorized_df = parser.auto_categorize(parsed_df)

                # Parsing summary
                psummary = parser.get_parsing_summary()
                st.success(
                    f"✅ Successfully parsed **{psummary['total_transactions']}** "
                    f"transactions spanning **{psummary['months_covered']} months** "
                    f"({psummary['date_range']})"
                )

                mc1, mc2, mc3, mc4 = st.columns(4)
                mc1.metric("Total Transactions", psummary["total_transactions"])
                mc2.metric("Credits", psummary["total_credits"])
                mc3.metric("Debits", psummary["total_debits"])
                mc4.metric("Months Covered", psummary["months_covered"])

                st.markdown("---")

                # Tabs: preview, breakdown, trend
                tab1, tab2, tab3 = st.tabs(
                    ["📋 Transactions", "📊 Category Breakdown", "📈 Monthly Trend"]
                )

                with tab1:
                    display_df = categorized_df[
                        ["date", "description", "amount", "type",
                         "category", "category_confidence"]
                    ].copy()
                    display_df["date"] = display_df["date"].dt.strftime("%d-%m-%Y")
                    display_df["amount"] = display_df["amount"].apply(
                        lambda x: f"₹{x:,.2f}"
                    )
                    display_df["category_confidence"] = (
                        display_df["category_confidence"].apply(lambda x: f"{x:.0%}")
                    )
                    display_df.columns = [
                        "Date", "Description", "Amount", "Type",
                        "Category", "Confidence",
                    ]
                    st.dataframe(display_df, use_container_width=True, height=400)

                with tab2:
                    cat_summary = parser.get_category_summary()
                    if len(cat_summary) > 0:
                        col_pie, col_bar = st.columns(2)
                        with col_pie:
                            fig = px.pie(
                                cat_summary, values="total", names="category",
                                title="Spending by Category",
                                color_discrete_sequence=px.colors.qualitative.Set3,
                                hole=0.35,
                            )
                            fig.update_layout(
                                paper_bgcolor="rgba(0,0,0,0)",
                                font=dict(color="#e2e8f0"),
                                height=400,
                                margin=dict(t=40, b=10, l=10, r=10),
                            )
                            st.plotly_chart(fig, use_container_width=True)

                        with col_bar:
                            fig = px.bar(
                                cat_summary, x="total", y="category",
                                orientation="h",
                                title="Spending Amount by Category",
                                color="category",
                                color_discrete_sequence=px.colors.qualitative.Set3,
                            )
                            fig.update_layout(
                                paper_bgcolor="rgba(0,0,0,0)",
                                plot_bgcolor="rgba(0,0,0,0)",
                                font=dict(color="#e2e8f0"),
                                height=400, showlegend=False,
                                xaxis_title="Total (₹)", yaxis_title="",
                                margin=dict(t=40, b=10, l=10, r=10),
                            )
                            st.plotly_chart(fig, use_container_width=True)

                with tab3:
                    monthly_summ = parser.get_monthly_summary()
                    if len(monthly_summ) > 0:
                        fig = go.Figure()
                        fig.add_trace(go.Bar(
                            x=monthly_summ["month"],
                            y=monthly_summ.get("credit", [0]),
                            name="Income", marker_color="#22c55e",
                        ))
                        fig.add_trace(go.Bar(
                            x=monthly_summ["month"],
                            y=monthly_summ.get("debit", [0]),
                            name="Expenses", marker_color="#ef4444",
                        ))
                        if "net_savings" in monthly_summ.columns:
                            fig.add_trace(go.Scatter(
                                x=monthly_summ["month"],
                                y=monthly_summ["net_savings"],
                                name="Net Savings",
                                line=dict(color="#6366f1", width=3),
                                mode="lines+markers",
                            ))
                        fig.update_layout(
                            barmode="group",
                            paper_bgcolor="rgba(0,0,0,0)",
                            plot_bgcolor="rgba(0,0,0,0)",
                            font=dict(color="#e2e8f0"),
                            height=350,
                            title="Monthly Income vs Expenses",
                            margin=dict(t=40, b=30, l=10, r=10),
                        )
                        st.plotly_chart(fig, use_container_width=True)

                st.markdown("---")

                # Supplementary info
                st.markdown("### 🔧 Supplementary Information")
                st.markdown(
                    "*These details help improve scoring accuracy "
                    "but are optional (defaults provided).*"
                )

                sup_c1, sup_c2, sup_c3 = st.columns(3)
                with sup_c1:
                    platform_rating = st.slider(
                        "Platform Rating (if gig worker)",
                        1.0, 5.0, 4.0, 0.1,
                        help="Your average rating on gig platforms.",
                    )
                with sup_c2:
                    active_days = st.slider(
                        "Active Work Days / Month",
                        1, 30, 20,
                        help="Average number of days you work per month.",
                    )
                with sup_c3:
                    st.markdown("")
                    generate_btn = st.button(
                        "🔮 Generate Credit Score",
                        use_container_width=True,
                        type="primary",
                    )

                if generate_btn:
                    with st.spinner(
                        "🧠 AI is analyzing your financial profile..."
                    ):
                        # Extract profile from transactions
                        profile = parser.extract_profile(
                            platform_rating=platform_rating,
                            active_days=active_days,
                        )

                        # Feature engineering
                        features = extract_all_features(profile)
                        for key, val in features.items():
                            if not isinstance(val, str):
                                profile[key] = val

                        # Compute base score
                        base_result = compute_base_score(profile)
                        for key, val in base_result.items():
                            profile[key] = val

                        # ML risk prediction
                        try:
                            risk_prob = model.predict_single(profile)
                        except Exception:
                            risk_prob = 0.25

                        # Final score
                        final = compute_final_score(
                            float(profile["base_trust_score"]), risk_prob,
                            profile
                        )

                    # ── Display Results ──
                    st.markdown("---")
                    st.markdown("## 🏆 Your Alternative Credit Score")

                    r1, r2 = st.columns([1, 1])
                    with r1:
                        st.plotly_chart(
                            create_gauge(
                                final["final_trust_score"],
                                final["grade"],
                                final["grade_color"],
                            ),
                            use_container_width=True,
                        )

                    with r2:
                        rm1, rm2, rm3 = st.columns(3)
                        rm1.metric(
                            "Final Score", f"{final['final_trust_score']:.0f}"
                        )
                        rm2.metric("Risk Level", f"{risk_prob:.1%}")
                        rm3.metric(
                            "Confidence", f"{final['confidence']:.0%}"
                        )
                        st.markdown(f"**Grade:** {final['grade']}")
                        st.markdown(
                            f"**Base Score:** {profile['base_trust_score']:.0f}"
                        )

                    # Sub-score breakdown
                    st.markdown("### 📋 Score Breakdown")
                    breakdown = get_score_breakdown(profile)

                    bk_cols = st.columns(4)
                    for i, (cat_name, cat_data) in enumerate(
                        breakdown.items()
                    ):
                        with bk_cols[i]:
                            sv = cat_data["score"]
                            cv = (
                                "#22c55e" if sv >= 70
                                else "#eab308" if sv >= 40
                                else "#ef4444"
                            )
                            st.markdown(
                                f'<div class="metric-card">'
                                f"<h3>{cat_name}</h3>"
                                f'<div class="val" style="color:{cv}">'
                                f"{sv:.1f}</div>"
                                f'<div style="color:#64748b; '
                                f'font-size:0.75rem">'
                                f"Weight: {cat_data['weight']}</div>"
                                f"</div>",
                                unsafe_allow_html=True,
                            )
                            st.progress(int(min(sv, 100)))

                    # AI Explanation
                    st.markdown("### 🧠 AI Explanation")
                    try:
                        explainer = ScoreExplainer(model)
                        explainer.initialize(df)
                        explanation = explainer.explain_single(profile)

                        col_e1, col_e2 = st.columns(2)
                        with col_e1:
                            st.markdown("#### ✅ Positive Factors")
                            for f in explanation.get(
                                "top_positive_factors", []
                            )[:5]:
                                st.markdown(
                                    f"- **{f['feature']}**: "
                                    f"{f['feature_value']:.2f}"
                                )
                        with col_e2:
                            st.markdown("#### ⚠️ Risk Factors")
                            for f in explanation.get(
                                "top_risk_factors", []
                            )[:5]:
                                st.markdown(
                                    f"- **{f['feature']}**: "
                                    f"{f['feature_value']:.2f}"
                                )
                        st.markdown("---")
                        st.markdown(
                            explanation.get("explanation_text", "")
                        )
                    except Exception as e:
                        st.warning(f"Explainability module: {e}")

                    # ── Loan Recommendations (Transaction-Based) ──
                    st.markdown("### 💳 Loan Recommendations")
                    upload_score = final["final_trust_score"]
                    user_inc = float(profile["mean_income"])
                    fixed_exp = float(profile.get("fixed_expenses", 0))
                    existing_emi_amt = 0
                    # Detect existing EMIs from parsed transactions
                    if parser.parsed_df is not None:
                        emi_txns = parser.parsed_df[
                            (parser.parsed_df["category"] == "EMI") &
                            (parser.parsed_df["type"] == "debit")
                        ]
                        if len(emi_txns) > 0:
                            months_with_emi = emi_txns["date"].dt.to_period("M").nunique()
                            if months_with_emi > 0:
                                existing_emi_amt = float(emi_txns["amount"].sum() / months_with_emi)

                    loan_recs = get_transaction_loan_recommendations(
                        score=upload_score,
                        monthly_income=user_inc,
                        monthly_expenses=fixed_exp,
                        existing_emi=existing_emi_amt,
                    )

                    # Tier & Pre-approval badge
                    tier = loan_recs["tier"]
                    pre_status = loan_recs["pre_approval_status"]
                    tier_color = tier["color"]
                    st.markdown(
                        f'<div style="background:{tier_color}22; border:1px solid {tier_color}; '
                        f'border-radius:8px; padding:12px 18px; margin-bottom:16px;">'
                        f'<span style="font-size:1.3rem; font-weight:bold; color:{tier_color};">'
                        f'{pre_status}</span> &nbsp;·&nbsp; '
                        f'Score: {upload_score:.0f} &nbsp;·&nbsp; '
                        f'Max {tier["max_simultaneous_loans"]} simultaneous loans &nbsp;·&nbsp; '
                        f'Total exposure up to ₹{loan_recs["max_total_exposure"]:,.0f}</div>',
                        unsafe_allow_html=True,
                    )

                    # Repayment capacity summary
                    rep = loan_recs["repayment_capacity"]
                    rc1, rc2, rc3, rc4 = st.columns(4)
                    rc1.metric("Monthly Income", f"₹{rep['monthly_income']:,.0f}")
                    rc2.metric("Fixed Expenses", f"₹{rep['monthly_expenses']:,.0f}")
                    rc3.metric("Existing EMI", f"₹{rep['existing_emi']:,.0f}")
                    rc4.metric("Max New EMI", f"₹{rep['max_new_emi']:,.0f}")

                    if rep["risk_flags"]:
                        for flag in rep["risk_flags"]:
                            st.warning(f"⚠️ {flag}")

                    # Eligible loans
                    if loan_recs["eligible_loans"]:
                        st.markdown(f"#### ✅ Eligible Loans ({loan_recs['total_eligible']})")

                        # Top 3 comparison
                        top_loans = compare_loans(loan_recs["eligible_loans"])
                        if top_loans:
                            st.markdown("##### 🏆 Best Loan Options")
                            tcols = st.columns(min(len(top_loans), 3))
                            for ti, tl in enumerate(top_loans):
                                with tcols[ti]:
                                    st.markdown(
                                        f'<div class="metric-card">'
                                        f'<h3>{tl["icon"]} {tl["name"]}</h3>'
                                        f'<div class="val" style="color:#22c55e">₹{tl["recommended_amount"]:,.0f}</div>'
                                        f'<div style="color:#94a3b8; font-size:0.85rem;">'
                                        f'{tl["effective_rate"]}% · {tl["suggested_tenure"]} months</div>'
                                        f'<div style="color:#64748b; font-size:0.8rem; margin-top:4px;">'
                                        f'EMI: ₹{tl["emi"]:,.0f}/month</div>'
                                        f'{"<div style=\'color:#22c55e; font-size:0.75rem;\'>" + tl["subsidy"][:60] + "...</div>" if tl.get("subsidy") else ""}'
                                        f'</div>',
                                        unsafe_allow_html=True,
                                    )

                        # Full loan table
                        with st.expander("📋 All Eligible Loans — Full Details"):
                            for loan in loan_recs["eligible_loans"]:
                                st.markdown(f"**{loan['icon']} {loan['name']}** ({loan['category']})")
                                lc1, lc2, lc3, lc4 = st.columns(4)
                                lc1.metric("Max Amount", f"₹{loan['max_loan_amount']:,.0f}")
                                lc2.metric("Interest Rate", f"{loan['effective_rate']}%")
                                lc3.metric("EMI", f"₹{loan['emi']:,.0f}/mo")
                                lc4.metric("Tenure", f"{loan['suggested_tenure']} mo")
                                st.caption(
                                    f"{loan['description']} · "
                                    f"Collateral: {'Yes' if loan['collateral_required'] else 'No'} · "
                                    f"Fee: {loan['processing_fee']} · "
                                    f"Total Interest: ₹{loan['total_interest']:,.0f}"
                                )
                                if loan.get("subsidy"):
                                    st.success(f"💰 Subsidy: {loan['subsidy']}")
                                if loan.get("interest_saved_via_subsidy", 0) > 0:
                                    st.info(f"💵 Interest saved via subsidy: ₹{loan['interest_saved_via_subsidy']:,.0f}")
                                st.markdown(f"📄 **Documents:** {', '.join(loan['documents'])}")
                                st.markdown(f"🏦 **Lenders:** {', '.join(loan['lenders'])}")
                                st.markdown("---")

                        # EMI Calculator
                        with st.expander("🧮 EMI Calculator"):
                            emi_c1, emi_c2, emi_c3 = st.columns(3)
                            with emi_c1:
                                emi_amount = st.number_input(
                                    "Loan Amount (₹)", min_value=1000,
                                    max_value=10000000, value=100000,
                                    step=10000, key="emi_calc_amt"
                                )
                            with emi_c2:
                                emi_rate = st.number_input(
                                    "Interest Rate (%)", min_value=1.0,
                                    max_value=40.0, value=12.0,
                                    step=0.5, key="emi_calc_rate"
                                )
                            with emi_c3:
                                emi_tenure = st.number_input(
                                    "Tenure (months)", min_value=1,
                                    max_value=360, value=24,
                                    step=6, key="emi_calc_tenure"
                                )

                            calc_emi = calculate_emi(emi_amount, emi_rate, emi_tenure)
                            total_payable = calc_emi * emi_tenure
                            total_int = total_payable - emi_amount

                            ec1, ec2, ec3 = st.columns(3)
                            ec1.metric("Monthly EMI", f"₹{calc_emi:,.0f}")
                            ec2.metric("Total Interest", f"₹{total_int:,.0f}")
                            ec3.metric("Total Payable", f"₹{total_payable:,.0f}")

                            # Repayment schedule preview
                            schedule = generate_repayment_schedule(
                                emi_amount, emi_rate, emi_tenure
                            )
                            if schedule:
                                sched_df = pd.DataFrame(schedule[:12])  # first 12 months
                                sched_df["emi"] = sched_df["emi"].apply(lambda x: f"₹{x:,.0f}")
                                sched_df["principal"] = sched_df["principal"].apply(lambda x: f"₹{x:,.0f}")
                                sched_df["interest"] = sched_df["interest"].apply(lambda x: f"₹{x:,.0f}")
                                sched_df["balance"] = sched_df["balance"].apply(lambda x: f"₹{x:,.0f}")
                                sched_df.columns = ["Month", "EMI", "Principal", "Interest", "Balance"]
                                st.markdown("**Repayment Schedule (first 12 months):**")
                                st.dataframe(sched_df, use_container_width=True, hide_index=True)
                    else:
                        st.error(
                            f"❌ **Not Eligible for Loans Currently** — "
                            f"Score: {upload_score:.0f}\n\n"
                            f"Build payment history for 3-6 months to qualify."
                        )

                    # Credit Improvement Path
                    if loan_recs.get("improvement_path"):
                        st.markdown("### 📈 Credit Improvement Path")
                        for imp in loan_recs["improvement_path"]:
                            if imp["type"] == "score_upgrade":
                                st.markdown(
                                    f"🎯 **{imp['title']}** "
                                    f"(+{imp['gap']:.0f} points needed)"
                                )
                                st.caption(imp.get("benefit", ""))
                                for action in imp.get("actions", []):
                                    st.markdown(f"  - {action}")
                            elif imp["type"] == "maintenance":
                                st.success(f"✅ {imp['title']}")
                                for action in imp.get("actions", []):
                                    st.markdown(f"  - {action}")

                    # Financial Literacy Tips
                    fin_tips = get_financial_tips(
                        score=upload_score,
                        eligible_loans=loan_recs.get("eligible_loans", [])
                    )
                    if fin_tips:
                        with st.expander("📚 Financial Literacy Tips"):
                            for tip in fin_tips:
                                st.markdown(f"{tip['icon']} **{tip['title']}**")
                                st.caption(tip["detail"])
                                st.markdown("")

            except Exception as e:
                st.error(f"❌ Error parsing file: {str(e)}")
                st.markdown(
                    "**Supported formats:**\n"
                    "- CSV with columns: Date, Description, "
                    "Debit, Credit, Balance\n"
                    "- Excel (.xlsx) with the same columns\n\n"
                    "**Tip:** Download the sample statement above "
                    "to see the expected format."
                )

    # ── Page: Alternative Score ──────────────────────────────────────────
    elif page == "🌍 Alternative Score":
        st.markdown("## 🌍 Alternative Credit Score — Beyond Bank Transactions")
        st.markdown(
            "Not everyone has a bank account or UPI history. "
            "**CrediVist scores ANYONE** using persona-specific alternative data — "
            "farmers, students, street vendors, homemakers, and more."
        )

        st.markdown("---")

        # Persona selection with cards
        st.markdown("### 👤 Select Your Profile")
        persona_cols = st.columns(len(PERSONAS))
        persona_keys = list(PERSONAS.keys())
        for i, (key, info) in enumerate(PERSONAS.items()):
            with persona_cols[i]:
                st.markdown(
                    f"<div style='background: linear-gradient(135deg, #1e293b, #334155); "
                    f"border-radius: 12px; padding: 16px; text-align: center; "
                    f"min-height: 140px; border: 1px solid #475569;'>"
                    f"<div style='font-size: 2rem'>{info['label'].split()[0]}</div>"
                    f"<div style='font-weight: 600; margin: 6px 0'>{info['label'][2:].strip()}</div>"
                    f"<div style='font-size: 0.75rem; color: #94a3b8'>{info['description']}</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

        selected_persona = st.selectbox(
            "Choose your persona",
            options=persona_keys,
            format_func=lambda k: PERSONAS[k]["label"],
            key="alt_persona",
        )

        st.markdown("---")
        persona_config = PERSONAS[selected_persona]

        # ── Two input modes: Auto Upload vs Manual Form ──
        input_tab1, input_tab2 = st.tabs([
            "📄 Auto Upload (Recommended)",
            "✏️ Manual Form",
        ])

        alt_result = None  # will hold score result from either tab

        # ── Tab 1: Auto Upload ──────────────────────────────────────────
        with input_tab1:
            st.markdown(f"### 📄 Upload Documents — {persona_config['label']}")
            st.markdown(
                "Upload your documents (PDF, CSV, Excel, or TXT) and CrediVist will "
                "**automatically extract** all relevant data and compute your score."
            )

            # Persona-specific document guidance
            doc_guidance = {
                "farmer": [
                    "Land record (RTC / Patta / Khata PDF)",
                    "PM-KISAN beneficiary statement",
                    "Mandi sale receipts",
                    "KCC statement or crop insurance documents",
                    "Utility bills (electricity, water, gas)",
                ],
                "student": [
                    "Marksheet / Transcript / Grade Card (PDF)",
                    "Scholarship award letters",
                    "Course certificates (NPTEL, Coursera, etc.)",
                    "Internship / Part-time work proof",
                ],
                "street_vendor": [
                    "Daily sales register (CSV / Excel)",
                    "Rent / stall fee receipts",
                    "Trade / vendor license copy",
                    "Utility bills",
                ],
                "homemaker": [
                    "Household expense diary (CSV / Excel)",
                    "SHG passbook / contribution record",
                    "Micro-enterprise receipts (if any)",
                    "Utility bills, skill certificates",
                ],
                "general_no_bank": [
                    "Aadhaar / PAN / Voter ID / Ration Card scans",
                    "Mobile recharge history",
                    "Utility bills (electricity, water, gas)",
                    "Rent receipts (if applicable)",
                ],
            }

            guidance = doc_guidance.get(selected_persona, [])
            if guidance:
                st.markdown("**Recommended documents to upload:**")
                for doc in guidance:
                    st.markdown(f"- {doc}")

            st.markdown("")

            # Sample document download
            sample_gen = SAMPLE_GENERATORS.get(selected_persona)
            if sample_gen:
                sample_text = sample_gen()
                st.download_button(
                    f"📥 Download Sample {persona_config['label'].split(maxsplit=1)[-1]} Document",
                    data=sample_text.encode("utf-8"),
                    file_name=f"sample_{selected_persona}_doc.txt",
                    mime="text/plain",
                    key=f"dl_sample_{selected_persona}",
                )

            st.markdown("")

            # File uploader — multiple files
            # Show OCR capability status
            try:
                from src.ocr_engine import get_ocr_capabilities
                caps = get_ocr_capabilities()
                cap_items = []
                if caps.get("tesseract"):
                    cap_items.append("\u2705 Tesseract OCR")
                else:
                    cap_items.append("\u274c Tesseract OCR (install for scanned docs)")
                if caps.get("pil"):
                    cap_items.append("\u2705 Image Processing")
                st.caption("OCR: " + " \u00b7 ".join(cap_items))
            except ImportError:
                pass

            uploaded_files = st.file_uploader(
                "Upload your documents",
                type=["pdf", "csv", "xlsx", "xls", "txt", "json",
                      "jpg", "jpeg", "png", "bmp", "tiff", "tif", "webp"],
                accept_multiple_files=True,
                key=f"alt_upload_{selected_persona}",
                help="Upload documents or scanned images. Supports PDF, images (JPG/PNG), CSV, Excel, TXT.",
            )

            # Auto-detect toggle
            auto_detect = st.checkbox(
                "🔮 Auto-detect persona from documents (overrides selection above)",
                value=False,
                key="auto_detect_persona",
            )

            if uploaded_files:
                if st.button("🔍 Analyze & Score", type="primary",
                             use_container_width=True, key="btn_analyze"):
                    # Prepare files
                    files = []
                    for uf in uploaded_files:
                        files.append((uf.name, uf.read()))
                        uf.seek(0)  # reset for potential re-read

                    persona_to_use = None if auto_detect else selected_persona

                    with st.spinner("Analyzing documents and extracting data..."):
                        analysis = analyze_documents(files, persona=persona_to_use)

                    # Show analysis summary
                    st.markdown("---")
                    st.markdown("### 📑 Document Analysis Summary")

                    as1, as2, as3, as4 = st.columns(4)
                    as1.metric("Files Processed", analysis["files_processed"])
                    as2.metric("Text Extracted", f"{analysis['total_text_length']:,} chars")
                    detected_label = PERSONAS.get(
                        analysis["detected_persona"], {}
                    ).get("label", analysis["detected_persona"])
                    as3.metric("Detected Persona", detected_label)
                    as4.metric("OCR Used", "Yes" if analysis.get("ocr_used") else "No")

                    # Show detected document types
                    doc_types = analysis.get("detected_document_types", [])
                    if doc_types:
                        type_labels = [t.replace('_', ' ').title() for t in doc_types]
                        st.info(f"\U0001f4c4 Documents identified: {', '.join(type_labels)}")

                    # Show per-file details
                    if analysis["document_summaries"]:
                        file_summary_data = []
                        for ds in analysis["document_summaries"]:
                            file_summary_data.append({
                                "File": ds["filename"],
                                "Doc Type": ds.get("document_type", "unknown").replace('_', ' ').title(),
                                "Text Length": f"{ds['text_length']:,}",
                                "OCR": "✓" if ds.get("ocr_used") else "✗",
                                "Has Table": "✓" if ds["has_table"] else "✗",
                                "Rows": ds["rows"],
                                "Amounts Found": ds["amounts_found"],
                                "Dates Found": ds["dates_found"],
                            })
                        st.dataframe(
                            pd.DataFrame(file_summary_data),
                            use_container_width=True, hide_index=True,
                        )

                    # Show warnings
                    for w in analysis.get("warnings", []):
                        st.warning(f"⚠️ {w}")

                    # Show extracted data in expander
                    with st.expander("🔎 View Extracted Data", expanded=False):
                        extracted = analysis["extracted_data"]
                        ext_items = []
                        for k, v in sorted(extracted.items()):
                            ext_items.append({"Field": k.replace("_", " ").title(), "Value": str(v)})
                        st.dataframe(
                            pd.DataFrame(ext_items),
                            use_container_width=True, hide_index=True,
                        )

                    # Compute score
                    final_persona = analysis["detected_persona"]
                    with st.spinner("Computing credit score..."):
                        alt_result = compute_persona_score(final_persona, analysis["extracted_data"])

                    # Store in session state for display below
                    st.session_state["alt_score_result"] = alt_result
                    st.session_state["alt_score_persona_config"] = PERSONAS[final_persona]

            # Check session state for previously computed result
            if "alt_score_result" in st.session_state and alt_result is None:
                alt_result = st.session_state.get("alt_score_result")
                persona_config = st.session_state.get("alt_score_persona_config", persona_config)

        # ── Tab 2: Manual Form ──────────────────────────────────────────
        with input_tab2:
            st.markdown(f"### ✏️ Manual Entry — {persona_config['label']}")
            st.markdown(
                f"*{persona_config['description']}* — "
                f"Fill in the details below. More data = higher confidence."
            )

            # Dynamically build form based on persona criteria
            form_fields = get_persona_form_fields(selected_persona)
            form_data = {}

            with st.form(key="alt_score_form"):
                for group in form_fields:
                    criterion = group["criterion"]
                    fields = group["fields"]

                    # Get the criterion label from registry
                    from src.alternative_profiles import CRITERIA_REGISTRY
                    scorer_fn = CRITERIA_REGISTRY.get(criterion)
                    if scorer_fn:
                        test_result = scorer_fn({})
                        section_label = test_result.get("label", criterion.replace('_', ' ').title())
                    else:
                        section_label = criterion.replace('_', ' ').title()

                    st.markdown(f"#### {section_label}")

                    # Create columns for fields (2 per row)
                    field_pairs = [fields[i:i+2] for i in range(0, len(fields), 2)]
                    for pair in field_pairs:
                        cols = st.columns(len(pair))
                        for j, field in enumerate(pair):
                            with cols[j]:
                                fkey = f"alt_{criterion}_{field['key']}"
                                if field["type"] == "boolean":
                                    default = field.get("default", False)
                                    form_data[field["key"]] = st.checkbox(
                                        field["label"], value=default, key=fkey
                                    )
                                elif field["type"] == "number":
                                    default = field.get("default", 0)
                                    min_val = field.get("min", 0)
                                    max_val = field.get("max", 100)
                                    # Use float for decimal fields
                                    if isinstance(default, float) and default != int(default):
                                        form_data[field["key"]] = st.number_input(
                                            field["label"], min_value=float(min_val),
                                            max_value=float(max_val),
                                            value=float(default), key=fkey
                                        )
                                    else:
                                        form_data[field["key"]] = st.number_input(
                                            field["label"], min_value=int(min_val),
                                            max_value=int(max_val),
                                            value=int(default), key=fkey
                                        )
                                elif field["type"] == "select":
                                    options = field.get("options", [])
                                    default = field.get("default", options[0] if options else "")
                                    default_idx = options.index(default) if default in options else 0
                                    form_data[field["key"]] = st.selectbox(
                                        field["label"], options=options,
                                        index=default_idx, key=fkey
                                    )
                                elif field["type"] == "text":
                                    default = field.get("default", "")
                                    val = st.text_input(
                                        field["label"], value=default, key=fkey
                                    )
                                    # Parse comma-separated into list if needed
                                    if "comma" in field["label"].lower():
                                        form_data[field["key"]] = [
                                            x.strip() for x in val.split(",") if x.strip()
                                        ]
                                    else:
                                        form_data[field["key"]] = val

                    st.markdown("")

                submitted = st.form_submit_button(
                    "🔍 Compute Alternative Credit Score",
                    use_container_width=True, type="primary"
                )

            if submitted:
                with st.spinner("Computing your alternative credit score..."):
                    alt_result = compute_persona_score(selected_persona, form_data)

        # ── Shared Results Display (works for both Auto Upload & Manual) ──
        if alt_result is not None:
            st.markdown("---")
            st.markdown("### 📊 Your Alternative Credit Score")

            # Score display
            sc1, sc2, sc3, sc4 = st.columns(4)
            sc1.metric("Trust Score", f"{alt_result['trust_score']:.0f} / 900")
            sc2.metric("Grade", alt_result["grade"])
            sc3.metric("Confidence", f"{alt_result['confidence']:.0%}")
            sc4.metric("Data Signals", f"{alt_result['filled_count']}/{alt_result['criteria_count']}")

            # Gauge chart
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=alt_result["trust_score"],
                domain={"x": [0, 1], "y": [0, 1]},
                title={"text": f"{alt_result['persona_label']} — Trust Score",
                       "font": {"size": 20}},
                gauge={
                    "axis": {"range": [300, 900], "tickwidth": 1},
                    "bar": {"color": alt_result["grade_color"]},
                    "bgcolor": "#1e293b",
                    "steps": [
                        {"range": [300, 400], "color": "#ef4444"},
                        {"range": [400, 500], "color": "#f97316"},
                        {"range": [500, 650], "color": "#eab308"},
                        {"range": [650, 750], "color": "#84cc16"},
                        {"range": [750, 900], "color": "#22c55e"},
                    ],
                    "threshold": {
                        "line": {"color": "white", "width": 4},
                        "thickness": 0.75,
                        "value": alt_result["trust_score"],
                    },
                },
            ))
            fig_gauge.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                font={"color": "#e2e8f0"},
                height=300,
            )
            st.plotly_chart(fig_gauge, use_container_width=True)

            # Criteria breakdown
            st.markdown("### 📋 Criteria Breakdown")
            breakdown = alt_result["criteria_breakdown"]
            display_persona = alt_result.get("persona", selected_persona)
            display_config = PERSONAS.get(display_persona, persona_config)
            weights = display_config["criteria_weights"]

            breakdown_data = []
            for criterion, info in breakdown.items():
                weight = weights.get(criterion, 0)
                breakdown_data.append({
                    "Criteria": info["label"],
                    "Score": f"{info['score']:.0%}",
                    "Weight": f"{weight:.0%}",
                    "Weighted": f"{info['score'] * weight:.2%}",
                    "Details": info["detail"],
                })

            st.dataframe(
                pd.DataFrame(breakdown_data),
                use_container_width=True, hide_index=True,
            )

            # Radar chart for criteria
            criteria_labels = [info["label"] for info in breakdown.values()]
            criteria_scores = [info["score"] * 100 for info in breakdown.values()]

            fig_radar = go.Figure(data=go.Scatterpolar(
                r=criteria_scores + [criteria_scores[0]],
                theta=criteria_labels + [criteria_labels[0]],
                fill="toself",
                fillcolor="rgba(99, 102, 241, 0.3)",
                line={"color": "#6366f1", "width": 2},
                marker={"size": 6, "color": "#818cf8"},
            ))
            fig_radar.update_layout(
                polar={
                    "radialaxis": {
                        "visible": True, "range": [0, 100],
                        "gridcolor": "#334155",
                        "tickfont": {"color": "#94a3b8"},
                    },
                    "angularaxis": {
                        "gridcolor": "#334155",
                        "tickfont": {"color": "#e2e8f0", "size": 10},
                    },
                    "bgcolor": "#0f172a",
                },
                paper_bgcolor="rgba(0,0,0,0)",
                showlegend=False,
                height=450,
                title={"text": "Criteria Performance Radar",
                       "font": {"color": "#e2e8f0", "size": 16}},
            )
            st.plotly_chart(fig_radar, use_container_width=True)

            # Improvement tips
            tips = get_improvement_tips(display_persona, alt_result)
            if tips:
                st.markdown("### 💡 How to Improve Your Score")
                for tip in tips:
                    impact_colors = {"high": "🔴", "medium": "🟡", "low": "🟢"}
                    impact_icon = impact_colors.get(tip["impact"], "⚪")
                    with st.expander(
                        f"{impact_icon} {tip['action']} (Current: {tip['current_score']:.0%})"
                    ):
                        st.markdown(tip["description"])
                        st.caption(f"Impact: {tip['impact'].upper()} · Criterion: {tip['criterion'].replace('_', ' ').title()}")
            else:
                st.success("🎉 Great job! All your criteria are above 50%.")

            # ── Loan Recommendations (Persona-Based) ──
            st.markdown("### 💳 Loan Recommendations")
            try:
                alt_score = alt_result["trust_score"]
                alt_persona_key = alt_result.get("persona", selected_persona)

                # Estimate income from persona data
                alt_form = alt_result.get("input_data", {})

                alt_loan_recs = get_persona_loan_recommendations(
                    persona=alt_persona_key,
                    score=alt_score,
                    persona_data=alt_form,
                )

                # Tier badge
                alt_tier = alt_loan_recs["tier"]
                alt_pre = alt_loan_recs["pre_approval_status"]
                atc = alt_tier["color"]
                st.markdown(
                    f'<div style="background:{atc}22; border:1px solid {atc}; '
                    f'border-radius:8px; padding:12px 18px; margin-bottom:16px;">'
                    f'<span style="font-size:1.3rem; font-weight:bold; color:{atc};">'
                    f'{alt_pre}</span> &nbsp;·&nbsp; '
                    f'Score: {alt_score:.0f} &nbsp;·&nbsp; '
                    f'Max {alt_tier["max_simultaneous_loans"]} simultaneous loans &nbsp;·&nbsp; '
                    f'Est. Income: ₹{alt_loan_recs.get("estimated_monthly_income", 0):,.0f}/mo</div>',
                    unsafe_allow_html=True,
                )

                # Eligible loans
                if alt_loan_recs["eligible_loans"]:
                    st.markdown(f"#### ✅ Eligible Loan Schemes ({alt_loan_recs['total_eligible']})")

                    # Top comparison cards
                    alt_top = compare_loans(alt_loan_recs["eligible_loans"])
                    if alt_top:
                        alt_tcols = st.columns(min(len(alt_top), 3))
                        for ti, tl in enumerate(alt_top):
                            with alt_tcols[ti]:
                                st.markdown(
                                    f'<div class="metric-card">'
                                    f'<h3>{tl["icon"]} {tl["name"]}</h3>'
                                    f'<div class="val" style="color:#22c55e">'
                                    f'₹{tl["recommended_amount"]:,.0f}</div>'
                                    f'<div style="color:#94a3b8; font-size:0.85rem;">'
                                    f'{tl["effective_rate"]}% · {tl["suggested_tenure"]} months</div>'
                                    f'<div style="color:#64748b; font-size:0.8rem; margin-top:4px;">'
                                    f'EMI: ₹{tl["emi"]:,.0f}/month</div>'
                                    f'{"<div style=\'color:#22c55e; font-size:0.75rem;\'>💰 " + tl["subsidy"][:50] + "...</div>" if tl.get("subsidy") else ""}'
                                    f'</div>',
                                    unsafe_allow_html=True,
                                )

                    # Full details expander
                    with st.expander("📋 All Eligible Loans — Full Details"):
                        for loan in alt_loan_recs["eligible_loans"]:
                            st.markdown(f"**{loan['icon']} {loan['name']}** ({loan['category']})")
                            alc1, alc2, alc3, alc4 = st.columns(4)
                            alc1.metric("Max Amount", f"₹{loan['max_loan_amount']:,.0f}")
                            alc2.metric("Interest Rate", f"{loan['effective_rate']}%")
                            alc3.metric("EMI", f"₹{loan['emi']:,.0f}/mo")
                            alc4.metric("Tenure", f"{loan['suggested_tenure']} mo")
                            st.caption(
                                f"{loan['description']} · "
                                f"Collateral: {'Yes' if loan['collateral_required'] else 'No'} · "
                                f"Fee: {loan['processing_fee']} · "
                                f"Total Interest: ₹{loan['total_interest']:,.0f}"
                            )
                            if loan.get("subsidy"):
                                st.success(f"💰 Subsidy: {loan['subsidy']}")
                            if loan.get("interest_saved_via_subsidy", 0) > 0:
                                st.info(f"💵 Interest saved: ₹{loan['interest_saved_via_subsidy']:,.0f}")
                            st.markdown(f"📄 **Documents:** {', '.join(loan['documents'])}")
                            st.markdown(f"🏦 **Lenders:** {', '.join(loan['lenders'])}")
                            # Eligibility criteria checklist
                            if loan.get("criteria_met") or loan.get("criteria_not_met"):
                                criteria_line = ""
                                for c in loan.get("criteria_met", []):
                                    criteria_line += f"✅ {c.replace('_', ' ').title()}  "
                                for c in loan.get("criteria_not_met", []):
                                    criteria_line += f"❌ {c.replace('_', ' ').title()}  "
                                st.markdown(f"📝 **Eligibility:** {criteria_line}")
                            st.markdown("---")

                    # EMI Calculator
                    with st.expander("🧮 EMI Calculator"):
                        aec1, aec2, aec3 = st.columns(3)
                        with aec1:
                            a_emi_amt = st.number_input(
                                "Loan Amount (₹)", min_value=1000,
                                max_value=10000000, value=50000,
                                step=5000, key="alt_emi_amt"
                            )
                        with aec2:
                            a_emi_rate = st.number_input(
                                "Interest Rate (%)", min_value=1.0,
                                max_value=40.0, value=10.0,
                                step=0.5, key="alt_emi_rate"
                            )
                        with aec3:
                            a_emi_ten = st.number_input(
                                "Tenure (months)", min_value=1,
                                max_value=360, value=12,
                                step=3, key="alt_emi_ten"
                            )

                        a_calc_emi = calculate_emi(a_emi_amt, a_emi_rate, a_emi_ten)
                        a_total = a_calc_emi * a_emi_ten
                        a_int = a_total - a_emi_amt

                        aec4, aec5, aec6 = st.columns(3)
                        aec4.metric("Monthly EMI", f"₹{a_calc_emi:,.0f}")
                        aec5.metric("Total Interest", f"₹{a_int:,.0f}")
                        aec6.metric("Total Payable", f"₹{a_total:,.0f}")
                else:
                    st.error(
                        f"❌ **Not Eligible for Loans** — Score: {alt_score:.0f}\n\n"
                        f"Build your alternative credit profile for 3-6 months."
                    )

                # Seasonal recommendations for farmers
                seasonal = get_seasonal_recommendations(alt_persona_key)
                if seasonal:
                    st.markdown("### 🌾 Seasonal Loan Recommendations")
                    for rec in seasonal:
                        st.info(
                            f"**{rec['season']}** — {rec['status']}\n\n"
                            f"🌱 Crops: {rec['crops']}\n\n{rec['advice']}"
                        )

                # Credit Improvement Path
                if alt_loan_recs.get("improvement_path"):
                    st.markdown("### 📈 Credit Improvement Path")
                    for imp in alt_loan_recs["improvement_path"]:
                        if imp["type"] == "score_upgrade":
                            st.markdown(
                                f"🎯 **{imp['title']}** "
                                f"(+{imp.get('gap', 0):.0f} points needed)"
                            )
                            st.caption(imp.get("benefit", ""))
                            for action in imp.get("actions", []):
                                st.markdown(f"  - {action}")
                        elif imp["type"] == "maintenance":
                            st.success(f"✅ {imp['title']}")
                            for action in imp.get("actions", []):
                                st.markdown(f"  - {action}")

                # Financial Tips
                alt_fin_tips = get_financial_tips(
                    persona=alt_persona_key,
                    score=alt_score,
                    eligible_loans=alt_loan_recs.get("eligible_loans", []),
                )
                if alt_fin_tips:
                    with st.expander("📚 Financial Literacy Tips"):
                        for tip in alt_fin_tips:
                            st.markdown(f"{tip['icon']} **{tip['title']}**")
                            st.caption(tip["detail"])
                            st.markdown("")

            except Exception as e:
                st.caption(f"Loan recommendation engine: {e}")

            # Comparison note
            st.markdown("---")
            st.info(
                "💡 **How this works:** Your score is computed purely from "
                "alternative data signals specific to your life situation — "
                "no bank account or CIBIL history needed. "
                "As you provide more data, confidence increases."
            )

    # ── Page: Score Builder ─────────────────────────────────────────────
    elif page == "🚀 Score Builder":
        st.markdown("## 🚀 Score Builder — Improve Your Credit Score")
        st.markdown(
            "See exactly which actions will boost your score the most. "
            "Select a user to get a **personalized improvement plan**."
        )

        # User selector
        user_ids = df["user_id"].tolist()
        selected_user = st.selectbox(
            "Select User for Improvement Plan", user_ids, index=0,
            key="builder_user"
        )
        user_row = df[df["user_id"] == selected_user].iloc[0]

        current_score = float(user_row["final_trust_score"])
        current_grade = user_row["grade"]
        current_color = user_row["grade_color"]

        # Current score summary
        bc1, bc2, bc3 = st.columns(3)
        bc1.metric("Current Score", f"{current_score:.0f}")
        bc2.metric("Current Grade", current_grade)
        bc3.metric("Risk Level", f"{float(user_row['risk_probability']):.1%}")

        st.markdown("---")

        # Analyze each feature's improvement potential
        improvements = []

        feature_actions = {
            "feat_income_stability": {
                "name": "Income Stability",
                "action": "Maintain consistent monthly income for 3+ months",
                "icon": "💰",
                "difficulty": "Medium",
                "timeframe": "3-6 months",
            },
            "feat_cash_flow_ratio": {
                "name": "Cash Flow Health",
                "action": "Reduce fixed expenses or increase income by 10%",
                "icon": "📊",
                "difficulty": "Medium",
                "timeframe": "1-3 months",
            },
            "feat_utility_score": {
                "name": "Utility Bill Payments",
                "action": "Pay all utility bills before the due date",
                "icon": "⚡",
                "difficulty": "Easy",
                "timeframe": "1-2 months",
            },
            "feat_emi_score": {
                "name": "EMI-like Behavior",
                "action": "Set up 2-3 recurring payments (SIP, subscriptions)",
                "icon": "🔄",
                "difficulty": "Easy",
                "timeframe": "1 month",
            },
            "feat_txn_regularity": {
                "name": "Transaction Regularity",
                "action": "Use digital payments consistently every week",
                "icon": "📱",
                "difficulty": "Easy",
                "timeframe": "1-2 months",
            },
            "feat_expense_score": {
                "name": "Expense Discipline",
                "action": "Shift spending toward essentials (food, transport, bills)",
                "icon": "🛒",
                "difficulty": "Medium",
                "timeframe": "1-2 months",
            },
            "feat_savings_score": {
                "name": "Savings Discipline",
                "action": "Start a recurring monthly SIP of ₹500+",
                "icon": "🏦",
                "difficulty": "Easy",
                "timeframe": "1 month",
            },
            "feat_work_reliability": {
                "name": "Work Reliability",
                "action": "Work 22+ days/month and maintain 4.5+ platform rating",
                "icon": "⭐",
                "difficulty": "Medium",
                "timeframe": "2-3 months",
            },
            "feat_income_diversity": {
                "name": "Income Diversity",
                "action": "Add a second gig platform (e.g., Swiggy + Uber)",
                "icon": "🔀",
                "difficulty": "Medium",
                "timeframe": "1-2 months",
            },
            "feat_shock_recovery": {
                "name": "Shock Recovery",
                "action": "Build a 1-month emergency buffer; recover quickly from dips",
                "icon": "🛡️",
                "difficulty": "Hard",
                "timeframe": "3-6 months",
            },
        }

        for feat_key, info in feature_actions.items():
            if feat_key in user_row.index:
                current_val = float(user_row[feat_key])
                # Simulate improvement to 0.85
                if current_val < 0.85:
                    improved_val = min(current_val + 0.20, 0.95)
                    gap = improved_val - current_val

                    # Estimate score impact (approximate via sub-score weights)
                    if feat_key in ["feat_income_stability", "feat_cash_flow_ratio", "feat_savings_score"]:
                        weight = 0.35
                    elif feat_key in ["feat_utility_score", "feat_emi_score"]:
                        weight = 0.30
                    elif feat_key in ["feat_txn_regularity", "feat_expense_score"]:
                        weight = 0.20
                    else:
                        weight = 0.15

                    estimated_points = gap * 100 * weight * 6  # scale to 300-900
                    improvements.append({
                        "feature": feat_key,
                        "name": info["name"],
                        "action": info["action"],
                        "icon": info["icon"],
                        "difficulty": info["difficulty"],
                        "timeframe": info["timeframe"],
                        "current": current_val,
                        "target": improved_val,
                        "estimated_points": estimated_points,
                    })

        # Sort by estimated impact
        improvements.sort(key=lambda x: x["estimated_points"], reverse=True)

        if improvements:
            # Top 3 impact actions
            st.markdown("### 🎯 Top Actions — Maximum Impact")

            for i, imp in enumerate(improvements[:3]):
                with st.container():
                    ac1, ac2, ac3, ac4 = st.columns([0.5, 3, 1.5, 1])
                    with ac1:
                        st.markdown(
                            f"<div style='font-size:2rem; text-align:center'>"
                            f"{imp['icon']}</div>",
                            unsafe_allow_html=True,
                        )
                    with ac2:
                        st.markdown(f"**{imp['name']}**")
                        st.markdown(f"{imp['action']}")
                        st.caption(
                            f"Difficulty: {imp['difficulty']} · "
                            f"Timeframe: {imp['timeframe']}"
                        )
                    with ac3:
                        st.markdown(
                            f"<div style='text-align:center'>"
                            f"<span style='color:#64748b'>Current</span><br>"
                            f"<b>{imp['current']:.0%}</b> → "
                            f"<b style='color:#22c55e'>{imp['target']:.0%}</b>"
                            f"</div>",
                            unsafe_allow_html=True,
                        )
                    with ac4:
                        st.markdown(
                            f"<div style='text-align:center'>"
                            f"<span style='color:#64748b'>Impact</span><br>"
                            f"<b style='color:#6366f1; font-size:1.3rem'>"
                            f"+{imp['estimated_points']:.0f}</b>"
                            f"<br><span style='font-size:0.7rem'>points</span>"
                            f"</div>",
                            unsafe_allow_html=True,
                        )
                    st.progress(
                        int(min(imp["current"] * 100, 100)),
                        text=f"{imp['current']:.0%} → {imp['target']:.0%}",
                    )
                    st.markdown("")

            # Projected score
            total_gain = sum(imp["estimated_points"] for imp in improvements[:3])
            projected = min(current_score + total_gain, 900)

            st.markdown("---")
            st.markdown("### 📈 Projected Score After Top 3 Actions")

            pc1, pc2 = st.columns(2)
            with pc1:
                st.plotly_chart(
                    create_gauge(current_score, current_grade, current_color),
                    use_container_width=True,
                )
                st.markdown(
                    "<div style='text-align:center; color:#94a3b8'>"
                    "Current</div>",
                    unsafe_allow_html=True,
                )
            with pc2:
                proj_final = compute_final_score(projected, 0.0)
                st.plotly_chart(
                    create_gauge(
                        projected,
                        proj_final["grade"],
                        proj_final["grade_color"],
                    ),
                    use_container_width=True,
                )
                st.markdown(
                    "<div style='text-align:center; color:#22c55e'>"
                    f"Projected (+{total_gain:.0f} points)</div>",
                    unsafe_allow_html=True,
                )

            # Grade progression
            next_grade_thresholds = [
                (750, "Excellent"), (650, "Good"),
                (500, "Fair"), (400, "Poor"),
            ]
            for threshold, grade_name in next_grade_thresholds:
                if current_score < threshold:
                    points_needed = threshold - current_score
                    st.info(
                        f"📍 You need **{points_needed:.0f} more points** "
                        f"to reach **{grade_name}** grade ({threshold}+)"
                    )
                    break

            # All improvements table
            st.markdown("---")
            st.markdown("### 📋 All Improvement Opportunities")
            all_imp_data = []
            for imp in improvements:
                all_imp_data.append({
                    "Action": f"{imp['icon']} {imp['name']}",
                    "What to Do": imp["action"],
                    "Current": f"{imp['current']:.0%}",
                    "Target": f"{imp['target']:.0%}",
                    "Est. Points": f"+{imp['estimated_points']:.0f}",
                    "Difficulty": imp["difficulty"],
                    "Timeframe": imp["timeframe"],
                })
            st.dataframe(
                pd.DataFrame(all_imp_data),
                use_container_width=True, hide_index=True,
            )
        else:
            st.success(
                "🎉 Outstanding! Your scores are already excellent "
                "across all criteria. Keep up the great work!"
            )


if __name__ == "__main__":
    main()
