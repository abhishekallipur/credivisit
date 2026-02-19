"""
CrediVist — Alternative Credit Scoring Engine
Main Streamlit Application — FinTech Dark Theme
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
    page_title="CrediVist — AI Credit Scoring",
    page_icon="₹",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─── FinTech Premium Design System ──────────────────────────────────────────
st.markdown("""
<style>
/* ── Import Google Fonts ── */
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800;900&display=swap');

/* ── Basteleur Font (local) ── */
@font-face {
    font-family: 'Basteleur';
    src: url('./app/static/fonts/Basteleur-Bold.otf') format('opentype');
    font-weight: 700;
    font-style: normal;
    font-display: swap;
}
@font-face {
    font-family: 'Basteleur';
    src: url('./app/static/fonts/Basteleur-Moonlight.otf') format('opentype');
    font-weight: 300;
    font-style: normal;
    font-display: swap;
}

/* ── CSS Variables ── */
:root {
    --navy: #0f172a;
    --navy-light: #1e293b;
    --indigo: #C9B59C;
    --indigo-dark: #8B7355;
    --indigo-glow: rgba(201, 181, 156, 0.15);
    --violet: #D9CFC7;
    --cyan: #B8A48E;
    --green: #10b981;
    --amber: #f59e0b;
    --red: #ef4444;
    --slate-50: #F9F8F6;
    --slate-100: #EFE9E3;
    --slate-200: #D9CFC7;
    --slate-400: #94a3b8;
    --slate-500: #64748b;
    --slate-600: #475569;
    --slate-700: #334155;
    --radius-sm: 8px;
    --radius-md: 12px;
    --radius-lg: 16px;
    --radius-xl: 20px;
    --radius-2xl: 24px;
    --shadow-sm: 0 1px 2px rgba(15, 23, 42, 0.04);
    --shadow-md: 0 4px 16px rgba(15, 23, 42, 0.06);
    --shadow-lg: 0 8px 32px rgba(15, 23, 42, 0.08);
    --shadow-xl: 0 16px 48px rgba(15, 23, 42, 0.10);
    --shadow-indigo: 0 4px 24px rgba(201, 181, 156, 0.2);
    --transition-fast: 0.15s cubic-bezier(0.4, 0, 0.2, 1);
    --transition-base: 0.25s cubic-bezier(0.4, 0, 0.2, 1);
    --transition-slow: 0.4s cubic-bezier(0.4, 0, 0.2, 1);
}

/* ── Keyframe Animations ── */
@keyframes fadeInUp {
    from { opacity: 0; transform: translateY(20px); }
    to { opacity: 1; transform: translateY(0); }
}
@keyframes fadeIn {
    from { opacity: 0; }
    to { opacity: 1; }
}
@keyframes slideInLeft {
    from { opacity: 0; transform: translateX(-16px); }
    to { opacity: 1; transform: translateX(0); }
}
@keyframes pulse-glow {
    0%, 100% { box-shadow: 0 0 20px rgba(201, 181, 156, 0.15); }
    50% { box-shadow: 0 0 40px rgba(201, 181, 156, 0.25); }
}
@keyframes gradient-shift {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}
@keyframes countUp {
    from { opacity: 0; transform: scale(0.8); }
    to { opacity: 1; transform: scale(1); }
}
@keyframes shimmer {
    0% { background-position: -200% center; }
    100% { background-position: 200% center; }
}
@keyframes float {
    0%, 100% { transform: translateY(0px); }
    50% { transform: translateY(-6px); }
}

/* ── Global ── */
html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
}
.stApp {
    background: #F9F8F6 !important;
    background-image:
        radial-gradient(ellipse at 20% 0%, rgba(201, 181, 156, 0.08) 0%, transparent 50%),
        radial-gradient(ellipse at 80% 100%, rgba(217, 207, 199, 0.06) 0%, transparent 50%),
        radial-gradient(ellipse at 50% 50%, rgba(184, 164, 142, 0.04) 0%, transparent 60%) !important;
    color: var(--navy) !important;
}

/* ── Base Text ── */
h1, h2, h3, h4, h5, h6 { color: var(--navy) !important; }
div[data-testid="stMarkdownContainer"],
.stMarkdown, .stMarkdownContainer, .stText, .stCaption,
.stAlert, .stExpander, .stRadio, .stCheckbox,
.stSelectbox, .stMultiSelect, .stSlider,
.stNumberInput, .stTextInput, .stTextArea {
    color: var(--navy) !important;
}
.stCaption, small { color: var(--slate-500) !important; }

/* ── Hide Sidebar Completely ── */
section[data-testid="stSidebar"] {
    display: none !important;
}
[data-testid="stSidebarCollapsedControl"] {
    display: none !important;
}

/* ── Top Navbar ── */
.navbar-container {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 8px 0;
}
.navbar-brand {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 4px 0;
}
.navbar-brand .logo-icon {
    width: 40px;
    height: 40px;
    background: #0f0f0f;
    border-radius: 12px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.3rem;
    font-weight: 900;
    color: #ffffff;
    box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    flex-shrink: 0;
}
.navbar-brand .brand-text {
    font-family: 'Basteleur', 'Plus Jakarta Sans', serif;
    font-size: 1.4rem;
    font-weight: 900;
    color: #0f0f0f;
    -webkit-text-fill-color: #0f0f0f;
    letter-spacing: -0.5px;
}
.navbar-brand .brand-sub {
    font-size: 0.55rem;
    color: #475569;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    font-weight: 700;
}
.navbar-status {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 8px 16px;
    background: linear-gradient(135deg, rgba(201,181,156,0.12), rgba(217,207,199,0.08));
    border: 1px solid rgba(201,181,156,0.2);
    border-radius: 50px;
    font-size: 0.72rem;
    color: var(--indigo-dark);
    font-weight: 600;
    letter-spacing: 0.3px;
}
.navbar-status .status-dot {
    width: 7px;
    height: 7px;
    background: #22c55e;
    border-radius: 50%;
    animation: pulse-glow 2s infinite;
}

/* ── Nav Button Tabs ── */
.nav-tabs {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 0;
    background: #f1f5f9;
    border-radius: 50px;
    padding: 4px 5px;
    border: 1px solid #e2e8f0;
    box-shadow: inset 0 1px 2px rgba(0,0,0,0.04);
}
.nav-tabs .nav-tab {
    background: transparent;
    border: none;
    color: #64748b;
    font-weight: 600;
    font-size: 0.8rem;
    padding: 9px 18px;
    border-radius: 50px;
    cursor: pointer;
    white-space: nowrap;
    font-family: 'Plus Jakarta Sans', 'Inter', sans-serif;
    transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
    letter-spacing: -0.01em;
}
.nav-tabs .nav-tab:hover {
    background: rgba(201, 181, 156, 0.12);
    color: #8B7355;
}
.nav-tabs .nav-tab.active {
    background: #ffffff;
    color: #6B5B45;
    font-weight: 700;
    box-shadow: 0 2px 8px rgba(201, 181, 156, 0.25), 0 1px 3px rgba(0,0,0,0.04);
}
/* Hide default Streamlit button styling in nav area */
.nav-btn-area .stButton > button {
    background: transparent !important;
    border: none !important;
    color: #64748b !important;
    font-weight: 600 !important;
    font-size: 0.82rem !important;
    padding: 8px 6px !important;
    border-radius: 50px !important;
    box-shadow: none !important;
    white-space: nowrap !important;
    width: 100% !important;
    transition: all 0.2s ease !important;
    font-family: 'Plus Jakarta Sans', 'Inter', sans-serif !important;
}
.nav-btn-area .stButton > button:hover {
    background: rgba(201, 181, 156, 0.12) !important;
    color: #8B7355 !important;
    border: none !important;
}
.nav-btn-area .stButton > button:focus {
    box-shadow: none !important;
    border: none !important;
    outline: none !important;
}
.nav-btn-active .stButton > button {
    background: #ffffff !important;
    color: #6B5B45 !important;
    font-weight: 700 !important;
    box-shadow: 0 2px 8px rgba(201, 181, 156, 0.25), 0 1px 3px rgba(0,0,0,0.04) !important;
}

.navbar-divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, var(--slate-200), var(--slate-200), transparent);
    margin: 0 0 1rem 0;
}

/* ── Glass Card ── */
.glass-card {
    background: #ffffff;
    border: 1px solid var(--slate-200);
    border-radius: var(--radius-xl);
    padding: 28px;
    margin-bottom: 16px;
    transition: var(--transition-base);
    box-shadow: 0 2px 12px rgba(201, 181, 156, 0.06), var(--shadow-sm);
    animation: fadeInUp 0.5s ease-out;
}
.glass-card:hover {
    border-color: rgba(201, 181, 156, 0.35);
    box-shadow: var(--shadow-lg);
    transform: translateY(-3px);
}

/* ── Hero Section ── */
.hero-section {
    text-align: center;
    padding: 2.5rem 0 2rem;
    animation: fadeIn 0.6s ease-out;
}
.hero-logo-container {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 18px;
    margin-bottom: 4px;
}
.hero-logo-icon {
    width: 72px;
    height: 72px;
    background: #0f0f0f;
    border-radius: 20px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 2.4rem;
    font-weight: 900;
    color: #ffffff;
    box-shadow: 0 8px 30px rgba(0,0,0,0.2);
    animation: float 3s ease-in-out infinite;
}
.hero-logo {
    font-family: 'Basteleur', 'Plus Jakarta Sans', serif;
    font-size: 4.5rem;
    font-weight: 900;
    color: #0f0f0f;
    -webkit-text-fill-color: #0f0f0f;
    margin-bottom: 0;
    letter-spacing: -2px;
    line-height: 1.1;
}
.hero-tagline {
    color: var(--slate-600);
    font-size: 1.15rem;
    font-weight: 400;
    margin-top: 12px;
    letter-spacing: 0.3px;
    line-height: 1.6;
    max-width: 600px;
    margin-left: auto;
    margin-right: auto;
}
.hero-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: linear-gradient(135deg, rgba(201, 181, 156, 0.12), rgba(217, 207, 199, 0.08));
    border: 1px solid rgba(201, 181, 156, 0.25);
    border-radius: 50px;
    padding: 8px 20px;
    color: var(--indigo-dark);
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    margin-bottom: 16px;
    animation: fadeInUp 0.4s ease-out;
}
.hero-badge::before {
    content: '';
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: var(--green);
    animation: pulse-glow 2s ease-in-out infinite;
}

/* ── Stat Cards ── */
.stat-card {
    background: #ffffff;
    border: 1px solid var(--slate-200);
    border-radius: var(--radius-lg);
    padding: 24px 20px;
    text-align: center;
    transition: var(--transition-base);
    box-shadow: 0 2px 12px rgba(201, 181, 156, 0.08);
    animation: fadeInUp 0.5s ease-out;
    position: relative;
    overflow: hidden;
}
.stat-card::after {
    content: '';
    position: absolute;
    bottom: 0; left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg, var(--indigo), var(--violet), var(--cyan));
    opacity: 0.6;
    transition: opacity var(--transition-base);
}
.stat-card:hover {
    border-color: rgba(201, 181, 156, 0.3);
    transform: translateY(-4px);
    box-shadow: var(--shadow-lg);
}
.stat-card:hover::after {
    opacity: 1;
}
.stat-number {
    font-size: 2.5rem;
    font-weight: 800;
    background: linear-gradient(135deg, var(--indigo), var(--violet));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0;
    line-height: 1.2;
}
.stat-label {
    color: var(--slate-500);
    font-size: 0.82rem;
    font-weight: 500;
    margin-top: 6px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

/* ── Feature Cards (Home) ── */
.feature-card {
    background: #ffffff;
    border: 1px solid var(--slate-200);
    border-radius: var(--radius-xl);
    padding: 32px 24px 28px;
    text-align: center;
    min-height: 240px;
    transition: var(--transition-slow);
    position: relative;
    overflow: hidden;
    box-shadow: var(--shadow-sm);
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: flex-start;
}
.feature-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 4px;
    background: linear-gradient(90deg, var(--indigo), var(--violet), var(--cyan));
    transform: scaleX(0);
    transition: transform var(--transition-slow);
    transform-origin: left;
}
.feature-card:hover {
    border-color: rgba(201, 181, 156, 0.3);
    box-shadow: var(--shadow-xl);
    transform: translateY(-6px);
}
.feature-card:hover::before {
    transform: scaleX(1);
}
.feature-icon {
    font-size: 2.8rem;
    margin-bottom: 16px;
    display: flex;
    align-items: center;
    justify-content: center;
    width: 72px;
    height: 72px;
    border-radius: var(--radius-lg);
    background: linear-gradient(135deg, rgba(201, 181, 156, 0.1), rgba(217, 207, 199, 0.08));
    transition: var(--transition-base);
}
.feature-card:hover .feature-icon {
    transform: scale(1.05);
    background: linear-gradient(135deg, rgba(201, 181, 156, 0.15), rgba(217, 207, 199, 0.12));
}
.feature-title {
    font-size: 1.05rem;
    font-weight: 700;
    color: var(--navy);
    margin-bottom: 8px;
}
.feature-desc {
    font-size: 0.85rem;
    color: var(--slate-500);
    line-height: 1.6;
}

/* ── Score Gauge Card ── */
.score-display {
    background: linear-gradient(145deg, rgba(201, 181, 156, 0.08), rgba(217, 207, 199, 0.05));
    border: 1px solid rgba(201, 181, 156, 0.2);
    border-radius: var(--radius-2xl);
    padding: 28px;
    text-align: center;
    animation: pulse-glow 3s ease-in-out infinite;
}
.score-big {
    font-size: 5rem;
    font-weight: 900;
    line-height: 1;
    margin: 0;
    animation: countUp 0.8s ease-out;
}
.score-grade {
    font-size: 1.1rem;
    font-weight: 600;
    margin-top: 6px;
    text-transform: uppercase;
    letter-spacing: 1px;
}

/* ── Metric Card (reusable) ── */
.metric-card {
    background: #ffffff;
    border: 1px solid var(--slate-200);
    border-radius: var(--radius-lg);
    padding: 20px;
    text-align: center;
    transition: var(--transition-base);
    box-shadow: var(--shadow-sm);
}
.metric-card:hover {
    border-color: rgba(201, 181, 156, 0.3);
    box-shadow: var(--shadow-md);
    transform: translateY(-2px);
}
.metric-card h3 {
    color: var(--slate-500);
    font-size: 0.75rem;
    font-weight: 600;
    margin-bottom: 6px;
    text-transform: uppercase;
    letter-spacing: 0.8px;
}
.metric-card .val {
    font-size: 1.8rem;
    font-weight: 800;
}

/* ── Loan Card ── */
.loan-card {
    background: #ffffff;
    border: 1px solid var(--slate-200);
    border-radius: var(--radius-lg);
    padding: 20px;
    margin-bottom: 12px;
    transition: var(--transition-base);
    box-shadow: var(--shadow-sm);
    position: relative;
}
.loan-card:hover {
    border-color: rgba(201, 181, 156, 0.3);
    box-shadow: var(--shadow-lg);
    transform: translateY(-3px);
}
.loan-card .loan-source {
    font-size: 0.7rem;
    color: var(--slate-500);
    margin-bottom: 6px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    font-weight: 600;
}
.loan-card .loan-name {
    font-size: 1.1rem;
    font-weight: 700;
    color: var(--navy);
}
.loan-card .loan-desc {
    font-size: 0.82rem;
    color: var(--slate-600);
    margin: 8px 0;
    line-height: 1.5;
}
.loan-tag {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 50px;
    font-size: 0.72rem;
    font-weight: 600;
    margin-right: 6px;
    margin-top: 6px;
}
.tag-rate { background: rgba(201, 181, 156, 0.12); color: var(--indigo-dark); }
.tag-amount { background: rgba(16, 185, 129, 0.08); color: #059669; }
.tag-tenure { background: rgba(245, 158, 11, 0.08); color: #b45309; }
.loan-card .loan-meta {
    font-size: 0.72rem;
    color: var(--slate-500);
    margin-top: 10px;
    padding-top: 10px;
    border-top: 1px solid var(--slate-100);
}

/* ── Verdict Banner ── */
.verdict-banner {
    border-radius: var(--radius-xl);
    padding: 28px;
    text-align: center;
    margin: 16px 0;
    animation: fadeInUp 0.5s ease-out;
}
.verdict-title {
    font-size: 1.8rem;
    font-weight: 800;
    letter-spacing: -0.5px;
}
.verdict-sub {
    font-size: 0.95rem;
    margin-top: 8px;
    line-height: 1.5;
}

/* ── Tier Badge ── */
.tier-badge {
    border-radius: var(--radius-lg);
    padding: 16px 24px;
    margin-bottom: 16px;
    display: flex;
    align-items: center;
    flex-wrap: wrap;
    gap: 8px;
}
.tier-badge .tier-text {
    font-size: 1.2rem;
    font-weight: 700;
}

/* ── Section Header ── */
.section-header {
    font-size: 1.5rem;
    font-weight: 700;
    color: var(--navy);
    margin: 1.5rem 0 1rem;
    display: flex;
    align-items: center;
    gap: 8px;
    animation: slideInLeft 0.4s ease-out;
}
.section-header .accent {
    background: linear-gradient(135deg, var(--indigo), var(--violet));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 800;
}

/* ── Persona Selector Card ── */
.persona-card {
    background: #ffffff;
    border: 1px solid var(--slate-200);
    border-radius: var(--radius-lg);
    padding: 20px 16px;
    text-align: center;
    min-height: 160px;
    transition: var(--transition-base);
    box-shadow: var(--shadow-sm);
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
}
.persona-card:hover {
    border-color: rgba(201, 181, 156, 0.35);
    box-shadow: var(--shadow-lg);
    transform: translateY(-4px);
}
.persona-icon {
    font-size: 2.4rem;
    margin-bottom: 4px;
}
.persona-name {
    font-weight: 700;
    color: var(--navy);
    margin: 6px 0 4px;
    font-size: 0.9rem;
}
.persona-desc {
    font-size: 0.75rem;
    color: var(--slate-500);
    line-height: 1.4;
}

/* ── Info Banner ── */
.info-banner {
    background: linear-gradient(135deg, rgba(201, 181, 156, 0.08), rgba(217, 207, 199, 0.05));
    border: 1px solid rgba(201, 181, 156, 0.2);
    border-radius: var(--radius-lg);
    padding: 20px 24px;
    color: var(--indigo-dark);
    font-size: 0.9rem;
    line-height: 1.6;
}

/* ── Trust Section ── */
.trust-section {
    background: linear-gradient(135deg, var(--navy) 0%, var(--navy-light) 100%);
    border-radius: var(--radius-2xl);
    padding: 40px 36px;
    margin: 2rem 0;
    color: white;
    position: relative;
    overflow: hidden;
}
.trust-section::before {
    content: '';
    position: absolute;
    top: -50%; right: -20%;
    width: 400px; height: 400px;
    border-radius: 50%;
    background: radial-gradient(circle, rgba(201, 181, 156, 0.15), transparent 70%);
}
.trust-badge {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    background: rgba(255, 255, 255, 0.1);
    border: 1px solid rgba(255, 255, 255, 0.15);
    border-radius: var(--radius-md);
    padding: 12px 18px;
    color: rgba(255, 255, 255, 0.9);
    font-size: 0.82rem;
    font-weight: 500;
    transition: var(--transition-base);
}
.trust-badge:hover {
    background: rgba(255, 255, 255, 0.15);
    transform: translateY(-2px);
}

/* ── Step Card ── */
.step-card {
    background: #ffffff;
    border: 1px solid var(--slate-200);
    border-radius: var(--radius-xl);
    padding: 32px 24px;
    text-align: center;
    transition: var(--transition-base);
    box-shadow: var(--shadow-sm);
    position: relative;
}
.step-card:hover {
    border-color: rgba(201, 181, 156, 0.3);
    box-shadow: var(--shadow-lg);
    transform: translateY(-4px);
}
.step-number {
    width: 52px;
    height: 52px;
    border-radius: 50%;
    background: linear-gradient(135deg, var(--indigo), var(--violet));
    display: inline-flex;
    align-items: center;
    justify-content: center;
    font-size: 1.3rem;
    font-weight: 800;
    color: white;
    margin-bottom: 16px;
    box-shadow: 0 4px 16px rgba(139, 115, 85, 0.25);
}

/* ── Architecture Card ── */
.arch-card {
    background: #ffffff;
    border: 1px solid var(--slate-200);
    border-radius: var(--radius-lg);
    padding: 24px 20px;
    text-align: center;
    flex: 1;
    min-width: 180px;
    transition: var(--transition-base);
}
.arch-card:hover {
    border-color: rgba(201, 181, 156, 0.3);
    box-shadow: var(--shadow-md);
    transform: translateY(-2px);
}
.arch-arrow {
    color: var(--indigo);
    font-size: 1.8rem;
    display: flex;
    align-items: center;
    padding: 0 8px;
    opacity: 0.6;
}

/* ── CTA Section ── */
.cta-section {
    background: linear-gradient(135deg, var(--indigo) 0%, var(--violet) 100%);
    border-radius: var(--radius-2xl);
    padding: 48px 40px;
    text-align: center;
    color: white;
    margin: 2rem 0;
    position: relative;
    overflow: hidden;
}
.cta-section::before {
    content: '';
    position: absolute;
    top: -30%; left: -10%;
    width: 300px; height: 300px;
    border-radius: 50%;
    background: rgba(255, 255, 255, 0.08);
}
.cta-section::after {
    content: '';
    position: absolute;
    bottom: -20%; right: -5%;
    width: 200px; height: 200px;
    border-radius: 50%;
    background: rgba(255, 255, 255, 0.05);
}

/* ── Streamlit Overrides ── */

/* Hide Streamlit top decoration bar & header */
header[data-testid="stHeader"] {
    background: transparent !important;
}
[data-testid="stDecoration"] {
    display: none !important;
}
[data-testid="stToolbar"] {
    display: none !important;
}
#MainMenu {
    display: none !important;
}

.stTabs [data-baseweb="tab-list"] {
    gap: 4px;
    background: var(--slate-100);
    border-radius: var(--radius-md);
    padding: 4px;
    border: 1px solid var(--slate-200);
}
.stTabs [data-baseweb="tab"] {
    border-radius: var(--radius-sm) !important;
    padding: 10px 22px !important;
    color: var(--slate-600) !important;
    font-weight: 600 !important;
    transition: var(--transition-base) !important;
    font-size: 0.88rem !important;
}
.stTabs [data-baseweb="tab"][aria-selected="true"] {
    background: white !important;
    color: var(--indigo-dark) !important;
    box-shadow: var(--shadow-sm) !important;
}
.stTabs [data-baseweb="tab-panel"] {
    padding-top: 20px;
}

button[kind="primary"], .stButton button[kind="primary"] {
    background: linear-gradient(135deg, var(--indigo) 0%, var(--violet) 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: var(--radius-md) !important;
    font-weight: 600 !important;
    padding: 12px 28px !important;
    font-size: 0.9rem !important;
    transition: var(--transition-base) !important;
    box-shadow: 0 4px 16px rgba(139, 115, 85, 0.2) !important;
    letter-spacing: 0.3px !important;
}
button[kind="primary"]:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 28px rgba(139, 115, 85, 0.25) !important;
}

button:not([kind="primary"]), .stButton button:not([kind="primary"]) {
    background: #ffffff !important;
    color: var(--navy) !important;
    border-radius: var(--radius-md) !important;
    border: 1px solid var(--slate-200) !important;
    transition: var(--transition-base) !important;
    font-weight: 500 !important;
}
button:not([kind="primary"]):hover {
    border-color: rgba(201, 181, 156, 0.4) !important;
    background: var(--indigo-glow) !important;
    color: var(--indigo-dark) !important;
}

div[data-testid="stExpander"] {
    background: #ffffff !important;
    border: 1px solid var(--slate-200) !important;
    border-radius: var(--radius-lg) !important;
    box-shadow: var(--shadow-sm) !important;
}

div[data-testid="stMetric"] {
    background: #ffffff;
    border: 1px solid var(--slate-200);
    border-radius: var(--radius-lg);
    padding: 16px 18px;
    box-shadow: var(--shadow-sm);
    transition: var(--transition-base);
}
div[data-testid="stMetric"]:hover {
    border-color: rgba(201, 181, 156, 0.25);
    box-shadow: var(--shadow-md);
}
div[data-testid="stMetric"] label {
    color: var(--slate-500) !important;
    font-weight: 600 !important;
    font-size: 0.72rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.8px !important;
}
div[data-testid="stMetric"] [data-testid="stMetricValue"] {
    color: var(--navy) !important;
    font-weight: 800 !important;
}

input, textarea, select, .stSelectbox > div > div,
.stTextInput > div > div > input,
.stNumberInput > div > div > input {
    background: #ffffff !important;
    border: 1px solid var(--slate-200) !important;
    border-radius: var(--radius-md) !important;
    color: var(--navy) !important;
    transition: var(--transition-base) !important;
}
input:focus, textarea:focus {
    border-color: var(--indigo) !important;
    box-shadow: 0 0 0 3px rgba(201, 181, 156, 0.15) !important;
}

div.stProgress > div > div {
    background: var(--slate-100) !important;
    border-radius: 50px !important;
    height: 8px !important;
}
div.stProgress > div > div > div {
    background: linear-gradient(90deg, var(--indigo), var(--violet)) !important;
    border-radius: 50px !important;
}

div[data-testid="stForm"] {
    background: #ffffff !important;
    border: 1px solid var(--slate-200) !important;
    border-radius: var(--radius-xl) !important;
    padding: 28px !important;
    box-shadow: var(--shadow-sm) !important;
}

/* ── Divider ── */
.divider {
    height: 1px;
    background: linear-gradient(90deg, transparent 0%, var(--slate-200) 50%, transparent 100%);
    margin: 2.5rem 0;
}

/* ── Footer ── */
.footer {
    text-align: center;
    padding: 3rem 0 1.5rem;
    color: var(--slate-500);
    font-size: 0.82rem;
    border-top: 1px solid var(--slate-100);
    margin-top: 2rem;
}
.footer a { color: var(--indigo); text-decoration: none; font-weight: 500; }
.footer a:hover { text-decoration: underline; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: var(--slate-200); border-radius: 50px; }
::-webkit-scrollbar-thumb:hover { background: var(--slate-400); }

/* ── Dataframe styling ── */
.stDataFrame { border-radius: var(--radius-lg) !important; overflow: hidden; }
.stDataFrame [data-testid="stDataFrameResizable"] {
    background: #ffffff !important;
}
.stDataFrame th {
    background: #EFE9E3 !important;
    color: #0f172a !important;
    font-weight: 600 !important;
}
.stDataFrame td {
    background: #ffffff !important;
    color: #0f172a !important;
}
.stDataFrame [class*="glideDataEditor"],
[data-testid="glideDataEditor"] {
    background: #ffffff !important;
    --gdg-bg-cell: #ffffff !important;
    --gdg-bg-header: #EFE9E3 !important;
    --gdg-bg-header-has-focus: #D9CFC7 !important;
    --gdg-text-dark: #0f172a !important;
    --gdg-text-header: #0f172a !important;
    --gdg-border-color: #D9CFC7 !important;
    --gdg-accent-color: #C9B59C !important;
    --gdg-accent-light: rgba(201,181,156,0.1) !important;
    --gdg-bg-cell-medium: #F9F8F6 !important;
}

/* ── Selection highlighting ── */
::selection {
    background: rgba(201, 181, 156, 0.25);
    color: var(--navy);
}

/* ── Download Button ── */
[data-testid="stDownloadButton"] button {
    background: linear-gradient(135deg, var(--indigo) 0%, var(--violet) 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: var(--radius-md) !important;
    font-weight: 600 !important;
    padding: 10px 24px !important;
    font-size: 0.88rem !important;
    transition: var(--transition-base) !important;
    box-shadow: 0 4px 16px rgba(139, 115, 85, 0.2) !important;
}
[data-testid="stDownloadButton"] button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 28px rgba(139, 115, 85, 0.25) !important;
}

/* ── File Uploader ── */
[data-testid="stFileUploader"] {
    background: #ffffff !important;
    border-radius: var(--radius-lg) !important;
}
[data-testid="stFileUploader"] section {
    background: #ffffff !important;
    border: 2px dashed var(--slate-300) !important;
    border-radius: var(--radius-lg) !important;
    padding: 24px !important;
}
[data-testid="stFileUploader"] section > div {
    color: var(--navy) !important;
}
[data-testid="stFileUploader"] section span,
[data-testid="stFileUploader"] section small,
[data-testid="stFileUploader"] section p,
[data-testid="stFileUploader"] section div {
    color: var(--slate-600) !important;
}
[data-testid="stFileUploader"] section button {
    background: white !important;
    color: var(--indigo) !important;
    border: 2px solid var(--indigo) !important;
    border-radius: var(--radius-md) !important;
    font-weight: 600 !important;
}
[data-testid="stFileUploader"] section button:hover {
    background: var(--indigo-glow) !important;
}
[data-testid="stFileUploaderDropzone"] {
    background: #f8fafc !important;
    border: 2px dashed var(--slate-300) !important;
    border-radius: var(--radius-lg) !important;
}
[data-testid="stFileUploaderDropzone"] div,
[data-testid="stFileUploaderDropzone"] span,
[data-testid="stFileUploaderDropzone"] small {
    color: var(--slate-600) !important;
}
[data-testid="stFileUploaderDropzoneInstructions"] div,
[data-testid="stFileUploaderDropzoneInstructions"] span,
[data-testid="stFileUploaderDropzoneInstructions"] small {
    color: var(--slate-500) !important;
}

/* ── Alerts / Info boxes ── */
[data-testid="stAlert"] {
    background: #ffffff !important;
    border-radius: var(--radius-md) !important;
}
[data-testid="stAlert"] p,
[data-testid="stAlert"] span {
    color: var(--navy) !important;
}

/* ── Code block fix ── */
[data-testid="stCode"], .stCodeBlock, code {
    background: #EFE9E3 !important;
    color: var(--navy) !important;
    border-radius: var(--radius-sm) !important;
}

/* ── Selectbox/Multiselect dropdowns ── */
[data-baseweb="select"] {
    background: #ffffff !important;
}
[data-baseweb="select"] > div {
    background: #ffffff !important;
    border: 1px solid var(--slate-200) !important;
    border-radius: var(--radius-md) !important;
    color: var(--navy) !important;
}
[data-baseweb="select"] svg {
    fill: var(--navy) !important;
    color: var(--navy) !important;
}
[data-baseweb="popover"],
[data-baseweb="popover"] > div,
[data-baseweb="menu"],
[role="listbox"],
[data-baseweb="select"] [role="listbox"] {
    background: #ffffff !important;
    background-color: #ffffff !important;
    border: 1px solid var(--slate-200) !important;
    border-radius: var(--radius-md) !important;
    box-shadow: 0 4px 20px rgba(0,0,0,0.08) !important;
}
[data-baseweb="popover"] li,
[data-baseweb="menu"] li,
[role="listbox"] li,
[role="option"],
[data-baseweb="menu"] [role="option"],
[data-baseweb="popover"] [role="option"] {
    color: var(--navy) !important;
    background: #ffffff !important;
    background-color: #ffffff !important;
}
[data-baseweb="popover"] li:hover,
[data-baseweb="menu"] li:hover,
[role="option"]:hover,
[role="option"][aria-selected="true"],
[data-baseweb="menu"] [role="option"]:hover {
    background: var(--indigo-glow) !important;
    background-color: rgba(201, 181, 156, 0.1) !important;
    color: var(--indigo-dark) !important;
}
/* Dropdown list container */
[data-baseweb="popover"] ul,
[data-baseweb="menu"] ul,
[role="listbox"] ul {
    background: #ffffff !important;
    background-color: #ffffff !important;
}

/* ── Radio/Checkbox ── */
.stRadio label, .stCheckbox label {
    color: var(--navy) !important;
}

/* ── Catch-all for dark text in main area ── */
.stMarkdown, .stMarkdown p, .stMarkdown span, .stMarkdown li,
.stMarkdown h1, .stMarkdown h2, .stMarkdown h3, .stMarkdown h4 {
    color: var(--navy);
}

/* ── Toast / Notifications ── */
[data-testid="stToast"] {
    background: #ffffff !important;
    color: var(--navy) !important;
    border: 1px solid var(--slate-200) !important;
}

/* ── Caption text ── */
.stCaption, [data-testid="stCaptionContainer"] {
    color: var(--slate-500) !important;
}
</style>
""", unsafe_allow_html=True)


# ─── Data Loading ───────────────────────────────────────────────────────────
@st.cache_data
def load_or_generate_data():
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
    df_feat = engineer_features(df)
    df_scored = compute_all_scores(df_feat)
    model = CreditRiskModel()
    metrics = model.train(df_scored)
    risk_probs = model.predict_risk(df_scored)
    df_scored["risk_probability"] = risk_probs
    final_scores = []
    for idx, row in df_scored.iterrows():
        fs = compute_final_score(row["base_trust_score"], row["risk_probability"], row)
        final_scores.append(fs)
    final_df = pd.DataFrame(final_scores)
    overlap_cols = [c for c in final_df.columns if c in df_scored.columns]
    final_df = final_df.drop(columns=overlap_cols, errors="ignore")
    df_scored = pd.concat([df_scored.reset_index(drop=True), final_df.reset_index(drop=True)], axis=1)
    return model, df_scored, metrics


# ─── Helper: Gauge Chart ───────────────────────────────────────────────────
def create_gauge(score, grade, color):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        number={"font": {"size": 56, "color": color, "family": "Plus Jakarta Sans, Inter"}},
        title={"text": f"<b>{grade}</b>", "font": {"size": 18, "color": color, "family": "Plus Jakarta Sans, Inter"}},
        gauge={
            "axis": {"range": [300, 900], "tickwidth": 2, "tickcolor": "#334155",
                     "tickvals": [300, 400, 500, 650, 750, 900],
                     "ticktext": ["300", "400", "500", "650", "750", "900"],
                     "tickfont": {"color": "#64748b"}},
            "bar": {"color": color, "thickness": 0.3},
            "bgcolor": "rgba(15,23,42,0.06)",
            "borderwidth": 0,
            "steps": [
                {"range": [300, 400], "color": "rgba(239,68,68,0.12)"},
                {"range": [400, 500], "color": "rgba(249,115,22,0.12)"},
                {"range": [500, 650], "color": "rgba(234,179,8,0.12)"},
                {"range": [650, 750], "color": "rgba(132,204,22,0.12)"},
                {"range": [750, 900], "color": "rgba(34,197,94,0.12)"},
            ],
            "threshold": {
                "line": {"color": "#0f172a", "width": 3},
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
        font={"color": "#0f172a", "family": "Plus Jakarta Sans, Inter"}
    )
    return fig


def create_subscore_radar(breakdown):
    categories = list(breakdown.keys())
    values = [breakdown[c]["score"] for c in categories]
    values.append(values[0])
    categories.append(categories[0])

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values, theta=categories, fill="toself",
        fillcolor="rgba(201,181,156,0.15)",
        line=dict(color="#C9B59C", width=2),
        marker=dict(size=6, color="#D9CFC7"),
        name="Sub-Scores"
    ))
    fig.update_layout(
        polar=dict(
            bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(visible=True, range=[0, 100],
                          tickfont=dict(size=10, color="#475569"),
                          gridcolor="rgba(201,181,156,0.1)"),
            angularaxis=dict(tickfont=dict(size=11, color="#475569"),
                           gridcolor="rgba(201,181,156,0.1)"),
        ),
        showlegend=False, height=350,
        margin=dict(t=30, b=30, l=60, r=60),
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def create_component_bars(breakdown):
    all_components = []
    for category, data in breakdown.items():
        for comp_name, comp_val in data["components"].items():
            all_components.append({
                "Category": category, "Component": comp_name, "Score": comp_val
            })
    comp_df = pd.DataFrame(all_components)
    fig = px.bar(
        comp_df, y="Component", x="Score", color="Category",
        orientation="h",
        color_discrete_sequence=["#C9B59C", "#D9CFC7", "#B8A48E", "#f59e0b"],
        height=400,
    )
    fig.update_layout(
        xaxis=dict(range=[0, 100], title="Score", gridcolor="rgba(201,181,156,0.1)"),
        yaxis=dict(title="", gridcolor="rgba(201,181,156,0.1)"),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#0f172a", family="Plus Jakarta Sans, Inter"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5,
                   font=dict(color="#475569")),
        margin=dict(t=40, b=20, l=10, r=10),
    )
    return fig


def create_income_chart(monthly_incomes):
    months = [f"Month {i+1}" for i in range(len(monthly_incomes))]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=months, y=monthly_incomes, mode="lines+markers",
        line=dict(color="#C9B59C", width=3),
        marker=dict(size=8, color="#D9CFC7"),
        fill="tozeroy", fillcolor="rgba(201,181,156,0.1)",
    ))
    fig.update_layout(
        height=250, xaxis=dict(title="", gridcolor="rgba(201,181,156,0.08)"),
        yaxis=dict(title="Income (₹)", gridcolor="rgba(201,181,156,0.08)"),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#0f172a", family="Plus Jakarta Sans, Inter"),
        margin=dict(t=10, b=30, l=10, r=10),
    )
    return fig


# ─── MAIN APP ──────────────────────────────────────────────────────────────
def main():
    # Load data & train model
    with st.spinner("Loading AI engine..."):
        raw_df = load_or_generate_data()
        model, df, metrics = train_model(raw_df)

    # ── Top Navigation Bar ───────────────────────────────────────────────
    if "current_page" not in st.session_state:
        st.session_state.current_page = "🏠 Home"

    nav_items = [
        ("🏠 Home", "Home"),
        ("📤 Score Me", "Score Me"),
        ("🌍 Alternative Score", "Alt Score"),
        ("🔍 Find Loans", "Loans"),
        ("🚀 Improve", "Improve"),
    ]

    brand_col, nav_col, status_col = st.columns([1.2, 3.6, 1.2])
    with brand_col:
        st.markdown("""
        <div class="navbar-brand">
            <div class="logo-icon">₹</div>
            <div>
                <div class="brand-text">CrediVist</div>
                <div class="brand-sub">AI Credit Engine</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    with nav_col:
        btn_cols = st.columns(len(nav_items))
        for i, (page_key, label) in enumerate(nav_items):
            with btn_cols[i]:
                is_active = st.session_state.current_page == page_key
                css_class = "nav-btn-active" if is_active else "nav-btn-area"
                st.markdown(f'<div class="nav-btn-area {css_class}">', unsafe_allow_html=True)
                if st.button(label, key=f"nav_{i}", use_container_width=True):
                    st.session_state.current_page = page_key
                    st.rerun()
                st.markdown('</div>', unsafe_allow_html=True)
    with status_col:
        st.markdown("""
        <div style="display:flex; justify-content:flex-end; padding-top:6px;">
            <div class="navbar-status">
                <div class="status-dot"></div>
                AI Engine Active
            </div>
        </div>
        """, unsafe_allow_html=True)
    st.markdown('<div class="navbar-divider"></div>', unsafe_allow_html=True)

    page = st.session_state.current_page

    # ════════════════════════════════════════════════════════════════════
    # PAGE: HOME
    # ════════════════════════════════════════════════════════════════════
    if page == "🏠 Home":
        # Hero
        st.markdown("""
        <div class="hero-section">
            <div class="hero-badge">AI-Powered Financial Inclusion</div>
            <div class="hero-logo-container">
                <div class="hero-logo-icon">₹</div>
                <div class="hero-logo">CrediVist</div>
            </div>
            <div class="hero-tagline">
                Alternative Credit Scoring for the Underbanked —
                <span style="color:#8B7355; font-weight:600;">No Bank Account Needed</span>
            </div>
            <div style="margin-top:24px; display:flex; gap:12px; justify-content:center; flex-wrap:wrap;">
                <div style="display:inline-flex;align-items:center;gap:6px;padding:8px 16px;
                     background:#ffffff;border:1px solid #e2e8f0;border-radius:50px;font-size:0.78rem;
                     color:#475569;font-weight:500;">
                    <span style="color:#22c55e;">✓</span> 5 Personas Supported
                </div>
                <div style="display:inline-flex;align-items:center;gap:6px;padding:8px 16px;
                     background:#ffffff;border:1px solid #e2e8f0;border-radius:50px;font-size:0.78rem;
                     color:#475569;font-weight:500;">
                    <span style="color:#22c55e;">✓</span> 34+ Loan Products
                </div>
                <div style="display:inline-flex;align-items:center;gap:6px;padding:8px 16px;
                     background:#ffffff;border:1px solid #e2e8f0;border-radius:50px;font-size:0.78rem;
                     color:#475569;font-weight:500;">
                    <span style="color:#22c55e;">✓</span> SHAP Explainability
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

        # Stats Row
        total_loans = len(get_all_loans_catalog())
        sc1, sc2, sc3, sc4 = st.columns(4)
        with sc1:
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-number">{len(PERSONAS)}</div>
                <div class="stat-label">Personas Supported</div>
            </div>""", unsafe_allow_html=True)
        with sc2:
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-number">{total_loans}+</div>
                <div class="stat-label">Loan Products</div>
            </div>""", unsafe_allow_html=True)
        with sc3:
            st.markdown("""
            <div class="stat-card">
                <div class="stat-number">300-900</div>
                <div class="stat-label">Score Range</div>
            </div>""", unsafe_allow_html=True)
        with sc4:
            st.markdown("""
            <div class="stat-card">
                <div class="stat-number">11+</div>
                <div class="stat-label">Data Signals</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("")

        # Mission
        st.markdown("""
        <div class="glass-card" style="text-align:center; max-width:800px; margin:0 auto; border-left:4px solid #C9B59C;">
            <div style="font-size:1.35rem; font-weight:800; color:#0f172a; margin-bottom:12px;
                 letter-spacing:-0.3px;">
                🌍 Our Mission
            </div>
            <div style="color:#475569; font-size:0.95rem; line-height:1.8;">
                <b style="color:#8B7355;">1.4 billion adults</b> worldwide lack access to formal credit.
                CrediVist bridges this gap using <b style="color:#8B7355;">AI and alternative data</b> —
                transaction patterns, utility bills, gig work history, farm records, and more —
                to build credit profiles for <b style="color:#8B7355;">anyone</b>, even without a bank account.
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("")

        # Feature Cards
        features = [
            ("📤", "Score Me", "Upload your bank or UPI statement and get an instant AI-powered credit score with full breakdown."),
            ("🌍", "Alternative Score", "No bank account? Score yourself as a farmer, student, vendor, homemaker, or gig worker using alternative data."),
            ("🔍", "Find Loans", "Browse 34+ loan products — government schemes, micro-credit, and more. Check eligibility instantly."),
            ("🚀", "Improve", "Get a personalized roadmap to improve your score. Simulate changes and see projected gains."),
            ("🧠", "AI Explainability", "Understand exactly why you got your score — powered by SHAP and ensemble ML models."),
            ("📄", "Smart OCR", "Upload scanned documents — land records, marksheets, licenses — and we extract data automatically."),
        ]
        for row_start in range(0, len(features), 3):
            cols = st.columns(3)
            for i, col in enumerate(cols):
                idx = row_start + i
                if idx < len(features):
                    icon, title, desc = features[idx]
                    with col:
                        st.markdown(f"""
                        <div class="feature-card">
                            <span class="feature-icon">{icon}</span>
                            <div class="feature-title">{title}</div>
                            <div class="feature-desc">{desc}</div>
                        </div>
                        """, unsafe_allow_html=True)

        st.markdown("")

        # How it Works
        st.markdown("""
        <div class="divider"></div>
        <div style="text-align:center; margin-bottom:24px;">
            <div style="font-size:0.75rem;font-weight:700;text-transform:uppercase;letter-spacing:2px;
                 color:#C9B59C;margin-bottom:8px;">Simple Process</div>
            <div style="font-size:1.6rem; font-weight:800; color:#0f172a; letter-spacing:-0.5px;">How It Works</div>
            <div style="color:#64748b; font-size:0.9rem; margin-top:6px;">Three simple steps to your credit score</div>
        </div>
        """, unsafe_allow_html=True)

        hw1, hw2, hw3 = st.columns(3)
        steps = [
            ("1", "Upload or Enter Data", "Bank statements, documents, or manual form — your choice.", "📤"),
            ("2", "AI Analyzes", "11+ signals processed by ensemble ML + SHAP explainability.", "🧠"),
            ("3", "Get Score & Loans", "Instant score, loan recommendations, and improvement roadmap.", "🎯"),
        ]
        for col, (num, title, desc, icon) in zip([hw1, hw2, hw3], steps):
            with col:
                st.markdown(f"""
                <div class="step-card">
                    <div class="step-number">{num}</div>
                    <div style="font-size:1.5rem;margin-bottom:8px;">{icon}</div>
                    <div style="font-weight:700; color:#0f172a; font-size:1.05rem; margin-bottom:8px;">{title}</div>
                    <div style="color:#64748b; font-size:0.85rem; line-height:1.6;">{desc}</div>
                </div>
                """, unsafe_allow_html=True)

        # Model Architecture
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        st.markdown('<div class="section-header">🤖 <span class="accent">Model Architecture</span></div>', unsafe_allow_html=True)

        st.markdown("""
        <div style="display:flex; flex-wrap:wrap; gap:12px; justify-content:center; align-items:stretch;">
            <div class="arch-card">
                <div style="font-size:2rem;margin-bottom:10px;">📊</div>
                <div style="font-weight:700; color:#0f172a; font-size:0.95rem; margin-bottom:6px;">Feature Engineering</div>
                <div style="color:#64748b; font-size:0.8rem; line-height:1.5;">11 features from transaction patterns, income stability, digital behavior</div>
            </div>
            <div class="arch-arrow">→</div>
            <div class="arch-card">
                <div style="font-size:2rem;margin-bottom:10px;">🧠</div>
                <div style="font-weight:700; color:#0f172a; font-size:0.95rem; margin-bottom:6px;">Ensemble ML</div>
                <div style="color:#64748b; font-size:0.8rem; line-height:1.5;">XGBoost (60%) + Logistic Regression (40%) weighted ensemble</div>
            </div>
            <div class="arch-arrow">→</div>
            <div class="arch-card">
                <div style="font-size:2rem;margin-bottom:10px;">📈</div>
                <div style="font-weight:700; color:#0f172a; font-size:0.95rem; margin-bottom:6px;">Score + Explainability</div>
                <div style="color:#64748b; font-size:0.8rem; line-height:1.5;">300-900 trust score with SHAP-based explanations per feature</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Feature Importance
        imp = model.get_feature_importance()
        if imp:
            st.markdown('<div class="section-header">🔬 <span class="accent">Feature Importance</span></div>', unsafe_allow_html=True)
            imp_df = pd.DataFrame({
                "Feature": [FEATURE_LABELS.get(k, k) for k in imp.keys()],
                "Importance": list(imp.values())
            })
            fig = px.bar(imp_df, x="Importance", y="Feature", orientation="h",
                         color="Importance",
                         color_continuous_scale=["#EFE9E3", "#D9CFC7", "#C9B59C"])
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#0f172a", family="Plus Jakarta Sans, Inter", size=13), height=600,
                margin=dict(t=10, b=40, l=200, r=10),
                yaxis=dict(autorange="reversed", gridcolor="rgba(201,181,156,0.08)",
                           tickfont=dict(color="#1e293b", size=13, family="Plus Jakarta Sans, Inter"),
                           title_font=dict(color="#1e293b", size=14)),
                xaxis=dict(gridcolor="rgba(201,181,156,0.08)",
                           tickfont=dict(color="#1e293b", size=12),
                           title_font=dict(color="#1e293b", size=14)),
            )
            st.plotly_chart(fig, use_container_width=True)

        # Footer
        st.markdown("""
        <div class="divider"></div>
        """, unsafe_allow_html=True)

        # Trust & Security Section
        st.markdown("""
        <div class="trust-section">
            <div style="text-align:center; position:relative; z-index:1;">
                <div style="font-size:0.72rem;font-weight:700;text-transform:uppercase;letter-spacing:2px;
                     color:rgba(255,255,255,0.55);margin-bottom:12px;">Why Trust CrediVist</div>
                <div style="font-size:1.6rem; font-weight:800; color:white; margin-bottom:8px; letter-spacing:-0.5px;">
                    Built for Trust, Designed for Transparency
                </div>
                <div style="color:rgba(255,255,255,0.7); font-size:0.9rem; max-width:600px; margin:0 auto 28px;line-height:1.6;">
                    Every score comes with a full explanation. No black boxes, no hidden rules.
                    Your data stays on your machine — we never store or share it.
                </div>
                <div style="display:flex; justify-content:center; gap:14px; flex-wrap:wrap;">
                    <div class="trust-badge">🔒 Local Processing</div>
                    <div class="trust-badge">🧠 SHAP Explainability</div>
                    <div class="trust-badge">📊 11+ Data Signals</div>
                    <div class="trust-badge">🛡️ No Data Storage</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # CTA Section
        st.markdown("""
        <div class="cta-section">
            <div style="position:relative; z-index:1;">
                <div style="font-size:1.8rem; font-weight:800; color:white; margin-bottom:10px; letter-spacing:-0.5px;">
                    Ready to Discover Your Credit Score?
                </div>
                <div style="color:rgba(255,255,255,0.85); font-size:1rem; margin-bottom:24px; max-width:500px; margin-left:auto; margin-right:auto; line-height:1.6;">
                    Upload a bank statement or fill a simple form — it takes less than 2 minutes.
                </div>
                <div style="display:flex; justify-content:center; gap:16px; flex-wrap:wrap;">
                    <div style="padding:12px 28px; background:white; color:#8B7355; border-radius:12px;
                         font-weight:700; font-size:0.9rem; cursor:pointer; transition:all 0.25s;
                         box-shadow:0 4px 16px rgba(0,0,0,0.15);">
                        📤 Score Me Now
                    </div>
                    <div style="padding:12px 28px; background:rgba(255,255,255,0.15); color:white;
                         border:1px solid rgba(255,255,255,0.3); border-radius:12px;
                         font-weight:600; font-size:0.9rem; cursor:pointer;">
                        🌍 No Bank Account? Try Alternative Score
                    </div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="footer">
            <div style="margin-bottom:12px;">
                <span style="font-family:'Basteleur','Plus Jakarta Sans',serif;font-size:1.1rem;font-weight:800;color:#0f0f0f;">
                    ₹ CrediVist
                </span>
            </div>
            AI-Powered Alternative Credit Scoring<br/>
            <span style="color:#475569;">Empowering the underbanked with fair, transparent credit access</span>
            <div style="margin-top:16px; display:flex; justify-content:center; gap:20px;
                 color:#475569; font-size:0.72rem;">
                <span style="font-family:'Basteleur','Plus Jakarta Sans',serif;">© 2026 CrediVist</span>
                <span>·</span>
                <span>Privacy Policy</span>
                <span>·</span>
                <span>Terms of Service</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ════════════════════════════════════════════════════════════════════
    # PAGE: SCORE ME  (Upload Bank Statement)
    # ════════════════════════════════════════════════════════════════════
    elif page == "📤 Score Me":
        st.markdown("""
        <div class="section-header">📤 <span class="accent">Upload & Get Your Score</span></div>
        <div style="color:#64748b; font-size:0.95rem; margin-bottom:8px; line-height:1.6;">
            Upload your bank or UPI transaction history (CSV/Excel) to get an
            <b style="color:#8B7355;">instant AI-powered credit score</b> with full breakdown and loan recommendations.
        </div>
        <div style="display:flex;gap:12px;margin-bottom:20px;flex-wrap:wrap;">
            <span style="display:inline-flex;align-items:center;gap:4px;font-size:0.75rem;color:#64748b;padding:4px 12px;background:#f8fafc;border-radius:50px;border:1px solid #e2e8f0;">📊 11 Signals</span>
            <span style="display:inline-flex;align-items:center;gap:4px;font-size:0.75rem;color:#64748b;padding:4px 12px;background:#f8fafc;border-radius:50px;border:1px solid #e2e8f0;">🧠 SHAP Explained</span>
            <span style="display:inline-flex;align-items:center;gap:4px;font-size:0.75rem;color:#64748b;padding:4px 12px;background:#f8fafc;border-radius:50px;border:1px solid #e2e8f0;">💳 Loan Match</span>
        </div>
        """, unsafe_allow_html=True)

        # File upload
        uploaded_file = st.file_uploader(
            "Upload your bank statement",
            type=["csv", "xlsx", "xls"],
            help="Your file should have columns for Date, Description, and Debit/Credit amounts.",
        )

        if uploaded_file is not None:
            try:
                parser = TransactionParser()
                file_ext = uploaded_file.name.rsplit(".", 1)[-1].lower()
                parsed_df = parser.parse_file(uploaded_file, file_ext)
                categorized_df = parser.auto_categorize(parsed_df)

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

                st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

                tab1, tab2, tab3 = st.tabs(
                    ["📋 Transactions", "📊 Category Breakdown", "📈 Monthly Trend"]
                )

                with tab1:
                    display_df = categorized_df[
                        ["date", "description", "amount", "type", "category", "category_confidence"]
                    ].copy()
                    display_df["date"] = display_df["date"].dt.strftime("%d-%m-%Y")
                    display_df["amount"] = display_df["amount"].apply(lambda x: f"₹{x:,.2f}")
                    display_df["category_confidence"] = display_df["category_confidence"].apply(lambda x: f"{x:.0%}")
                    display_df.columns = ["Date", "Description", "Amount", "Type", "Category", "Confidence"]
                    st.dataframe(display_df, use_container_width=True, height=400)

                with tab2:
                    cat_summary = parser.get_category_summary()
                    if len(cat_summary) > 0:
                        col_pie, col_bar = st.columns(2)
                        with col_pie:
                            fig = px.pie(
                                cat_summary, values="total", names="category",
                                title="Spending by Category",
                                color_discrete_sequence=["#C9B59C", "#D9CFC7", "#B8A48E", "#B8A48E",
                                                        "#22c55e", "#eab308", "#f97316", "#ef4444"],
                                hole=0.4,
                            )
                            fig.update_layout(
                                paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#0f172a", family="Plus Jakarta Sans, Inter"),
                                height=400, margin=dict(t=40, b=10, l=10, r=10),
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        with col_bar:
                            fig = px.bar(
                                cat_summary, x="total", y="category", orientation="h",
                                title="Spending Amount by Category", color="category",
                                color_discrete_sequence=["#C9B59C", "#D9CFC7", "#B8A48E", "#B8A48E",
                                                        "#22c55e", "#eab308", "#f97316", "#ef4444"],
                            )
                            fig.update_layout(
                                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                font=dict(color="#0f172a", family="Plus Jakarta Sans, Inter"),
                                height=400, showlegend=False,
                                xaxis_title="Total (₹)", yaxis_title="",
                                margin=dict(t=40, b=10, l=10, r=10),
                                xaxis=dict(gridcolor="rgba(201,181,156,0.08)"),
                            )
                            st.plotly_chart(fig, use_container_width=True)

                with tab3:
                    monthly_summ = parser.get_monthly_summary()
                    if len(monthly_summ) > 0:
                        fig = go.Figure()
                        fig.add_trace(go.Bar(x=monthly_summ["month"], y=monthly_summ.get("credit", [0]),
                                           name="Income", marker_color="#22c55e"))
                        fig.add_trace(go.Bar(x=monthly_summ["month"], y=monthly_summ.get("debit", [0]),
                                           name="Expenses", marker_color="#ef4444"))
                        if "net_savings" in monthly_summ.columns:
                            fig.add_trace(go.Scatter(
                                x=monthly_summ["month"], y=monthly_summ["net_savings"],
                                name="Net Savings", line=dict(color="#C9B59C", width=3),
                                mode="lines+markers",
                            ))
                        fig.update_layout(
                            barmode="group", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                            font=dict(color="#0f172a", family="Plus Jakarta Sans, Inter"), height=350,
                            title="Monthly Income vs Expenses", margin=dict(t=40, b=30, l=10, r=10),
                            xaxis=dict(gridcolor="rgba(201,181,156,0.08)"),
                            yaxis=dict(gridcolor="rgba(201,181,156,0.08)"),
                        )
                        st.plotly_chart(fig, use_container_width=True)

                st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

                # Supplementary info
                st.markdown('<div class="section-header">🔧 Additional Details</div>', unsafe_allow_html=True)
                st.markdown("""
                <div style="color:#475569; font-size:0.85rem; margin-bottom:12px;">
                    These details improve scoring accuracy but are <b>optional</b> (defaults provided).
                </div>
                """, unsafe_allow_html=True)

                sup_c1, sup_c2, sup_c3 = st.columns(3)
                with sup_c1:
                    platform_rating = st.slider("Platform Rating (if gig worker)", 1.0, 5.0, 4.0, 0.1,
                                               help="Your average rating on gig platforms.")
                with sup_c2:
                    active_days = st.slider("Active Work Days / Month", 1, 30, 20,
                                           help="Average number of days you work per month.")
                with sup_c3:
                    st.markdown("")
                    generate_btn = st.button("🔮 Generate Credit Score", use_container_width=True, type="primary")

                if generate_btn:
                    with st.spinner("🧠 AI is analyzing your financial profile..."):
                        profile = parser.extract_profile(
                            platform_rating=platform_rating, active_days=active_days)
                        features = extract_all_features(profile)
                        for key, val in features.items():
                            if not isinstance(val, str):
                                profile[key] = val
                        base_result = compute_base_score(profile)
                        for key, val in base_result.items():
                            profile[key] = val
                        try:
                            risk_prob = model.predict_single(profile)
                        except Exception:
                            risk_prob = 0.25
                        final = compute_final_score(float(profile["base_trust_score"]), risk_prob, profile)

                    # ── Display Results ──
                    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
                    st.markdown('<div class="section-header">🏆 <span class="accent">Your Credit Score</span></div>',
                               unsafe_allow_html=True)

                    r1, r2 = st.columns([1, 1])
                    with r1:
                        st.plotly_chart(
                            create_gauge(final["final_trust_score"], final["grade"], final["grade_color"]),
                            use_container_width=True)
                    with r2:
                        rm1, rm2, rm3 = st.columns(3)
                        rm1.metric("Final Score", f"{final['final_trust_score']:.0f}")
                        rm2.metric("Risk Level", f"{risk_prob:.1%}")
                        rm3.metric("Confidence", f"{final['confidence']:.0%}")
                        st.markdown(f"**Grade:** {final['grade']}")
                        st.markdown(f"**Base Score:** {profile['base_trust_score']:.0f}")

                    # Sub-score breakdown
                    st.markdown('<div class="section-header">📋 Score Breakdown</div>', unsafe_allow_html=True)
                    breakdown = get_score_breakdown(profile)
                    bk_cols = st.columns(4)
                    for i, (cat_name, cat_data) in enumerate(breakdown.items()):
                        with bk_cols[i]:
                            sv = cat_data["score"]
                            cv = "#22c55e" if sv >= 70 else "#eab308" if sv >= 40 else "#ef4444"
                            st.markdown(
                                f'<div class="metric-card">'
                                f'<h3>{cat_name}</h3>'
                                f'<div class="val" style="color:{cv}">{sv:.1f}</div>'
                                f'<div style="color:#475569; font-size:0.72rem">Weight: {cat_data["weight"]}</div>'
                                f'</div>', unsafe_allow_html=True)
                            st.progress(int(min(sv, 100)))

                    # AI Explanation
                    st.markdown('<div class="section-header">🧠 <span class="accent">AI Explanation</span></div>',
                               unsafe_allow_html=True)
                    try:
                        explainer = ScoreExplainer(model)
                        explainer.initialize(df)
                        explanation = explainer.explain_single(profile)
                        col_e1, col_e2 = st.columns(2)
                        with col_e1:
                            st.markdown("#### ✅ Positive Factors")
                            for f in explanation.get("top_positive_factors", [])[:5]:
                                st.markdown(f"- **{f['feature']}**: {f['feature_value']:.2f}")
                        with col_e2:
                            st.markdown("#### ⚠️ Risk Factors")
                            for f in explanation.get("top_risk_factors", [])[:5]:
                                st.markdown(f"- **{f['feature']}**: {f['feature_value']:.2f}")
                        st.markdown("---")
                        st.markdown(explanation.get("explanation_text", ""))
                    except Exception as e:
                        st.warning(f"Explainability module: {e}")

                    # ── Loan Recommendations ──
                    st.markdown('<div class="section-header">💳 <span class="accent">Loan Recommendations</span></div>',
                               unsafe_allow_html=True)
                    upload_score = final["final_trust_score"]
                    user_inc = float(profile["mean_income"])
                    fixed_exp = float(profile.get("fixed_expenses", 0))
                    existing_emi_amt = 0
                    if parser.parsed_df is not None:
                        emi_txns = parser.parsed_df[
                            (parser.parsed_df["category"] == "EMI") & (parser.parsed_df["type"] == "debit")]
                        if len(emi_txns) > 0:
                            months_with_emi = emi_txns["date"].dt.to_period("M").nunique()
                            if months_with_emi > 0:
                                existing_emi_amt = float(emi_txns["amount"].sum() / months_with_emi)

                    loan_recs = get_transaction_loan_recommendations(
                        score=upload_score, monthly_income=user_inc,
                        monthly_expenses=fixed_exp, existing_emi=existing_emi_amt)

                    # Tier badge
                    tier = loan_recs["tier"]
                    pre_status = loan_recs["pre_approval_status"]
                    tc = tier["color"]
                    st.markdown(
                        f'<div class="tier-badge" style="background:{tc}15; border:1px solid {tc}40;">'
                        f'<span class="tier-text" style="color:{tc};">{pre_status}</span>'
                        f' &nbsp;·&nbsp; Score: {upload_score:.0f}'
                        f' &nbsp;·&nbsp; Max {tier["max_simultaneous_loans"]} loans'
                        f' &nbsp;·&nbsp; Exposure up to ₹{loan_recs["max_total_exposure"]:,.0f}</div>',
                        unsafe_allow_html=True)

                    # Repayment capacity
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
                                        f'<div style="color:#475569; font-size:0.85rem;">'
                                        f'{tl["effective_rate"]}% · {tl["suggested_tenure"]} months</div>'
                                        f'<div style="color:#64748b; font-size:0.8rem; margin-top:4px;">'
                                        f'EMI: ₹{tl["emi"]:,.0f}/month</div>'
                                        f'{"<div style=\'color:#22c55e; font-size:0.75rem;\'>" + tl["subsidy"][:60] + "...</div>" if tl.get("subsidy") else ""}'
                                        f'</div>', unsafe_allow_html=True)

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
                                    f"Total Interest: ₹{loan['total_interest']:,.0f}")
                                if loan.get("subsidy"):
                                    st.success(f"💰 Subsidy: {loan['subsidy']}")
                                if loan.get("interest_saved_via_subsidy", 0) > 0:
                                    st.info(f"💵 Interest saved via subsidy: ₹{loan['interest_saved_via_subsidy']:,.0f}")
                                st.markdown(f"📄 **Documents:** {', '.join(loan['documents'])}")
                                st.markdown(f"🏦 **Lenders:** {', '.join(loan['lenders'])}")
                                st.markdown("---")

                        with st.expander("🧮 EMI Calculator"):
                            emi_c1, emi_c2, emi_c3 = st.columns(3)
                            with emi_c1:
                                emi_amount = st.number_input("Loan Amount (₹)", min_value=1000,
                                    max_value=10000000, value=100000, step=10000, key="emi_calc_amt")
                            with emi_c2:
                                emi_rate = st.number_input("Interest Rate (%)", min_value=1.0,
                                    max_value=40.0, value=12.0, step=0.5, key="emi_calc_rate")
                            with emi_c3:
                                emi_tenure = st.number_input("Tenure (months)", min_value=1,
                                    max_value=360, value=24, step=6, key="emi_calc_tenure")
                            calc_emi = calculate_emi(emi_amount, emi_rate, emi_tenure)
                            total_payable = calc_emi * emi_tenure
                            total_int = total_payable - emi_amount
                            ec1, ec2, ec3 = st.columns(3)
                            ec1.metric("Monthly EMI", f"₹{calc_emi:,.0f}")
                            ec2.metric("Total Interest", f"₹{total_int:,.0f}")
                            ec3.metric("Total Payable", f"₹{total_payable:,.0f}")
                            schedule = generate_repayment_schedule(emi_amount, emi_rate, emi_tenure)
                            if schedule:
                                sched_df = pd.DataFrame(schedule[:12])
                                sched_df["emi"] = sched_df["emi"].apply(lambda x: f"₹{x:,.0f}")
                                sched_df["principal"] = sched_df["principal"].apply(lambda x: f"₹{x:,.0f}")
                                sched_df["interest"] = sched_df["interest"].apply(lambda x: f"₹{x:,.0f}")
                                sched_df["balance"] = sched_df["balance"].apply(lambda x: f"₹{x:,.0f}")
                                sched_df.columns = ["Month", "EMI", "Principal", "Interest", "Balance"]
                                st.markdown("**Repayment Schedule (first 12 months):**")
                                st.dataframe(sched_df, use_container_width=True, hide_index=True)
                    else:
                        st.error(
                            f"❌ **Not Eligible for Loans Currently** — Score: {upload_score:.0f}\n\n"
                            f"Build payment history for 3-6 months to qualify.")

                    # Credit Improvement Path
                    if loan_recs.get("improvement_path"):
                        st.markdown('<div class="section-header">📈 Credit Improvement Path</div>', unsafe_allow_html=True)
                        for imp_item in loan_recs["improvement_path"]:
                            if imp_item["type"] == "score_upgrade":
                                st.markdown(f"🎯 **{imp_item['title']}** (+{imp_item['gap']:.0f} points needed)")
                                st.caption(imp_item.get("benefit", ""))
                                for action in imp_item.get("actions", []):
                                    st.markdown(f"  - {action}")
                            elif imp_item["type"] == "maintenance":
                                st.success(f"✅ {imp_item['title']}")
                                for action in imp_item.get("actions", []):
                                    st.markdown(f"  - {action}")

                    # Financial Tips
                    fin_tips = get_financial_tips(score=upload_score, eligible_loans=loan_recs.get("eligible_loans", []))
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
                    "- CSV with columns: Date, Description, Debit, Credit, Balance\n"
                    "- Excel (.xlsx) with the same columns\n\n"
                    "**Tip:** Download the sample statement above to see the expected format.")

    # ════════════════════════════════════════════════════════════════════
    # PAGE: ALTERNATIVE SCORE
    # ════════════════════════════════════════════════════════════════════
    elif page == "🌍 Alternative Score":
        st.markdown("""
        <div class="section-header">🌍 <span class="accent">Alternative Credit Score</span></div>
        <div style="color:#64748b; font-size:0.95rem; margin-bottom:8px; line-height:1.6;">
            No bank account? No problem. <b style="color:#8B7355;">CrediVist scores ANYONE</b> using
            persona-specific alternative data — farmers, students, vendors, homemakers, and more.
        </div>
        <div style="display:flex;gap:12px;margin-bottom:12px;flex-wrap:wrap;">
            <span style="display:inline-flex;align-items:center;gap:4px;font-size:0.75rem;color:#64748b;padding:4px 12px;background:#f8fafc;border-radius:50px;border:1px solid #e2e8f0;">🌾 Farmer</span>
            <span style="display:inline-flex;align-items:center;gap:4px;font-size:0.75rem;color:#64748b;padding:4px 12px;background:#f8fafc;border-radius:50px;border:1px solid #e2e8f0;">🎓 Student</span>
            <span style="display:inline-flex;align-items:center;gap:4px;font-size:0.75rem;color:#64748b;padding:4px 12px;background:#f8fafc;border-radius:50px;border:1px solid #e2e8f0;">🏪 Vendor</span>
            <span style="display:inline-flex;align-items:center;gap:4px;font-size:0.75rem;color:#64748b;padding:4px 12px;background:#f8fafc;border-radius:50px;border:1px solid #e2e8f0;">🏠 Homemaker</span>
            <span style="display:inline-flex;align-items:center;gap:4px;font-size:0.75rem;color:#64748b;padding:4px 12px;background:#f8fafc;border-radius:50px;border:1px solid #e2e8f0;">👤 General</span>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

        # Persona selection cards
        st.markdown('<div class="section-header">👤 Select Your Profile</div>', unsafe_allow_html=True)
        persona_cols = st.columns(len(PERSONAS))
        persona_keys = list(PERSONAS.keys())
        for i, (key, info) in enumerate(PERSONAS.items()):
            with persona_cols[i]:
                st.markdown(
                    f'<div class="persona-card">'
                    f'<div class="persona-icon">{info["label"].split()[0]}</div>'
                    f'<div class="persona-name">{info["label"][2:].strip()}</div>'
                    f'<div class="persona-desc">{info["description"]}</div>'
                    f'</div>', unsafe_allow_html=True)

        selected_persona = st.selectbox(
            "Choose your persona", options=persona_keys,
            format_func=lambda k: PERSONAS[k]["label"], key="alt_persona")

        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        persona_config = PERSONAS[selected_persona]

        # Two input modes
        input_tab1, input_tab2 = st.tabs(["📄 Auto Upload (Recommended)", "✏️ Manual Form"])
        alt_result = None

        # ── Tab 1: Auto Upload ──
        with input_tab1:
            st.markdown(f'<div class="section-header">📄 Upload Documents — {persona_config["label"]}</div>',
                       unsafe_allow_html=True)
            st.markdown("""
            <div style="color:#475569; font-size:0.88rem; margin-bottom:12px;">
                Upload your documents (PDF, CSV, Excel, or TXT) and CrediVist will
                <b style="color:#8B7355;">automatically extract</b> all relevant data.
            </div>
            """, unsafe_allow_html=True)

            doc_guidance = {
                "farmer": ["Land record (RTC / Patta / Khata PDF)", "PM-KISAN beneficiary statement",
                           "Mandi sale receipts", "KCC statement or crop insurance documents", "Utility bills"],
                "student": ["Marksheet / Transcript / Grade Card (PDF)", "Scholarship award letters",
                           "Course certificates (NPTEL, Coursera, etc.)", "Internship / Part-time work proof"],
                "street_vendor": ["Daily sales register (CSV / Excel)", "Rent / stall fee receipts",
                                  "Trade / vendor license copy", "Utility bills"],
                "homemaker": ["Household expense diary (CSV / Excel)", "SHG passbook / contribution record",
                             "Micro-enterprise receipts (if any)", "Utility bills, skill certificates"],
                "general_no_bank": ["Aadhaar / PAN / Voter ID / Ration Card scans", "Mobile recharge history",
                                    "Utility bills (electricity, water, gas)", "Rent receipts (if applicable)"],
            }
            guidance = doc_guidance.get(selected_persona, [])
            if guidance:
                st.markdown("**Recommended documents:**")
                for doc in guidance:
                    st.markdown(f"- {doc}")

            st.markdown("")

            try:
                from src.ocr_engine import get_ocr_capabilities
                caps = get_ocr_capabilities()
                cap_items = []
                if caps.get("tesseract"):
                    cap_items.append("✅ Tesseract OCR")
                else:
                    cap_items.append("❌ Tesseract OCR (install for scanned docs)")
                if caps.get("pil"):
                    cap_items.append("✅ Image Processing")
                st.caption("OCR: " + " · ".join(cap_items))
            except ImportError:
                pass

            uploaded_files = st.file_uploader(
                "Upload your documents",
                type=["pdf", "csv", "xlsx", "xls", "txt", "json",
                      "jpg", "jpeg", "png", "bmp", "tiff", "tif", "webp"],
                accept_multiple_files=True, key=f"alt_upload_{selected_persona}",
                help="Upload documents or scanned images. Supports PDF, images, CSV, Excel, TXT.")

            auto_detect = st.checkbox("🔮 Auto-detect persona from documents", value=False, key="auto_detect_persona")

            if uploaded_files:
                if st.button("🔍 Analyze & Score", type="primary", use_container_width=True, key="btn_analyze"):
                    files = []
                    for uf in uploaded_files:
                        files.append((uf.name, uf.read()))
                        uf.seek(0)
                    persona_to_use = None if auto_detect else selected_persona
                    with st.spinner("Analyzing documents and extracting data..."):
                        analysis = analyze_documents(files, persona=persona_to_use)

                    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
                    st.markdown('<div class="section-header">📑 Document Analysis Summary</div>', unsafe_allow_html=True)

                    as1, as2, as3, as4 = st.columns(4)
                    as1.metric("Files Processed", analysis["files_processed"])
                    as2.metric("Text Extracted", f"{analysis['total_text_length']:,} chars")
                    detected_label = PERSONAS.get(analysis["detected_persona"], {}).get("label", analysis["detected_persona"])
                    as3.metric("Detected Persona", detected_label)
                    as4.metric("OCR Used", "Yes" if analysis.get("ocr_used") else "No")

                    doc_types = analysis.get("detected_document_types", [])
                    if doc_types:
                        type_labels = [t.replace('_', ' ').title() for t in doc_types]
                        st.info(f"📄 Documents identified: {', '.join(type_labels)}")

                    if analysis["document_summaries"]:
                        file_summary_data = []
                        for ds in analysis["document_summaries"]:
                            file_summary_data.append({
                                "File": ds["filename"],
                                "Doc Type": ds.get("document_type", "unknown").replace('_', ' ').title(),
                                "Text Length": f"{ds['text_length']:,}",
                                "OCR": "✓" if ds.get("ocr_used") else "✗",
                                "Has Table": "✓" if ds["has_table"] else "✗",
                                "Rows": ds["rows"], "Amounts Found": ds["amounts_found"],
                                "Dates Found": ds["dates_found"],
                            })
                        st.dataframe(pd.DataFrame(file_summary_data), use_container_width=True, hide_index=True)

                    for w in analysis.get("warnings", []):
                        st.warning(f"⚠️ {w}")

                    with st.expander("🔎 View Extracted Data", expanded=False):
                        extracted = analysis["extracted_data"]
                        ext_items = []
                        for k, v in sorted(extracted.items()):
                            ext_items.append({"Field": k.replace("_", " ").title(), "Value": str(v)})
                        st.dataframe(pd.DataFrame(ext_items), use_container_width=True, hide_index=True)

                    final_persona = analysis["detected_persona"]
                    with st.spinner("Computing credit score..."):
                        alt_result = compute_persona_score(final_persona, analysis["extracted_data"])
                    st.session_state["alt_score_result"] = alt_result
                    st.session_state["alt_score_persona_config"] = PERSONAS[final_persona]

            if "alt_score_result" in st.session_state and alt_result is None:
                alt_result = st.session_state.get("alt_score_result")
                persona_config = st.session_state.get("alt_score_persona_config", persona_config)

        # ── Tab 2: Manual Form ──
        with input_tab2:
            st.markdown(f'<div class="section-header">✏️ Manual Entry — {persona_config["label"]}</div>',
                       unsafe_allow_html=True)
            st.markdown(
                f'<div style="color:#475569; font-size:0.88rem; margin-bottom:12px;>'
                f'<em>{persona_config["description"]}</em> — Fill in the details below. More data = higher confidence.'
                f'</div>', unsafe_allow_html=True)

            form_fields = get_persona_form_fields(selected_persona)
            form_data = {}

            with st.form(key="alt_score_form"):
                for group in form_fields:
                    criterion = group["criterion"]
                    fields = group["fields"]
                    from src.alternative_profiles import CRITERIA_REGISTRY
                    scorer_fn = CRITERIA_REGISTRY.get(criterion)
                    if scorer_fn:
                        test_result = scorer_fn({})
                        section_label = test_result.get("label", criterion.replace('_', ' ').title())
                    else:
                        section_label = criterion.replace('_', ' ').title()
                    st.markdown(f"#### {section_label}")
                    field_pairs = [fields[i:i+2] for i in range(0, len(fields), 2)]
                    for pair in field_pairs:
                        cols = st.columns(len(pair))
                        for j, field in enumerate(pair):
                            with cols[j]:
                                fkey = f"alt_{criterion}_{field['key']}"
                                if field["type"] == "boolean":
                                    form_data[field["key"]] = st.checkbox(field["label"], value=field.get("default", False), key=fkey)
                                elif field["type"] == "number":
                                    default = field.get("default", 0)
                                    min_val = field.get("min", 0)
                                    max_val = field.get("max", 100)
                                    if isinstance(default, float) and default != int(default):
                                        form_data[field["key"]] = st.number_input(field["label"], min_value=float(min_val), max_value=float(max_val), value=float(default), key=fkey)
                                    else:
                                        form_data[field["key"]] = st.number_input(field["label"], min_value=int(min_val), max_value=int(max_val), value=int(default), key=fkey)
                                elif field["type"] == "select":
                                    options = field.get("options", [])
                                    default = field.get("default", options[0] if options else "")
                                    default_idx = options.index(default) if default in options else 0
                                    form_data[field["key"]] = st.selectbox(field["label"], options=options, index=default_idx, key=fkey)
                                elif field["type"] == "text":
                                    default = field.get("default", "")
                                    val = st.text_input(field["label"], value=default, key=fkey)
                                    if "comma" in field["label"].lower():
                                        form_data[field["key"]] = [x.strip() for x in val.split(",") if x.strip()]
                                    else:
                                        form_data[field["key"]] = val
                    st.markdown("")
                submitted = st.form_submit_button("🔍 Compute Alternative Credit Score", use_container_width=True, type="primary")

            if submitted:
                with st.spinner("Computing your alternative credit score..."):
                    alt_result = compute_persona_score(selected_persona, form_data)

        # ── Shared Results Display ──
        if alt_result is not None:
            st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
            st.markdown('<div class="section-header">📊 <span class="accent">Your Alternative Credit Score</span></div>',
                       unsafe_allow_html=True)

            sc1, sc2, sc3, sc4 = st.columns(4)
            sc1.metric("Trust Score", f"{alt_result['trust_score']:.0f} / 900")
            sc2.metric("Grade", alt_result["grade"])
            sc3.metric("Confidence", f"{alt_result['confidence']:.0%}")
            sc4.metric("Data Signals", f"{alt_result['filled_count']}/{alt_result['criteria_count']}")

            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number+delta", value=alt_result["trust_score"],
                domain={"x": [0, 1], "y": [0, 1]},
                title={"text": f"{alt_result['persona_label']} — Trust Score",
                       "font": {"size": 20, "family": "Plus Jakarta Sans, Inter"}},
                gauge={
                    "axis": {"range": [300, 900], "tickwidth": 1,
                             "tickfont": {"color": "#64748b"}},
                    "bar": {"color": alt_result["grade_color"]},
                    "bgcolor": "rgba(15,23,42,0.06)",
                    "steps": [
                        {"range": [300, 400], "color": "rgba(239,68,68,0.12)"},
                        {"range": [400, 500], "color": "rgba(249,115,22,0.12)"},
                        {"range": [500, 650], "color": "rgba(234,179,8,0.12)"},
                        {"range": [650, 750], "color": "rgba(132,204,22,0.12)"},
                        {"range": [750, 900], "color": "rgba(34,197,94,0.12)"},
                    ],
                    "threshold": {"line": {"color": "#0f172a", "width": 4}, "thickness": 0.75, "value": alt_result["trust_score"]},
                },
            ))
            fig_gauge.update_layout(paper_bgcolor="rgba(0,0,0,0)", font={"color": "#0f172a", "family": "Plus Jakarta Sans, Inter"}, height=300)
            st.plotly_chart(fig_gauge, use_container_width=True)

            # Criteria breakdown
            st.markdown('<div class="section-header">📋 Criteria Breakdown</div>', unsafe_allow_html=True)
            breakdown = alt_result["criteria_breakdown"]
            display_persona = alt_result.get("persona", selected_persona)
            display_config = PERSONAS.get(display_persona, persona_config)
            weights = display_config["criteria_weights"]

            breakdown_data = []
            for criterion, info in breakdown.items():
                weight = weights.get(criterion, 0)
                breakdown_data.append({
                    "Criteria": info["label"], "Score": f"{info['score']:.0%}",
                    "Weight": f"{weight:.0%}", "Weighted": f"{info['score'] * weight:.2%}",
                    "Details": info["detail"],
                })
            st.dataframe(pd.DataFrame(breakdown_data), use_container_width=True, hide_index=True)

            # Radar chart
            criteria_labels = [info["label"] for info in breakdown.values()]
            criteria_scores = [info["score"] * 100 for info in breakdown.values()]
            fig_radar = go.Figure(data=go.Scatterpolar(
                r=criteria_scores + [criteria_scores[0]],
                theta=criteria_labels + [criteria_labels[0]],
                fill="toself", fillcolor="rgba(201, 181, 156, 0.2)",
                line={"color": "#C9B59C", "width": 2},
                marker={"size": 6, "color": "#D9CFC7"},
            ))
            fig_radar.update_layout(
                polar={
                    "radialaxis": {"visible": True, "range": [0, 100],
                                  "gridcolor": "rgba(201,181,156,0.1)",
                                  "tickfont": {"color": "#64748b"}},
                    "angularaxis": {"gridcolor": "rgba(201,181,156,0.1)",
                                   "tickfont": {"color": "#475569", "size": 10}},
                    "bgcolor": "rgba(0,0,0,0)",
                },
                paper_bgcolor="rgba(0,0,0,0)", showlegend=False, height=450,
                title={"text": "Criteria Performance Radar", "font": {"color": "#0f172a", "size": 16, "family": "Plus Jakarta Sans, Inter"}},
            )
            st.plotly_chart(fig_radar, use_container_width=True)

            # Improvement tips
            tips = get_improvement_tips(display_persona, alt_result)
            if tips:
                st.markdown('<div class="section-header">💡 How to Improve</div>', unsafe_allow_html=True)
                for tip in tips:
                    impact_colors = {"high": "🔴", "medium": "🟡", "low": "🟢"}
                    impact_icon = impact_colors.get(tip["impact"], "⚪")
                    with st.expander(f"{impact_icon} {tip['action']} (Current: {tip['current_score']:.0%})"):
                        st.markdown(tip["description"])
                        st.caption(f"Impact: {tip['impact'].upper()} · Criterion: {tip['criterion'].replace('_', ' ').title()}")
            else:
                st.success("🎉 Great job! All your criteria are above 50%.")

            # ── Loan Recommendations (Persona-Based) ──
            st.markdown('<div class="section-header">💳 <span class="accent">Loan Recommendations</span></div>',
                       unsafe_allow_html=True)
            try:
                alt_score = alt_result["trust_score"]
                alt_persona_key = alt_result.get("persona", selected_persona)
                alt_form = alt_result.get("input_data", {})
                alt_loan_recs = get_persona_loan_recommendations(
                    persona=alt_persona_key, score=alt_score, persona_data=alt_form)

                alt_tier = alt_loan_recs["tier"]
                alt_pre = alt_loan_recs["pre_approval_status"]
                atc = alt_tier["color"]
                st.markdown(
                    f'<div class="tier-badge" style="background:{atc}15; border:1px solid {atc}40;">'
                    f'<span class="tier-text" style="color:{atc};">{alt_pre}</span>'
                    f' &nbsp;·&nbsp; Score: {alt_score:.0f}'
                    f' &nbsp;·&nbsp; Max {alt_tier["max_simultaneous_loans"]} loans'
                    f' &nbsp;·&nbsp; Est. Income: ₹{alt_loan_recs.get("estimated_monthly_income", 0):,.0f}/mo</div>',
                    unsafe_allow_html=True)

                if alt_loan_recs["eligible_loans"]:
                    st.markdown(f"#### ✅ Eligible Loan Schemes ({alt_loan_recs['total_eligible']})")
                    alt_top = compare_loans(alt_loan_recs["eligible_loans"])
                    if alt_top:
                        alt_tcols = st.columns(min(len(alt_top), 3))
                        for ti, tl in enumerate(alt_top):
                            with alt_tcols[ti]:
                                st.markdown(
                                    f'<div class="metric-card">'
                                    f'<h3>{tl["icon"]} {tl["name"]}</h3>'
                                    f'<div class="val" style="color:#22c55e">₹{tl["recommended_amount"]:,.0f}</div>'
                                    f'<div style="color:#475569; font-size:0.85rem;">'
                                    f'{tl["effective_rate"]}% · {tl["suggested_tenure"]} months</div>'
                                    f'<div style="color:#64748b; font-size:0.8rem; margin-top:4px;">'
                                    f'EMI: ₹{tl["emi"]:,.0f}/month</div>'
                                    f'{"<div style=\'color:#22c55e; font-size:0.75rem;\'>💰 " + tl["subsidy"][:50] + "...</div>" if tl.get("subsidy") else ""}'
                                    f'</div>', unsafe_allow_html=True)

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
                                f"Total Interest: ₹{loan['total_interest']:,.0f}")
                            if loan.get("subsidy"):
                                st.success(f"💰 Subsidy: {loan['subsidy']}")
                            if loan.get("interest_saved_via_subsidy", 0) > 0:
                                st.info(f"💵 Interest saved: ₹{loan['interest_saved_via_subsidy']:,.0f}")
                            st.markdown(f"📄 **Documents:** {', '.join(loan['documents'])}")
                            st.markdown(f"🏦 **Lenders:** {', '.join(loan['lenders'])}")
                            if loan.get("criteria_met") or loan.get("criteria_not_met"):
                                criteria_line = ""
                                for c in loan.get("criteria_met", []):
                                    criteria_line += f"✅ {c.replace('_', ' ').title()}  "
                                for c in loan.get("criteria_not_met", []):
                                    criteria_line += f"❌ {c.replace('_', ' ').title()}  "
                                st.markdown(f"📝 **Eligibility:** {criteria_line}")
                            st.markdown("---")

                    with st.expander("🧮 EMI Calculator"):
                        aec1, aec2, aec3 = st.columns(3)
                        with aec1:
                            a_emi_amt = st.number_input("Loan Amount (₹)", min_value=1000, max_value=10000000, value=50000, step=5000, key="alt_emi_amt")
                        with aec2:
                            a_emi_rate = st.number_input("Interest Rate (%)", min_value=1.0, max_value=40.0, value=10.0, step=0.5, key="alt_emi_rate")
                        with aec3:
                            a_emi_ten = st.number_input("Tenure (months)", min_value=1, max_value=360, value=12, step=3, key="alt_emi_ten")
                        a_calc_emi = calculate_emi(a_emi_amt, a_emi_rate, a_emi_ten)
                        a_total = a_calc_emi * a_emi_ten
                        a_int = a_total - a_emi_amt
                        aec4, aec5, aec6 = st.columns(3)
                        aec4.metric("Monthly EMI", f"₹{a_calc_emi:,.0f}")
                        aec5.metric("Total Interest", f"₹{a_int:,.0f}")
                        aec6.metric("Total Payable", f"₹{a_total:,.0f}")
                else:
                    st.error(f"❌ **Not Eligible for Loans** — Score: {alt_score:.0f}\n\nBuild your alternative credit profile for 3-6 months.")

                seasonal = get_seasonal_recommendations(alt_persona_key)
                if seasonal:
                    st.markdown('<div class="section-header">🌾 Seasonal Recommendations</div>', unsafe_allow_html=True)
                    for rec in seasonal:
                        st.info(f"**{rec['season']}** — {rec['status']}\n\n🌱 Crops: {rec['crops']}\n\n{rec['advice']}")

                if alt_loan_recs.get("improvement_path"):
                    st.markdown('<div class="section-header">📈 Credit Improvement Path</div>', unsafe_allow_html=True)
                    for imp_item in alt_loan_recs["improvement_path"]:
                        if imp_item["type"] == "score_upgrade":
                            st.markdown(f"🎯 **{imp_item['title']}** (+{imp_item.get('gap', 0):.0f} points needed)")
                            st.caption(imp_item.get("benefit", ""))
                            for action in imp_item.get("actions", []):
                                st.markdown(f"  - {action}")
                        elif imp_item["type"] == "maintenance":
                            st.success(f"✅ {imp_item['title']}")
                            for action in imp_item.get("actions", []):
                                st.markdown(f"  - {action}")

                alt_fin_tips = get_financial_tips(persona=alt_persona_key, score=alt_score, eligible_loans=alt_loan_recs.get("eligible_loans", []))
                if alt_fin_tips:
                    with st.expander("📚 Financial Literacy Tips"):
                        for tip in alt_fin_tips:
                            st.markdown(f"{tip['icon']} **{tip['title']}**")
                            st.caption(tip["detail"])
                            st.markdown("")

            except Exception as e:
                st.caption(f"Loan recommendation engine: {e}")

            st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
            st.markdown("""
            <div class="info-banner">
                💡 <b>How this works:</b> Your score is computed purely from alternative data signals
                specific to your life situation — no bank account or CIBIL history needed.
                As you provide more data, confidence increases.
            </div>
            """, unsafe_allow_html=True)

    # ════════════════════════════════════════════════════════════════════
    # PAGE: FIND LOANS
    # ════════════════════════════════════════════════════════════════════
    elif page == "🔍 Find Loans":
        st.markdown("""
        <div class="section-header">🔍 <span class="accent">Find Loans & Check Eligibility</span></div>
        <div style="color:#64748b; font-size:0.95rem; margin-bottom:12px; line-height:1.6;">
            Browse <b style="color:#8B7355;">34+ loan products</b> across transaction-based and persona-specific schemes.
            Search, filter, and check if <b style="color:#8B7355;">you</b> are eligible — with instant gap analysis.
        </div>
        """, unsafe_allow_html=True)

        search_tab, eligibility_tab = st.tabs(["🗂️ Browse & Search Loans", "✅ Check My Eligibility"])

        # === TAB 1: Browse & Search ===
        with search_tab:
            st.markdown('<div class="section-header">🗂️ Loan Catalog</div>', unsafe_allow_html=True)

            fc1, fc2, fc3, fc4 = st.columns(4)
            with fc1:
                search_query = st.text_input("🔎 Search by keyword", placeholder="e.g. KCC, Mudra, education...", key="loan_search_query")
            with fc2:
                all_categories = ["All"] + get_loan_categories()
                cat_filter = st.selectbox("📂 Category", all_categories, key="loan_cat_filter")
            with fc3:
                source_options = ["All", "Transaction-based (Bank History)", "Persona-based (No Bank)"]
                source_sel = st.selectbox("📋 Loan Type", source_options, key="loan_source_filter")
            with fc4:
                persona_options = ["All", "Farmer", "Student", "Street Vendor", "Homemaker", "General (No Bank)"]
                persona_sel = st.selectbox("👤 Persona", persona_options, key="loan_persona_filter")

            with st.expander("⚙️ Advanced Filters"):
                afc1, afc2, afc3 = st.columns(3)
                with afc1:
                    collateral_opts = ["Any", "No Collateral Only", "Collateral Required"]
                    collateral_sel = st.selectbox("🔒 Collateral", collateral_opts, key="loan_collateral_filter")
                with afc2:
                    subsidy_only = st.checkbox("💰 Subsidized loans only", key="loan_subsidy_filter")
                with afc3:
                    max_interest = st.slider("📉 Max interest rate (%)", 0.0, 42.0, 0.0, 0.5, key="loan_max_rate")
                min_loan_amount = st.number_input("💵 Minimum loan amount (₹)", min_value=0, value=0, step=5000, key="loan_min_amount")

            source_map = {"All": "", "Transaction-based (Bank History)": "transaction", "Persona-based (No Bank)": "persona"}
            persona_map = {"All": "", "Farmer": "farmer", "Student": "student", "Street Vendor": "street_vendor", "Homemaker": "homemaker", "General (No Bank)": "general_no_bank"}
            collateral_map = {"Any": "", "No Collateral Only": "no", "Collateral Required": "yes"}

            filtered_loans = search_loans(
                query=search_query, category="" if cat_filter == "All" else cat_filter,
                source_filter=source_map.get(source_sel, ""),
                persona_filter=persona_map.get(persona_sel, ""),
                collateral_filter=collateral_map.get(collateral_sel, ""),
                subsidy_filter=subsidy_only, max_rate=max_interest,
                min_amount=float(min_loan_amount))

            total_catalog = len(get_all_loans_catalog())
            st.markdown(f"**Showing {len(filtered_loans)} of {total_catalog} loans**"
                       + (f' matching "{search_query}"' if search_query else ""))

            if not filtered_loans:
                st.warning("No loans match your filters. Try broadening your search.")
            else:
                for i in range(0, len(filtered_loans), 3):
                    cols = st.columns(3)
                    for j, col in enumerate(cols):
                        if i + j < len(filtered_loans):
                            loan = filtered_loans[i + j]
                            with col:
                                if loan["source"] == "transaction":
                                    badge = "🏦 Transaction-based"
                                else:
                                    persona_label = (loan.get("persona", "") or "").replace("_", " ").title()
                                    badge = f"🌍 {persona_label}"
                                rate_low, rate_high = loan["interest_range"]
                                amt_low, amt_high = loan["amount_range"]
                                tenure_low, tenure_high = loan["tenure_range"]
                                st.markdown(f"""<div class="loan-card">
<div class="loan-source">{badge}</div>
<div class="loan-name">{loan['icon']} {loan['name']}</div>
<div class="loan-desc">{loan['description']}</div>
<div>
<span class="loan-tag tag-rate">{rate_low}% – {rate_high}%</span>
<span class="loan-tag tag-amount">₹{amt_low:,} – ₹{amt_high:,}</span>
<span class="loan-tag tag-tenure">{tenure_low} – {tenure_high} mo</span>
</div>
<div class="loan-meta">{"🔓 No Collateral" if not loan.get("collateral") else "🔒 Collateral Required"}{"  •  💰 <b>Subsidized</b>" if loan.get("subsidy") else ""} &nbsp;|&nbsp; Min Score: {loan.get("min_score", "N/A")} &nbsp;|&nbsp; {loan.get("category", "N/A")}</div>
</div>""", unsafe_allow_html=True)
                                with st.expander(f"📋 Details — {loan['name']}", expanded=False):
                                    st.markdown(f"**Lenders:** {', '.join(loan.get('lenders', []))}")
                                    st.markdown(f"**Documents:** {', '.join(loan.get('documents', []))}")
                                    if loan.get("subsidy"):
                                        st.success(f"**Subsidy:** {loan['subsidy']}")
                                    if loan.get("eligibility_criteria"):
                                        st.info("**Special Criteria:** " + ", ".join(c.replace("_", " ").title() for c in loan["eligibility_criteria"]))

        # === TAB 2: Check Eligibility ===
        with eligibility_tab:
            st.markdown("""
            <div class="section-header">✅ Check Your Eligibility</div>
            <div style="color:#475569; font-size:0.88rem; margin-bottom:12px;">
                Select any loan, enter your details, and get an
                <b style="color:#8B7355;">instant eligibility verdict</b> with gap analysis.
            </div>
            """, unsafe_allow_html=True)

            st.markdown("#### Step 1: Choose a Loan")
            elig_c1, elig_c2 = st.columns(2)
            with elig_c1:
                elig_source = st.radio("Are you checking with bank history or without?",
                    ["🏦 With Bank History (Transaction-based)", "🌍 Without Bank History (Persona-based)"],
                    key="elig_source_radio")
            with elig_c2:
                if "Persona" in elig_source:
                    elig_persona_opts = {"Farmer": "farmer", "Student": "student",
                        "Street Vendor": "street_vendor", "Homemaker": "homemaker",
                        "General (No Bank Account)": "general_no_bank"}
                    elig_persona_label = st.selectbox("Select your persona", list(elig_persona_opts.keys()), key="elig_persona_sel")
                    elig_persona = elig_persona_opts[elig_persona_label]
                else:
                    elig_persona = ""
                    st.info("Transaction-based loans — no persona needed")

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
                selected_loan_name = st.selectbox("🏷️ Select Loan", list(loan_options.keys()), key="elig_loan_sel")
                selected_loan_key = loan_options[selected_loan_name]

                if elig_source_key == "transaction":
                    sel_loan = TRANSACTION_LOANS[selected_loan_key]
                else:
                    sel_loan = PERSONA_LOANS.get(elig_persona, {}).get(selected_loan_key, {})

                if sel_loan:
                    rate_l, rate_h = sel_loan["interest_range"]
                    amt_l, amt_h = sel_loan["amount_range"]
                    st.markdown(f"""
<div class="glass-card" style="border-left:4px solid #C9B59C; padding:14px 18px;">
    <b>{sel_loan.get('icon','')} {sel_loan['name']}</b> — {sel_loan['description']}<br/>
    <span style="color:#8B7355;">Rate: {rate_l}%–{rate_h}%</span> &nbsp;|&nbsp;
    <span style="color:#86efac;">Amount: ₹{amt_l:,}–₹{amt_h:,}</span> &nbsp;|&nbsp;
    Min Score: {sel_loan.get('min_score', 'N/A')}
    {f" &nbsp;|&nbsp; <b style='color:#fde047;'>Subsidy: {sel_loan['subsidy']}</b>" if sel_loan.get('subsidy') else ""}
</div>""", unsafe_allow_html=True)

                st.markdown("#### Step 2: Enter Your Details")
                det_c1, det_c2, det_c3 = st.columns(3)
                with det_c1:
                    elig_score = st.number_input("Your Trust Score (300–900)", min_value=300, max_value=900, value=550, step=10, key="elig_score_input")
                with det_c2:
                    elig_income = st.number_input("Monthly Income (₹)", min_value=0, value=15000, step=1000, key="elig_income_input")
                with det_c3:
                    elig_expenses = st.number_input("Monthly Expenses (₹)", min_value=0, value=5000, step=500, key="elig_expenses_input")

                det_c4, det_c5, det_c6 = st.columns(3)
                with det_c4:
                    elig_existing_emi = st.number_input("Existing EMI (₹/month)", min_value=0, value=0, step=500, key="elig_emi_input")
                with det_c5:
                    elig_desired_amount = st.number_input("Desired Loan Amount (₹, 0 = auto)", min_value=0, value=0, step=5000, key="elig_desired_amount")
                with det_c6:
                    elig_desired_tenure = st.number_input("Desired Tenure (months, 0 = auto)", min_value=0, value=0, step=3, key="elig_desired_tenure")

                elig_persona_data = {}
                if elig_source_key == "persona" and elig_persona:
                    criteria = sel_loan.get("eligibility_criteria", []) if sel_loan else []
                    if criteria:
                        st.markdown("**Persona-Specific Details:**")
                        pc_cols = st.columns(min(len(criteria), 3))
                        for ci, criterion in enumerate(criteria):
                            with pc_cols[ci % len(pc_cols)]:
                                label = criterion.replace("_", " ").title()
                                if criterion in ("owns_land", "has_license", "is_shg_member", "has_enterprise", "has_internship", "is_group_member", "has_warehouse_receipt"):
                                    elig_persona_data[criterion] = st.checkbox(label, value=False, key=f"elig_pd_{criterion}")
                                elif criterion in ("land_acres",):
                                    elig_persona_data[criterion] = st.number_input(f"{label} (acres)", min_value=0.0, value=2.0, step=0.5, key=f"elig_pd_{criterion}")
                                elif criterion in ("crops_per_year",):
                                    elig_persona_data[criterion] = st.number_input(label, min_value=1, value=2, step=1, key=f"elig_pd_{criterion}")
                                elif criterion in ("years_in_trade",):
                                    elig_persona_data[criterion] = st.number_input(label, min_value=0, value=2, step=1, key=f"elig_pd_{criterion}")
                                elif criterion in ("score_value",):
                                    elig_persona_data[criterion] = elig_score
                                else:
                                    elig_persona_data[criterion] = st.text_input(label, key=f"elig_pd_{criterion}")

                st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
                if st.button("🔍 Check My Eligibility", type="primary", use_container_width=True, key="elig_check_btn"):
                    result = check_loan_eligibility(
                        loan_key=selected_loan_key, source=elig_source_key, persona=elig_persona,
                        score=float(elig_score), monthly_income=float(elig_income),
                        monthly_expenses=float(elig_expenses), existing_emi=float(elig_existing_emi),
                        persona_data=elig_persona_data, desired_amount=float(elig_desired_amount),
                        desired_tenure=int(elig_desired_tenure))

                    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

                    verdict = result["verdict"]
                    verdict_config = {
                        "ELIGIBLE": ("✅ You Are Eligible!", "#22c55e", "rgba(34,197,94,0.08)", "rgba(34,197,94,0.25)"),
                        "ELIGIBLE_WITH_CAUTION": ("⚠️ Eligible with Conditions", "#eab308", "rgba(234,179,8,0.08)", "rgba(234,179,8,0.25)"),
                        "MICRO_ONLY": ("🔸 Eligible for Micro Amount Only", "#f97316", "rgba(249,115,22,0.08)", "rgba(249,115,22,0.25)"),
                        "NOT_ELIGIBLE": ("❌ Not Eligible Currently", "#ef4444", "rgba(239,68,68,0.08)", "rgba(239,68,68,0.25)"),
                        "LOAN_NOT_FOUND": ("❓ Loan Not Found", "#64748b", "rgba(100,116,139,0.08)", "rgba(100,116,139,0.25)"),
                    }
                    v_title, v_color, v_bg, v_border = verdict_config.get(verdict, ("❓ Unknown", "#64748b", "rgba(100,116,139,0.08)", "rgba(100,116,139,0.25)"))

                    st.markdown(f"""
<div class="verdict-banner" style="background:{v_bg}; border:2px solid {v_border};">
    <div class="verdict-title" style="color:{v_color};">{v_title}</div>
    <div class="verdict-sub" style="color:#475569;">
        {result['loan_icon']} <b>{result['loan_name']}</b> &nbsp;|&nbsp;
        Score: {result['score_used']:.0f} ({result['tier']}) &nbsp;|&nbsp;
        Income: ₹{elig_income:,}/mo
    </div>
</div>""", unsafe_allow_html=True)

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

                    if result["gap_analysis"]:
                        st.markdown("##### 📊 Gap Analysis")
                        gap_df = pd.DataFrame(result["gap_analysis"])
                        gap_df.columns = [c.replace("_", " ").title() for c in gap_df.columns]
                        st.dataframe(gap_df, use_container_width=True, hide_index=True)

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

                        doc_c1, doc_c2 = st.columns(2)
                        with doc_c1:
                            st.markdown("**📄 Documents Needed:**")
                            for doc in ld.get("documents_needed", []):
                                st.markdown(f"- {doc}")
                        with doc_c2:
                            st.markdown("**🏦 Available Lenders:**")
                            for lender in ld.get("lenders", []):
                                st.markdown(f"- {lender}")

                        with st.expander("📅 Repayment Schedule (first 12 months)"):
                            schedule = generate_repayment_schedule(ld["actual_amount"], ld["effective_rate"], ld["actual_tenure_months"])
                            if schedule:
                                show_months = min(12, len(schedule))
                                sched_df = pd.DataFrame(schedule[:show_months])
                                sched_df.columns = ["Month", "EMI (₹)", "Principal (₹)", "Interest (₹)", "Balance (₹)"]
                                st.dataframe(sched_df, use_container_width=True, hide_index=True)
                                fig_sched = go.Figure()
                                fig_sched.add_trace(go.Bar(x=[s["month"] for s in schedule[:show_months]], y=[s["principal"] for s in schedule[:show_months]], name="Principal", marker_color="#22c55e"))
                                fig_sched.add_trace(go.Bar(x=[s["month"] for s in schedule[:show_months]], y=[s["interest"] for s in schedule[:show_months]], name="Interest", marker_color="#ef4444"))
                                fig_sched.update_layout(
                                    barmode="stack", height=300, title="Monthly EMI Breakdown",
                                    xaxis_title="Month", yaxis_title="Amount (₹)",
                                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                    font=dict(color="#0f172a", family="Plus Jakarta Sans, Inter"),
                                    xaxis=dict(gridcolor="rgba(201,181,156,0.08)"),
                                    yaxis=dict(gridcolor="rgba(201,181,156,0.08)"),
                                )
                                st.plotly_chart(fig_sched, use_container_width=True)

                    if result["improvement_steps"]:
                        st.markdown("##### 🛤️ Next Steps")
                        for step in result["improvement_steps"]:
                            icon = "✅" if verdict == "ELIGIBLE" else "💡"
                            st.markdown(f"- {icon} {step}")

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

    # ════════════════════════════════════════════════════════════════════
    # PAGE: IMPROVE (Combined Simulator + Builder)
    # ════════════════════════════════════════════════════════════════════
    elif page == "🚀 Improve":
        st.markdown("""
        <div class="section-header">🚀 <span class="accent">Improve Your Score</span></div>
        <div style="color:#64748b; font-size:0.95rem; margin-bottom:12px; line-height:1.6;">
            Simulate score changes in real-time <b style="color:#8B7355;">and</b> get a personalized
            improvement roadmap. Two powerful tools, one clear goal.
        </div>
        """, unsafe_allow_html=True)

        improve_tab1, improve_tab2 = st.tabs(["🧪 Score Simulator", "🎯 Score Builder"])

        # === TAB 1: Score Simulator ===
        with improve_tab1:
            st.markdown("""
            <div class="section-header">🧪 Score Simulator</div>
            <div style="color:#475569; font-size:0.85rem; margin-bottom:12px;">
                Adjust parameters to see how they affect your trust score in real-time.
            </div>
            """, unsafe_allow_html=True)

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
                sim_row = pd.Series({
                    "feat_income_stability": income_stability, "feat_income_trend": income_trend,
                    "feat_cash_flow_ratio": cash_flow, "feat_income_diversity": income_diversity,
                    "feat_utility_score": utility_score, "feat_emi_score": emi_score,
                    "feat_txn_regularity": txn_regularity, "feat_expense_score": expense_ratio,
                    "feat_savings_score": savings_score, "feat_work_reliability": work_rel,
                    "feat_shock_recovery": shock_recovery, "recharge_regularity": recharge_reg,
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
                try:
                    risk_prob = model.predict_single(sim_row)
                except Exception:
                    risk_prob = 0.2
                final = compute_final_score(base_score, risk_prob, sim_row)

                st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
                r1, r2 = st.columns([1, 1])
                with r1:
                    st.plotly_chart(create_gauge(final["final_trust_score"], final["grade"], final["grade_color"]), use_container_width=True)
                with r2:
                    rm1, rm2, rm3 = st.columns(3)
                    rm1.metric("Final Score", f"{final['final_trust_score']:.0f}")
                    rm2.metric("Risk Probability", f"{risk_prob:.1%}")
                    rm3.metric("Confidence", f"{final['confidence']:.0%}")
                    st.markdown(f"**Grade:** {final['grade']}")
                    st.markdown(f"**Base Score:** {base_score:.0f}")
                    breakdown = get_score_breakdown(pd.Series(base_result))
                    for cat_name, cat_data in breakdown.items():
                        score_pct = cat_data["score"]
                        bar_color = "#22c55e" if score_pct >= 70 else "#eab308" if score_pct >= 40 else "#ef4444"
                        st.markdown(f"**{cat_name}**: {score_pct:.1f}/100 ({cat_data['weight']})")
                        st.progress(int(min(score_pct, 100)))

                st.markdown('<div class="section-header">💳 Loan Eligibility</div>', unsafe_allow_html=True)
                score_val = final["final_trust_score"]
                if score_val >= 750:
                    st.success(f"✅ **Eligible for Premium Loans** — Score: {score_val:.0f}\n\nUp to ₹{mean_income * 6:,.0f} | Interest: 10-12% | Tenure: 24 months")
                elif score_val >= 650:
                    st.info(f"✅ **Eligible for Standard Loans** — Score: {score_val:.0f}\n\nUp to ₹{mean_income * 4:,.0f} | Interest: 14-16% | Tenure: 12 months")
                elif score_val >= 500:
                    st.warning(f"⚠️ **Eligible for Micro Loans** — Score: {score_val:.0f}\n\nUp to ₹{mean_income * 2:,.0f} | Interest: 18-22% | Tenure: 6 months")
                else:
                    st.error(f"❌ **Not Eligible Currently** — Score: {score_val:.0f}\n\nRecommendation: Build payment history for 3-6 months.")

        # === TAB 2: Score Builder ===
        with improve_tab2:
            st.markdown("""
            <div class="section-header">🎯 Score Builder</div>
            <div style="color:#475569; font-size:0.85rem; margin-bottom:12px;">
                See exactly which actions will boost your score the most.
                Select a user to get a <b style="color:#8B7355;">personalized improvement plan</b>.
            </div>
            """, unsafe_allow_html=True)

            user_ids = df["user_id"].tolist()
            selected_user = st.selectbox("Select User for Improvement Plan", user_ids, index=0, key="builder_user")
            user_row = df[df["user_id"] == selected_user].iloc[0]

            current_score = float(user_row["final_trust_score"])
            current_grade = user_row["grade"]
            current_color = user_row["grade_color"]

            bc1, bc2, bc3 = st.columns(3)
            bc1.metric("Current Score", f"{current_score:.0f}")
            bc2.metric("Current Grade", current_grade)
            bc3.metric("Risk Level", f"{float(user_row['risk_probability']):.1%}")

            st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

            improvements = []
            feature_actions = {
                "feat_income_stability": {"name": "Income Stability", "action": "Maintain consistent monthly income for 3+ months", "icon": "💰", "difficulty": "Medium", "timeframe": "3-6 months"},
                "feat_cash_flow_ratio": {"name": "Cash Flow Health", "action": "Reduce fixed expenses or increase income by 10%", "icon": "📊", "difficulty": "Medium", "timeframe": "1-3 months"},
                "feat_utility_score": {"name": "Utility Bill Payments", "action": "Pay all utility bills before the due date", "icon": "⚡", "difficulty": "Easy", "timeframe": "1-2 months"},
                "feat_emi_score": {"name": "EMI-like Behavior", "action": "Set up 2-3 recurring payments (SIP, subscriptions)", "icon": "🔄", "difficulty": "Easy", "timeframe": "1 month"},
                "feat_txn_regularity": {"name": "Transaction Regularity", "action": "Use digital payments consistently every week", "icon": "📱", "difficulty": "Easy", "timeframe": "1-2 months"},
                "feat_expense_score": {"name": "Expense Discipline", "action": "Shift spending toward essentials (food, transport, bills)", "icon": "🛒", "difficulty": "Medium", "timeframe": "1-2 months"},
                "feat_savings_score": {"name": "Savings Discipline", "action": "Start a recurring monthly SIP of ₹500+", "icon": "🏦", "difficulty": "Easy", "timeframe": "1 month"},
                "feat_work_reliability": {"name": "Work Reliability", "action": "Work 22+ days/month and maintain 4.5+ platform rating", "icon": "⭐", "difficulty": "Medium", "timeframe": "2-3 months"},
                "feat_income_diversity": {"name": "Income Diversity", "action": "Add a second gig platform (e.g., Swiggy + Uber)", "icon": "🔀", "difficulty": "Medium", "timeframe": "1-2 months"},
                "feat_shock_recovery": {"name": "Shock Recovery", "action": "Build a 1-month emergency buffer; recover quickly from dips", "icon": "🛡️", "difficulty": "Hard", "timeframe": "3-6 months"},
            }

            for feat_key, info in feature_actions.items():
                if feat_key in user_row.index:
                    current_val = float(user_row[feat_key])
                    if current_val < 0.85:
                        improved_val = min(current_val + 0.20, 0.95)
                        gap = improved_val - current_val
                        if feat_key in ["feat_income_stability", "feat_cash_flow_ratio", "feat_savings_score"]:
                            weight = 0.35
                        elif feat_key in ["feat_utility_score", "feat_emi_score"]:
                            weight = 0.30
                        elif feat_key in ["feat_txn_regularity", "feat_expense_score"]:
                            weight = 0.20
                        else:
                            weight = 0.15
                        estimated_points = gap * 100 * weight * 6
                        improvements.append({
                            "feature": feat_key, "name": info["name"], "action": info["action"],
                            "icon": info["icon"], "difficulty": info["difficulty"],
                            "timeframe": info["timeframe"], "current": current_val,
                            "target": improved_val, "estimated_points": estimated_points,
                        })

            improvements.sort(key=lambda x: x["estimated_points"], reverse=True)

            if improvements:
                st.markdown('<div class="section-header">🎯 Top Actions — Maximum Impact</div>', unsafe_allow_html=True)

                for i, imp in enumerate(improvements[:3]):
                    with st.container():
                        ac1, ac2, ac3, ac4 = st.columns([0.5, 3, 1.5, 1])
                        with ac1:
                            st.markdown(f"<div style='font-size:2rem; text-align:center'>{imp['icon']}</div>", unsafe_allow_html=True)
                        with ac2:
                            st.markdown(f"**{imp['name']}**")
                            st.markdown(f"{imp['action']}")
                            st.caption(f"Difficulty: {imp['difficulty']} · Timeframe: {imp['timeframe']}")
                        with ac3:
                            st.markdown(
                                f"<div style='text-align:center'><span style='color:#64748b'>Current</span><br>"
                                f"<b>{imp['current']:.0%}</b> → <b style='color:#22c55e'>{imp['target']:.0%}</b></div>",
                                unsafe_allow_html=True)
                        with ac4:
                            st.markdown(
                                f"<div style='text-align:center'><span style='color:#64748b'>Impact</span><br>"
                                f"<b style='color:#C9B59C; font-size:1.3rem'>+{imp['estimated_points']:.0f}</b>"
                                f"<br><span style='font-size:0.7rem'>points</span></div>",
                                unsafe_allow_html=True)
                        st.progress(int(min(imp["current"] * 100, 100)), text=f"{imp['current']:.0%} → {imp['target']:.0%}")
                        st.markdown("")

                total_gain = sum(imp["estimated_points"] for imp in improvements[:3])
                projected = min(current_score + total_gain, 900)

                st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
                st.markdown('<div class="section-header">📈 Projected Score After Top 3 Actions</div>', unsafe_allow_html=True)

                pc1, pc2 = st.columns(2)
                with pc1:
                    st.plotly_chart(create_gauge(current_score, current_grade, current_color), use_container_width=True)
                    st.markdown("<div style='text-align:center; color:#475569'>Current</div>", unsafe_allow_html=True)
                with pc2:
                    proj_final = compute_final_score(projected, 0.0)
                    st.plotly_chart(create_gauge(projected, proj_final["grade"], proj_final["grade_color"]), use_container_width=True)
                    st.markdown(f"<div style='text-align:center; color:#22c55e'>Projected (+{total_gain:.0f} points)</div>", unsafe_allow_html=True)

                next_grade_thresholds = [(750, "Excellent"), (650, "Good"), (500, "Fair"), (400, "Poor")]
                for threshold, grade_name in next_grade_thresholds:
                    if current_score < threshold:
                        points_needed = threshold - current_score
                        st.info(f"📍 You need **{points_needed:.0f} more points** to reach **{grade_name}** grade ({threshold}+)")
                        break

                st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
                st.markdown('<div class="section-header">📋 All Improvement Opportunities</div>', unsafe_allow_html=True)
                all_imp_data = []
                for imp in improvements:
                    all_imp_data.append({
                        "Action": f"{imp['icon']} {imp['name']}", "What to Do": imp["action"],
                        "Current": f"{imp['current']:.0%}", "Target": f"{imp['target']:.0%}",
                        "Est. Points": f"+{imp['estimated_points']:.0f}",
                        "Difficulty": imp["difficulty"], "Timeframe": imp["timeframe"],
                    })
                st.dataframe(pd.DataFrame(all_imp_data), use_container_width=True, hide_index=True)
            else:
                st.success("🎉 Outstanding! Your scores are already excellent across all criteria. Keep up the great work!")


if __name__ == "__main__":
    main()
