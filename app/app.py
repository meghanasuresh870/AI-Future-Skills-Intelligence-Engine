import streamlit as st
import pandas as pd
import plotly.express as px
import sys
import os

# =========================================================
# PATH SETUP
# =========================================================

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(CURRENT_DIR)

sys.path.append(PROJECT_DIR)

from data_engine import load_data, filter_data

from skill_engine import (
    top_future_skills,
    declining_skills,
    reskilling_priorities,
    impact_distribution,
    role_skill_gap,
)

from ai_engine import (
    explain_skill,
    explain_role,
    answer_executive_question,
)

# =========================================================
# RAG + GEMINI
# =========================================================

from rag.rag_pipeline import ask_ai
# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="AI Future Skills Intelligence Engine",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =========================================================
# GLOBAL CSS
# =========================================================

st.markdown(
    """
    <style>

    /* =========================================
       GLOBAL APP
       ========================================= */

    .stApp {
        background:
            radial-gradient(
                circle at 10% 0%,
                rgba(37, 99, 235, 0.18),
                transparent 30%
            ),
            radial-gradient(
                circle at 90% 10%,
                rgba(124, 58, 237, 0.16),
                transparent 28%
            ),
            #070b14;

        color: #ffffff;
    }

    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1400px;
    }

    /* =========================================
       SIDEBAR
       ========================================= */

    [data-testid="stSidebar"] {
        background:
            linear-gradient(
                180deg,
                #080d18 0%,
                #0b1222 100%
            );

        border-right: 1px solid #1e3a5f;
    }

    [data-testid="stSidebar"] * {
        color: #f8fafc !important;
    }

    [data-testid="stSidebar"] label {
        color: #dbeafe !important;
        font-weight: 600;
    }

    /* =========================================
       HERO
       ========================================= */

    .hero {
        background:
            linear-gradient(
                135deg,
                #020617 0%,
                #0f1f4d 45%,
                #172554 75%,
                #1d4ed8 100%
            );

        padding: 32px 35px;
        border-radius: 20px;
        margin-bottom: 25px;

        border: 1px solid #2563eb;

        box-shadow:
            0 10px 35px rgba(0, 0, 0, 0.45),
            0 0 35px rgba(37, 99, 235, 0.16);
    }

    .hero h1 {
        color: #ffffff !important;
        font-size: 36px;
        margin: 0;
        font-weight: 800;
        letter-spacing: -0.5px;
    }

    .hero p {
        color: #bfdbfe !important;
        margin-top: 10px;
        font-size: 15px;
        line-height: 1.6;
    }

    /* =========================================
       METRIC CARDS
       ========================================= */

    [data-testid="stMetric"] {
        background:
            linear-gradient(
                145deg,
                #0d1628,
                #111c31
            );

        border: 1px solid #243b5a;
        border-radius: 16px;

        padding: 20px;

        box-shadow:
            0 8px 22px rgba(0, 0, 0, 0.35),
            inset 0 1px 0 rgba(255,255,255,0.03);

        transition: 0.2s ease;
    }

    [data-testid="stMetric"]:hover {
        border-color: #3b82f6;

        box-shadow:
            0 10px 28px rgba(0, 0, 0, 0.45),
            0 0 20px rgba(59, 130, 246, 0.12);
    }

    [data-testid="stMetricLabel"] {
        color: #94a3b8 !important;
        font-weight: 600;
    }

    [data-testid="stMetricValue"] {
        color: #ffffff !important;
        font-weight: 800;
    }

    /* =========================================
       SECTION TITLES
       ========================================= */

    .section-title {
        color: #ffffff !important;
        font-size: 25px;
        font-weight: 800;

        margin-top: 28px;
        margin-bottom: 15px;

        letter-spacing: -0.3px;
    }

    .main p,
    .main li {
        color: #cbd5e1;
    }

    .main label {
        color: #e2e8f0 !important;
    }

    /* =========================================
       TABS
       ========================================= */

    button[data-baseweb="tab"] {
        color: #94a3b8 !important;
        font-weight: 600;
        background: transparent !important;
    }

    button[data-baseweb="tab"][aria-selected="true"] {
        color: #60a5fa !important;
        font-weight: 800;
    }

    /* =========================================
       ALERTS
       ========================================= */

    [data-testid="stAlert"] {
        background: #0d1728 !important;

        border-radius: 14px;

        border: 1px solid #29476b;

        color: #e2e8f0 !important;
    }

    [data-testid="stAlert"] * {
        color: #e2e8f0 !important;
    }

    /* =========================================
       DATAFRAME
       ========================================= */

    [data-testid="stDataFrame"] {
        border-radius: 14px;
        overflow: hidden;

        border: 1px solid #29476b;

        background: #0b1220;
    }

    /* =========================================
       INPUTS
       ========================================= */

    input,
    textarea {
        color: #ffffff !important;

        background: #0d1728 !important;

        border: 1px solid #29476b !important;

        border-radius: 10px !important;
    }

    input::placeholder,
    textarea::placeholder {
        color: #64748b !important;
    }

    /* =========================================
       SELECT BOX
       ========================================= */

    div[data-baseweb="select"] > div {
        background: #0d1728 !important;

        border: 1px solid #29476b !important;

        color: #ffffff !important;

        border-radius: 10px !important;
    }

    div[data-baseweb="select"] * {
        color: #ffffff !important;
    }

    /* =========================================
       BUTTONS
       ========================================= */

    .stButton > button {
        background:
            linear-gradient(
                135deg,
                #2563eb,
                #4f46e5
            );

        color: #ffffff !important;

        border: 1px solid #60a5fa;

        border-radius: 10px;

        font-weight: 700;

        padding: 8px 20px;

        box-shadow:
            0 5px 15px rgba(37, 99, 235, 0.25);

        transition: 0.2s ease;
    }

    .stButton > button:hover {
        background:
            linear-gradient(
                135deg,
                #3b82f6,
                #6366f1
            );

        border-color: #93c5fd;

        transform: translateY(-1px);

        box-shadow:
            0 8px 20px rgba(59, 130, 246, 0.3);
    }

    /* =========================================
       EXPANDERS
       ========================================= */

    [data-testid="stExpander"] {
        background: #0b1324 !important;

        border: 1px solid #29476b !important;

        border-radius: 14px !important;
    }

    [data-testid="stExpander"] summary {
        color: #ffffff !important;
        font-weight: 700;
    }

    /* =========================================
       CHECKBOX / RADIO
       ========================================= */

    [data-testid="stCheckbox"] label,
    [data-testid="stRadio"] label {
        color: #e2e8f0 !important;
    }

    /* =========================================
       DIVIDER
       ========================================= */

    hr {
        border-color: #1e3a5f !important;
    }

    /* =========================================
       INFO CARD
       ========================================= */

    .info-card {
        background:
            linear-gradient(
                145deg,
                #0b1628,
                #10213d
            );

        border: 1px solid #1d4ed8;

        border-radius: 14px;

        padding: 18px 20px;

        margin: 10px 0 20px 0;

        color: #cbd5e1;
    }

    .info-card strong {
        color: #ffffff;
    }

    /* =========================================
       AI RESULT CARD
       ========================================= */

    .ai-result {
        background:
            linear-gradient(
                145deg,
                #0b1324,
                #101d35
            );

        border-left: 4px solid #3b82f6;

        border-top: 1px solid #29476b;
        border-right: 1px solid #29476b;
        border-bottom: 1px solid #29476b;

        border-radius: 14px;

        padding: 22px;

        margin-top: 18px;

        color: #cbd5e1;

        line-height: 1.7;
    }

    .ai-result-title {
        color: #ffffff;

        font-size: 18px;

        font-weight: 800;

        margin-bottom: 10px;
    }

    /* =========================================
       FOOTER
       ========================================= */

    .footer {
        text-align: center;

        margin-top: 50px;

        padding: 25px;

        color: #64748b !important;

        font-size: 13px;

        border-top: 1px solid #1e3a5f;
    }

    .footer span {
        color: #93c5fd !important;

        font-size: 15px;

        font-weight: 700;
    }

    </style>
    """,
    unsafe_allow_html=True,
)

# =========================================================
# LOAD DATA
# =========================================================

DEFAULT_DATA_PATH = os.path.join(
    PROJECT_DIR,
    "data",
    "skills_intelligence.csv"
)

try:
    df = load_data(DEFAULT_DATA_PATH)
except Exception as e:
    st.error(f"Unable to load dataset: {e}")
    st.stop()

# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    # Sidebar title
    st.markdown(
        """
        <div style="
            color:#ffffff;
            font-size:22px;
            font-weight:800;
            margin-bottom:22px;
        ">
            🚀 Intelligence Filters
        </div>
        """,
        unsafe_allow_html=True,
    )

    # -----------------------------------------------------
    # FILTERS
    # -----------------------------------------------------

    st.markdown(
        "<div style='color:#dbeafe;font-weight:700;margin-bottom:6px;'>Industry</div>",
        unsafe_allow_html=True,
    )

    industries = ["All"] + sorted(
        df["Industry"].dropna().unique().tolist()
    )

    selected_industry = st.selectbox(
        "Select Industry",
        industries,
        label_visibility="collapsed",
    )

    st.markdown(
        "<div style='color:#dbeafe;font-weight:700;margin-top:12px;margin-bottom:6px;'>Role</div>",
        unsafe_allow_html=True,
    )

    roles = ["All"] + sorted(
        df["Role"].dropna().unique().tolist()
    )

    selected_role = st.selectbox(
        "Select Role",
        roles,
        label_visibility="collapsed",
    )

    st.markdown(
        "<div style='color:#dbeafe;font-weight:700;margin-top:12px;margin-bottom:6px;'>AI Impact</div>",
        unsafe_allow_html=True,
    )

    impacts = ["All"] + sorted(
        df["AI_Impact"].dropna().unique().tolist()
    )

    selected_impact = st.selectbox(
        "Select AI Impact",
        impacts,
        label_visibility="collapsed",
    )

    st.markdown("---")

    # -----------------------------------------------------
    # DATASET MANAGEMENT
    # -----------------------------------------------------

    st.markdown(
        """
        <div style="
            color:#ffffff;
            font-size:17px;
            font-weight:800;
            margin-bottom:12px;
        ">
            📂 Dataset Management
        </div>
        """,
        unsafe_allow_html=True,
    )

    uploaded_file = st.file_uploader(
        "Upload Intelligence CSV",
        type=["csv"],
    )

    st.caption("CSV files up to 200MB")

    if uploaded_file is not None:

        try:

            uploaded_df = pd.read_csv(uploaded_file)

            required_columns = [
                "Industry",
                "Process",
                "Activity",
                "Role",
                "Current_Skill",
                "AI_Impact",
                "Future_Skill",
                "Impact_Score",
                "Reskilling_Level",
                "Evidence",
            ]

            missing_columns = [
                col
                for col in required_columns
                if col not in uploaded_df.columns
            ]

            if missing_columns:

                st.error(
                    "Missing columns: "
                    + ", ".join(missing_columns)
                )

            else:

                df = uploaded_df

                st.success(
                    f"Dataset loaded successfully: {len(df)} records"
                )

        except Exception as e:

            st.error(
                f"Unable to read uploaded CSV: {e}"
            )

    # -----------------------------------------------------
    # INFO BOX
    # -----------------------------------------------------

    st.markdown(
        """
        <div style="
            background:linear-gradient(
                135deg,
                #0b3b73,
                #1d4ed8
            );
            border:1px solid #2563eb;
            border-radius:12px;
            padding:16px;
            margin-top:25px;
            color:#dbeafe;
            font-size:13px;
            line-height:1.6;
        ">
            💡 Use the filters to explore AI impact,
            future skills, declining skills and
            reskilling priorities.
        </div>
        """,
        unsafe_allow_html=True,
    )


# =========================================================
# APPLY FILTERS
# =========================================================

filtered_df = filter_data(
    df,
    industry=selected_industry,
    role=selected_role,
    impact=selected_impact,
)

# =========================================================
# HERO
# =========================================================

st.markdown(
    """
<div class="hero">
    <h1>🚀 AI Future Skills Intelligence Engine</h1>
    <p>
        Discover how AI is changing jobs, skills and activities —
        and identify the capabilities needed for the future.
    </p>
</div>
""",
    unsafe_allow_html=True,
)

# =========================================================
# KPI CARDS
# =========================================================

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric(
        "Intelligence Records",
        len(filtered_df)
    )

with col2:
    st.metric(
        "Roles",
        filtered_df["Role"].nunique()
    )

with col3:
    st.metric(
        "Current Skills",
        filtered_df["Current_Skill"].nunique()
    )

with col4:
    st.metric(
        "Future Skills",
        filtered_df["Future_Skill"].nunique()
    )

with col5:
    st.metric(
        "Processes",
        filtered_df["Process"].nunique()
    )

# =========================================================
# NAVIGATION TABS
# =========================================================

tabs = st.tabs(
    [
        "📊 Executive Overview",
        "🚀 Future Skills",
        "📉 Declining Skills",
        "🎯 Reskilling",
        "👤 Role Intelligence",
        "🤖 Ask AI",
    ]
)

# =========================================================
# TAB 1 — EXECUTIVE OVERVIEW
# =========================================================

with tabs[0]:

    st.markdown(
        '<div class="section-title">AI Impact Overview</div>',
        unsafe_allow_html=True,
    )

    if filtered_df.empty:

        st.warning(
            "No intelligence data available for the selected filters."
        )

    else:

        chart_col1, chart_col2 = st.columns(2)

        # ---------------------------------------------
        # DONUT / PIE CHART
        # ---------------------------------------------

        with chart_col1:

            impact_df = impact_distribution(filtered_df)

            fig_pie = px.pie(
                impact_df,
                names="AI_Impact",
                values="Count",
                hole=0.55,
                title="AI Impact Distribution",
            )

            fig_pie.update_traces(
                textposition="inside",
                textinfo="percent",
            )

            fig_pie.update_layout(
                paper_bgcolor="#0b1220",
                plot_bgcolor="#0b1f3a",
                font=dict(color="#ffffff"),
                legend=dict(
                    font=dict(color="#ffffff")
                ),
                title_font=dict(
                    color="#ffffff",
                    size=15
                ),
            )

            st.plotly_chart(
                fig_pie,
                use_container_width=True
            )

        # ---------------------------------------------
        # IMPACT CATEGORY BAR CHART
        # ---------------------------------------------

        with chart_col2:

            impact_df = (
                filtered_df["AI_Impact"]
                .value_counts()
                .reset_index()
            )

            impact_df.columns = [
                "AI_Impact",
                "Count"
            ]

            fig_bar = px.bar(
                impact_df,
                x="AI_Impact",
                y="Count",
                text="Count",
                title="Skills by AI Impact Category",
            )

            fig_bar.update_traces(
                textposition="outside"
            )

            fig_bar.update_layout(
                paper_bgcolor="#0b1220",
                plot_bgcolor="#0b1f3a",
                font=dict(color="#ffffff"),
                title_font=dict(
                    color="#ffffff",
                    size=15
                ),
                xaxis=dict(
                    title="AI Impact",
                    color="#cbd5e1"
                ),
                yaxis=dict(
                    title="Count",
                    color="#cbd5e1"
                ),
            )

            st.plotly_chart(
                fig_bar,
                use_container_width=True
            )

    # ---------------------------------------------
    # EXECUTIVE SUMMARY — NORMAL TEXT
    # ---------------------------------------------

    st.markdown(
        '<div class="section-title">Executive Summary</div>',
        unsafe_allow_html=True,
    )

    if filtered_df.empty:

        st.write(
            "No summary is available because the selected filters "
            "returned no records."
        )

    else:

        declining_count = len(
            filtered_df[
                filtered_df["AI_Impact"] == "Declining"
            ]
        )

        high_reskill = len(
            filtered_df[
                filtered_df["Reskilling_Level"] == "High"
            ]
        )

        avg_impact = filtered_df[
            "Impact_Score"
        ].mean()

        top_skill_df = top_future_skills(
            filtered_df,
            top_n=3
        )

        if not top_skill_df.empty:
            top_skill = top_skill_df.iloc[0]["Future_Skill"]
        else:
            top_skill = "future AI capabilities"

        st.markdown(
            f"""
            AI is significantly changing the activities represented
            in the current intelligence dataset.

            The average AI impact score is
            **{avg_impact:.1f}/5**.

            The strongest future capability identified is
            **{top_skill}**.

            There are **{declining_count}** intelligence records associated
            with declining skills, while **{high_reskill}** records have
            a high reskilling requirement.

            These insights are calculated directly from the underlying
            relationship between process, activity, role, current skill,
            AI impact and future skill.
            """
        )

# =========================================================
# TAB 2 — FUTURE SKILLS
# =========================================================

with tabs[1]:

    st.markdown(
        '<div class="section-title">🚀 Future Skills Intelligence</div>',
        unsafe_allow_html=True,
    )

    if filtered_df.empty:

        st.warning("No data available.")

    else:

        future_df = top_future_skills(
            filtered_df,
            top_n=10,
        )

        # Future skills table
        st.dataframe(
            future_df,
            use_container_width=True,
            hide_index=True,
        )

        st.markdown("---")

        # Explanation heading
        st.markdown(
            '<div class="section-title">💡 Why is this skill important?</div>',
            unsafe_allow_html=True,
        )

        if not future_df.empty:

            selected_skill = st.selectbox(
                "Select a future skill",
                future_df["Future_Skill"].tolist(),
                key="future_skill_analysis",
            )

            # Generate explanation
            explanation = explain_skill(
                filtered_df,
                selected_skill,
            )

            # Explanation card
            with st.container(border=True):
                st.markdown("### 🔎 Intelligence Explanation")
                st.write(explanation)

# =========================================================
# TAB 3 — DECLINING SKILLS
# =========================================================

with tabs[2]:

    st.markdown(
        '<div class="section-title">📉 Declining Skills</div>',
        unsafe_allow_html=True,
    )

    declining_df = declining_skills(
        filtered_df
    )

    if declining_df.empty:

        st.info(
            "No declining skills identified for the selected filters."
        )

    else:

        st.markdown(
            """
            Skills in this section are associated with
            activities where automation or AI is reducing
            the need for manual execution.
            """
        )

        st.dataframe(
            declining_df,
            use_container_width=True,
            hide_index=True,
        )

        fig_declining = px.bar(
            declining_df,
            x="Current_Skill",
            y="Impact_Score",
            text="Impact_Score",
            title="Automation Impact on Declining Skills",
        )

        fig_declining.update_traces(
            textposition="outside"
        )

        fig_declining.update_layout(
            paper_bgcolor="#0b1220",
            plot_bgcolor="#0b1f3a",
            font=dict(color="#ffffff"),
            title_font=dict(
                color="#ffffff",
                size=15
            ),
            xaxis=dict(
                color="#cbd5e1"
            ),
            yaxis=dict(
                title="Impact Score",
                color="#cbd5e1"
            ),
        )

        st.plotly_chart(
            fig_declining,
            use_container_width=True
        )

# =========================================================
# TAB 4 — RESKILLING
# =========================================================

with tabs[3]:

    st.markdown(
        '<div class="section-title">🎯 Reskilling Priorities</div>',
        unsafe_allow_html=True,
    )

    reskill_df = reskilling_priorities(
        filtered_df,
        top_n=10
    )

    if reskill_df.empty:

        st.info(
            "No reskilling priorities identified."
        )

    else:

        st.markdown(
            """
            These roles receive higher reskilling priority when
            their activities have stronger AI impact and connect
            to multiple future capabilities.
            """
        )

        st.dataframe(
            reskill_df,
            use_container_width=True,
            hide_index=True,
        )

        fig_reskill = px.bar(
            reskill_df,
            x="Role",
            y="Reskilling_Score",
            text="Reskilling_Score",
            title="Role Reskilling Priority",
        )

        fig_reskill.update_traces(
            textposition="outside"
        )

        fig_reskill.update_layout(
            paper_bgcolor="#0b1220",
            plot_bgcolor="#0b1f3a",
            font=dict(color="#ffffff"),
            title_font=dict(
                color="#ffffff",
                size=15
            ),
            xaxis=dict(
                color="#cbd5e1"
            ),
            yaxis=dict(
                title="Reskilling Score",
                color="#cbd5e1"
            ),
        )

        st.plotly_chart(
            fig_reskill,
            use_container_width=True
        )

# =========================================================
# TAB 5 — ROLE INTELLIGENCE
# =========================================================

with tabs[4]:

    st.markdown(
        '<div class="section-title">👤 Role Intelligence</div>',
        unsafe_allow_html=True,
    )

    # Get available roles based on current filters
    available_roles = sorted(
        filtered_df["Role"]
        .dropna()
        .unique()
        .tolist()
    )

    if not available_roles:

        st.info(
            "No roles available for the selected filters."
        )

    else:

        # Role selection
        selected_role_analysis = st.selectbox(
            "Select a role",
            available_roles,
            key="role_analysis",
        )

        # Get role skill mapping
        role_data = role_skill_gap(
            filtered_df,
            selected_role_analysis,
        )

        # Display role intelligence table
        st.dataframe(
            role_data,
            use_container_width=True,
            hide_index=True,
        )

        # Explanation section
        st.markdown(
            '<div class="section-title">🔎 Role Explanation</div>',
            unsafe_allow_html=True,
        )

        # Generate explanation
        role_explanation = explain_role(
            filtered_df,
            selected_role_analysis,
        )

        # Display explanation card
        with st.container(border=True):
            st.write(role_explanation)

# =========================================================
# TAB 6 — ASK AI
# =========================================================

with tabs[5]:

    st.markdown(
        '<div class="section-title">🤖 Ask AI</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        Ask questions about future skills, automation impact,
        declining capabilities and reskilling priorities.
        """
    )

    question = st.text_input(
        "Ask your question",
        placeholder="Example: What are the top future skills?"
    )

    if st.button(
        "🔍 Analyze",
        key="analyze_question"
    ):

        if not question.strip():

            st.warning(
                "Please enter a question first."
            )

        elif filtered_df.empty:

            st.warning(
                "No intelligence data matches the selected filters."
            )

        else:

            with st.spinner("Analyzing intelligence..."):

                try:

                    # -----------------------------------------
                    # PREPARE PROJECT INTELLIGENCE
                    # -----------------------------------------

                    # Calculate intelligence first
                    top_skills_df = top_future_skills(
                        filtered_df,
                        top_n=5
                    )

                    declining_df = declining_skills(
                        filtered_df
                    )

                    reskilling_df = reskilling_priorities(
                        filtered_df,
                        top_n=5
                    )

                    # Build calculated intelligence context
                    intelligence_context = f"""
                    CALCULATED TOP FUTURE SKILLS:

                    {top_skills_df[
                        [
                            "Future_Skill",
                            "Impact_Score",
                            "Priority_Score",
                            "Activities",
                            "Roles"
                        ]
                    ].to_string(index=False)}


                    CALCULATED DECLINING SKILLS:

                    {declining_df[
                        [
                            "Current_Skill",
                            "Impact_Score",
                            "Activities",
                            "Roles"
                        ]
                    ].to_string(index=False)}


                    CALCULATED RESKILLING PRIORITIES:

                    {reskilling_df[
                        [
                            "Role",
                            "Average_Impact",
                            "Reskilling_Score",
                            "Future_Skills",
                            "Activities"
                        ]
                    ].to_string(index=False)}


                    DETAILED INTELLIGENCE MAPPINGS:

                    {filtered_df[
                        [
                            "Process",
                            "Activity",
                            "Role",
                            "Current_Skill",
                            "AI_Impact",
                            "Future_Skill",
                            "Impact_Score",
                            "Reskilling_Level",
                            "Evidence"
                        ]
                    ].to_string(index=False)}
                    """

                    # -----------------------------------------
                    # RAG + GEMINI + DATASET
                    # -----------------------------------------

                    result = ask_ai(
                        question,
                        intelligence_context=intelligence_context
                    )

                    answer = result["answer"]
                    sources = result["sources"]

                    # -----------------------------------------
                    # RESULT
                    # -----------------------------------------

                    st.success(
                        "Analysis completed"
                    )

                    st.markdown(
                        """
                        <div class="ai-result">
                            <div class="ai-result-title">
                                💡 Intelligence Result
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    # Display AI answer only once
                    st.markdown(
                        answer,
                        unsafe_allow_html=False,
                    )

                    # -----------------------------------------
                    # KNOWLEDGE SOURCES
                    # -----------------------------------------

                    st.markdown(
    '<div class="section-title">📚 Knowledge Sources</div>',
    unsafe_allow_html=True,
)

                    for source in sources:

                        st.caption(
                            f"📄 {source['source']} "
                            f"(relevance: {source['score']:.2f})"
                        )

                except Exception as e:

                    st.error(
                        f"Unable to generate AI response: {e}"
                    )


# =========================================================
# FOOTER
# =========================================================

st.markdown(
    """
    <div class="footer">
        <span>Developed by Meghana Kolla</span>
    </div>
    """,
    unsafe_allow_html=True,
)