"""app.py - Smart Learning Recommendation System.

An explainable, content-based recommendation system designed for
AI Project 3: AI Recommendation Logic.

Features:
- Multi-criteria preference matching (70% interest, 20% skill, 10% format)
- Deterministic ranking with alphabetical secondary tie-breaking
- Transparent mathematical explainability for every recommendation
- Demonstration catalog explorer
- Session-based user feedback capture
"""

import streamlit as st
from datetime import datetime
from typing import List, Dict, Any

from data import (
    get_all_resources,
    get_available_interests,
    get_skill_levels,
    get_available_formats,
    SAMPLE_RESOURCES,
)
from recommender import (
    recommend_resources,
    WEIGHT_INTEREST,
    WEIGHT_SKILL,
    WEIGHT_FORMAT,
    SCORE_SKILL_EXACT,
    SCORE_SKILL_ONE_AWAY,
    SCORE_SKILL_TWO_AWAY,
    SCORE_SKILL_ALL_LEVELS,
    SCORE_FORMAT_EXACT,
    SCORE_FORMAT_ANY,
    SCORE_FORMAT_DIFFERENT,
    RecommendationOutcome,
    ScoredResource,
)

# Page configuration
st.set_page_config(
    page_title="Smart Learning Recommendation System",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom styling for high-contrast, polished UI
CUSTOM_CSS = """
<style>
    /* Main container styling */
    .main-header {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        color: #ffffff;
        padding: 1.75rem 2rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        border-left: 6px solid #3b82f6;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    .main-header h1 {
        margin: 0;
        font-size: 2.1rem;
        font-weight: 700;
        letter-spacing: -0.02em;
    }
    .main-header p {
        margin: 0.4rem 0 0 0;
        color: #94a3b8;
        font-size: 1.05rem;
    }
    
    /* Recommendation card styling */
    .rec-card {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1.25rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.04);
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }
    .rec-card:hover {
        box-shadow: 0 6px 12px rgba(0,0,0,0.08);
        border-color: #cbd5e1;
    }
    
    /* Score display badges */
    .score-badge {
        display: inline-block;
        padding: 0.35rem 0.85rem;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 0.95rem;
        text-align: center;
    }
    .score-high {
        background-color: #ecfdf5;
        color: #065f46;
        border: 1px solid #a7f3d0;
    }
    .score-mid {
        background-color: #eff6ff;
        color: #1e40af;
        border: 1px solid #bfdbfe;
    }
    .score-low {
        background-color: #fffbeb;
        color: #92400e;
        border: 1px solid #fde68a;
    }
    
    /* Metadata tag pills */
    .tag-pill {
        display: inline-block;
        background-color: #f1f5f9;
        color: #334155;
        padding: 0.2rem 0.6rem;
        border-radius: 6px;
        font-size: 0.82rem;
        font-weight: 500;
        margin-right: 0.35rem;
        margin-bottom: 0.35rem;
    }
    .tag-matched {
        background-color: #dbeafe;
        color: #1d4ed8;
        border: 1px solid #93c5fd;
        font-weight: 600;
    }
    
    /* Formula callout */
    .formula-box {
        background-color: #f8fafc;
        border-left: 4px solid #3b82f6;
        padding: 0.85rem 1.15rem;
        border-radius: 6px;
        font-family: 'Consolas', monospace;
        font-size: 0.9rem;
        margin: 0.5rem 0;
        color: #0f172a;
    }
    
    /* Subtitle muted */
    .meta-subtext {
        font-size: 0.85rem;
        color: #64748b;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# Initialize session state for user feedback and controls
if "user_ratings" not in st.session_state:
    st.session_state.user_ratings = {}

if "input_interests" not in st.session_state:
    st.session_state.input_interests = ["Python", "Machine Learning"]

if "input_skill" not in st.session_state:
    st.session_state.input_skill = "Intermediate"

if "input_format" not in st.session_state:
    st.session_state.input_format = "Project"

if "input_k" not in st.session_state:
    st.session_state.input_k = 5


def reset_filters() -> None:
    """Reset preference filters back to standard defaults."""
    st.session_state.input_interests = ["Python", "Machine Learning"]
    st.session_state.input_skill = "Intermediate"
    st.session_state.input_format = "Project"
    st.session_state.input_k = 5
    st.toast("Filters reset to defaults!", icon="🔄")


# Application Header
st.markdown(
    """
    <div class="main-header">
        <h1>🎓 Smart Learning Recommendation System</h1>
        <p>AI Recommendation Logic • Transparent, Content-Based Preference Matching Engine</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Tabs: Recommendations, Algorithm & Methodology, Catalog Explorer, User Feedback Log
tab_recs, tab_algo, tab_catalog, tab_feedback = st.tabs([
    "🎯 Recommendations",
    "📐 Recommendation Logic & Formula",
    "📚 Demonstration Catalog",
    "📝 Feedback & Evaluation Log"
])

# Load resources
catalog = get_all_resources()
available_interests = get_available_interests()
available_formats = get_available_formats()
skill_levels = get_skill_levels()

# =============================================================================
# SIDEBAR: PREFERENCE CONTROLS
# =============================================================================
with st.sidebar:
    st.header("⚙️ Learner Preferences")
    st.caption("Customize your profile to compute deterministic compatibility scores.")

    # Input 1: Interest selection
    chosen_interests = st.multiselect(
        "1. Select Your Interests (Tags)",
        options=available_interests,
        key="input_interests",
        help="Select one or more topics you wish to study.",
    )

    # Input 2: Skill level
    chosen_skill = st.selectbox(
        "2. Your Current Skill Level",
        options=skill_levels,
        key="input_skill",
        help="Select your current proficiency.",
    )

    # Input 3: Learning format
    chosen_format = st.selectbox(
        "3. Preferred Learning Format",
        options=available_formats,
        key="input_format",
        help="Choose 'Any' or a specific format like Video, Article, or Project.",
    )

    # Input 4: Top-K recommendations
    chosen_top_k = st.slider(
        "4. Number of Recommendations (k)",
        min_value=1,
        max_value=10,
        key="input_k",
        help="Limit of highest-scoring resources to display (1 to 10).",
    )

    st.markdown("---")
    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        try:
            run_recommendations = st.button("🚀 Recommend", width="stretch", type="primary")
        except TypeError:
            run_recommendations = st.button("🚀 Recommend", use_container_width=True, type="primary")
    with col_btn2:
        try:
            st.button("🔄 Reset", on_click=reset_filters, width="stretch")
        except TypeError:
            st.button("🔄 Reset", on_click=reset_filters, use_container_width=True)

    st.markdown("---")
    st.caption("ℹ️ **Honest AI Disclosure**")
    st.caption(
        "This system uses an explainable multi-attribute matching algorithm. "
        "No black-box neural networks or external web APIs are used."
    )


# =============================================================================
# TAB 1: RECOMMENDATIONS DISPLAY
# =============================================================================
with tab_recs:
    # Validate user inputs
    if not chosen_interests:
        st.warning(
            "⚠️ **No interests currently selected.** Please choose at least one interest tag "
            "from the sidebar to compute relevant matches. Below are closest alternative resources "
            "based purely on your skill level and format preferences.",
            icon="⚠️",
        )

    # Execute recommendation engine
    outcome: RecommendationOutcome = recommend_resources(
        catalog=catalog,
        user_interests=chosen_interests,
        user_skill_level=chosen_skill,
        preferred_format=chosen_format,
        top_k=chosen_top_k,
    )

    # Header metrics
    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
    with metric_col1:
        st.metric("Total Catalog Items", len(catalog))
    with metric_col2:
        st.metric("Selected Interests", len(chosen_interests))
    with metric_col3:
        st.metric("Target Skill Level", chosen_skill)
    with metric_col4:
        st.metric("Preferred Format", chosen_format)

    st.markdown("---")

    # Scenario A: Direct interest matches found
    if outcome.has_interest_matches:
        st.subheader(f"✨ Top {len(outcome.recommendations)} Recommended Learning Resources")
        st.caption(
            f"Ranked deterministically by compatibility score for: "
            f"**{', '.join(chosen_interests)}** | Level: **{chosen_skill}** | Format: **{chosen_format}**"
        )

        for idx, rec in enumerate(outcome.recommendations, start=1):
            with st.container():
                # Score pill class based on threshold
                if rec.final_score >= 80.0:
                    badge_class = "score-high"
                    badge_icon = "🟢"
                elif rec.final_score >= 50.0:
                    badge_class = "score-mid"
                    badge_icon = "🔵"
                else:
                    badge_class = "score-low"
                    badge_icon = "🟡"

                # Recommendation card layout
                c_head, c_score = st.columns([4, 1])
                with c_head:
                    st.markdown(
                        f"### #{idx}. {rec.title} <span class='meta-subtext'>({rec.id})</span>",
                        unsafe_allow_html=True,
                    )
                    st.write(rec.description)
                with c_score:
                    st.markdown(
                        f"<div style='text-align: right;'>"
                        f"<span class='score-badge {badge_class}'>{badge_icon} {rec.final_score:.1f}% Match</span>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )

                # Badges row: Format, Duration, Skill level
                col_b1, col_b2, col_b3 = st.columns(3)
                with col_b1:
                    st.caption(f"📁 **Format:** {rec.format}")
                with col_b2:
                    st.caption(f"⏱️ **Duration:** {rec.duration}")
                with col_b3:
                    st.caption(f"📊 **Resource Level:** {rec.skill_level}")

                # Matched interests tag rendering
                matched_html = " ".join([
                    f"<span class='tag-pill tag-matched'>✓ {t}</span>" for t in rec.matched_interests
                ])
                other_tags = [t for t in rec.tags if t not in rec.matched_interests]
                other_html = " ".join([
                    f"<span class='tag-pill'>{t}</span>" for t in other_tags
                ])
                st.markdown(
                    f"**Tags:** {matched_html} {other_html}",
                    unsafe_allow_html=True,
                )

                # Skills covered
                st.caption(f"🎯 **Skills Covered:** {', '.join(rec.skills_covered)}")

                # Recommendation reason
                st.info(f"💡 **Recommendation Rationale:** {rec.recommendation_reason}")

                # Transparent math calculation breakdown
                with st.expander(f"🔍 Inspect Scoring Math for '{rec.title}'"):
                    st.markdown(
                        f"""
                        **Component Scores (0–100 scale):**
                        - **Interest Match (70% weight):** `{rec.interest_score:.1f} pts` 
                          ({len(rec.matched_interests)} of {len(chosen_interests)} selected interests matched)
                        - **Skill Compatibility (20% weight):** `{rec.skill_score:.1f} pts` 
                          ({rec.skill_compatibility_label})
                        - **Format Compatibility (10% weight):** `{rec.format_score:.1f} pts` 
                          ({rec.format_compatibility_label})
                        
                        **Deterministic Formula Calculation:**
                        """
                    )
                    st.markdown(
                        f"<div class='formula-box'>Final Score = (0.70 × {rec.interest_score:.1f}) + "
                        f"(0.20 × {rec.skill_score:.1f}) + (0.10 × {rec.format_score:.1f}) = {rec.final_score:.1f}%</div>",
                        unsafe_allow_html=True,
                    )

                # Interactive user feedback mechanism
                with st.expander(f"⭐ Rate & Provide Feedback for '{rec.title}'"):
                    f_col1, f_col2, f_col3 = st.columns([1, 3, 1])
                    with f_col1:
                        user_rating = st.selectbox(
                            "Rating (1-5)",
                            options=[5, 4, 3, 2, 1],
                            key=f"rating_{rec.id}",
                        )
                    with f_col2:
                        user_comment = st.text_input(
                            "Comments (Optional)",
                            placeholder="e.g. Exactly what I was looking for!",
                            key=f"comment_{rec.id}",
                        )
                    with f_col3:
                        st.write("")
                        st.write("")
                        if st.button("Submit Feedback", key=f"btn_{rec.id}"):
                            st.session_state.user_ratings[rec.id] = {
                                "resource_id": rec.id,
                                "resource_title": rec.title,
                                "rating": user_rating,
                                "comment": user_comment.strip() or "No comment provided",
                                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                "match_score": rec.final_score,
                            }
                            st.success(f"Feedback recorded for {rec.id}!")

                st.markdown("<hr style='margin: 1.5rem 0; border: none; border-top: 1px solid #e2e8f0;'>", unsafe_allow_html=True)

    # Scenario B: No direct interest matches found
    else:
        st.error(
            "❌ **No Direct Interest Matches Found**\n\n"
            "None of the resources in the catalog match your specified interests. "
            "To maintain honesty, the algorithm does not fabricate false matches (Interest Score = 0.0%).",
            icon="ℹ️",
        )
        st.subheader("Closest Alternative Resources (Matched by Skill Level & Format)")
        st.caption(
            "These alternatives are ranked purely on how well they fit your skill level and format preferences."
        )

        for idx, alt in enumerate(outcome.fallback_alternatives, start=1):
            with st.container():
                st.markdown(
                    f"#### Alternative #{idx}. {alt.title} <span class='meta-subtext'>({alt.id})</span> "
                    f"<span class='score-badge score-low'>0.0% Interest Match • Fallback</span>",
                    unsafe_allow_html=True,
                )
                st.write(alt.description)
                st.warning(
                    f"⚠️ **Notice:** This resource does **not** match your selected interest tags "
                    f"(Interest Score: 0.0%). It is presented as the closest pedagogical alternative based solely "
                    f"on: **{alt.skill_compatibility_label}** and **{alt.format_compatibility_label}**.",
                    icon="ℹ️",
                )
                col_a1, col_a2, col_a3 = st.columns(3)
                with col_a1:
                    st.caption(f"📁 **Format:** {alt.format}")
                with col_a2:
                    st.caption(f"📊 **Resource Level:** {alt.skill_level}")
                with col_a3:
                    st.caption(f"🎯 **Skill Score:** {alt.skill_score:.1f} pts | **Format Score:** {alt.format_score:.1f} pts")
                st.caption(f"🏷️ **Catalog Tags:** {', '.join(alt.tags)}")
                st.markdown("<hr style='margin: 1rem 0; border-top: 1px dashed #cbd5e1;'>", unsafe_allow_html=True)


# =============================================================================
# TAB 2: ALGORITHM & METHODOLOGY
# =============================================================================
with tab_algo:
    st.subheader("📐 Recommendation Methodology & Mathematical Formulation")
    st.markdown(
        """
        The **Smart Learning Recommendation System** is built on an explainable, deterministic,
        content-based recommendation model. Unlike opaque neural network models that require large
        unlabeled training corpuses, this system evaluates individual items transparently against
        a multi-criteria objective function.
        """
    )

    st.markdown("### 1. Weighted Objective Function")
    st.markdown(
        """
        Every catalog resource $R$ is evaluated with respect to the active user profile $U$:
        
        $$\\text{Final Score}(R, U) = w_{\\text{interest}} \\cdot S_{\\text{interest}} + w_{\\text{skill}} \\cdot S_{\\text{skill}} + w_{\\text{format}} \\cdot S_{\\text{format}}$$
        
        Where the weights are strictly defined as:
        - **$w_{\\text{interest}} = 0.70$ (70%)** — Dominant relevance factor.
        - **$w_{\\text{skill}} = 0.20$ (20%)** — Pedagogical difficulty suitability.
        - **$w_{\\text{format}} = 0.10$ (10%)** — Learning style delivery preference.
        """
    )

    st.markdown("### 2. Component Scoring Rubrics")
    col_sc1, col_sc2 = st.columns(2)

    with col_sc1:
        st.markdown("#### A. Interest Match Score ($S_{\\text{interest}}$)")
        st.markdown(
            """
            Given user selected unique interests $I_U$ and resource tags $T_R$:
            
            $$S_{\\text{interest}} = 100 \\times \\frac{|I_U \\cap T_R|}{|I_U|}$$
            
            - If $|I_U \\cap T_R| = 0$, $S_{\\text{interest}} = 0.0$.
            - Duplicate selections do not inflate the score.
            - Range: $[0.0, 100.0]$.
            """
        )

        st.markdown("#### B. Format Match Score ($S_{\\text{format}}$)")
        st.markdown(
            """
            - User selected **'Any'**: **100.0 points** for all formats.
            - **Exact format match** (e.g., Video == Video): **100.0 points**.
            - **Different format** (e.g., Video vs Article): **40.0 points**.
            - Range: $[0.0, 100.0]$.
            """
        )

    with col_sc2:
        st.markdown("#### C. Skill Level Compatibility Score ($S_{\\text{skill}}$)")
        st.markdown(
            """
            Skill levels are ordered on a discrete hierarchy:
            $$\\text{Beginner (0)} \\longleftrightarrow \\text{Intermediate (1)} \\longleftrightarrow \\text{Advanced (2)}$$
            
            - **Exact match** (distance = 0): **100.0 points**
            - **One step apart** (distance = 1): **60.0 points**
            - **Two steps apart** (distance = 2): **25.0 points**
            - Resource designated **'All Levels'**: **85.0 points**
            - Range: $[0.0, 100.0]$.
            """
        )

    st.markdown("### 3. Deterministic Ranking & Tie-Breaking")
    st.markdown(
        """
        Recommendations are ranked by tuple key:
        `(-final_score, resource_title.lower())`
        
        This guarantees:
        1. Higher scores always precede lower scores.
        2. Equal-scoring resources are sorted strictly alphabetically by title, preventing random or non-deterministic ordering.
        """
    )


# =============================================================================
# TAB 3: CATALOG EXPLORER
# =============================================================================
with tab_catalog:
    st.subheader("📚 Curated Demonstration Catalog")
    st.caption(
        f"Browse the complete local catalog of {len(catalog)} learning resources across 12 computer science and AI domains."
    )

    search_query = st.text_input("🔍 Search catalog by title or keyword", placeholder="e.g. Python, Network, Algorithm")
    filter_tag = st.selectbox("Filter by Category Tag", options=["All Categories"] + available_interests)

    filtered_items = catalog
    if filter_tag != "All Categories":
        filtered_items = [item for item in filtered_items if filter_tag in item["tags"]]
    if search_query:
        sq = search_query.lower()
        filtered_items = [
            item for item in filtered_items
            if sq in item["title"].lower() or sq in item["description"].lower() or any(sq in t.lower() for t in item["tags"])
        ]

    st.write(f"Showing **{len(filtered_items)}** matching resources:")

    # Table view
    table_data = []
    for item in filtered_items:
        table_data.append({
            "ID": item["id"],
            "Title": item["title"],
            "Skill Level": item["skill_level"],
            "Format": item["format"],
            "Duration": item["duration"],
            "Tags": ", ".join(item["tags"]),
        })
    try:
        st.dataframe(table_data, width="stretch", hide_index=True)
    except TypeError:
        st.dataframe(table_data, use_container_width=True, hide_index=True)


# =============================================================================
# TAB 4: FEEDBACK & EVALUATION LOG
# =============================================================================
with tab_feedback:
    st.subheader("📝 Session User Feedback Log")
    st.caption(
        "Demonstration feature: User ratings and comments submitted during this session are stored "
        "in Streamlit session state for live presentation."
    )

    ratings_dict = st.session_state.user_ratings

    if ratings_dict:
        st.success(f"Recorded **{len(ratings_dict)}** resource ratings during this session.")
        feedback_list = list(ratings_dict.values())
        st.table(feedback_list)

        avg_rating = sum(r["rating"] for r in feedback_list) / len(feedback_list)
        st.metric("Average Feedback Rating", f"{avg_rating:.2f} / 5.0 ⭐")
    else:
        st.info("No feedback submitted yet. Rate any recommended resource in the Recommendations tab to see it logged here.")

    st.markdown("---")
    st.caption(
        "⚠️ **Academic Integrity Notice:** In accordance with Project 3 specifications, "
        "feedback is logged locally for demonstration. We do not make false claims that these ratings "
        "permanently retrain or re-weight the offline deterministic algorithm."
    )
