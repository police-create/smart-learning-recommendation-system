# Smart Learning Recommendation System
> **AI Project 3 — AI Recommendation Logic**  
> *DecodeLabs Artificial Intelligence Specification*

An explainable, deterministic, content-based recommendation system that suggests high-relevance learning resources based on a learner's topic interests, current skill level, and format preferences. Built with pure Python algorithmic logic and an interactive Streamlit user interface.

---

## 1. Problem Statement

With the proliferation of online educational material, students and self-learners often face information overload. Generic search engines lack contextual alignment with an individual learner's prerequisite proficiency level and preferred medium of instruction. Furthermore, many contemporary "AI" systems utilize opaque black-box models or pretend to train complex neural networks when a mathematically transparent, explainable preference-matching model is more appropriate, deterministic, and educational.

The **Smart Learning Recommendation System** addresses this problem by implementing a transparent, multi-criteria recommendation engine that scores and ranks resources strictly according to an explainable formula.

---

## 2. Project Objectives

1. **User Preference Modeling**: Ingest learner choices across multiple technical interest tags, proficiency level, and format preference.
2. **Deterministic Recommendation Logic**: Score resources using a weighted multi-criteria similarity algorithm without random variation or external dependencies.
3. **Complete Explainability**: Provide mathematical transparency for every recommendation, including exact point breakdowns and natural language rationale.
4. **Resilient Edge Case Handling**: Gracefully handle duplicate inputs, no-match scenarios, and tie-breaking without crashes or fabricated matches.
5. **Interactive Demonstration**: Offer an interface built with Streamlit featuring real-time filtering, catalog exploration, and session feedback logging.

---

## 3. Main Features

- **Multi-Attribute Weighted Scoring**:
  - **70%** Topic Interest Match
  - **20%** Skill-Level Compatibility
  - **10%** Learning-Format Alignment
- **Deterministic Ranking**: Highest scores ranked first; alphabetical title sorting breaks ties deterministically.
- **Deep Explainability Accordion**: For every recommended resource, inspect the exact mathematical formula and step-by-step arithmetic.
- **Honest Fallback Handling**: If no resources match the user's selected interests, the engine displays an explicit notice (0.0% interest score) and surfaces closest alternatives based on skill level and format compatibility.
- **Catalog Explorer**: Browse, search, and filter the complete 40-resource demonstration dataset.
- **Session-Based Feedback**: Learners can rate recommendations (1–5 stars) and submit comments stored in Streamlit session state.
- **Zero Cloud / API Dependencies**: Requires no API keys, paid credits, or internet connection after installing dependencies.

---

## 4. Technology Stack

- **Language**: Python 3.10+ (Verified on Python 3.14)
- **Web Interface**: Streamlit (v1.30+)
- **Recommendation Core**: Pure Python (Standard Library: `dataclasses`, `typing`, `copy`, `set`, `math`)
- **Testing Framework**: Pytest (v8.0+)
- **Data Structures**: Typed Python dictionaries and frozen dataclasses

---

## 5. Local Demonstration Dataset Structure

The catalog in `data.py` comprises **exactly 40 curated learning resources** spanning all 12 core computer science and AI disciplines:
- Python
- Artificial Intelligence
- Machine Learning
- Data Science
- Cybersecurity
- Web Development
- Cloud Computing
- DevOps
- Databases
- Computer Networks
- Data Structures and Algorithms
- Mathematics

Each resource entry includes the following structured schema:

| Field | Type | Description | Example |
|---|---|---|---|
| `id` | `str` | Unique resource identifier | `"RES-004"` |
| `title` | `str` | Name of the learning resource | `"Hands-On Machine Learning with Scikit-Learn"` |
| `description` | `str` | Pedagogical summary | `"Practical guide to supervised and unsupervised learning..."` |
| `tags` | `List[str]` | Core domain categories | `["Machine Learning", "Python", "Data Science"]` |
| `skill_level` | `str` | Difficulty tier | `"Beginner"`, `"Intermediate"`, `"Advanced"`, or `"All Levels"` |
| `format` | `str` | Instructional medium | `"Video"`, `"Article"`, or `"Project"` |
| `duration` | `str` | Estimated time to complete | `"14 hours"` |
| `skills_covered` | `List[str]` | Granular topics and concepts | `["Cross-Validation", "Feature Engineering", ...]` |

---

## 6. Recommendation Methodology & Exact Scoring Formula

The recommendation engine evaluates candidate resources $R$ against user preferences $U$ using the following weighted objective function:

$$\text{Final Score}(R, U) = (0.70 \times S_{\text{interest}}) + (0.20 \times S_{\text{skill}}) + (0.10 \times S_{\text{format}})$$

All component scores $S_i$ are constrained to $[0.0, 100.0]$, ensuring the final score remains strictly in $[0.0, 100.0]$. The final score is rounded to **1 decimal place**.

### A. Interest Match Score ($S_{\text{interest}}$)
Measures the coverage of the user's selected interests against the resource tags using deduplicated set overlap:

$$S_{\text{interest}} = 100.0 \times \frac{|I_U \cap T_R|}{|I_U|}$$

- If $|I_U \cap T_R| = 0$, $S_{\text{interest}} = 0.0$.
- Duplicate user interests do not artificially inflate the score.

### B. Skill Level Compatibility Score ($S_{\text{skill}}$)
Mapped along a discrete educational progression: $\text{Beginner (0)} \leftrightarrow \text{Intermediate (1)} \leftrightarrow \text{Advanced (2)}$.

| Relationship | Points Awarded | Description |
|---|---|---|
| **Resource marked 'All Levels'** | **85.0 pts** | Accessible to all learners |
| **Exact Match** ($\Delta = 0$) | **100.0 pts** | Resource level matches learner level |
| **1 Step Away** ($\Delta = 1$) | **60.0 pts** | Slight stretch or gentle review |
| **2 Steps Away** ($\Delta = 2$) | **25.0 pts** | Substantial pedagogical gap |

### C. Learning Format Score ($S_{\text{format}}$)
| User Preference | Resource Format | Points Awarded |
|---|---|---|
| **Any** | Any format | **100.0 pts** |
| Specific format | Exact match | **100.0 pts** |
| Specific format | Different format | **40.0 pts** |

### D. Worked Example Verification (Produces Exactly 51.0)
Consider a learner with:
- **Interests**: `["Python", "Cybersecurity"]` (2 selected interests)
- **Skill Level**: `Beginner`
- **Preferred Format**: `Video`

Evaluating an intermediate resource with tags `["Python"]` and format `Article`:
1. **Interest Match**: 1 of 2 matches $\rightarrow S_{\text{interest}} = 100.0 \times \frac{1}{2} = 50.0\text{ pts}$  
   Contribution: $0.70 \times 50.0 = 35.0$
2. **Skill Compatibility**: `Intermediate` vs `Beginner` (1 step away) $\rightarrow S_{\text{skill}} = 60.0\text{ pts}$  
   Contribution: $0.20 \times 60.0 = 12.0$
3. **Format Match**: `Article` vs preferred `Video` (different format) $\rightarrow S_{\text{format}} = 40.0\text{ pts}$  
   Contribution: $0.10 \times 40.0 = 4.0$
4. **Final Weighted Score**:
   $$\text{Final Score} = 35.0 + 12.0 + 4.0 = \mathbf{51.0\%}$$

### E. Deterministic Tie-Breaking
When two resources achieve identical final scores, they are sorted alphabetically by title (`title.lower()`), guaranteeing reproducible rankings across runs.

---

## 7. Project Directory Structure

```text
ai-recommendation-project/
├── app.py                  # Streamlit web user interface
├── recommender.py          # Core recommendation engine & scoring functions
├── data.py                 # 40 curated demonstration resources & helpers
├── requirements.txt        # Minimal dependency list
├── README.md               # Complete project documentation & guide
├── .gitignore              # Git ignore rules for Python / virtual envs
└── tests/
    ├── __init__.py         # Test package initializer
    └── test_recommender.py # Pytest automated test suite (26 tests)
```

---

## 8. Windows PowerShell Installation & Execution Guide

Follow these exact steps in PowerShell:

### Step 1: Open the Project Directory
```powershell
cd "c:\Users\Vundavalli.Sree Ram\OneDrive\Desktop\ai-recommendation-project"
```

### Step 2: Create and Activate a Python Virtual Environment
```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
```
*(If PowerShell script execution is restricted on your machine, activate using `.\.venv\Scripts\activate.bat` or run directly with `py`)*

### Step 3: Install Dependencies
```powershell
python -m pip install -r requirements.txt
```

### Step 4: Run Automated Tests
```powershell
python -m pytest -v
```
*Expected result: 26 passed in < 0.20s.*

### Step 5: Launch the Streamlit Application
```powershell
python -m streamlit run app.py
```

### Step 6: Access the Application
Open your browser to the local URL displayed in the console:
```text
http://localhost:8501
```

---

## 9. Faculty / Evaluator Demonstration Script

When presenting this project to faculty or examiners, follow this 5-minute demonstration sequence:

1. **Explain the Architecture**:
   - Point out that this is an **explainable content-based recommendation system**.
   - Show the separation of concerns: `data.py` (catalog data), `recommender.py` (scoring engine), `app.py` (UI), and `tests/test_recommender.py` (automated verification).

2. **Run the Automated Tests**:
   - Execute `python -m pytest -v` in terminal to prove all 26 tests pass, covering distance rubrics, deduplication, tie-breaking, boundary limits, and the worked example.

3. **Demonstrate Preference Matching**:
   - In the sidebar, select **Interests**: `Python` and `Machine Learning`.
   - Set **Skill Level**: `Intermediate`.
   - Set **Preferred Format**: `Project`.
   - Click **🚀 Recommend**.
   - Observe that `#1. Hands-On Machine Learning with Scikit-Learn (RES-004)` receives a **100.0% Match**.

4. **Inspect the Explainability Formula**:
   - Click the expander **"🔍 Inspect Scoring Math"** under the #1 card.
   - Show the evaluator the exact calculation:
     $$\text{Final Score} = (0.70 \times 100.0) + (0.20 \times 100.0) + (0.10 \times 100.0) = 100.0\%$$

5. **Demonstrate Changing Preferences (Dynamic Adaptation)**:
   - Change **Skill Level** from `Intermediate` to `Beginner`.
   - Observe how the skill score drops from 100.0 pts to 60.0 pts (1 step away), and the final score recalculates to **92.0%** deterministically.

6. **Demonstrate Honest No-Match Behavior**:
   - Deselect all interests.
   - Show the warning message: the system does not fabricate false matches, but instead displays closest alternatives ranked on skill level and format.

7. **Demonstrate Session Feedback**:
   - Submit a 5-star rating on a card.
   - Switch to the **📝 Feedback & Evaluation Log** tab to show the logged rating and computed average.

---

## 10. Limitations of the Demonstration

- **Fixed Catalog**: Recommendations are drawn from the curated 40-item local catalog rather than dynamic external web scraping.
- **Stateless Ratings**: Submitted user feedback is stored in Streamlit session state for live demonstration and resets when the app is restarted.
- **Cold-Start Simplicity**: The system assumes the user can self-declare their interest tags and skill level.

---

## 11. Suggested Future Improvements

1. **Persistent Relational / SQLite Storage**: Store user ratings and learning history across sessions in a lightweight SQLite database.
2. **Hybrid Collaborative Filtering**: Blend item-based collaborative filtering with content-based matching once multi-user interaction data is collected.
3. **Interactive Knowledge Assessment**: Administer a 3-question diagnostic quiz to infer skill level rather than relying solely on self-reporting.
4. **Prerequisite Graph Routing**: Model resources as a Directed Acyclic Graph (DAG) to generate structured multi-step learning paths.
