"""test_recommender.py - Comprehensive Unit & Integration Test Suite.

Verifies all 20 required specifications from the DecodeLabs Project 3 Brief:
1. Resource with matching interests receives expected interest score.
2. Resource with no matching interests receives an interest score of zero.
3. Exact skill-level matching produces 100 points.
4. Adjacent skill levels produce 60 points.
5. Skill levels two steps apart produce 25 points.
6. An all-levels resource receives the defined compatibility score (85 points).
7. Exact format matching produces 100 points.
8. Selecting Any gives all formats 100 compatibility points.
9. A different format produces 40 points.
10. The weighted final score is calculated correctly.
11. Final scores stay within the valid range (0 to 100).
12. Recommendations are sorted from highest to lowest score.
13. Equal-scoring resources are sorted deterministically.
14. Returned recommendation counts respect the requested limit.
15. Resources contain the required fields.
16. Invalid or empty preference selections are handled appropriately.
17. No-match cases are handled without crashes.
18. Duplicate interests do not inflate a score.
19. The ranking is reproducible for identical inputs.
20. The recommendation engine does not modify the original catalog.
Plus additional tests for data catalog integrity, boundaries, and conversions.
"""

import copy
import pytest
from data import get_all_resources, get_available_interests, SAMPLE_RESOURCES
from recommender import (
    calculate_interest_score,
    calculate_skill_score,
    calculate_format_score,
    calculate_weighted_final_score,
    score_resource,
    recommend_resources,
    SCORE_SKILL_EXACT,
    SCORE_SKILL_ONE_AWAY,
    SCORE_SKILL_TWO_AWAY,
    SCORE_SKILL_ALL_LEVELS,
    SCORE_FORMAT_EXACT,
    SCORE_FORMAT_ANY,
    SCORE_FORMAT_DIFFERENT,
)


# ---------------------------------------------------------------------------
# Test 1: Matching interests receive expected interest score
# ---------------------------------------------------------------------------
def test_matching_interests_receive_expected_score():
    """Test 1: Interest match score = 100 * (matched / selected)."""
    user_interests = ["Python", "Machine Learning"]
    resource_tags = ["Python", "Machine Learning", "Data Science"]
    score, matched, unmatched = calculate_interest_score(resource_tags, user_interests)

    # 2 out of 2 matched -> 100.0%
    assert score == 100.0
    assert matched == ["Machine Learning", "Python"]
    assert unmatched == []

    # 1 out of 2 matched -> 50.0%
    resource_tags_single = ["Python", "Web Development"]
    score_single, matched_s, unmatched_s = calculate_interest_score(
        resource_tags_single, user_interests
    )
    assert score_single == 50.0
    assert matched_s == ["Python"]
    assert unmatched_s == ["Machine Learning"]


# ---------------------------------------------------------------------------
# Test 2: No matching interests receive an interest score of zero
# ---------------------------------------------------------------------------
def test_no_matching_interests_receive_zero():
    """Test 2: When no interests match, interest score must be strictly 0.0."""
    user_interests = ["Cybersecurity", "Cloud Computing"]
    resource_tags = ["Python", "Web Development"]
    score, matched, unmatched = calculate_interest_score(resource_tags, user_interests)

    assert score == 0.0
    assert matched == []
    assert sorted(unmatched) == sorted(user_interests)


# ---------------------------------------------------------------------------
# Test 3: Exact skill-level matching produces 100 points
# ---------------------------------------------------------------------------
def test_exact_skill_level_produces_100_points():
    """Test 3: Beginner==Beginner, Intermediate==Intermediate, Advanced==Advanced -> 100."""
    for level in ["Beginner", "Intermediate", "Advanced"]:
        score, label = calculate_skill_score(level, level)
        assert score == SCORE_SKILL_EXACT
        assert score == 100.0
        assert "Exact Match" in label


# ---------------------------------------------------------------------------
# Test 4: Adjacent skill levels produce 60 points
# ---------------------------------------------------------------------------
def test_adjacent_skill_levels_produce_60_points():
    """Test 4: 1 level away -> 60 points."""
    # Beginner to Intermediate (diff 1)
    score1, label1 = calculate_skill_score("Beginner", "Intermediate")
    assert score1 == SCORE_SKILL_ONE_AWAY
    assert score1 == 60.0

    # Intermediate to Beginner (diff 1)
    score2, label2 = calculate_skill_score("Intermediate", "Beginner")
    assert score2 == 60.0

    # Intermediate to Advanced (diff 1)
    score3, label3 = calculate_skill_score("Intermediate", "Advanced")
    assert score3 == 60.0

    # Advanced to Intermediate (diff 1)
    score4, label4 = calculate_skill_score("Advanced", "Intermediate")
    assert score4 == 60.0


# ---------------------------------------------------------------------------
# Test 5: Skill levels two steps apart produce 25 points
# ---------------------------------------------------------------------------
def test_skill_levels_two_steps_apart_produce_25_points():
    """Test 5: Beginner to Advanced (diff 2) -> 25 points."""
    score1, label1 = calculate_skill_score("Beginner", "Advanced")
    assert score1 == SCORE_SKILL_TWO_AWAY
    assert score1 == 25.0

    score2, label2 = calculate_skill_score("Advanced", "Beginner")
    assert score2 == SCORE_SKILL_TWO_AWAY
    assert score2 == 25.0


# ---------------------------------------------------------------------------
# Test 6: All-levels resource receives 85 points
# ---------------------------------------------------------------------------
def test_all_levels_resource_receives_85_points():
    """Test 6: Resource marked 'All Levels' receives 85 points regardless of user level."""
    for user_level in ["Beginner", "Intermediate", "Advanced"]:
        score, label = calculate_skill_score("All Levels", user_level)
        assert score == SCORE_SKILL_ALL_LEVELS
        assert score == 85.0
        assert "All Levels Friendly" in label


# ---------------------------------------------------------------------------
# Test 7: Exact format matching produces 100 points
# ---------------------------------------------------------------------------
def test_exact_format_matching_produces_100_points():
    """Test 7: Video==Video, Article==Article, Project==Project -> 100."""
    for fmt in ["Video", "Article", "Project"]:
        score, label = calculate_format_score(fmt, fmt)
        assert score == SCORE_FORMAT_EXACT
        assert score == 100.0
        assert "Exact Format Match" in label


# ---------------------------------------------------------------------------
# Test 8: Selecting Any gives all formats 100 compatibility points
# ---------------------------------------------------------------------------
def test_selecting_any_format_gives_100_points():
    """Test 8: Preferred format 'Any' gives 100.0 to Video, Article, and Project."""
    for res_format in ["Video", "Article", "Project"]:
        score, label = calculate_format_score(res_format, "Any")
        assert score == SCORE_FORMAT_ANY
        assert score == 100.0
        assert "Any Format Preferred" in label


# ---------------------------------------------------------------------------
# Test 9: Different format produces 40 points
# ---------------------------------------------------------------------------
def test_different_format_produces_40_points():
    """Test 9: Mismatch format -> 40 points."""
    score1, label1 = calculate_format_score("Video", "Article")
    assert score1 == SCORE_FORMAT_DIFFERENT
    assert score1 == 40.0

    score2, label2 = calculate_format_score("Project", "Video")
    assert score2 == 40.0

    score3, label3 = calculate_format_score("Article", "Project")
    assert score3 == 40.0


# ---------------------------------------------------------------------------
# Test 10: Weighted final score is calculated correctly
# ---------------------------------------------------------------------------
def test_weighted_final_score_calculated_correctly():
    """Test 10: Final = 0.70 * interest + 0.20 * skill + 0.10 * format."""
    # Perfect score: 70 + 20 + 10 = 100.0
    perfect = calculate_weighted_final_score(100.0, 100.0, 100.0)
    assert perfect == 100.0

    # Specific known values:
    # 0.70 * 50.0 = 35.0
    # 0.20 * 60.0 = 12.0
    # 0.10 * 40.0 = 4.0
    # Expected sum = 51.0
    score = calculate_weighted_final_score(50.0, 60.0, 40.0)
    assert score == 51.0

    # Another test case:
    # 0.70 * 100.0 = 70.0
    # 0.20 * 25.0 = 5.0
    # 0.10 * 100.0 = 10.0
    # Expected sum = 85.0
    score2 = calculate_weighted_final_score(100.0, 25.0, 100.0)
    assert score2 == 85.0


# ---------------------------------------------------------------------------
# Test 11: Final scores stay strictly within [0, 100] range
# ---------------------------------------------------------------------------
def test_final_scores_stay_within_valid_range():
    """Test 11: Edge tests asserting bounds [0.0, 100.0]."""
    zero_score = calculate_weighted_final_score(0.0, 0.0, 0.0)
    assert zero_score == 0.0

    max_score = calculate_weighted_final_score(100.0, 100.0, 100.0)
    assert max_score == 100.0

    # Test clamping on extreme inputs
    clamped_low = calculate_weighted_final_score(-20.0, -10.0, -5.0)
    assert clamped_low == 0.0

    clamped_high = calculate_weighted_final_score(150.0, 120.0, 110.0)
    assert clamped_high == 100.0


# ---------------------------------------------------------------------------
# Test 12: Recommendations are sorted from highest to lowest score
# ---------------------------------------------------------------------------
def test_recommendations_sorted_descending():
    """Test 12: All returned items must have monotonically non-increasing final scores."""
    catalog = get_all_resources()
    outcome = recommend_resources(
        catalog=catalog,
        user_interests=["Python", "Machine Learning"],
        user_skill_level="Intermediate",
        preferred_format="Project",
        top_k=10,
    )

    recs = outcome.recommendations
    assert len(recs) > 0
    scores = [r.final_score for r in recs]
    assert scores == sorted(scores, reverse=True)


# ---------------------------------------------------------------------------
# Test 13: Equal-scoring resources are sorted deterministically
# ---------------------------------------------------------------------------
def test_equal_scoring_resources_sorted_deterministically():
    """Test 13: Tie breaker is title alphabetical ascending."""
    tied_catalog = [
        {
            "id": "T-01",
            "title": "Zebra Fundamentals",
            "description": "Zebra learning",
            "tags": ["Python"],
            "skill_level": "Beginner",
            "format": "Video",
            "duration": "1 hour",
            "skills_covered": ["Python"],
        },
        {
            "id": "T-02",
            "title": "Alpha Python",
            "description": "Alpha learning",
            "tags": ["Python"],
            "skill_level": "Beginner",
            "format": "Video",
            "duration": "1 hour",
            "skills_covered": ["Python"],
        },
        {
            "id": "T-03",
            "title": "Beta Python",
            "description": "Beta learning",
            "tags": ["Python"],
            "skill_level": "Beginner",
            "format": "Video",
            "duration": "1 hour",
            "skills_covered": ["Python"],
        },
    ]

    outcome = recommend_resources(
        catalog=tied_catalog,
        user_interests=["Python"],
        user_skill_level="Beginner",
        preferred_format="Video",
        top_k=3,
    )

    titles = [r.title for r in outcome.recommendations]
    # All scores are 100.0, so titles should be sorted alphabetically: Alpha, Beta, Zebra
    assert titles == ["Alpha Python", "Beta Python", "Zebra Fundamentals"]


# ---------------------------------------------------------------------------
# Test 14: Returned recommendation counts respect the requested limit
# ---------------------------------------------------------------------------
def test_recommendation_counts_respect_requested_limit():
    """Test 14: top_k parameter must constrain output count."""
    catalog = get_all_resources()
    for k in [1, 3, 5, 8]:
        outcome = recommend_resources(
            catalog=catalog,
            user_interests=["Python", "Data Science", "Machine Learning"],
            user_skill_level="Intermediate",
            preferred_format="Any",
            top_k=k,
        )
        assert len(outcome.recommendations) == k


# ---------------------------------------------------------------------------
# Test 15: Resources contain the required fields
# ---------------------------------------------------------------------------
def test_resources_contain_required_fields():
    """Test 15: All catalog resources and scored items contain all necessary fields."""
    catalog = get_all_resources()
    assert len(catalog) >= 30

    required_raw_fields = {
        "id", "title", "description", "tags", "skill_level",
        "format", "duration", "skills_covered"
    }

    for item in catalog:
        for field in required_raw_fields:
            assert field in item, f"Missing {field} in resource {item.get('id')}"
            assert item[field] is not None

    outcome = recommend_resources(
        catalog=catalog,
        user_interests=["Python"],
        user_skill_level="Beginner",
        preferred_format="Video",
        top_k=3,
    )

    required_scored_fields = {
        "id", "title", "description", "tags", "skill_level", "format",
        "duration", "skills_covered", "final_score", "interest_score",
        "skill_score", "format_score", "matched_interests",
        "unmatched_user_interests", "skill_compatibility_label",
        "format_compatibility_label", "recommendation_reason",
        "score_breakdown_formula"
    }

    for scored in outcome.recommendations:
        scored_dict = scored.to_dict()
        for field in required_scored_fields:
            assert field in scored_dict, f"Missing scored field {field}"


# ---------------------------------------------------------------------------
# Test 16: Invalid or empty preference selections are handled appropriately
# ---------------------------------------------------------------------------
def test_invalid_or_empty_preferences_handled():
    """Test 16: Empty inputs return safe fallback; invalid inputs raise descriptive ValueError."""
    catalog = get_all_resources()

    # Empty user interests: should not crash, returns fallback alternatives
    outcome_empty = recommend_resources(
        catalog=catalog,
        user_interests=[],
        user_skill_level="Beginner",
        preferred_format="Video",
        top_k=5,
    )
    assert outcome_empty.has_interest_matches is False
    assert len(outcome_empty.recommendations) == 0
    assert len(outcome_empty.fallback_alternatives) == 5

    # Invalid skill level raises ValueError
    with pytest.raises(ValueError, match="Invalid.*skill level"):
        recommend_resources(
            catalog=catalog,
            user_interests=["Python"],
            user_skill_level="Expert",  # Invalid
            preferred_format="Video",
        )

    # Invalid format raises ValueError
    with pytest.raises(ValueError, match="Invalid.*format"):
        recommend_resources(
            catalog=catalog,
            user_interests=["Python"],
            user_skill_level="Beginner",
            preferred_format="Podcast",  # Invalid
        )

    # Invalid top_k < 1 or > 10 raises ValueError
    with pytest.raises(ValueError, match="top_k must be an integer between 1 and 10"):
        recommend_resources(
            catalog=catalog,
            user_interests=["Python"],
            user_skill_level="Beginner",
            preferred_format="Video",
            top_k=0,
        )

    with pytest.raises(ValueError, match="top_k must be an integer between 1 and 10"):
        recommend_resources(
            catalog=catalog,
            user_interests=["Python"],
            user_skill_level="Beginner",
            preferred_format="Video",
            top_k=11,
        )


# ---------------------------------------------------------------------------
# Test 17: No-match cases are handled without crashes
# ---------------------------------------------------------------------------
def test_no_match_cases_handled_without_crash():
    """Test 17: When user interests do not match any item in catalog, do not fabricate match."""
    catalog = get_all_resources()
    outcome = recommend_resources(
        catalog=catalog,
        user_interests=["NonExistentFantasyInterest", "QuantumAlchemy"],
        user_skill_level="Beginner",
        preferred_format="Video",
        top_k=5,
    )

    assert outcome.has_interest_matches is False
    assert len(outcome.recommendations) == 0
    # Fallback alternatives are provided based on skill and format matching
    assert len(outcome.fallback_alternatives) == 5
    for alt in outcome.fallback_alternatives:
        assert alt.interest_score == 0.0


# ---------------------------------------------------------------------------
# Test 18: Duplicate interests do not inflate a score
# ---------------------------------------------------------------------------
def test_duplicate_interests_do_not_inflate_score():
    """Test 18: Passing ["Python", "Python", "Python"] is identical to ["Python"]."""
    resource_tags = ["Python", "Web Development"]

    score_single, _, _ = calculate_interest_score(resource_tags, ["Python"])
    score_dup, _, _ = calculate_interest_score(
        resource_tags, ["Python", "Python", "Python", "  Python  "]
    )

    assert score_single == 100.0
    assert score_dup == 100.0

    # Test with partial match
    score_single_part, _, _ = calculate_interest_score(
        ["Python"], ["Python", "Cybersecurity"]
    )
    score_dup_part, _, _ = calculate_interest_score(
        ["Python"], ["Python", "Python", "Cybersecurity", "Cybersecurity"]
    )
    assert score_single_part == 50.0
    assert score_dup_part == 50.0


# ---------------------------------------------------------------------------
# Test 19: Ranking is reproducible for identical inputs
# ---------------------------------------------------------------------------
def test_ranking_reproducible_for_identical_inputs():
    """Test 19: Calling recommend_resources multiple times with same input yields identical results."""
    catalog = get_all_resources()
    interests = ["Python", "Databases"]
    skill = "Intermediate"
    fmt = "Article"

    run1 = recommend_resources(catalog, interests, skill, fmt, top_k=5)
    run2 = recommend_resources(catalog, interests, skill, fmt, top_k=5)

    assert [r.id for r in run1.recommendations] == [r.id for r in run2.recommendations]
    assert [r.final_score for r in run1.recommendations] == [r.final_score for r in run2.recommendations]
    assert [r.title for r in run1.recommendations] == [r.title for r in run2.recommendations]


# ---------------------------------------------------------------------------
# Test 20: Recommendation engine does not modify original catalog
# ---------------------------------------------------------------------------
def test_recommendation_engine_does_not_modify_catalog():
    """Test 20: Catalog before and after recommendation call must be identical."""
    catalog = get_all_resources()
    catalog_snapshot = copy.deepcopy(catalog)

    _ = recommend_resources(
        catalog=catalog,
        user_interests=["Artificial Intelligence", "DevOps"],
        user_skill_level="Advanced",
        preferred_format="Project",
        top_k=7,
    )

    assert catalog == catalog_snapshot


# ---------------------------------------------------------------------------
# Test 21: Data catalog integrity check (exactly 40 items, unique IDs, 12 categories)
# ---------------------------------------------------------------------------
def test_data_catalog_integrity():
    """Ensure catalog has exactly 40 items, unique IDs and titles, and all 12 categories."""
    assert len(SAMPLE_RESOURCES) == 40

    ids = [r["id"] for r in SAMPLE_RESOURCES]
    assert len(set(ids)) == 40, "Resource IDs must be unique"

    titles = [r["title"] for r in SAMPLE_RESOURCES]
    assert len(set(titles)) == 40, "Resource titles must be unique"

    required_categories = {
        "Python",
        "Artificial Intelligence",
        "Machine Learning",
        "Data Science",
        "Cybersecurity",
        "Web Development",
        "Cloud Computing",
        "DevOps",
        "Databases",
        "Computer Networks",
        "Data Structures and Algorithms",
        "Mathematics",
    }

    available = set(get_available_interests())
    for category in required_categories:
        assert category in available, f"Missing required category: {category}"


# ---------------------------------------------------------------------------
# Test 22: Score breakdown formula string correctness
# ---------------------------------------------------------------------------
def test_score_breakdown_formula_string():
    """Ensure the explainability formula string contains the exact weights and arithmetic."""
    res = {
        "id": "T-99",
        "title": "Math for AI",
        "description": "Math concepts",
        "tags": ["Mathematics", "Artificial Intelligence"],
        "skill_level": "Beginner",
        "format": "Video",
        "duration": "2 hours",
        "skills_covered": ["Algebra"],
    }
    scored = score_resource(
        res,
        user_interests=["Mathematics"],
        user_skill_level="Beginner",
        preferred_format="Video",
    )
    # Interest: 100 * (1/1) = 100.0
    # Skill: Beginner==Beginner = 100.0
    # Format: Video==Video = 100.0
    # Final = 70 + 20 + 10 = 100.0
    assert scored.final_score == 100.0
    assert "0.70 × 100.0" in scored.score_breakdown_formula
    assert "0.20 × 100.0" in scored.score_breakdown_formula
    assert "0.10 × 100.0" in scored.score_breakdown_formula
    assert "= 100.0%" in scored.score_breakdown_formula


# ---------------------------------------------------------------------------
# Test 23: Mandatory Check 3 — Worked example produces exactly 51.0
# ---------------------------------------------------------------------------
def test_worked_example_produces_51_0():
    """Verify that the worked example from specification produces exactly 51.0.

    Worked Example Specification:
    - User interests: 2 selected interests.
    - Resource matches 1 of the 2 interests -> Interest score = 100 * (1/2) = 50.0
      70% of 50.0 = 35.0
    - Skill level: 1 step away -> Skill score = 60.0
      20% of 60.0 = 12.0
    - Format: Different format -> Format score = 40.0
      10% of 40.0 = 4.0
    - Final score = 35.0 + 12.0 + 4.0 = 51.0
    """
    # Mathematical formula check
    final_score = calculate_weighted_final_score(50.0, 60.0, 40.0)
    assert final_score == 51.0

    # End-to-end check with score_resource
    example_resource = {
        "id": "EX-01",
        "title": "Intermediate Python Projects",
        "description": "Hands on project",
        "tags": ["Python"],  # matches 1 of ["Python", "Cybersecurity"]
        "skill_level": "Intermediate",  # 1 step away from Beginner
        "format": "Article",  # different from preferred Video
        "duration": "3 hours",
        "skills_covered": ["Python"],
    }
    scored = score_resource(
        resource=example_resource,
        user_interests=["Python", "Cybersecurity"],  # 2 interests
        user_skill_level="Beginner",  # 1 step from Intermediate (60 pts)
        preferred_format="Video",  # different from Article (40 pts)
    )

    assert scored.interest_score == 50.0
    assert scored.skill_score == 60.0
    assert scored.format_score == 40.0
    assert scored.final_score == 51.0


# ---------------------------------------------------------------------------
# Test 24: Mandatory Check 5 — Only resources matching >= 1 interest in normal recs
# ---------------------------------------------------------------------------
def test_only_resources_matching_at_least_one_interest_appear_in_normal_recommendations():
    """Confirm that only resources matching at least one selected interest appear in normal recommendations."""
    catalog = get_all_resources()

    # Query for a specific niche interest
    outcome = recommend_resources(
        catalog=catalog,
        user_interests=["Cybersecurity"],
        user_skill_level="Beginner",
        preferred_format="Any",
        top_k=10,
    )

    assert outcome.has_interest_matches is True
    assert len(outcome.recommendations) > 0
    # Every single returned recommendation MUST have interest_score > 0.0
    for rec in outcome.recommendations:
        assert rec.interest_score > 0.0
        assert len(rec.matched_interests) >= 1
        assert "Cybersecurity" in rec.tags


# ---------------------------------------------------------------------------
# Test 25: Mandatory Check 6 — Fallback alternatives have 0 interest score
# ---------------------------------------------------------------------------
def test_no_match_fallback_displays_closest_alternatives_with_zero_interest_score():
    """Confirm that the no-match fallback never misrepresents items as interest matches."""
    catalog = get_all_resources()
    outcome = recommend_resources(
        catalog=catalog,
        user_interests=["UnknownFuturisticConcept"],
        user_skill_level="Advanced",
        preferred_format="Video",
        top_k=5,
    )

    assert outcome.has_interest_matches is False
    assert len(outcome.recommendations) == 0
    assert len(outcome.fallback_alternatives) == 5

    for alt in outcome.fallback_alternatives:
        assert alt.interest_score == 0.0
        assert alt.matched_interests == []
        # Fallback is scored purely on skill and format
        assert alt.skill_score in [25.0, 60.0, 85.0, 100.0]
        assert alt.format_score in [40.0, 100.0]


# ---------------------------------------------------------------------------
# Test 26: Mandatory Check 8 — Session ratings do not alter recommendation scores
# ---------------------------------------------------------------------------
def test_user_ratings_do_not_change_recommendation_scores_or_ranking():
    """Confirm that recording ratings has zero impact on scores or rankings."""
    catalog = get_all_resources()
    user_interests = ["Python", "Machine Learning"]
    skill = "Intermediate"
    fmt = "Project"

    # Pre-rating recommendation run
    run_before = recommend_resources(catalog, user_interests, skill, fmt, top_k=5)

    # Simulate user submitting ratings in session state
    session_ratings = {
        run_before.recommendations[0].id: {"rating": 1, "comment": "Disliked"},
        run_before.recommendations[1].id: {"rating": 5, "comment": "Loved it"},
    }
    assert len(session_ratings) == 2

    # Post-rating recommendation run
    run_after = recommend_resources(catalog, user_interests, skill, fmt, top_k=5)

    assert [r.id for r in run_before.recommendations] == [r.id for r in run_after.recommendations]
    assert [r.final_score for r in run_before.recommendations] == [r.final_score for r in run_after.recommendations]

