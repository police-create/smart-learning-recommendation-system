"""recommender.py - Explainable AI Recommendation Engine for Learning Resources.

This module implements a deterministic, content-based recommendation algorithm
using multi-criteria preference matching:
- Interest match (70% weight)
- Skill-level compatibility (20% weight)
- Learning-format compatibility (10% weight)

It contains zero neural networks or external dependencies, prioritizing
explainability, mathematical rigor, and testability.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Set
import copy


# Standard skill hierarchy for distance calculations
SKILL_LEVEL_HIERARCHY: Dict[str, int] = {
    "Beginner": 0,
    "Intermediate": 1,
    "Advanced": 2,
}

VALID_USER_SKILL_LEVELS: Set[str] = {"Beginner", "Intermediate", "Advanced"}
VALID_FORMATS: Set[str] = {"Video", "Article", "Project", "Any"}

# Scoring weights defined in project specification
WEIGHT_INTEREST: float = 0.70
WEIGHT_SKILL: float = 0.20
WEIGHT_FORMAT: float = 0.10

# Skill scoring rubric
SCORE_SKILL_EXACT: float = 100.0
SCORE_SKILL_ONE_AWAY: float = 60.0
SCORE_SKILL_TWO_AWAY: float = 25.0
SCORE_SKILL_ALL_LEVELS: float = 85.0

# Format scoring rubric
SCORE_FORMAT_EXACT: float = 100.0
SCORE_FORMAT_ANY: float = 100.0
SCORE_FORMAT_DIFFERENT: float = 40.0


@dataclass(frozen=True)
class ScoredResource:
    """Represents a catalog resource evaluated against user preferences."""
    id: str
    title: str
    description: str
    tags: List[str]
    skill_level: str
    format: str
    duration: str
    skills_covered: List[str]
    final_score: float
    interest_score: float
    skill_score: float
    format_score: float
    matched_interests: List[str]
    unmatched_user_interests: List[str]
    skill_compatibility_label: str
    format_compatibility_label: str
    recommendation_reason: str
    score_breakdown_formula: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert the scored resource to a standard dictionary representation."""
        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "tags": list(self.tags),
            "skill_level": self.skill_level,
            "format": self.format,
            "duration": self.duration,
            "skills_covered": list(self.skills_covered),
            "final_score": self.final_score,
            "interest_score": self.interest_score,
            "skill_score": self.skill_score,
            "format_score": self.format_score,
            "matched_interests": list(self.matched_interests),
            "unmatched_user_interests": list(self.unmatched_user_interests),
            "skill_compatibility_label": self.skill_compatibility_label,
            "format_compatibility_label": self.format_compatibility_label,
            "recommendation_reason": self.recommendation_reason,
            "score_breakdown_formula": self.score_breakdown_formula,
        }


@dataclass
class RecommendationOutcome:
    """Complete outcome of a recommendation run."""
    recommendations: List[ScoredResource]
    has_interest_matches: bool
    fallback_alternatives: List[ScoredResource] = field(default_factory=list)
    total_evaluated: int = 0
    query_summary: Dict[str, Any] = field(default_factory=dict)


def calculate_interest_score(
    resource_tags: List[str],
    user_interests: List[str]
) -> tuple[float, List[str], List[str]]:
    """Calculate the interest-match score (0-100) using deduplicated set overlap.

    Formula:
        interest-match score = 100 * (matched user interests / total user interests)
        If no interests match, interest-match score is strictly 0.

    Args:
        resource_tags: Tags associated with the resource.
        user_interests: User's chosen interests (may contain duplicates).

    Returns:
        tuple: (score, matched_interests_list, unmatched_user_interests_list)
    """
    # Deduplicate and normalize user interests
    unique_user_interests: Set[str] = {
        i.strip() for i in user_interests if i and i.strip()
    }

    if not unique_user_interests:
        return 0.0, [], []

    resource_tag_set: Set[str] = {t.strip() for t in resource_tags if t and t.strip()}
    matched_set: Set[str] = unique_user_interests.intersection(resource_tag_set)
    unmatched_set: Set[str] = unique_user_interests.difference(resource_tag_set)

    matched_list: List[str] = sorted(list(matched_set))
    unmatched_list: List[str] = sorted(list(unmatched_set))

    if not matched_list:
        return 0.0, [], unmatched_list

    raw_score = 100.0 * (len(matched_set) / len(unique_user_interests))
    # Constrain strictly within [0.0, 100.0]
    bounded_score = max(0.0, min(100.0, round(raw_score, 4)))
    return bounded_score, matched_list, unmatched_list


def calculate_skill_score(resource_level: str, user_level: str) -> tuple[float, str]:
    """Calculate skill compatibility score (0-100) and rationale.

    Rubric:
        - Resource marked 'All Levels': 85.0 points
        - Exact match: 100.0 points
        - One level away: 60.0 points
        - Two levels away: 25.0 points

    Args:
        resource_level: 'Beginner', 'Intermediate', 'Advanced', or 'All Levels'.
        user_level: 'Beginner', 'Intermediate', or 'Advanced'.

    Returns:
        tuple: (score, description_label)
    """
    cleaned_user_level = user_level.strip().title()
    if cleaned_user_level not in VALID_USER_SKILL_LEVELS:
        raise ValueError(
            f"Invalid user skill level '{user_level}'. Must be one of: {sorted(list(VALID_USER_SKILL_LEVELS))}"
        )

    cleaned_resource_level = resource_level.strip()

    if cleaned_resource_level.title() == "All Levels":
        return SCORE_SKILL_ALL_LEVELS, "All Levels Friendly (85 pts)"

    if cleaned_resource_level.title() not in SKILL_LEVEL_HIERARCHY:
        # Default fallback for unclassified resource level
        return 50.0, f"Unclassified Level '{cleaned_resource_level}' (50 pts)"

    resource_rank = SKILL_LEVEL_HIERARCHY[cleaned_resource_level.title()]
    user_rank = SKILL_LEVEL_HIERARCHY[cleaned_user_level]

    distance = abs(resource_rank - user_rank)

    if distance == 0:
        return SCORE_SKILL_EXACT, f"Exact Match ({cleaned_resource_level}) (100 pts)"
    elif distance == 1:
        return SCORE_SKILL_ONE_AWAY, f"1 Step Apart ({cleaned_resource_level} vs {cleaned_user_level}) (60 pts)"
    elif distance == 2:
        return SCORE_SKILL_TWO_AWAY, f"2 Steps Apart ({cleaned_resource_level} vs {cleaned_user_level}) (25 pts)"
    else:
        return 0.0, "Mismatched Level (0 pts)"


def calculate_format_score(resource_format: str, preferred_format: str) -> tuple[float, str]:
    """Calculate format compatibility score (0-100) and rationale.

    Rubric:
        - User selected 'Any': 100.0 points for all formats
        - Exact format match: 100.0 points
        - Different format: 40.0 points

    Args:
        resource_format: Format of the resource ('Video', 'Article', 'Project').
        preferred_format: 'Video', 'Article', 'Project', or 'Any'.

    Returns:
        tuple: (score, description_label)
    """
    cleaned_preferred = preferred_format.strip().title()
    if cleaned_preferred not in VALID_FORMATS:
        raise ValueError(
            f"Invalid preferred format '{preferred_format}'. Must be one of: {sorted(list(VALID_FORMATS))}"
        )

    cleaned_res_format = resource_format.strip().title()

    if cleaned_preferred == "Any":
        return SCORE_FORMAT_ANY, f"Any Format Preferred (100 pts - {cleaned_res_format})"

    if cleaned_res_format == cleaned_preferred:
        return SCORE_FORMAT_EXACT, f"Exact Format Match ({cleaned_res_format}) (100 pts)"
    else:
        return SCORE_FORMAT_DIFFERENT, f"Alternative Format ({cleaned_res_format} vs {cleaned_preferred}) (40 pts)"


def calculate_weighted_final_score(
    interest_score: float,
    skill_score: float,
    format_score: float
) -> float:
    """Calculate the final weighted score rounded to 1 decimal place.

    Formula:
        Final = 0.70 * interest + 0.20 * skill + 0.10 * format

    Args:
        interest_score: 0.0 to 100.0
        skill_score: 0.0 to 100.0
        format_score: 0.0 to 100.0

    Returns:
        float: Weighted final score rounded to 1 decimal place.
    """
    # Enforce strict input bounds
    c_interest = max(0.0, min(100.0, interest_score))
    c_skill = max(0.0, min(100.0, skill_score))
    c_format = max(0.0, min(100.0, format_score))

    raw_final = (
        (WEIGHT_INTEREST * c_interest) +
        (WEIGHT_SKILL * c_skill) +
        (WEIGHT_FORMAT * c_format)
    )

    bounded_final = max(0.0, min(100.0, raw_final))
    return round(bounded_final, 1)


def generate_recommendation_reason(
    title: str,
    matched_interests: List[str],
    skill_label: str,
    format_label: str,
    final_score: float,
    interest_score: float
) -> str:
    """Generate a clear, human-readable natural language explanation for why this item was recommended."""
    if matched_interests:
        interest_phrase = f"covers your target interest(s): {', '.join(matched_interests)} ({interest_score:.1f}% match)"
    else:
        interest_phrase = "does not directly match your specified interest tags"

    reason = (
        f"Recommended with a {final_score:.1f}% compatibility score because it {interest_phrase}. "
        f"Skill compatibility: {skill_label}. Format: {format_label}."
    )
    return reason


def score_resource(
    resource: Dict[str, Any],
    user_interests: List[str],
    user_skill_level: str,
    preferred_format: str
) -> ScoredResource:
    """Evaluate a single resource against user criteria and return a ScoredResource."""
    interest_score, matched_ints, unmatched_ints = calculate_interest_score(
        resource.get("tags", []), user_interests
    )
    skill_score, skill_label = calculate_skill_score(
        resource.get("skill_level", ""), user_skill_level
    )
    format_score, format_label = calculate_format_score(
        resource.get("format", ""), preferred_format
    )

    final_score = calculate_weighted_final_score(interest_score, skill_score, format_score)

    reason = generate_recommendation_reason(
        resource.get("title", ""),
        matched_ints,
        skill_label,
        format_label,
        final_score,
        interest_score,
    )

    breakdown_formula = (
        f"({WEIGHT_INTEREST:.2f} × {interest_score:.1f}) + "
        f"({WEIGHT_SKILL:.2f} × {skill_score:.1f}) + "
        f"({WEIGHT_FORMAT:.2f} × {format_score:.1f}) = {final_score:.1f}%"
    )

    return ScoredResource(
        id=str(resource.get("id", "")),
        title=str(resource.get("title", "")),
        description=str(resource.get("description", "")),
        tags=list(resource.get("tags", [])),
        skill_level=str(resource.get("skill_level", "")),
        format=str(resource.get("format", "")),
        duration=str(resource.get("duration", "")),
        skills_covered=list(resource.get("skills_covered", [])),
        final_score=final_score,
        interest_score=round(interest_score, 1),
        skill_score=round(skill_score, 1),
        format_score=round(format_score, 1),
        matched_interests=matched_ints,
        unmatched_user_interests=unmatched_ints,
        skill_compatibility_label=skill_label,
        format_compatibility_label=format_label,
        recommendation_reason=reason,
        score_breakdown_formula=breakdown_formula,
    )


def recommend_resources(
    catalog: List[Dict[str, Any]],
    user_interests: List[str],
    user_skill_level: str,
    preferred_format: str = "Any",
    top_k: int = 5
) -> RecommendationOutcome:
    """Score, rank, and return top recommendations from catalog.

    Deterministic sorting:
        Primary: final_score descending (-final_score)
        Secondary: resource title ascending (case-insensitive) for stable tie-breaking.

    Args:
        catalog: List of resource dictionaries (will not be modified).
        user_interests: List of selected interest tags.
        user_skill_level: 'Beginner', 'Intermediate', or 'Advanced'.
        preferred_format: 'Video', 'Article', 'Project', or 'Any'.
        top_k: Maximum number of recommendations to return (must be >= 1).

    Returns:
        RecommendationOutcome containing top_k recommendations, flags, and fallbacks.
    """
    if not (1 <= top_k <= 10):
        raise ValueError(f"top_k must be an integer between 1 and 10, got {top_k}")

    # Validate inputs
    cleaned_skill = user_skill_level.strip().title()
    if cleaned_skill not in VALID_USER_SKILL_LEVELS:
        raise ValueError(
            f"Invalid skill level '{user_skill_level}'. Choose from {sorted(list(VALID_USER_SKILL_LEVELS))}"
        )

    cleaned_format = preferred_format.strip().title()
    if cleaned_format not in VALID_FORMATS:
        raise ValueError(
            f"Invalid format '{preferred_format}'. Choose from {sorted(list(VALID_FORMATS))}"
        )

    # Protect input catalog from modification
    safe_catalog = copy.deepcopy(catalog)

    # Score every resource in the catalog
    scored_items: List[ScoredResource] = [
        score_resource(item, user_interests, cleaned_skill, cleaned_format)
        for item in safe_catalog
    ]

    # Deterministic sorting: highest score first, then title alphabetically
    scored_items.sort(key=lambda item: (-item.final_score, item.title.lower()))

    # Check whether any item matched user interests
    has_matches = any(item.interest_score > 0.0 for item in scored_items)

    # When interests were provided and at least one matched
    if has_matches:
        # Filter items with interest match > 0 for primary recommendations
        interest_matched_items = [item for item in scored_items if item.interest_score > 0.0]
        primary_results = interest_matched_items[:top_k]
        fallback_results = []
    else:
        # No interest match found at all (or empty user interests)
        primary_results = []
        # Fallbacks: top ranked items based on skill & format compatibility
        fallback_results = scored_items[:top_k]

    return RecommendationOutcome(
        recommendations=primary_results,
        has_interest_matches=has_matches,
        fallback_alternatives=fallback_results,
        total_evaluated=len(safe_catalog),
        query_summary={
            "interests": sorted(list({i.strip() for i in user_interests if i and i.strip()})),
            "skill_level": cleaned_skill,
            "preferred_format": cleaned_format,
            "top_k": top_k,
        },
    )
