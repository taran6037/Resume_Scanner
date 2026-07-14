from backend.schemas.job import ScoringWeights
from pipeline.matching.semantic_matcher import _core_tokens
from config.pipeline_config import DEGREE_LEVEL_MAP

REQUIRED_SKILL_WEIGHT    = 0.75
PREFERRED_SKILL_WEIGHT   = 0.25
TRANSFERABLE_CREDIT      = 0.5
EXPERIENCE_RELEVANCE_WEIGHT = 0.6
EXPERIENCE_YEARS_WEIGHT     = 0.4
EXPERIENCE_UNKNOWN_YEARS_SCORE = 50


def _skill_tier(skill, required, preferred):
    skill_tokens = set(_core_tokens(skill))
    for req in required:
        req_tokens = set(_core_tokens(req))
        if req_tokens and (skill_tokens <= req_tokens or req_tokens <= skill_tokens):
            return "required"
    for pref in preferred:
        pref_tokens = set(_core_tokens(pref))
        if pref_tokens and (skill_tokens <= pref_tokens or pref_tokens <= skill_tokens):
            return "preferred"
    return "preferred"


def calculate_skill_score(skill_matches, skill_gaps, criteria, is_transferable=False) -> int:
    required  = criteria.get("skills", {}).get("required", [])
    preferred = criteria.get("skills", {}).get("preferred", [])

    if not required and not preferred:
        return 100

    matched_required = matched_preferred = 0
    gap_required = gap_preferred = 0

    for skill in skill_matches:
        if _skill_tier(skill, required, preferred) == "required":
            matched_required += 1
        else:
            matched_preferred += 1

    for skill in skill_gaps:
        if _skill_tier(skill, required, preferred) == "required":
            gap_required += 1
        else:
            gap_preferred += 1

    total_required  = matched_required + gap_required
    total_preferred = matched_preferred + gap_preferred

    required_ratio  = matched_required / total_required  if total_required  else 1.0
    preferred_ratio = matched_preferred / total_preferred if total_preferred else 1.0

    if is_transferable and total_required and gap_required:
        required_ratio = min(1.0, required_ratio + (gap_required / total_required) * TRANSFERABLE_CREDIT)

    if total_required and total_preferred:
        score = required_ratio * REQUIRED_SKILL_WEIGHT + preferred_ratio * PREFERRED_SKILL_WEIGHT
    elif total_required:
        score = required_ratio
    else:
        score = preferred_ratio

    return round(max(0.0, min(1.0, score)) * 100)


def calculate_experience_score(experience_relevance, total_experience_years, criteria) -> int:
    required_years = criteria.get("experience_years")
    relevance = max(0, min(100, experience_relevance))

    if not required_years:
        return round(relevance)

    if total_experience_years is None:
        years_score = EXPERIENCE_UNKNOWN_YEARS_SCORE
    else:
        years_score = max(0, min(100, (total_experience_years / required_years) * 100))

    score = relevance * EXPERIENCE_RELEVANCE_WEIGHT + years_score * EXPERIENCE_YEARS_WEIGHT
    return round(max(0, min(100, score)))


def _candidate_degree_level(education_entries: list) -> int:
    """Extract the highest degree level from a candidate's education list."""
    highest = 0
    for entry in (education_entries or []):
        degree = (entry.get("degree") or "").lower()
        for keyword, level in DEGREE_LEVEL_MAP.items():
            if keyword in degree:
                highest = max(highest, level)
    return highest


def check_education_gate(
    education_match,
    education_entries: list,
    education_requirement: str,
) -> dict:
    """
    Returns:
      passed       : bool  — whether candidate clears the gate
      flagged      : bool  — passes but below preferred level (soft warning)
      gate_note    : str   — human-readable explanation shown on dashboard
    """

    # No requirement — everyone passes silently
    if education_requirement == "none":
        return {"passed": True, "flagged": False, "gate_note": None}

    # Candidate degree level from extracted education entries
    candidate_level = _candidate_degree_level(education_entries)

    # Required level from the dropdown selection
    level_map = {
        "diploma_preferred":   1, "diploma_required":   1,
        "bachelors_preferred": 2, "bachelors_required": 2,
        "masters_preferred":   3, "masters_required":   3,
        "phd_preferred":       4, "phd_required":       4,
    }
    label_map = {
        1: "Diploma", 2: "Bachelor's", 3: "Master's", 4: "PhD"
    }
    required_level = level_map.get(education_requirement, 0)
    is_preferred   = education_requirement.endswith("_preferred")
    required_label = label_map.get(required_level, "")

    # Education couldn't be verified from resume
    if candidate_level == 0:
        if education_match is True:
            # Qwen says it matched even though we couldn't parse a degree level
            return {
                "passed":    True,
                "flagged":   True,
                "gate_note": f"{required_label} {'preferred' if is_preferred else 'required'} — education verified by AI but degree level unconfirmed.",
            }
        # Unverifiable — pass with flag rather than exclude unfairly
        return {
            "passed":    True,
            "flagged":   True,
            "gate_note": f"{required_label} {'preferred' if is_preferred else 'required'} — could not verify education from resume. Manual review recommended.",
        }

    candidate_label = label_map.get(candidate_level, f"Level {candidate_level}")

    if is_preferred:
        # Preferred: candidate must be at least ONE level below required to pass
        if candidate_level >= required_level:
            return {"passed": True, "flagged": False, "gate_note": None}
        elif candidate_level == required_level - 1:
            return {
                "passed":    True,
                "flagged":   True,
                "gate_note": f"{required_label} preferred — candidate holds {candidate_label}.",
            }
        else:
            return {
                "passed":    False,
                "flagged":   False,
                "gate_note": f"{required_label} preferred — candidate holds {candidate_label}, which is below the minimum threshold.",
            }
    else:
        # Required: candidate must meet or exceed the level
        if candidate_level >= required_level:
            return {"passed": True, "flagged": False, "gate_note": None}
        else:
            return {
                "passed":    False,
                "flagged":   False,
                "gate_note": f"{required_label} required — candidate holds {candidate_label}.",
            }


def calculate_fit_score(skill_score, experience_score, semantic_score, weights: ScoringWeights = None) -> int:
    """3-component weighted score — education is no longer part of the formula."""
    weights = weights or ScoringWeights()
    if isinstance(weights, dict):
        weights = ScoringWeights(**weights)

    # Normalise the 3 weights in case they don't sum to exactly 1.0
    total_w = weights.skills + weights.experience + weights.llm
    if total_w <= 0:
        total_w = 1.0
    w_skills = weights.skills     / total_w
    w_exp    = weights.experience / total_w
    w_llm    = weights.llm        / total_w

    total = (
        skill_score    * w_skills
        + experience_score * w_exp
        + semantic_score   * w_llm
    )
    return round(max(0, min(100, total)))


def score_candidate(match, profile: dict, criteria: dict, weights: ScoringWeights = None) -> dict:
    skill_score = calculate_skill_score(
        match.skill_matches, match.skill_gaps, criteria, match.is_transferable
    )
    experience_score = calculate_experience_score(
        match.experience_relevance, profile.get("total_experience_years"), criteria
    )
    fit_score = calculate_fit_score(
        skill_score, experience_score, match.semantic_score, weights
    )

    # Education gate check
    weights_obj = weights or ScoringWeights()
    if isinstance(weights_obj, dict):
        weights_obj = ScoringWeights(**weights_obj)

    education_requirement = weights_obj.education_requirement
    education_entries     = profile.get("education", [])
    gate = check_education_gate(
        match.education_match,
        education_entries,
        education_requirement,
    )

    return {
        "fit_score":            fit_score,
        "skill_score":          skill_score,
        "experience_score":     experience_score,
        "semantic_score":       match.semantic_score,
        "education_gate_passed": gate["passed"],
        "education_flagged":    gate["flagged"],
        "education_gate_note":  gate["gate_note"],
        "education_requirement": education_requirement,
    }


def rank_candidates(scored_results: list[dict]) -> list[dict]:
    """
    Split into passed/excluded by education gate,
    rank passed candidates by fit_score,
    return both lists separately.
    """
    passed   = [r for r in scored_results if r.get("education_gate_passed", True)]
    excluded = [r for r in scored_results if not r.get("education_gate_passed", True)]

    ranked = sorted(passed, key=lambda r: r["fit_score"], reverse=True)
    for position, result in enumerate(ranked, 1):
        result["rank_position"] = position

    return ranked, excluded