import logging
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent.parent
RESUMES_DIR = PROJECT_ROOT / "data" / "resumes"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
ALLOWED_EXT = {".pdf", ".docx", ".doc", ".txt"}


def parse_resume(path: Path):
    from pipeline.ingestion.validator import validate_file
    from pipeline.ingestion.router import route_file
    from pipeline.parsing.cleaner import clean_text
    from pipeline.parsing.ner_extractor import extract_entities
    from pipeline.parsing.structured_extractor import extract_structured_profile

    validation = validate_file(path)
    if not validation.is_valid:
        raise ValueError(validation.error)

    route = route_file(validation)
    clean = clean_text(route.raw_text, was_ocr=route.is_scanned)
    ner = extract_entities(clean.clean_text)
    profile = extract_structured_profile(
        ner_result=ner, clean_text=clean.clean_text, parser_version="v1"
    )
    return profile.model_dump()


def match_one(profile: dict, criteria: dict):
    from pipeline.matching.semantic_matcher import semantic_match
    from pipeline.matching.scoring import score_candidate

    match   = semantic_match(profile, criteria)
    weights = criteria.get("scoring_weights")
    scores  = score_candidate(match, profile, criteria, weights)

    education   = profile.get("education") or []
    institution = education[0].get("institution") if education else None

    return {
        **scores,
        "institution":     institution,
        "experience":      profile.get("experience") or [],
        "projects":        profile.get("projects") or [],
        "skill_matches":   match.skill_matches,
        "skill_gaps":      match.skill_gaps,
        "is_transferable": match.is_transferable,
        "education_match": match.education_match,
        "education_notes": match.education_notes,
        "reasoning":       match.reasoning,
    }


def run_full_batch(jd_text: str, on_progress=None, custom_weights: dict | None = None) -> dict:
    from pipeline.jd.jd_extractor import extract_jd
    from pipeline.matching.scoring import rank_candidates

    criteria = extract_jd(jd_text, custom_weights=custom_weights)

    files = [f for f in sorted(RESUMES_DIR.iterdir())
              if f.is_file() and f.suffix.lower() in ALLOWED_EXT]

    results = []
    failures = []
    total = len(files)

    for i, path in enumerate(files, 1):
        if on_progress:
            on_progress(i, total, path.name)

        try:
            profile = parse_resume(path)
        except Exception as e:
            failures.append({"file": path.name, "stage": "parsing", "error": str(e)[:150]})
            continue

        name = profile.get("contact", {}).get("name") or "Unknown"

        try:
            m = match_one(profile, criteria)
        except Exception as e:
            failures.append({"file": path.name, "stage": "matching", "error": str(e)[:150]})
            continue

        results.append({
            "file": path.name,
            "name": name,
            **m,
        })

    ranked, excluded = rank_candidates(results)

    output = {
        "jd_criteria":      criteria,
        "ranking":          ranked,
        "excluded":         excluded,
        "excluded_count":   len(excluded),
        "failures":         failures,
    }


    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = PROCESSED_DIR / f"batch_ranking_{ts}.json"
    import json
    out_path.write_text(json.dumps(output, indent=2, default=str))

    return output