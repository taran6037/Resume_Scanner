import sys
import json
import time
import argparse
import logging
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.WARNING, format="%(levelname)s | %(message)s")

RESUMES_DIR   = PROJECT_ROOT / "data" / "resumes"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
ALLOWED_EXT   = {".pdf", ".docx", ".doc", ".txt"}


def banner(text):
    print(f"\n{'='*64}\n  {text}\n{'='*64}")


def parse_resume(path: Path) -> dict | None:
    from pipeline.ingestion.validator import validate_file
    from pipeline.ingestion.router import route_file
    from pipeline.parsing.cleaner import clean_text
    from pipeline.parsing.ner_extractor import extract_entities
    from pipeline.parsing.structured_extractor import extract_structured_profile

    try:
        validation = validate_file(path)
        if not validation.is_valid:
            print(f"    ✗ Invalid: {validation.error}")
            return None

        route = route_file(validation)
        clean = clean_text(route.raw_text, was_ocr=route.is_scanned)
        ner   = extract_entities(clean.clean_text)
        profile = extract_structured_profile(
            ner_result=ner, clean_text=clean.clean_text, parser_version="v1"
        )
        return profile.model_dump()

    except Exception as e:
        print(f"    ✗ Parse failed: {type(e).__name__}: {str(e)[:80]}")
        return None


def match_one(profile: dict, criteria: dict):
    from pipeline.matching.semantic_matcher import semantic_match
    from pipeline.matching.scoring import score_candidate

    result = {
        "fit_score": None,
        "skill_score": None,
        "experience_score": None,
        "semantic_score": None,
        "education_score": None,
        "skill_matches": [],
        "skill_gaps": [],
        "reasoning": "",
        "is_transferable": False,
        "education_match": None,
        "education_notes": None,
        "error": None,
    }

    try:
        match = semantic_match(profile, criteria)
        weights = criteria.get("scoring_weights")
        scores = score_candidate(match, profile, criteria, weights)
        result.update(scores)
        result["skill_matches"]   = match.skill_matches
        result["skill_gaps"]      = match.skill_gaps
        result["reasoning"]       = match.reasoning
        result["is_transferable"] = match.is_transferable
        result["education_match"] = match.education_match
        result["education_notes"] = match.education_notes
    except Exception as e:
        result["error"] = f"Qwen match failed: {str(e)[:60]}"

    return result


def run_batch(jd_text: str):
    from pipeline.jd.jd_extractor import extract_jd
    from pipeline.utils.ollama_client import check_ollama_health
    from pipeline.matching.scoring import rank_candidates

    banner("PRE-FLIGHT CHECKS")
    oh = check_ollama_health()
    print(f"  Ollama : {'OK' if oh['status']=='ok' else 'FAIL — ' + oh['message']}")
    if oh["status"] != "ok":
        sys.exit(1)

    files = [f for f in sorted(RESUMES_DIR.iterdir())
             if f.is_file() and f.suffix.lower() in ALLOWED_EXT
             and not f.name.startswith("demo_resume_")]

    if not files:
        print(f"\n  No resume files found in {RESUMES_DIR}")
        sys.exit(1)

    print(f"  Resumes : {len(files)} files found")

    banner("PROCESSING JOB DESCRIPTION (once)")
    criteria = extract_jd(jd_text)
    print(f"  Role      : {criteria.get('role_type')}")
    print(f"  Seniority : {criteria.get('seniority')}")
    print(f"  Required  : {criteria.get('skills', {}).get('required', [])}")
    print(f"  Preferred : {criteria.get('skills', {}).get('preferred', [])}")

    results  = []
    failures = []

    for i, path in enumerate(files, 1):
        banner(f"RESUME {i}/{len(files)} — {path.name}")

        profile = parse_resume(path)
        if profile is None:
            failures.append({"file": path.name, "stage": "parsing"})
            continue

        name   = profile.get("contact", {}).get("name") or "Unknown"
        skills = profile.get("skills", {}).get("technical", [])[:6]
        print(f"    Name   : {name}")
        print(f"    Skills : {skills}")

        print(f"    Matching against JD (Qwen)... ", end="", flush=True)
        m = match_one(profile, criteria)

        if m["error"]:
            print(f"\n    ⚠ {m['error']}")
            failures.append({"file": path.name, "stage": "matching", "error": m["error"]})
            continue

        final = m["fit_score"]
        print(f"\r    ✓ Skills: {m['skill_score']}/100   "
              f"Experience: {m['experience_score']}/100   "
              f"Semantic: {m['semantic_score']}/100   "
              f"Education: {m['education_score']}/100   Fit: {final}/100")

        results.append({
            "file": path.name,
            "name": name,
            "fit_score": final,
            "skill_score": m["skill_score"],
            "experience_score": m["experience_score"],
            "semantic_score": m["semantic_score"],
            "education_score": m["education_score"],
            "skill_matches": m["skill_matches"],
            "skill_gaps": m["skill_gaps"],
            "is_transferable": m["is_transferable"],
            "education_match": m["education_match"],
            "education_notes": m["education_notes"],
            "reasoning": m["reasoning"],
        })

        time.sleep(1)

    results = rank_candidates(results)

    banner("FINAL RANKING")
    print(f"  {'Rank':<5}{'Score':<8}{'Name':<22}{'File'}")
    print(f"  {'-'*60}")
    for r in results:
        print(f"  {r['rank_position']:<5}{r['fit_score']:<8}{(r['name'] or '')[:20]:<22}{r['file'][:24]}")

    if failures:
        print(f"\n  {'-'*60}")
        print(f"  FAILED ({len(failures)}):")
        for f in failures:
            print(f"    - {f['file']} (at {f['stage']})"
                  + (f" — {f.get('error','')}" if f.get('error') else ""))

    banner("DETAILED BREAKDOWN (top to bottom)")
    for r in results:
        print(f"\n  #{r['rank_position']} — {r['name']} ({r['file']})  →  {r['fit_score']}/100")
        print(f"     Skill score    : {r['skill_score']}/100")
        print(f"     Experience score: {r['experience_score']}/100")
        print(f"     Semantic score : {r['semantic_score']}/100")
        print(f"     Education score: {r['education_score']}/100")
        print(f"     Matched skills : {', '.join(r['skill_matches']) or 'none'}")
        print(f"     Missing skills : {', '.join(r['skill_gaps']) or 'none'}")
        print(f"     Transferable   : {r['is_transferable']}")
        if r['education_match'] is not None:
            print(f"     Education match: {r['education_match']} — {r['education_notes']}")
        print(f"     Why: {r['reasoning']}")

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = PROCESSED_DIR / f"batch_ranking_{ts}.json"
    out.write_text(json.dumps({
        "jd_criteria": criteria,
        "ranking": results,
        "failures": failures,
    }, indent=2, default=str))

    banner("DONE")
    print(f"  Ranked : {len(results)} resumes")
    print(f"  Failed : {len(failures)} resumes")
    print(f"  Saved  : {out.relative_to(PROJECT_ROOT)}")
    print()


def main():
    parser = argparse.ArgumentParser(description="Batch parse + rank resumes against one JD")
    parser.add_argument("--jd", type=str, required=True)
    args = parser.parse_args()

    jd_path = Path(args.jd)
    jd_text = jd_path.read_text(encoding="utf-8") if jd_path.exists() else args.jd

    run_batch(jd_text)


if __name__ == "__main__":
    main()
