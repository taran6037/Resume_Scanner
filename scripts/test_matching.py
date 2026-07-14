import sys
import json
import argparse
import logging
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.WARNING, format="%(asctime)s | %(levelname)s | %(message)s")


def print_header(title):
    print(f"\n{'─'*60}")
    print(f"  {title}")
    print(f"{'─'*60}")


def run_test(profile: dict, criteria: dict):
    from pipeline.matching.semantic_matcher import semantic_match

    print_header("SEMANTIC MATCHING (Qwen)")
    print("  ⏳ Calling Qwen for deep semantic analysis...")
    match = semantic_match(profile, criteria)

    print(f"  ✓ Semantic score      : {match.semantic_score}/100")
    print(f"    Experience relevance: {match.experience_relevance}/100")
    print(f"    Transferable skills : {match.is_transferable}")
    print(f"    Skill matches       : {match.skill_matches}")
    print(f"    Skill gaps          : {match.skill_gaps}")
    print(f"    Reasoning           : {match.reasoning}")

    print_header("FINAL MATCH SUMMARY")
    print(f"  Semantic score (Qwen analysis) : {match.semantic_score}/100")
    print(f"  Experience relevance           : {match.experience_relevance}/100")
    print(f"  Skills matched                 : {', '.join(match.skill_matches) or 'none'}")
    print(f"  Skills missing                 : {', '.join(match.skill_gaps) or 'none'}")
    print(f"  Reasoning                      : {match.reasoning}")
    print("─"*60)


def main():
    parser = argparse.ArgumentParser(description="Test semantic matching (layer 6)")
    parser.add_argument("--profile", type=str, help="Path to parsed profile JSON")
    parser.add_argument("--jd",      type=str, help="Path to structured criteria JSON")
    parser.add_argument("--demo",    action="store_true", help="Run with dummy data")
    args = parser.parse_args()

    if args.demo:
        profile = {
            "summary": "Backend developer with 2 years experience in Python and FastAPI",
            "skills": {
                "technical": ["Python", "FastAPI", "PostgreSQL", "Redis", "Docker"],
                "tools": ["GitHub Actions", "VS Code", "Postman"],
                "soft": ["problem solving", "teamwork"]
            },
            "experience": [
                {
                    "company": "Tech Mahindra",
                    "title": "Backend Developer",
                    "start_date": "Jun 2023",
                    "end_date": "Present",
                    "responsibilities": [
                        "Built REST APIs using FastAPI",
                        "Optimized PostgreSQL queries",
                        "Implemented Redis caching"
                    ],
                    "achievements": ["Reduced API response time by 35%"]
                }
            ]
        }
        criteria = {
            "role_type": "backend",
            "seniority": "mid",
            "skills": {
                "required": ["Python", "FastAPI", "PostgreSQL", "Docker"],
                "preferred": ["Kubernetes", "Kafka", "Redis"]
            },
            "experience_years": 2,
            "responsibilities": [
                "Design and build REST APIs",
                "Optimize database performance",
                "Collaborate with frontend team"
            ]
        }
        print("\n  Running in DEMO mode with built-in sample data")
        run_test(profile, criteria)
        return

    if not args.profile or not args.jd:
        print("Error: provide --profile and --jd paths, or use --demo")
        parser.print_help()
        sys.exit(1)

    profile  = json.loads(Path(args.profile).read_text())
    criteria = json.loads(Path(args.jd).read_text())
    run_test(profile, criteria)


if __name__ == "__main__":
    main()
