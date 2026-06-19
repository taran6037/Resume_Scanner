# scripts/run_pipeline.py
#
# End-to-end test harness for layers 1, 2, and 3.
#
# Flow:
#   LAYER 1  file → validate → route → raw text
#   LAYER 2  raw text → clean → ner (regex, stays local) → Qwen (no PII)
#   LAYER 3  JD text → clean → Qwen → structured_criteria
#
# Usage:
#   python scripts/run_pipeline.py --demo
#   python scripts/run_pipeline.py --resume data/resumes/cv.pdf --jd "JD text here"
#   python scripts/run_pipeline.py --resume data/resumes/cv.pdf --jd data/resumes/jd.txt

import sys
import json
import argparse
import logging
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level   = logging.INFO,
    format  = "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt = "%H:%M:%S",
)
logger = logging.getLogger("run_pipeline")


# ─── Demo data ────────────────────────────────────────────────────────────────

DEMO_JD = """
Senior Backend Engineer - Platform Team
Tech Mahindra | Pune, India | Hybrid (3 days onsite)

We are building the next generation of our enterprise communication platform
serving 10M+ users. We need a Senior Backend Engineer who can own critical
services end to end — from design through deployment.

Requirements (Must Have):
- 3+ years of professional experience in backend development
- Strong proficiency in Python with FastAPI or Django REST Framework
- Deep understanding of PostgreSQL — query optimization, indexing, partitioning
- Redis for caching, pub/sub, and distributed locking
- Docker and Docker Compose for local and production environments
- Experience designing RESTful and event-driven APIs
- Solid understanding of authentication — JWT, OAuth2, API keys
- Git workflow — branching strategy, code reviews, PR conventions

Requirements (Nice to Have):
- Kubernetes or AWS ECS for container orchestration
- Message queues — RabbitMQ or Apache Kafka
- CI/CD pipelines using GitHub Actions or Jenkins
- Experience with vector databases (ChromaDB, Pinecone, Weaviate)
- Exposure to ML pipelines or LLM integrations (LangChain, Ollama)
- Monitoring — Prometheus, Grafana, ELK stack

Responsibilities:
- Own and lead the design of 2-3 core backend microservices
- Write production-grade Python code with full test coverage (pytest, 80%+)
- Optimize database performance — currently handling 500k queries/day
- Collaborate with frontend, ML team, and DevOps on system design
- Conduct technical interviews and mentor junior engineers

Education:
B.Tech or M.Tech in Computer Science or related field preferred.
"""

DEMO_RESUME_TEXT = """
Taranpreet Singh
taran@example.com | +91-9876543210
linkedin.com/in/taran-singh | github.com/taran-dev
Pune, Maharashtra, India

A backend developer with 2+ years of experience building scalable REST APIs
and distributed systems. Passionate about clean code, performance optimization,
and AI-powered applications.

Python, FastAPI, Django, PostgreSQL, Redis, Docker, Git, REST APIs, SQLAlchemy
GitHub Actions, VS Code, Postman, pgAdmin, Linux, pytest
Problem solving, Team collaboration, Technical documentation

Tech Mahindra — Backend Developer
June 2023 to Present, Pune India
Built and maintained REST API endpoints using FastAPI serving 50k requests per day
Reduced API response time by 35% by implementing Redis caching layer
Integrated PostgreSQL with SQLAlchemy ORM, optimized slow queries by 40%
Set up CI/CD pipeline using GitHub Actions, reducing deployment time by 60%
Participated in daily standups, code reviews, and sprint planning

Infosys — Software Engineer Intern
January 2023 to May 2023, Bangalore India
Developed internal HR management tool using Django and PostgreSQL
Wrote unit tests achieving 85% code coverage using pytest
Collaborated with a team of 5 engineers on API design and documentation

B.Tech Computer Science and Engineering
Punjab University, 2019 to 2023, CGPA 8.6 out of 10

AI Resume Scanner project
An intelligent hiring platform using Python FastAPI ChromaDB and Ollama
that automates resume screening with bias detection and explainability
github.com/taran-dev/resume-scanner

Redis Cache Optimizer project
A Python library for automatic cache invalidation and TTL management
Technologies used Python Redis Docker

AWS Certified Developer Associate 2024
Python for Data Science Coursera 2023
Docker Fundamentals Docker Inc 2023
"""


# ─── Resume pipeline (layers 1 + 2) ──────────────────────────────────────────

def run_resume_pipeline(resume_path: str | Path) -> dict:
    """
    Runs a resume file through layer 1 (ingestion) and layer 2 (parsing).
    Returns parsed_profile as a dict.
    """
    from pipeline.ingestion.validator import validate_file
    from pipeline.ingestion.router import route_file
    from pipeline.parsing.cleaner import clean_text
    from pipeline.parsing.ner_extractor import extract_entities
    from pipeline.parsing.structured_extractor import extract_structured_profile

    print_header("LAYER 1 — INGESTION")

    # ── Validate ──────────────────────────────────────────────────────────
    print(f"  Validating: {Path(resume_path).name}")
    validation = validate_file(resume_path)

    if not validation.is_valid:
        print(f"  ✗ Validation failed: {validation.error}")
        sys.exit(1)

    print(f"  ✓ Valid | {validation.file_size_mb} MB | {validation.extension}")

    if validation.warnings:
        for w in validation.warnings:
            print(f"  ⚠ {w}")

    # ── Route to parser ────────────────────────────────────────────────────
    print(f"  Routing to parser...")
    route = route_file(validation)
    print(f"  ✓ Parser used  : {route.parser_used}")
    print(f"    Words found  : {route.word_count}")
    print(f"    Scanned PDF  : {route.is_scanned}")

    if route.warnings:
        for w in route.warnings:
            print(f"  ⚠ {w}")

    print_header("LAYER 2 — PARSING")

    # ── Clean ─────────────────────────────────────────────────────────────
    print("  Cleaning text...")
    clean = clean_text(route.raw_text, was_ocr=route.is_scanned)
    print(f"  ✓ Cleaned | {clean.original_length} → {clean.cleaned_length} chars")

    if clean.injection_flags:
        print(f"  ⚠ SECURITY — injection patterns removed: {clean.injection_flags}")

    # ── NER (regex + spaCy — stays local, no Qwen) ────────────────────────
    print("  Extracting contact info (regex — stays local)...")
    ner = extract_entities(clean.clean_text)
    print(f"  ✓ Name    : {ner.contact.name}")
    print(f"    Email   : {ner.contact.email}")
    print(f"    Phone   : {ner.contact.phone}")
    print(f"    LinkedIn: {ner.contact.linkedin}")
    print(f"    GitHub  : {ner.contact.github}")
    print(f"    spaCy   : {'yes' if ner.spacy_used else 'no (regex-only mode)'}")
    print(f"  ✓ Contact info extracted — will NOT be sent to Qwen")

    # ── Qwen structured extraction (no PII, no keyword sections) ──────────
    print("  Sending clean text to Qwen (contact lines stripped)...")
    print("  ⏳ Qwen is thinking — this takes 20-60 seconds on first run...")
    profile = extract_structured_profile(
        ner_result   = ner,
        clean_text   = clean.clean_text,
        parser_version = "v1",
    )

    print(f"  ✓ Extraction complete!")
    print(f"    Technical skills : {profile.skills.technical}")
    print(f"    Tools            : {profile.skills.tools}")
    print(f"    Soft skills      : {profile.skills.soft}")
    print(f"    Experience       : {len(profile.experience)} jobs | "
          f"{profile.total_experience_years} years total")
    print(f"    Education        : {len(profile.education)} entries")
    print(f"    Projects         : {len(profile.projects)}")
    print(f"    Certifications   : {len(profile.certifications)}")
    print(f"    Confidence       : {profile.extraction_confidence}")

    return profile.model_dump()


# ─── JD pipeline (layer 3) ────────────────────────────────────────────────────

def run_jd_pipeline(jd_text: str) -> dict:
    """
    Runs JD text through layer 3.
    Returns structured_criteria as a dict.
    """
    from pipeline.jd.jd_extractor import extract_jd

    print_header("LAYER 3 — JD UNDERSTANDING")

    print("  Cleaning JD text...")
    print("  ⏳ Qwen is extracting JD criteria — 10-30 seconds...")

    criteria = extract_jd(jd_text)

    print(f"  ✓ JD extraction complete!")
    print(f"    Role type        : {criteria.get('role_type')}")
    print(f"    Seniority        : {criteria.get('seniority')}")
    print(f"    Experience min   : {criteria.get('experience_years')} years")
    print(f"    Required skills  : {criteria.get('skills', {}).get('required', [])}")
    print(f"    Preferred skills : {criteria.get('skills', {}).get('preferred', [])}")
    print(f"    Responsibilities : {len(criteria.get('responsibilities', []))} items")
    print(f"    Confidence       : {criteria.get('extraction_confidence')}")

    return criteria


# ─── Demo mode ────────────────────────────────────────────────────────────────

def run_demo_mode():
    """Runs the full pipeline with built-in sample data — no files needed."""
    import tempfile, os

    print("\n" + "="*60)
    print("  DEMO MODE — built-in sample resume + JD")
    print("="*60)

    resumes_dir = PROJECT_ROOT / "data" / "resumes"
    resumes_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".txt",
        delete=False,
        dir=resumes_dir,
        prefix="demo_resume_"
    ) as f:
        f.write(DEMO_RESUME_TEXT)
        demo_path = f.name

    try:
        profile  = run_resume_pipeline(demo_path)
        criteria = run_jd_pipeline(DEMO_JD)
        save_and_summarize(profile, criteria, "demo")
    finally:
        os.unlink(demo_path)


# ─── Save + summary ───────────────────────────────────────────────────────────

def save_and_summarize(profile: dict, criteria: dict, label: str):
    """Saves outputs to data/processed/ and prints final summary."""

    output_dir = PROJECT_ROOT / "data" / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp    = datetime.now().strftime("%Y%m%d_%H%M%S")
    profile_path = output_dir / f"{label}_profile_{timestamp}.json"
    jd_path      = output_dir / f"{label}_jd_{timestamp}.json"

    profile_path.write_text(json.dumps(profile, indent=2, default=str))
    jd_path.write_text(json.dumps(criteria, indent=2, default=str))

    print_header("PIPELINE COMPLETE")
    print(f"  Profile saved → {profile_path.relative_to(PROJECT_ROOT)}")
    print(f"  JD saved      → {jd_path.relative_to(PROJECT_ROOT)}")
    print()
    print("  RESUME SUMMARY")
    print(f"    Name     : {profile.get('contact', {}).get('name')}")
    print(f"    Email    : {profile.get('contact', {}).get('email')}")
    print(f"    Skills   : {profile.get('skills', {}).get('technical', [])[:5]}")
    print(f"    Exp yrs  : {profile.get('total_experience_years')}")
    print(f"    Jobs     : {len(profile.get('experience', []))}")
    print()
    print("  JD SUMMARY")
    print(f"    Role     : {criteria.get('role_type')}")
    print(f"    Seniority: {criteria.get('seniority')}")
    print(f"    Skills   : {criteria.get('skills', {}).get('required', [])[:5]}")
    print(f"    Exp min  : {criteria.get('experience_years')} years")
    print("="*60 + "\n")


# ─── Helpers ──────────────────────────────────────────────────────────────────

def print_header(title: str):
    print(f"\n{'─'*60}")
    print(f"  {title}")
    print(f"{'─'*60}")


def check_ollama():
    from pipeline.utils.ollama_client import check_ollama_health
    health = check_ollama_health()
    print("\n" + "="*60)
    print("  Ollama Health Check")
    print("="*60)
    if health["status"] == "ok":
        print(f"  ✓ {health['message']}")
    else:
        print(f"  ✗ {health['message']}")
        print("\n  Fix:")
        print("    1. ollama serve")
        print("    2. ollama pull qwen3:8b")
        sys.exit(1)


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="AI Resume Scanner — pipeline layers 1-3",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--resume", type=str,
                        help="Path to resume (.pdf / .docx / .doc / .txt)")
    parser.add_argument("--jd", type=str,
                        help="Path to JD text file OR raw JD text string")
    parser.add_argument("--demo", action="store_true",
                        help="Run with built-in sample data (no files needed)")
    parser.add_argument("--no-health-check", action="store_true",
                        help="Skip Ollama health check")

    args = parser.parse_args()

    if not args.no_health_check:
        check_ollama()

    if args.demo:
        run_demo_mode()
        return

    if not args.resume:
        print("Error: --resume is required (or use --demo)")
        parser.print_help()
        sys.exit(1)

    if not args.jd:
        print("Error: --jd is required (or use --demo)")
        parser.print_help()
        sys.exit(1)

    # JD can be a file path or raw text
    jd_path = Path(args.jd)
    jd_text = jd_path.read_text(encoding="utf-8") if jd_path.exists() else args.jd

    label    = Path(args.resume).stem
    profile  = run_resume_pipeline(args.resume)
    criteria = run_jd_pipeline(jd_text)
    save_and_summarize(profile, criteria, label)


if __name__ == "__main__":
    main()