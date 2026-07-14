# AI-Powered Resume Scanner & Candidate Matchmaking Platform

An end-to-end resume screening pipeline that runs entirely on local infrastructure. It ingests unstructured resumes and a job description, extracts structured data using a locally hosted LLM (Qwen3:8b via Ollama), verifies every AI claim against the raw source text, and produces a ranked, explained, recruiter-tunable shortlist of candidates — with per-candidate PDF reports and AI-generated interview questions.

**Core principle: the LLM proposes, deterministic code verifies and decides.** No unverified model claim ever reaches a score. No candidate data ever leaves the machine.

> Built for the TechnoRise 2.0 Internship Program (COMMS-4007) · Tech Mahindra
> **Team:** Taranpreet Singh, Vishesh Bhandari · **Manager:** Naveen Bathla · **Mentor:** Pankaj Bhagwat

---

## Features

- **Multi-format ingestion** — PDF, DOCX, DOC, TXT, with automatic OCR fallback (Tesseract) for scanned documents
- **Security-first validation** — file-size caps, extension whitelisting, and magic-byte signature verification before any parsing
- **Prompt-injection defense** — detects and strips hidden instruction-override text (including white-on-white tricks) before the model sees anything, logging each attempt
- **Schema-enforced AI extraction** — strict Pydantic-validated JSON output with a self-correcting retry loop that always re-sends full source context
- **Guardrailed semantic matching** — bidirectional skill validation against raw resume text, deterministic GPA comparison, institution abbreviation expansion, multi-entry degree scanning
- **Recruiter-tunable scoring** — three weight sliders (Skills / Experience / LLM semantic) that must sum to 100%, flowing end-to-end into the fit score
- **Education hard gate** — a nine-option requirement selector (No requirement → PhD required); candidates below a *required* threshold are excluded from the ranking, not just penalized. Unverifiable education passes with a manual-review flag, never silent exclusion
- **Explainable dashboard** — every fit score decomposed into components, text-verified skill chips, education verification seals, plain-language reasoning
- **Per-candidate PDF reports** — with an optional AI-generated interview question bank (15 questions across 4 categories, anchored to the candidate's actual employers and projects)
- **Collective ranking report** — one-click PDF with a cover summary table and a page per candidate
- **Fully local & deterministic** — CPU-pinned inference at temperature 0; identical inputs produce identical rankings, and nothing is sent to any external API

---

## Architecture

```
Resume files ──► [1] Ingestion & Validation      (validator.py, router.py)
                      │  size/extension/magic-byte checks, OCR routing
                      ▼
                 [2] Parsing & Sanitization      (pdf/docx/ocr_parser.py, cleaner.py, ner_extractor.py)
                      │  injection defense, normalization, contact separation
                      ▼
JD text ───────► [3] JD Processing               (jd_reader.py, jd_extractor.py)
                      │  boilerplate stripping, structured criteria + weights
                      ▼
                 [4] AI Structured Extraction    (structured_extractor.py, ollama_client.py)
                      │  schema-enforced JSON, retry loop, section-aware truncation
                      ▼
                 [5] Semantic Matching           (semantic_matcher.py)
                      │  guardrails: skill verification, deterministic overrides
                      ▼
                 [6] Scoring & Ranking           (scoring.py)
                      │  weighted fit score → education hard gate → rank
                      ▼
                 [7] Dashboard & Reports         (demo_server.py, dashboard.html, job_intake.html)
                         score breakdown, PDF reports, interview questions
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Language model | Qwen3:8b via Ollama (CPU-only, local) |
| Backend | Python 3.12, FastAPI, Uvicorn |
| Schema validation | Pydantic v2 |
| PDF parsing | PyMuPDF, pdfplumber |
| DOCX parsing | python-docx |
| OCR | Tesseract (pytesseract) |
| NER | spaCy (`en_core_web_lg`) |
| PDF generation | reportlab |
| Frontend | HTML / CSS / vanilla JavaScript (served by FastAPI) |
| Config | Central `config/pipeline_config.py` |

---

## Project Structure

```
resumescanner/
├── backend/
│   ├── demo_server.py          # FastAPI app — all routes, PDF generation
│   └── schemas/
│       ├── candidate.py        # Candidate profile Pydantic models
│       └── job.py              # StructuredCriteria, ScoringWeights, education levels
├── config/
│   └── pipeline_config.py      # Every tunable parameter, single source of truth
├── data/
│   ├── resumes/                # Drop resume files here
│   ├── processed/              # Batch ranking JSON outputs
│   └── prompts/
│       ├── jd_extraction_v1.txt
│       └── semantic_matching_v1.txt
├── frontend/
│   ├── job_intake.html         # JD form, weight sliders, education selector
│   └── dashboard.html          # Ranking view, reports, interview questions
├── pipeline/
│   ├── ingestion/              # validator.py, router.py
│   ├── parsing/                # pdf/docx/ocr parsers, cleaner, NER,
│   │                           # section_detector, structured_extractor
│   ├── jd/                     # jd_reader.py, jd_extractor.py
│   ├── matching/               # semantic_matcher.py, scoring.py,
│   │                           # batch_runner.py, interview_questions.py
│   └── utils/
│       └── ollama_client.py    # Shared LLM client — schema enforcement, retries
├── scripts/
│   └── batch_rank.py           # CLI batch runner (no server needed)
├── debug.py                    # Single-resume diagnostic harness
├── docker-compose.yml
├── requirements.txt
└── README.md
```

---

## Setup

### Prerequisites

- **Python 3.12+**
- **Ollama** — [ollama.com](https://ollama.com)
- **Tesseract OCR** — required only for scanned PDFs
  - Windows: [UB Mannheim builds](https://github.com/UB-Mannheim/tesseract/wiki)
- ~8 GB free RAM for CPU inference

### Installation

```bash
# 1. Clone and enter the project
git clone <repo-url>
cd resumescanner

# 2. Install Python dependencies
pip install -r requirements.txt

# 3. Download the spaCy model
python -m spacy download en_core_web_lg

# 4. Pull the LLM
ollama pull qwen3:8b
```

### Running

```bash
# Terminal 1 — start Ollama
ollama serve

# Terminal 2 — start the web server
python -m uvicorn backend.demo_server:app --reload
```

Then open:

| Page | URL |
|---|---|
| Job intake (start here) | http://127.0.0.1:8000/intake |
| Ranking dashboard | http://127.0.0.1:8000/dashboard |

**Usage:** drop resumes into `data/resumes/`, paste a job description on the intake page, set your scoring weights and education requirement, and click **Run on all resumes**. Results appear on the dashboard when processing completes.

### CLI alternative (no server)

```bash
python scripts/batch_rank.py --jd "path/to/job_description.txt"
# or pass the JD text directly as the argument
```

Prints per-resume progress, the final ranking table, and a detailed per-candidate breakdown to the terminal. Output JSON is saved to `data/processed/`.

### Single-resume debugging

```bash
python debug.py "data/resumes/some_resume.pdf"
```

Runs one resume through parsing → section detection → extraction only (no JD, no matching), showing detected sections, the raw education text found, and the final extracted fields. Invaluable for diagnosing extraction issues on a specific file.

---

## How Scoring Works

Each candidate receives four signals, of which three are weighted:

| Component | Default weight | Source |
|---|---|---|
| **Skill score** | 40% | Computed by code from text-verified skill matches/gaps. Required-tier skills weighted 75/25 over preferred within the component; partial "transferable credit" for closely related skills |
| **Experience score** | 34% | 60% model relevance judgment + 40% deterministic years-vs-required ratio; neutral default when years can't be extracted |
| **LLM semantic score** | 26% | The model's guardrailed holistic fit judgment (0–100, hard floor applied) |
| **Education** | — (gate) | Not a weight. Checked after scoring against the selected requirement level; failing a *required* level excludes the candidate from the ranking |

`fit_score = skills×w₁ + experience×w₂ + semantic×w₃`, with weights normalized to sum to 1.0. Excluded candidates remain fully scored and visible in a collapsible dashboard section.

---

## Key Configuration (`config/pipeline_config.py`)

| Setting | Default | Purpose |
|---|---|---|
| `RESUME_TEXT_LIMIT` | 2500 | Character budget for model input (section-aware: education/skills always survive) |
| `LLM_NUM_PREDICT` | 2048 | Max output tokens — raised from 1024 after skill-dense resumes truncated mid-JSON |
| `LLM_NUM_CTX` | 4096 | Model context window |
| `OLLAMA_TIMEOUT` | 480s | Per-request ceiling, tuned for CPU inference |
| `OLLAMA_MAX_RETRIES` | 3 | Schema-validation retry attempts (each re-sends full source text) |
| `SCORE_FLOOR` | 8 | Minimum semantic score, prevents single harsh judgments |
| `DEFAULT_WEIGHT_*` | 0.40/0.34/0.26 | Skills / Experience / LLM defaults |

CPU inference is pinned at the request level (`"num_gpu": 0`) — do not remove this on low-VRAM machines; partial GPU offload of an 8B model on ≤4 GB VRAM causes crashes and timeouts.

---

## API Reference

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/run` | Start a batch. Body: `{jd_text, scoring_weights: {skills, experience, llm, education_requirement}}` |
| `GET` | `/api/run-status` | Poll batch progress (`status`, `current`, `total`, `current_file`) |
| `GET` | `/api/rankings/latest` | Latest batch result — criteria, ranked + excluded candidates |
| `GET/POST` | `/api/candidate-report/{pool}/{index}` | Per-candidate PDF. `?include_questions=true` generates questions fresh; POST body `{questions}` reuses cached ones |
| `GET` | `/api/full-report` | Collective PDF — cover summary + one page per ranked candidate |
| `POST` | `/api/interview-questions/{pool}/{index}` | Generate 15 resume-anchored interview questions |

`{pool}` is `ranking` or `excluded`; `{index}` is the candidate's position in that list.

---

## Known Constraints

- **CPU-only inference is deliberately slow-but-stable** — a few minutes per batch on typical hardware. The design ports unchanged to GPU-equipped machines.
- **Context window bounds per-call resume text** — mitigated by section-aware truncation that guarantees high-signal sections survive.
- **Extraction quality depends on source quality** — OCR fallback and OCR-aware cleaning help, but unusual layouts can still degrade field extraction (hence the manual-review flag rather than silent exclusion on unverifiable education).
- Persistent audit logging (PostgreSQL) and authentication are scoped for the production-deployment phase; batch results currently persist as JSON in `data/processed/`.

---

## Version History

| Tag | Milestone |
|---|---|
| `v1.0.0` | Core pipeline validated end-to-end (ingestion → ranking → dashboard) |
| `v1.1.0` | Education hard gate, per-candidate & collective PDF reports, interview-question generation, weight-pipeline fixes |

---

*TechnoRise 2.0 · COMMS-4007 · Tech Mahindra*
