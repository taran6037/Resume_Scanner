# config/pipeline_config.py
#
# CENTRAL CONFIGURATION FILE — ALL PIPELINE PARAMETERS LIVE HERE
#
# This is the single source of truth for every tunable value in the pipeline.
# If you need to change any limit, threshold, model name, or setting —
# change it here and nowhere else. Every pipeline file imports from here.
#
# SECTIONS:
#   1. File ingestion settings    (validator.py, router.py)
#   2. PDF parsing settings       (pdf_parser.py)
#   3. OCR settings               (ocr_parser.py)
#   4. DOCX parsing settings      (docx_parser.py)
#   5. Text cleaning settings     (cleaner.py)
#   6. NER extraction settings    (ner_extractor.py)
#   7. Resume extraction settings (structured_extractor.py)
#   8. JD processing settings     (jd_reader.py, jd_extractor.py)
#   9. Ollama / LLM settings      (ollama_client.py)
#  10. Scoring weights            (job.py schema)
#  11. Pipeline metadata          (run_pipeline.py)


# ═══════════════════════════════════════════════════════════════════════════════
# 1. FILE INGESTION SETTINGS
# Used by: pipeline/ingestion/validator.py, pipeline/ingestion/router.py
# ═══════════════════════════════════════════════════════════════════════════════

# Maximum resume file size allowed — files larger than this are rejected.
# Increase if you expect very long resumes with embedded images.
MAX_FILE_SIZE_MB = 10

# Derived value — do not change this, change MAX_FILE_SIZE_MB above.
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024

# File extensions the pipeline accepts.
# Any file with an extension not in this set is rejected at the validator.
# Add new formats here if you want to support them in future.
ALLOWED_EXTENSIONS = {".pdf", ".docx", ".doc", ".txt"}

# Magic byte signatures — the first few bytes of a file that identify its
# true format. Used to catch files that have been renamed (e.g. a .exe
# disguised as a .pdf). None means no magic byte check (e.g. .txt files).
MAGIC_BYTES = {
    ".pdf":  [b"%PDF"],            # PDF always starts with %PDF
    ".docx": [b"PK\x03\x04"],     # DOCX is a ZIP — starts with PK
    ".doc":  [b"\xd0\xcf\x11\xe0"], # Legacy DOC — Compound File Binary
    ".txt":  None,                 # Plain text has no magic bytes
}

# If the text extracted from a PDF has fewer than this many characters,
# the router treats it as a scanned PDF and falls back to OCR.
# Set lower if legitimate resumes are being incorrectly sent to OCR.
OCR_FALLBACK_THRESHOLD = 100


# ═══════════════════════════════════════════════════════════════════════════════
# 2. PDF PARSING SETTINGS
# Used by: pipeline/parsing/pdf_parser.py
# ═══════════════════════════════════════════════════════════════════════════════

# If a parser returns fewer than this many characters per page on average,
# it is treated as having failed and the other parser is tried.
# A real resume page always has well over 50 characters.
MIN_CHARS_PER_PAGE = 50

# pdfplumber character grouping tolerances (in points/pixels).
# x_tolerance: how close two characters must be horizontally to be the same word.
# y_tolerance: how close two characters must be vertically to be the same line.
# Increase if words are being split incorrectly. Decrease if words are merging.
PDF_X_TOLERANCE = 3
PDF_Y_TOLERANCE = 3


# ═══════════════════════════════════════════════════════════════════════════════
# 3. OCR SETTINGS
# Used by: pipeline/parsing/ocr_parser.py
# ═══════════════════════════════════════════════════════════════════════════════

# Resolution to render PDF pages before running OCR.
# 300 DPI is the standard minimum for good OCR accuracy.
# Increase to 400+ for better accuracy on low-quality scans (slower).
# Decrease to 150-200 for faster processing if scan quality is good.
OCR_RENDER_DPI = 300

# Tesseract engine and page segmentation mode.
# --oem 3: use LSTM neural net engine (most accurate, requires tessdata)
# --psm 1: automatic page segmentation with orientation detection
#          best for mixed-layout documents like resumes
# Change --psm to 3 (fully automatic, no OSD) if OSD causes errors.
TESSERACT_CONFIG = "--oem 3 --psm 1"

# Language pack Tesseract uses for OCR.
# "eng" = English. Add more with "eng+hin" for Hindi+English resumes.
TESSERACT_LANG = "eng"


# ═══════════════════════════════════════════════════════════════════════════════
# 4. DOCX PARSING SETTINGS
# Used by: pipeline/parsing/docx_parser.py
# ═══════════════════════════════════════════════════════════════════════════════

# Estimated average words per page in a resume.
# Used to calculate approximate page count (DOCX does not store it).
# Most resumes have 400-500 words per page.
DOCX_WORDS_PER_PAGE = 450


# ═══════════════════════════════════════════════════════════════════════════════
# 5. TEXT CLEANING SETTINGS
# Used by: pipeline/parsing/cleaner.py
# ═══════════════════════════════════════════════════════════════════════════════

# Maximum number of consecutive blank lines allowed in cleaned text.
# More than this are collapsed to MAX_BLANK_LINES blank lines.
MAX_BLANK_LINES = 2

# Minimum length of a garbage OCR line (all symbols, no letters/digits).
# Lines shorter than this are kept even if they look like symbols.
MIN_GARBAGE_LINE_LENGTH = 5


# ═══════════════════════════════════════════════════════════════════════════════
# 6. NER EXTRACTION SETTINGS
# Used by: pipeline/parsing/ner_extractor.py
# ═══════════════════════════════════════════════════════════════════════════════

# spaCy models to try in order of preference (most accurate first).
# The first one found installed on the machine is used.
# If none are found, the extractor falls back to regex-only mode.
SPACY_MODELS = ["en_core_web_lg", "en_core_web_md", "en_core_web_sm"]

# How many characters of the resume to send to spaCy for NER.
# Names and company names are always near the top of a resume.
# Processing the full text would be slow with no benefit.
SPACY_TEXT_LIMIT = 3000

# Maximum number of organization names to collect from spaCy.
# Beyond this the list becomes noise.
MAX_ORGS_DETECTED = 10

# Valid word count range for a candidate name.
# Names shorter than MIN or longer than MAX words are rejected.
# Prevents single words or full sentences being treated as names.
NAME_MIN_WORDS = 2
NAME_MAX_WORDS = 4

# How many lines from the top of the resume to search for the name
# when using the heuristic fallback (no spaCy).
NAME_HEURISTIC_LINES = 5

# Maximum characters a line can have to be considered a name line.
# Lines longer than this are summaries or descriptions, not names.
NAME_MAX_LINE_LENGTH = 60

# Only strip the candidate's name from the first N lines of the resume
# before sending text to Qwen. Beyond these lines, the name might appear
# as part of a company name or reference and should not be removed.
CONTACT_STRIP_NAME_LINES = 3


# ═══════════════════════════════════════════════════════════════════════════════
# 7. RESUME EXTRACTION SETTINGS
# Used by: pipeline/parsing/structured_extractor.py
# ═══════════════════════════════════════════════════════════════════════════════

# Maximum characters of resume text sent to Qwen for extraction.
# 2500 chars ≈ 600-700 tokens — enough for a full resume.
# Increase carefully — larger 8B models degrade on very long prompts.
RESUME_TEXT_LIMIT = 2500

# Parser version string stored in every ParsedProfile for traceability.
# Increment this when you make significant changes to the parsing logic
# so you can identify which version produced any given profile in the DB.
PARSER_VERSION = "v1"

# Confidence score thresholds — stored in ParsedProfile.extraction_confidence.
# Scores at or below CONFIDENCE_REVIEW_THRESHOLD flag the record for manual review.
CONFIDENCE_ATTEMPT_1 = 1.0   # succeeded on first try — highest confidence
CONFIDENCE_ATTEMPT_2 = 0.8   # succeeded on second try after one retry
CONFIDENCE_ATTEMPT_3 = 0.6   # succeeded on third try — lowest acceptable
CONFIDENCE_REVIEW_THRESHOLD = 0.6   # at or below this → flag for manual review


# ═══════════════════════════════════════════════════════════════════════════════
# 8. JD PROCESSING SETTINGS
# Used by: pipeline/jd/jd_reader.py, pipeline/jd/jd_extractor.py
# ═══════════════════════════════════════════════════════════════════════════════

# Minimum JD length — anything shorter is probably not a real job description.
MIN_JD_LENGTH_CHARS = 50
MIN_JD_LENGTH_WORDS = 20

# Maximum JD length sent to Qwen.
# JDs longer than this are truncated to keep the model focused.
# We cut at the last sentence ending within this limit.
MAX_JD_LENGTH_CHARS = 6000

# How far back from the truncation point we search for a sentence boundary.
# e.g. 0.8 means we search the last 20% of the truncated text for a period.
JD_TRUNCATION_SENTENCE_SEARCH = 0.8

# Path to the JD extraction prompt template file.
# Version in the filename — increment when you update the prompt.
JD_PROMPT_PATH = "data/prompts/jd_extraction_v1.txt"


# ═══════════════════════════════════════════════════════════════════════════════
# 9. OLLAMA / LLM SETTINGS
# Used by: pipeline/utils/ollama_client.py
# ═══════════════════════════════════════════════════════════════════════════════

# URL where Ollama is running. Change this if Ollama runs on a different host
# (e.g. a separate server on your network).
OLLAMA_BASE_URL = "http://localhost:11434"

# The model to use for all LLM calls.
# Change this to switch models — e.g. "llama3:8b" or "mistral:7b".
OLLAMA_MODEL = "qwen3:8b"

# How long to wait for Ollama to respond before giving up (seconds).
# Increase on slow machines. Decrease if you want faster failure detection.
OLLAMA_TIMEOUT = 300

# Maximum number of times to retry a failed LLM call.
# Each retry uses a corrective prompt showing the model what went wrong.
OLLAMA_MAX_RETRIES = 3

# Seconds to wait between retry attempts.
# Gives the model time to recover between calls.
OLLAMA_RETRY_DELAY = 2

# LLM inference parameters.
# temperature: 0.0 = fully deterministic output (best for extraction tasks).
#              Higher values add randomness — not suitable for structured extraction.
# num_predict: maximum tokens to generate in one response.
#              Resume JSON fits in ~500 tokens. 1024 gives enough headroom.
# num_ctx:     context window size in tokens. 4096 is enough for our prompts.
#              Larger values use more RAM. Do not increase on low-memory machines.
LLM_TEMPERATURE = 0.0
LLM_NUM_PREDICT = 1024
LLM_NUM_CTX     = 4096

# Whether to disable Qwen3's internal thinking/reasoning mode.
# True = faster responses (recommended for extraction tasks).
# False = Qwen reasons before answering (useful for complex reasoning, slow).
LLM_DISABLE_THINKING = True


# ═══════════════════════════════════════════════════════════════════════════════
# 10. SCORING WEIGHTS
# Used by: backend/schemas/job.py (ScoringWeights default)
# ═══════════════════════════════════════════════════════════════════════════════

# Default weights for candidate scoring — must sum to 1.0.
# Recruiters can override these per job via the API.
# These match the architecture diagram exactly.
DEFAULT_WEIGHT_SKILLS     = 0.40   # 40% — technical skill match
DEFAULT_WEIGHT_EXPERIENCE = 0.25   # 25% — years + relevance of experience
DEFAULT_WEIGHT_LLM        = 0.25   # 25% — Qwen semantic reasoning score
DEFAULT_WEIGHT_EDUCATION  = 0.10   # 10% — degree + field match


# ═══════════════════════════════════════════════════════════════════════════════
# 11. PIPELINE METADATA
# Used by: scripts/run_pipeline.py
# ═══════════════════════════════════════════════════════════════════════════════

# Where to read resume files from.
RESUMES_INPUT_DIR = "data/resumes"

# Where to save parsed profile and JD JSON outputs.
PROCESSED_OUTPUT_DIR = "data/processed"

# Where prompt template files are stored.
PROMPTS_DIR = "data/prompts"




# ═══════════════════════════════════════════════════════════════════════════════
# 12. EMBEDDING SETTINGS
# Used by: pipeline/matching/embedder.py
# ═══════════════════════════════════════════════════════════════════════════════

# Sentence Transformers model for generating embeddings.
# MiniLM-L6-v2 produces 384-dim vectors — fast, good quality, runs on CPU.
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Device to run the model on. "cpu" works on all machines.
# Change to "cuda" if you have a GPU for faster embedding.
EMBEDDING_DEVICE = "cpu"


# ═══════════════════════════════════════════════════════════════════════════════
# 13. CHROMADB SETTINGS
# Used by: pipeline/matching/vector_store.py
# ═══════════════════════════════════════════════════════════════════════════════

# ChromaDB HTTP client settings — connects to the Docker container.
CHROMA_HOST = "localhost"
CHROMA_PORT = 8001

# Collection names — one for candidates, one for jobs.
CHROMA_CANDIDATE_COLLECTION = "candidates"
CHROMA_JD_COLLECTION        = "jobs"

# How many similar candidates to return in a search by default.
CHROMA_TOP_K = 10


# ═══════════════════════════════════════════════════════════════════════════════
# 14. SEMANTIC MATCHING SETTINGS
# Used by: pipeline/matching/semantic_matcher.py
# ═══════════════════════════════════════════════════════════════════════════════

# Path to the semantic matching prompt template.
SEMANTIC_MATCHING_PROMPT_PATH = "data/prompts/semantic_matching_v1.txt"

# Score below which a match is considered poor and flagged.
SEMANTIC_SCORE_MIN_THRESHOLD = 30