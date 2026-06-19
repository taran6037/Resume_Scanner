# pipeline/utils/ollama_client.py
#
# PURPOSE: Shared Ollama client used by every LLM call in the pipeline.
#
# USED BY:
#   pipeline/parsing/structured_extractor.py  — resume -> parsed_profile JSON
#   pipeline/jd/jd_extractor.py               — JD text -> structured_criteria JSON
#
# WHAT THIS FILE HANDLES:
#   1. Making the HTTP call to the local Ollama server
#   2. Forcing JSON output (Ollama format="json" mode)
#   3. Extracting valid JSON from messy LLM responses
#   4. Validating the JSON against a Pydantic schema
#   5. Retrying up to 3 times with a corrective prompt on failure
#   6. Returning a confidence score based on which attempt succeeded
#
# ALL TUNABLE VALUES (URL, model, timeout, retries, LLM params) come from
# config/pipeline_config.py — do not hardcode anything here.

import json
import re
import time
import logging
from typing import Optional, Type
import requests
from pydantic import BaseModel, ValidationError

# All configuration imported from the central config file.
# To change model, timeout, retries, or LLM parameters — edit pipeline_config.py.
from config.pipeline_config import (
    OLLAMA_BASE_URL,        # URL where Ollama is running
    OLLAMA_MODEL,           # model name e.g. "qwen3:8b"
    OLLAMA_TIMEOUT,         # seconds before giving up on a response
    OLLAMA_MAX_RETRIES,     # how many times to retry on failure
    OLLAMA_RETRY_DELAY,     # seconds to wait between retries
    LLM_TEMPERATURE,        # 0.0 = deterministic, higher = more random
    LLM_NUM_PREDICT,        # max tokens to generate per response
    LLM_NUM_CTX,            # context window size in tokens
    LLM_DISABLE_THINKING,   # True = disable Qwen3 thinking mode (faster)
    CONFIDENCE_ATTEMPT_1,   # confidence score if succeeded on attempt 1
    CONFIDENCE_ATTEMPT_2,   # confidence score if succeeded on attempt 2
    CONFIDENCE_ATTEMPT_3,   # confidence score if succeeded on attempt 3
)

logger = logging.getLogger(__name__)


# ─── Core Ollama call ─────────────────────────────────────────────────────────

def call_ollama(
    prompt: str,
    system_prompt: str,
    temperature: float = LLM_TEMPERATURE,
    timeout: int = OLLAMA_TIMEOUT,
) -> str:
    """
    Makes a single raw HTTP call to the Ollama /api/generate endpoint.
    Returns the model's response as a plain string.

    Args:
        prompt:        The user prompt to send to the model.
        system_prompt: Standing instruction that sets the model's role.
        temperature:   Controls randomness. 0.0 = fully deterministic.
                       Default comes from config (LLM_TEMPERATURE = 0.0).
        timeout:       How many seconds to wait before giving up.
                       Default comes from config (OLLAMA_TIMEOUT = 300).

    Returns:
        The model's response as a string — may be JSON, text, or garbage.
        Callers use extract_json() to parse it.

    Raises:
        OllamaConnectionError — if Ollama is not running or not reachable.
        OllamaTimeoutError    — if Ollama does not respond in time.
        OllamaError           — for any other HTTP error.
    """

    # Build the request payload.
    # format="json" tells Ollama to constrain output to valid JSON structure.
    # This is Ollama's built-in JSON mode — reduces but does not eliminate bad output.
    payload = {
        "model":  OLLAMA_MODEL,
        "prompt": prompt,
        "system": system_prompt,
        "format": "json",      # Ollama JSON mode — forces JSON structure
        "stream": False,       # get the full response at once, not token by token
        "think":  not LLM_DISABLE_THINKING,   # False = disable Qwen3 thinking mode

        # LLM inference parameters — all from config/pipeline_config.py
        "options": {
            "temperature": temperature,     # 0.0 = deterministic extraction
            "num_predict": LLM_NUM_PREDICT, # max tokens to generate (1024)
            "num_ctx":     LLM_NUM_CTX,     # context window (4096 tokens)
            "top_p":       0.9,             # nucleus sampling threshold
        }
    }

    try:
        # POST the request to Ollama running locally.
        # timeout=timeout applies to the entire request — connection + response.
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json=payload,
            timeout=timeout
        )

        # raise_for_status() throws an exception if HTTP status is 4xx or 5xx.
        response.raise_for_status()

        # Parse the response JSON and extract the model's text.
        # .get("response", "") returns empty string if "response" key is missing.
        # .strip() removes leading/trailing whitespace.
        data = response.json()
        return data.get("response", "").strip()

    except requests.exceptions.ConnectionError:
        # Ollama is not running or the URL is wrong.
        raise OllamaConnectionError(
            f"Cannot reach Ollama at {OLLAMA_BASE_URL}. "
            "Make sure Ollama is running: `ollama serve`"
        )
    except requests.exceptions.Timeout:
        # Model took too long to respond.
        # Common on slow machines or with very long prompts.
        raise OllamaTimeoutError(
            f"Ollama did not respond within {timeout}s. "
            "Try a shorter prompt or increase OLLAMA_TIMEOUT in pipeline_config.py."
        )
    except requests.exceptions.HTTPError as e:
        # Any other HTTP error — 404 (model not found), 500 (server error), etc.
        raise OllamaError(f"Ollama HTTP error: {e}")


# ─── JSON extraction ──────────────────────────────────────────────────────────

def extract_json(text: str) -> Optional[dict]:
    """
    Pulls a valid JSON object out of a model response.

    Handles three real output patterns from Qwen:
      Pattern 1 — JSON wrapped in markdown fences:  ```json { ... } ```
      Pattern 2 — JSON embedded in explanation text: "Here is the result: {...}"
      Pattern 3 — Clean JSON with nothing else:      { "key": "value" }

    Args:
        text: Raw string from call_ollama().

    Returns:
        A Python dict if valid JSON was found, None otherwise.
    """
    if not text:
        return None

    # Pattern 1: strip markdown code fences if present.
    # Qwen sometimes wraps output in ```json ... ``` despite being told not to.
    # re.DOTALL makes "." match newlines too — needed for multi-line JSON.
    fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence_match:
        text = fence_match.group(1)   # extract just the JSON part

    # Pattern 2: find the outermost { } block in the text.
    # This handles "Here is the extracted data: { ... } Hope that helps!"
    # We grab everything from the first { to the last }.
    brace_match = re.search(r"\{.*\}", text, re.DOTALL)
    if brace_match:
        candidate = brace_match.group(0)
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass   # not valid JSON — fall through to pattern 3

    # Pattern 3: try the entire text as-is.
    # This handles the ideal case where output is pure clean JSON.
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None   # nothing worked — return None, caller will retry


# ─── Validated call with retry ────────────────────────────────────────────────

def call_ollama_for_schema(
    prompt: str,
    system_prompt: str,
    schema: Type[BaseModel],
    context_label: str = "extraction",
) -> tuple[BaseModel, float]:
    """
    The main function used by structured_extractor and jd_extractor.
    Calls Ollama, extracts JSON, validates against a Pydantic schema,
    and retries with a corrective prompt on failure.

    Args:
        prompt:        The extraction prompt with resume or JD text injected.
        system_prompt: Standing role instruction for the model.
        schema:        Pydantic model class to validate the response against.
                       e.g. _LLMExtractionOutput or StructuredCriteria
        context_label: Used in log messages to identify which call this is.
                       e.g. "resume extraction" or "JD extraction"

    Returns:
        (validated_model_instance, confidence_score)
        confidence_score: 1.0 on first attempt, 0.8 on second, 0.6 on third.
        A lower score flags the record for manual review.

    Raises:
        ExtractionFailedError if all retry attempts are exhausted.
        OllamaConnectionError / OllamaTimeoutError passed through immediately.
    """
    last_error: Optional[str] = None
    last_raw:   Optional[str] = None

    # Loop through attempts. OLLAMA_MAX_RETRIES comes from config (currently 3).
    for attempt in range(1, OLLAMA_MAX_RETRIES + 1):

        # On the first attempt use the original prompt.
        # On retries use a corrective prompt that shows the model what went wrong.
        active_prompt = (
            _build_corrective_prompt(last_raw, last_error, schema)
            if attempt > 1
            else prompt
        )

        logger.debug(f"[{context_label}] Attempt {attempt}/{OLLAMA_MAX_RETRIES}")

        # Make the Ollama call.
        # If it throws a connection or timeout error we do not retry —
        # those are infrastructure errors, not model output errors.
        try:
            raw_response = call_ollama(active_prompt, system_prompt)
            last_raw     = raw_response   # save for corrective prompt on next retry
        except (OllamaConnectionError, OllamaTimeoutError):
            raise   # immediately re-raise — no point retrying infra errors

        # Try to extract a JSON object from the response.
        parsed_dict = extract_json(raw_response)

        if parsed_dict is None:
            # No valid JSON found in the response at all.
            last_error = "Response contained no valid JSON object."
            logger.warning(
                f"[{context_label}] Attempt {attempt}: no JSON found. "
                f"Raw response (first 200 chars): {raw_response[:200]}"
            )
            # Wait before retrying — OLLAMA_RETRY_DELAY from config (currently 2s).
            if attempt < OLLAMA_MAX_RETRIES:
                time.sleep(OLLAMA_RETRY_DELAY)
            continue   # move to next attempt

        # JSON was found — now validate it against the Pydantic schema.
        # This catches structural errors: wrong types, missing fields, etc.
        try:
            validated  = schema(**parsed_dict)   # throws ValidationError if invalid
            confidence = _confidence_from_attempt(attempt)
            logger.info(
                f"[{context_label}] Succeeded on attempt {attempt}. "
                f"Confidence: {confidence}"
            )
            return validated, confidence   # success — return immediately

        except ValidationError as e:
            # JSON was valid but did not match our schema structure.
            last_error = str(e)
            logger.warning(
                f"[{context_label}] Attempt {attempt}: Pydantic validation failed. "
                f"Errors: {last_error[:300]}"
            )
            if attempt < OLLAMA_MAX_RETRIES:
                time.sleep(OLLAMA_RETRY_DELAY)
            continue   # move to next attempt

    # All attempts exhausted without a valid response.
    # Raise so the pipeline marks this record as status="failed".
    raise ExtractionFailedError(
        f"[{context_label}] Failed after {OLLAMA_MAX_RETRIES} attempts. "
        f"Last error: {last_error}. "
        f"Last raw response (first 500 chars): {str(last_raw)[:500]}"
    )


# ─── Corrective prompt builder ────────────────────────────────────────────────

def _build_corrective_prompt(
    last_raw:   Optional[str],
    last_error: Optional[str],
    schema:     Type[BaseModel],
) -> str:
    """
    Builds a retry prompt that tells the model exactly what went wrong
    and shows the correct structure it should return.

    More effective than repeating the original prompt because it gives
    the model specific feedback about the failure.
    """
    # Generate a minimal example of the correct structure from the schema.
    schema_example = _schema_to_example(schema)

    return f"""Your previous response was invalid.
Error: {last_error}
Previous response: {str(last_raw)[:300]}

Return ONLY a valid JSON object matching this structure exactly:
{json.dumps(schema_example, indent=2)}"""


def _schema_to_example(schema: Type[BaseModel]) -> dict:
    """
    Generates a minimal placeholder dict from a Pydantic schema.
    Used to show the model what structure is expected on retry.
    Each field gets a type-appropriate placeholder value.
    """
    example = {}
    for field_name, field_info in schema.model_fields.items():
        annotation = field_info.annotation

        # Map Python type annotations to appropriate placeholder values.
        if annotation in (str, Optional[str]):
            example[field_name] = f"<{field_name}>"   # e.g. "<company>"
        elif annotation in (int, Optional[int]):
            example[field_name] = 0
        elif annotation in (float, Optional[float]):
            example[field_name] = 0.0
        elif annotation in (bool,):
            example[field_name] = False
        elif hasattr(annotation, "__origin__") and annotation.__origin__ is list:
            example[field_name] = []                   # empty list for list fields
        else:
            example[field_name] = {}                   # empty dict for nested models

    return example


# ─── Confidence scoring ───────────────────────────────────────────────────────

def _confidence_from_attempt(attempt: int) -> float:
    """
    Returns a confidence score based on which attempt succeeded.
    All scores come from config/pipeline_config.py so they can be tuned centrally.

    Stored in ParsedProfile.extraction_confidence in the database.
    Records at or below CONFIDENCE_REVIEW_THRESHOLD are flagged for manual review.
    """
    scores = {
        1: CONFIDENCE_ATTEMPT_1,   # 1.0 — first attempt, highest confidence
        2: CONFIDENCE_ATTEMPT_2,   # 0.8 — one retry needed
        3: CONFIDENCE_ATTEMPT_3,   # 0.6 — two retries needed, lowest acceptable
    }
    # 0.5 fallback for any attempt beyond 3 — should not happen but safe.
    return scores.get(attempt, 0.5)


# ─── Health check ─────────────────────────────────────────────────────────────

def check_ollama_health() -> dict:
    """
    Checks whether Ollama is running and the configured model is available.
    Called by run_pipeline.py at startup before any extraction begins.

    Returns a dict with:
        "status":  "ok" or "error"
        "model":   the model name from config
        "message": human-readable description of the result
    """
    try:
        # GET /api/tags returns all models installed in Ollama.
        # timeout=5 — should be near-instant if Ollama is running.
        response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        response.raise_for_status()
        data = response.json()

        # Check if our configured model is in the list of installed models.
        available_models = [m["name"] for m in data.get("models", [])]
        model_available  = any(OLLAMA_MODEL in m for m in available_models)

        if not model_available:
            return {
                "status":  "error",
                "model":   OLLAMA_MODEL,
                "message": (
                    f"Model '{OLLAMA_MODEL}' not found in Ollama. "
                    f"Run: ollama pull {OLLAMA_MODEL}. "
                    f"Currently installed: {available_models}"
                )
            }

        return {
            "status":  "ok",
            "model":   OLLAMA_MODEL,
            "message": f"Ollama is running. '{OLLAMA_MODEL}' is available."
        }

    except requests.exceptions.ConnectionError:
        return {
            "status":  "error",
            "model":   OLLAMA_MODEL,
            "message": (
                f"Ollama not reachable at {OLLAMA_BASE_URL}. "
                "Start it with: ollama serve"
            )
        }


# ─── Custom exceptions ────────────────────────────────────────────────────────

class OllamaError(Exception):
    """
    Base class for all Ollama-related errors.
    Catch this to handle any Ollama error regardless of specific type.
    """

class OllamaConnectionError(OllamaError):
    """
    Raised when the Ollama server cannot be reached.
    Usually means Ollama is not running — fix with: ollama serve
    """

class OllamaTimeoutError(OllamaError):
    """
    Raised when Ollama does not respond within the configured timeout.
    Fix: increase OLLAMA_TIMEOUT in pipeline_config.py, or shorten the prompt.
    """

class ExtractionFailedError(OllamaError):
    """
    Raised when all retry attempts are exhausted without a valid response.
    The pipeline catches this and marks the candidate/job as status="failed"
    so a human can review it manually instead of storing bad data.
    """