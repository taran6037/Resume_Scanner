import json
import re
import time
import logging
from typing import Optional, Type
import requests
from pydantic import BaseModel, ValidationError

from config.pipeline_config import (
    OLLAMA_BASE_URL,        
    OLLAMA_MODEL,           
    OLLAMA_TIMEOUT,        
    OLLAMA_MAX_RETRIES,     
    OLLAMA_RETRY_DELAY,     
    LLM_TEMPERATURE,        
    LLM_NUM_PREDICT,        
    LLM_NUM_CTX,            
    LLM_DISABLE_THINKING,   
    CONFIDENCE_ATTEMPT_1,   
    CONFIDENCE_ATTEMPT_2,   
    CONFIDENCE_ATTEMPT_3,   
)

logger = logging.getLogger(__name__)

def call_ollama(
    prompt: str,
    system_prompt: str,
    temperature: float = LLM_TEMPERATURE,
    timeout: int = OLLAMA_TIMEOUT,
) -> str:

    payload = {
        "model":  OLLAMA_MODEL,
        "prompt": prompt,
        "system": system_prompt,
        "format": "json",
        "stream": False,       
        "think":  not LLM_DISABLE_THINKING,  
        "options": {
            "temperature": temperature,    
            "num_predict": LLM_NUM_PREDICT, 
            "num_ctx":     LLM_NUM_CTX,    
            "top_p":       0.9,             
        }
    }

    try:
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json=payload,
            timeout=timeout
        )
        response.raise_for_status()
        data = response.json()
        return data.get("response", "").strip()

    except requests.exceptions.ConnectionError:
        raise OllamaConnectionError(
            f"Cannot reach Ollama at {OLLAMA_BASE_URL}. "
            "Make sure Ollama is running: `ollama serve`"
        )
    except requests.exceptions.Timeout:
        raise OllamaTimeoutError(
            f"Ollama did not respond within {timeout}s. "
            "Try a shorter prompt or increase OLLAMA_TIMEOUT in pipeline_config.py."
        )
    except requests.exceptions.HTTPError as e:
        raise OllamaError(f"Ollama HTTP error: {e}")

def extract_json(text: str) -> Optional[dict]:
    if not text:
        return None

    fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence_match:
        text = fence_match.group(1)   

    brace_match = re.search(r"\{.*\}", text, re.DOTALL)
    if brace_match:
        candidate = brace_match.group(0)
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None   

def call_ollama_for_schema(
    prompt: str,
    system_prompt: str,
    schema: Type[BaseModel],
    context_label: str = "extraction",
) -> tuple[BaseModel, float]:
    last_error: Optional[str] = None
    last_raw:   Optional[str] = None

    for attempt in range(1, OLLAMA_MAX_RETRIES + 1):

        active_prompt = (
            _build_corrective_prompt(last_raw, last_error, schema)
            if attempt > 1
            else prompt
        )

        logger.debug(f"[{context_label}] Attempt {attempt}/{OLLAMA_MAX_RETRIES}")

        try:
            raw_response = call_ollama(active_prompt, system_prompt)
            last_raw     = raw_response
        except (OllamaConnectionError, OllamaTimeoutError):
            raise

        parsed_dict = extract_json(raw_response)

        if parsed_dict is None:
            last_error = "Response contained no valid JSON object."
            logger.warning(
                f"[{context_label}] Attempt {attempt}: no JSON found. "
                f"Raw response (first 200 chars): {raw_response[:200]}"
            )
            if attempt < OLLAMA_MAX_RETRIES:
                time.sleep(OLLAMA_RETRY_DELAY)
            continue  
        try:
            validated  = schema(**parsed_dict)  
            confidence = _confidence_from_attempt(attempt)
            logger.info(
                f"[{context_label}] Succeeded on attempt {attempt}. "
                f"Confidence: {confidence}"
            )
            return validated, confidence 

        except ValidationError as e:
            last_error = str(e)
            logger.warning(
                f"[{context_label}] Attempt {attempt}: Pydantic validation failed. "
                f"Errors: {last_error[:300]}"
            )
            if attempt < OLLAMA_MAX_RETRIES:
                time.sleep(OLLAMA_RETRY_DELAY)
            continue   

    raise ExtractionFailedError(
        f"[{context_label}] Failed after {OLLAMA_MAX_RETRIES} attempts. "
        f"Last error: {last_error}. "
        f"Last raw response (first 500 chars): {str(last_raw)[:500]}"
    )


def _build_corrective_prompt(
    last_raw:   Optional[str],
    last_error: Optional[str],
    schema:     Type[BaseModel],
) -> str:
    schema_example = _schema_to_example(schema)

    return f"""Your previous response was invalid.
Error: {last_error}
Previous response: {str(last_raw)[:300]}

Return ONLY a valid JSON object matching this structure exactly:
{json.dumps(schema_example, indent=2)}"""


def _schema_to_example(schema: Type[BaseModel]) -> dict:
    example = {}
    for field_name, field_info in schema.model_fields.items():
        annotation = field_info.annotation
        if annotation in (str, Optional[str]):
            example[field_name] = f"<{field_name}>"   
        elif annotation in (int, Optional[int]):
            example[field_name] = 0
        elif annotation in (float, Optional[float]):
            example[field_name] = 0.0
        elif annotation in (bool,):
            example[field_name] = False
        elif hasattr(annotation, "__origin__") and annotation.__origin__ is list:
            example[field_name] = []                
        else:
            example[field_name] = {}                   
    return example



def _confidence_from_attempt(attempt: int) -> float:
    scores = {
        1: CONFIDENCE_ATTEMPT_1,  
        2: CONFIDENCE_ATTEMPT_2,   
        3: CONFIDENCE_ATTEMPT_3,   
    }
    return scores.get(attempt, 0.5)


def check_ollama_health() -> dict:
    
    try:
        response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        response.raise_for_status()
        data = response.json()
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
