"""
OpenAI-compatible chat-completions client helpers.
"""
import time
from typing import Any, Dict, List, Optional

from .config import normalize_openai_base_url, require_llm_api_key, require_llm_base_url


def _create_client(api_key: Optional[str], base_url: Optional[str], timeout: int):
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise ImportError(
            "The openai package is required for LLM calls. Install dependencies from requirements.txt."
        ) from exc

    resolved_api_key = (api_key or require_llm_api_key()).strip()
    resolved_base_url = normalize_openai_base_url(base_url or require_llm_base_url())
    return OpenAI(api_key=resolved_api_key, base_url=resolved_base_url, timeout=timeout)


def _message_content_to_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                parts.append(str(item.get("text", "")))
            elif hasattr(item, "text"):
                parts.append(str(item.text))
            else:
                parts.append(str(item))
        return "".join(parts)
    return str(content)


def chat_completion_content(
    model: str,
    messages: List[Dict[str, Any]],
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    temperature: float = 0.0,
    response_format: Optional[Dict[str, Any]] = None,
    max_tokens: Optional[int] = None,
    timeout: int = 60,
    max_retries: int = 3,
    retry_delay: int = 2,
    request_label: str = "LLM request",
) -> str:
    """Call an OpenAI-compatible chat-completions endpoint and return message content."""
    client = _create_client(api_key=api_key, base_url=base_url, timeout=timeout)
    payload: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
    }
    if response_format is not None:
        payload["response_format"] = response_format
    if max_tokens is not None:
        payload["max_tokens"] = max_tokens

    last_error = None
    for attempt in range(max_retries):
        try:
            print(
                f"[REQUEST] {request_label} (attempt {attempt + 1}/{max_retries}, timeout={timeout}s) "
                f"with model '{model}'",
                flush=True,
            )
            response = client.chat.completions.create(**payload)
            return _message_content_to_text(response.choices[0].message.content)
        except Exception as exc:
            last_error = exc
            if attempt < max_retries - 1:
                wait_time = retry_delay * (2 ** attempt)
                print(
                    f"[RETRY] Attempt {attempt + 1}/{max_retries} failed: "
                    f"{type(exc).__name__}: {exc}. Retrying in {wait_time}s..."
                )
                time.sleep(wait_time)
            else:
                break

    raise RuntimeError(f"{request_label} failed after {max_retries} attempts: {last_error}") from last_error
