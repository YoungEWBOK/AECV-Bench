"""
OpenAI-compatible chat-completions client helpers.
"""
import json
import os
import time
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional

from .config import (
    get_llm_api_mode,
    get_llm_model,
    normalize_openai_base_url,
    require_llm_api_key,
    require_llm_base_url,
)


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


def _chat_content_to_responses_content(content: Any) -> Any:
    """Convert chat-completions content blocks to Responses API content blocks."""
    if isinstance(content, str):
        return [{"type": "input_text", "text": content}]
    if not isinstance(content, list):
        return [{"type": "input_text", "text": str(content)}]

    converted = []
    for item in content:
        if not isinstance(item, dict):
            converted.append({"type": "input_text", "text": str(item)})
            continue
        item_type = item.get("type")
        if item_type == "text":
            converted.append({"type": "input_text", "text": str(item.get("text", ""))})
        elif item_type == "image_url":
            image_url = item.get("image_url", {})
            if isinstance(image_url, dict):
                url = image_url.get("url", "")
            else:
                url = str(image_url)
            converted.append({"type": "input_image", "image_url": url})
        elif item_type in {"input_text", "input_image"}:
            converted.append(item)
        else:
            text = item.get("text")
            if text is not None:
                converted.append({"type": "input_text", "text": str(text)})
            else:
                converted.append({"type": "input_text", "text": json.dumps(item, ensure_ascii=False)})
    return converted


def _chat_messages_to_responses_input(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Convert chat-completions messages to Responses API input."""
    responses_input = []
    for message in messages:
        role = message.get("role", "user")
        # Responses API accepts user/assistant/system/developer in current OpenAI
        # clients, but some gateways are stricter. Preserve common roles and map
        # unknown ones to user.
        if role not in {"user", "assistant", "system", "developer"}:
            role = "user"
        responses_input.append(
            {
                "role": role,
                "content": _chat_content_to_responses_content(message.get("content", "")),
            }
        )
    return responses_input


def _response_format_to_responses_text(response_format: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Best-effort conversion from chat response_format to Responses API text format."""
    if not response_format:
        return None
    if response_format.get("type") == "json_object":
        return {"format": {"type": "json_object"}}
    if response_format.get("type") == "json_schema":
        schema_payload = dict(response_format.get("json_schema") or {})
        return {"format": {"type": "json_schema", **schema_payload}}
    return None


def _parse_responses_sse_stream(response) -> str:
    """Parse streamed /v1/responses SSE events and return assistant text."""
    text_parts = []
    last_done_text = ""

    for raw_line in response:
        line = raw_line.decode("utf-8", errors="replace").strip()
        if not line or not line.startswith("data: "):
            continue
        data = line[6:].strip()
        if data == "[DONE]":
            break
        try:
            event = json.loads(data)
        except json.JSONDecodeError:
            continue

        event_type = event.get("type")
        if event_type == "response.output_text.delta":
            delta = event.get("delta", "")
            if delta:
                text_parts.append(delta)
        elif event_type == "response.output_text.done":
            last_done_text = event.get("text", "") or last_done_text
        elif event_type == "response.completed" and not text_parts:
            extracted = _extract_responses_completed_text(event.get("response"))
            if extracted:
                text_parts.append(extracted)

    if not text_parts and last_done_text:
        text_parts.append(last_done_text)
    return "".join(text_parts).strip()


def _extract_responses_completed_text(response_obj: Any) -> str:
    """Extract text from a non-stream-style Responses object nested in an SSE event."""
    if not isinstance(response_obj, dict):
        return ""
    direct_text = response_obj.get("output_text")
    if isinstance(direct_text, str) and direct_text.strip():
        return direct_text.strip()

    parts = []
    for output_item in response_obj.get("output", []) or []:
        if not isinstance(output_item, dict):
            continue
        for content_item in output_item.get("content", []) or []:
            if not isinstance(content_item, dict):
                continue
            if content_item.get("type") in {"output_text", "text"}:
                text = content_item.get("text", "")
                if text:
                    parts.append(str(text))
    return "".join(parts).strip()


def _call_responses_stream(
    model: str,
    messages: List[Dict[str, Any]],
    api_key: Optional[str],
    base_url: Optional[str],
    temperature: float,
    response_format: Optional[Dict[str, Any]],
    max_tokens: Optional[int],
    timeout: int,
) -> str:
    """Call an OpenAI Responses-compatible endpoint over SSE."""
    resolved_api_key = (api_key or require_llm_api_key()).strip()
    resolved_base_url = normalize_openai_base_url(base_url or require_llm_base_url())
    url = _responses_url(resolved_base_url)
    payload: Dict[str, Any] = {
        "model": _resolve_model_name(model),
        "input": _chat_messages_to_responses_input(messages),
        "stream": True,
    }
    if temperature is not None:
        payload["temperature"] = temperature
    if max_tokens is not None:
        payload["max_output_tokens"] = max_tokens
    text_format = _response_format_to_responses_text(response_format)
    if text_format is not None and _use_responses_text_format():
        payload["text"] = text_format

    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    headers = {
        "Authorization": f"Bearer {resolved_api_key}",
        "Content-Type": "application/json",
    }
    request = urllib.request.Request(url, data=data, headers=headers, method="POST")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return _parse_responses_sse_stream(response)


def _responses_url(base_url: str) -> str:
    """Build /v1/responses URL from either a root URL or a /v1 base URL."""
    normalized = base_url.rstrip("/")
    if normalized.endswith("/v1"):
        return f"{normalized}/responses"
    return f"{normalized}/v1/responses"


def _resolve_model_name(model: str) -> str:
    """Resolve optional environment-backed model placeholders."""
    normalized = (model or "").strip()
    if normalized in {"", "env", "$XMAPI_MODEL", "${XMAPI_MODEL}", "$LLM_MODEL", "${LLM_MODEL}"}:
        resolved = get_llm_model()
        if not resolved:
            raise ValueError("Model is empty and no XMAPI_MODEL/LLM_MODEL environment variable is set.")
        return resolved
    return normalized


def _use_responses_text_format() -> bool:
    """Whether to send Responses API text.format response constraints."""
    value = os.getenv("LLM_RESPONSES_USE_TEXT_FORMAT", "").strip().lower()
    return value in {"1", "true", "yes", "on"}


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
    """Call the configured LLM endpoint and return message content.

    Supports both OpenAI-compatible /chat/completions and XMAPI-style
    /v1/responses streamed SSE. Select the latter with XMAPI_BASE_URL or
    LLM_API_MODE=responses.
    """
    api_mode = get_llm_api_mode()
    client = None
    payload: Dict[str, Any] = {}
    if api_mode == "chat_completions":
        client = _create_client(api_key=api_key, base_url=base_url, timeout=timeout)
        payload = {
            "model": _resolve_model_name(model),
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
            if api_mode == "responses":
                content = _call_responses_stream(
                    model=model,
                    messages=messages,
                    api_key=api_key,
                    base_url=base_url,
                    temperature=temperature,
                    response_format=response_format,
                    max_tokens=max_tokens,
                    timeout=timeout,
                )
            else:
                response = client.chat.completions.create(**payload)
                content = _message_content_to_text(response.choices[0].message.content)

            if content:
                return content
            raise RuntimeError("No assistant text was returned")
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            last_error = RuntimeError(f"HTTP {exc.code}: {body}")
        except urllib.error.URLError as exc:
            last_error = RuntimeError(f"Request failed: {exc}")
        except Exception as exc:
            last_error = exc

        if attempt < max_retries - 1:
            wait_time = retry_delay * (2 ** attempt)
            print(
                f"[RETRY] Attempt {attempt + 1}/{max_retries} failed: "
                f"{type(last_error).__name__}: {last_error}. Retrying in {wait_time}s..."
            )
            time.sleep(wait_time)
        else:
            break

    raise RuntimeError(f"{request_label} failed after {max_retries} attempts: {last_error}") from last_error
