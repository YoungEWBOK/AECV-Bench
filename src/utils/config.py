"""
Configuration management for API keys and settings.
"""
import os
from typing import Optional
from pathlib import Path

LLM_API_KEY_ENV_NAMES = (
    "XMAPI_API_KEY",
    "OPENAI_API_KEY",
    "API_KEY",
    "api_key",
    "LLM_API_KEY",
    "OPENAI_COMPATIBLE_API_KEY",
    "OPEN_ROUTER_API_KEY",
)

LLM_BASE_URL_ENV_NAMES = (
    "XMAPI_BASE_URL",
    "OPENAI_BASE_URL",
    "OPENAI_API_BASE",
    "BASE_URL",
    "base_url",
    "LLM_BASE_URL",
    "OPENAI_COMPATIBLE_BASE_URL",
    "OPEN_ROUTER_BASE_URL",
)

LLM_API_MODE_ENV_NAMES = (
    "LLM_API_MODE",
    "OPENAI_COMPATIBLE_API_MODE",
    "XMAPI_API_MODE",
)

LLM_MODEL_ENV_NAMES = (
    "XMAPI_MODEL",
    "OPENAI_MODEL",
    "LLM_MODEL",
    "MODEL",
)


def load_dotenv_if_available():
    """Load .env file if python-dotenv is available and .env exists."""
    try:
        from dotenv import load_dotenv
        env_file = Path('.env')
        if env_file.exists():
            load_dotenv(env_file)
    except ImportError:
        pass


def get_api_key(env_var_name: str) -> Optional[str]:
    """
    Get API key from environment variable.
    
    Args:
        env_var_name: Environment variable name (e.g., 'OPEN_ROUTER_API_KEY')
    
    Returns:
        API key string or None if not set
    """
    # Load .env file each time to pick up any changes
    load_dotenv_if_available()
    return os.getenv(env_var_name)


def get_first_config_value(*env_var_names: str) -> Optional[str]:
    """Return the first non-empty environment value from the provided names."""
    load_dotenv_if_available()
    for env_var_name in env_var_names:
        value = os.getenv(env_var_name)
        if value and value.strip():
            return value.strip()
    return None


def require_api_key(env_var_name: str, provider_name: str = None) -> str:
    """
    Require an API key to be set, raise error if not.
    
    Args:
        env_var_name: Environment variable name
        provider_name: Human-readable provider name for error message
        
    Returns:
        The API key string
        
    Raises:
        ValueError: If key is None or empty
    """
    key = get_api_key(env_var_name)
    if not key or not key.strip():
        provider_display = provider_name or env_var_name
        raise ValueError(
            f"{provider_display} API key is required. "
            f"Please set {env_var_name} as an environment variable or create a .env file."
        )
    return key.strip()


def get_llm_api_key() -> Optional[str]:
    """Get the API key for the configured LLM endpoint."""
    return get_first_config_value(*LLM_API_KEY_ENV_NAMES)


def require_llm_api_key() -> str:
    """Require the single OpenAI-compatible LLM API key."""
    key = get_llm_api_key()
    if not key:
        raise ValueError(
            "LLM API key is required. "
            "Set XMAPI_API_KEY, OPENAI_API_KEY, or API_KEY as an environment variable or in .env."
        )
    return key


def get_llm_base_url() -> Optional[str]:
    """Get the base URL for the configured LLM endpoint."""
    base_url = get_first_config_value(*LLM_BASE_URL_ENV_NAMES)
    return normalize_openai_base_url(base_url) if base_url else None


def require_llm_base_url() -> str:
    """Require the single OpenAI-compatible LLM base URL."""
    base_url = get_llm_base_url()
    if not base_url:
        raise ValueError(
            "LLM base URL is required. "
            "Set XMAPI_BASE_URL, OPENAI_BASE_URL, or BASE_URL as an environment variable or in .env."
        )
    return base_url


def normalize_openai_base_url(base_url: str) -> str:
    """Normalize a base URL or legacy endpoint URL."""
    normalized = base_url.strip().rstrip("/")
    for suffix in ("/chat/completions", "/responses"):
        if normalized.endswith(suffix):
            normalized = normalized[: -len(suffix)]
    return normalized


def get_llm_api_mode() -> str:
    """Return the API transport mode: 'chat_completions' or 'responses'."""
    explicit = get_first_config_value(*LLM_API_MODE_ENV_NAMES)
    if explicit:
        normalized = explicit.strip().lower().replace("-", "_")
        if normalized in {"responses", "response", "responses_stream", "xmapi"}:
            return "responses"
        if normalized in {"chat", "chat_completions", "chat_completion", "openai"}:
            return "chat_completions"
        raise ValueError(
            f"Unsupported LLM API mode '{explicit}'. Use 'chat_completions' or 'responses'."
        )

    # XMAPI's gateway currently returns useful text only from streamed
    # /v1/responses events, so prefer that mode when XMAPI_BASE_URL is used.
    load_dotenv_if_available()
    if os.getenv("XMAPI_BASE_URL", "").strip():
        return "responses"
    return "chat_completions"


def get_llm_model(default: Optional[str] = None) -> Optional[str]:
    """Get an optional default model from environment variables."""
    return get_first_config_value(*LLM_MODEL_ENV_NAMES) or default


# Legacy functions for backward compatibility
def get_open_router_api_key() -> Optional[str]:
    return get_llm_api_key()


def get_cohere_api_key() -> Optional[str]:
    return get_llm_api_key()

