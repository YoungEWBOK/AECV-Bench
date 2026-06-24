"""
Backward-compatible Replicate analyzer wrapper.

All LLM calls are routed through the configured OpenAI-compatible endpoint.
"""
from typing import Dict, Optional, Union

from ..models.plan_elements import get_json_schema
from ..utils.config import require_llm_api_key, require_llm_base_url
from .openrouter import analyze_floorplan_prompt_based


class ReplicateAnalyzer:
    """Compatibility surface for older Replicate-configured benchmark entries."""

    def __init__(self, max_retries: int = 3, timeout: int = 300):
        self.max_retries = max_retries
        self.timeout = timeout

    def analyze_floorplan(
        self,
        image_path: str,
        model_name: str,
        json_schema: Dict = None,
        replicate_api_token: Optional[str] = None,
        temperature: float = 0.1,
        max_new_tokens: int = 2048,
        top_p: float = 0.9,
        repetition_penalty: float = 1.1,
        max_retries: int = None,
        url: Optional[str] = None,
        **kwargs,
    ) -> Union[str, Dict]:
        del max_new_tokens, top_p, repetition_penalty, kwargs

        return analyze_floorplan_prompt_based(
            image_path=image_path,
            model_name=model_name,
            json_schema=json_schema or get_json_schema(),
            open_router_api_key=replicate_api_token or require_llm_api_key(),
            url=url or require_llm_base_url(),
            temperature=temperature,
        )


_replicate_analyzer = ReplicateAnalyzer()


def analyze_floorplan_replicate(
    image_path: str,
    model_name: str,
    json_schema: Dict = None,
    replicate_api_token: Optional[str] = None,
    temperature: float = 0.1,
    max_new_tokens: int = 2048,
    top_p: float = 0.9,
    repetition_penalty: float = 1.1,
    max_retries: int = 3,
    url: Optional[str] = None,
    **kwargs,
) -> Union[str, Dict]:
    """Analyze a floor plan through the configured OpenAI-compatible endpoint."""
    return _replicate_analyzer.analyze_floorplan(
        image_path=image_path,
        model_name=model_name,
        json_schema=json_schema,
        replicate_api_token=replicate_api_token,
        temperature=temperature,
        max_new_tokens=max_new_tokens,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        max_retries=max_retries,
        url=url,
        **kwargs,
    )
