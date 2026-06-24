"""
Backward-compatible Cohere analyzer wrapper.

All LLM calls are routed through the configured OpenAI-compatible endpoint.
"""
import os
from typing import Dict, Optional

from ..models.plan_elements import get_json_schema
from ..parsers.json_parser import extract_json_counts
from ..parsers.text_parser import parse_counts_from_text_improved
from ..utils.config import require_llm_api_key, require_llm_base_url
from ..utils.image_utils import encode_image_to_base64, get_image_mime_type
from ..utils.openai_compatible import chat_completion_content


class CohereAnalyzer:
    """Compatibility surface for older Cohere-configured benchmark entries."""

    def __init__(self, max_retries: int = 3, timeout: int = 60):
        self.max_retries = max_retries
        self.timeout = timeout
        self.max_tokens = 2000

    def _build_prompt(self, image_name: str) -> str:
        return (
            f"FLOOR PLAN ANALYSIS\n"
            f"Image: {image_name}\n\n"
            "I have attached a floor plan image for you to analyze. Please examine the image carefully and count the following elements:\n\n"
            "1. DOORS - Count all door openings, gaps in walls, entrance points (both interior and exterior doors)\n"
            "2. WINDOWS - Count all window openings shown on exterior walls (rectangular symbols on walls)\n"
            "3. SPACES - Count every distinct room or area (bedrooms, living room, kitchen, bathrooms, closets, hallways, any enclosed space)\n"
            "4. BEDROOMS - Count only rooms that are clearly bedrooms (sleeping areas)\n"
            "5. TOILETS - Count bathrooms, WCs, toilet rooms\n\n"
            "The image is attached below. Please analyze it and provide your counts.\n\n"
            "End your response with this exact format:\n"
            'FINAL COUNTS: {"Door": X, "Window": Y, "Space": Z, "Bedroom": A, "Toilet": B}\n\n'
            "Replace X, Y, Z, A, B with the actual numbers you counted from the image."
        )

    def _process_response(self, content: str, image_name: str) -> Dict:
        json_result = extract_json_counts(content, image_name)
        if json_result:
            return json_result
        return parse_counts_from_text_improved(content, image_name)

    def analyze_floorplan(
        self,
        image_path: str,
        model_name: str,
        json_schema: Dict = None,
        cohere_api_key: Optional[str] = None,
        url: Optional[str] = None,
        temperature: float = 0.0,
        max_retries: int = None,
    ) -> Dict:
        json_schema = json_schema or get_json_schema()
        del json_schema

        base64_image = encode_image_to_base64(image_path)
        data_url = f"data:{get_image_mime_type(image_path)};base64,{base64_image}"
        image_name = os.path.basename(image_path)
        prompt_text = self._build_prompt(image_name)

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt_text},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            }
        ]

        content = chat_completion_content(
            model=model_name,
            messages=messages,
            api_key=cohere_api_key or require_llm_api_key(),
            base_url=url or require_llm_base_url(),
            temperature=temperature,
            max_tokens=self.max_tokens,
            timeout=self.timeout,
            max_retries=max_retries or self.max_retries,
            request_label=f"Compatibility floor-plan analysis for '{image_path}'",
        )
        return self._process_response(content, image_name)


_cohere_analyzer = CohereAnalyzer()


def analyze_floorplan_cohere(
    image_path: str,
    model_name: str,
    json_schema: Dict = None,
    cohere_api_key: Optional[str] = None,
    url: Optional[str] = None,
    temperature: float = 0.0,
    max_retries: int = 3,
) -> Dict:
    """Analyze a floor plan through the configured OpenAI-compatible endpoint."""
    return _cohere_analyzer.analyze_floorplan(
        image_path=image_path,
        model_name=model_name,
        json_schema=json_schema,
        cohere_api_key=cohere_api_key,
        url=url,
        temperature=temperature,
        max_retries=max_retries,
    )
