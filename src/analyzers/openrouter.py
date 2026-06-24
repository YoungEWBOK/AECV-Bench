"""
OpenAI-compatible API analyzer for floor plan analysis.
"""
import json
import re
from typing import Dict, Union, Optional
from ..utils.image_utils import encode_image_to_base64
from ..models.plan_elements import get_json_schema
from .prompts import COUNTING_RULES
from ..parsers.json_parser import extract_json_from_response
from ..utils.config import require_llm_api_key, require_llm_base_url
from ..utils.openai_compatible import chat_completion_content
from ..utils.prompt_strategies import (
    build_counting_prompt,
    build_counting_reflection_prompt,
    normalize_prompt_strategy,
)
from ..skill_evolution.contracts import SkillLibrary

try:
    from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
    HAS_TENACITY = True
except ImportError:
    HAS_TENACITY = False


class OpenRouterAnalyzer:
    """
    Configurable OpenAI-compatible API analyzer that eliminates code duplication.
    Supports both standard JSON schema enforcement and Claude-style prompt-based extraction.
    """
    
    def __init__(self, use_schema_format: bool = True):
        """
        Initialize the analyzer.
        
        Args:
            use_schema_format: If True, uses JSON schema enforcement in API request.
                             If False, uses prompt-based JSON extraction (Claude-style).
        """
        self.use_schema_format = use_schema_format
        self.max_retries = 3
        self.retry_delay = 2
        self.timeout = 60

    def _validate_inputs(self, image_path: str, model_name: str, open_router_api_key: Optional[str], json_schema: Optional[Dict]):
        """Validate input parameters."""
        if not model_name:
            raise ValueError("Model name is required")
        return json_schema or get_json_schema()

    def _build_prompt(self, json_schema: Dict, prompt_strategy: str = "one_shot", skill_context: str = "") -> str:
        """Build the prompt text based on configuration."""
        if self.use_schema_format:
            # Standard format - schema enforcement via API
            intro_text = (
                "You are an expert floor-plan analyst trained to interpret architectural "
                "and engineering drawings in any language.\n\n"
                "Your task is to fully understand the drawing first, then count the elements exactly "
                "according to the rules below, and then return ONLY the JSON strictly following the provided schema."
            )
        else:
            # Claude format - schema in prompt
            schema_str = json.dumps(json_schema, indent=2)
            intro_text = (
                f"You are an expert floor-plan analyst trained to interpret architectural "
                f"and engineering drawings in any language.\n\n"
                f"Your task is to fully understand the drawing first, then count the elements exactly "
                f"according to the rules below, and then return ONLY the JSON strictly following the provided schema:\n\n"
                f"{schema_str}\n\n"
            )
        
        return build_counting_prompt(intro_text + "\n\n" + COUNTING_RULES, prompt_strategy, skill_context=skill_context)

    def _build_payload(self, model_name: str, prompt_text: str, data_url: str, temperature: float, json_schema: Dict) -> Dict:
        """Build the request payload based on configuration."""
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt_text
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": data_url
                        }
                    }
                ]
            }
        ]

        payload = {
            "model": model_name,
            "messages": messages,
            "temperature": temperature
        }

        # Add schema enforcement for standard format
        if self.use_schema_format:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "PlanElements",
                    "strict": True,
                    "schema": json_schema,
                }
            }

        return payload

    def _validate_content(self, content: str, image_path: str) -> str:
        """Validate content returned by the API."""
        # Check for empty content
        if not content or (self.use_schema_format and not content.strip()):
            raise ValueError(f"Empty response from API for {image_path}")
        
        return content

    def _process_response(self, content: str, image_path: str) -> Union[str, Dict]:
        """Process the response content based on configuration."""
        if self.use_schema_format:
            # Standard format - return content directly (should be valid JSON)
            return content
        else:
            # Claude format - extract JSON from potentially mixed content
            if content:
                # First try direct JSON parsing
                try:
                    return json.loads(content)
                except json.JSONDecodeError:
                    # If direct parsing fails, try to extract JSON from text
                    extracted_json = extract_json_from_response(content, image_path)
                    if extracted_json:
                        return extracted_json
                    # If extraction fails, try to find JSON in code blocks
                    json_block = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', content, re.DOTALL)
                    if json_block:
                        try:
                            return json.loads(json_block.group(1))
                        except json.JSONDecodeError:
                            pass
                    # Last resort: raise error with content preview
                    raise ValueError(f"Could not extract valid JSON from response. Content preview: {content[:200]}...")
            else:
                raise ValueError("Empty response from API")

    def analyze_floorplan(
        self,
        image_path: str,
        model_name: str,
        json_schema: Dict = None,
        open_router_api_key: str = None,
        url: str = None,
        temperature: float = 0.0,
        timeout: int = None,
        max_retries: int = None,
        retry_delay: int = None,
        prompt_strategy: str = "one_shot",
        skill_library_path: str = "",
        max_skills_per_question: int = 4,
        skill_statuses=None,
    ) -> Union[str, Dict]:
        """
        Analyze floor plan image using OpenRouter API.
        
        This is the main method that coordinates all steps of the analysis process.
        """
        # Step 1: Validate inputs
        validated_schema = self._validate_inputs(image_path, model_name, open_router_api_key, json_schema)
        api_key = open_router_api_key.strip() if open_router_api_key and open_router_api_key.strip() else require_llm_api_key()
        base_url = url or require_llm_base_url()
        prompt_strategy = normalize_prompt_strategy(prompt_strategy)
        skill_context = ""
        if skill_library_path:
            skill_library = SkillLibrary.load(skill_library_path)
            skill_context = skill_library.format_for_prompt(
                question="Count doors, windows, spaces, bedrooms, and toilets in the full floor plan.",
                qa_type="object_counting",
                task="object_counting",
                max_skills=max_skills_per_question,
                statuses=skill_statuses or ("accepted",),
            )
        
        # Step 2: Prepare image
        base64_image = encode_image_to_base64(image_path)
        from ..utils.image_utils import get_image_mime_type
        mime_type = get_image_mime_type(image_path)
        data_url = f"data:{mime_type};base64,{base64_image}"
        
        # Step 3: Build prompt
        prompt_text = self._build_prompt(validated_schema, prompt_strategy, skill_context=skill_context)
        
        # Step 4: Build payload
        payload = self._build_payload(model_name, prompt_text, data_url, temperature, validated_schema)
        
        # Step 5: Make request with retries
        content = chat_completion_content(
            model=payload["model"],
            messages=payload["messages"],
            api_key=api_key,
            base_url=base_url,
            temperature=payload["temperature"],
            response_format=payload.get("response_format"),
            timeout=timeout or self.timeout,
            max_retries=max_retries or self.max_retries,
            retry_delay=retry_delay or self.retry_delay,
            request_label=f"Floor-plan analysis for '{image_path}'",
        )
        content = self._validate_content(content, image_path)

        if prompt_strategy == "two_pass_reflection":
            first_result = self._process_response(content, image_path)
            reflection_prompt = build_counting_reflection_prompt(
                self._build_prompt(validated_schema, "one_shot", skill_context=skill_context),
                first_result,
            )
            reflection_payload = self._build_payload(
                model_name,
                reflection_prompt,
                data_url,
                temperature,
                validated_schema,
            )
            content = chat_completion_content(
                model=reflection_payload["model"],
                messages=reflection_payload["messages"],
                api_key=api_key,
                base_url=base_url,
                temperature=reflection_payload["temperature"],
                response_format=reflection_payload.get("response_format"),
                timeout=timeout or self.timeout,
                max_retries=max_retries or self.max_retries,
                retry_delay=retry_delay or self.retry_delay,
                request_label=f"Floor-plan reflection for '{image_path}'",
            )
            content = self._validate_content(content, image_path)
        
        # Step 6: Process response
        return self._process_response(content, image_path)


# Create global analyzer instances
_standard_analyzer = OpenRouterAnalyzer(use_schema_format=True)
_prompt_based_analyzer = OpenRouterAnalyzer(use_schema_format=False)


def analyze_floorplan(
    image_path: str,
    model_name: str,
    json_schema: Dict = None,
    open_router_api_key: str = None,
    url: str = None,
    temperature: float = 0.0,
    timeout: int = None,
    max_retries: int = None,
    retry_delay: int = None,
    prompt_strategy: str = "one_shot",
    skill_library_path: str = "",
    max_skills_per_question: int = 4,
    skill_statuses=None,
) -> Union[str, Dict]:
    """
    Sends a floor-plan image to an OpenAI-compatible chat-completions endpoint and returns the JSON response.

    Parameters:
    - image_path: Path to the floor-plan image file.
    - model_name: The model identifier to use (e.g., "google/gemini-2.0-flash-001").
    - json_schema: A dict defining the JSON schema expected in the response.
    - open_router_api_key: API key override. If omitted, OPENAI_API_KEY/API_KEY is used.
    - url: Base URL override. If omitted, OPENAI_BASE_URL/BASE_URL is used.
    - temperature: Sampling temperature (default 0.0 for deterministic).

    Returns:
    - The parsed JSON content from the API response (the "content" field under choices[0].message).

    Raises:
    - RuntimeError if the API request fails.
    """
    return _standard_analyzer.analyze_floorplan(
        image_path=image_path,
        model_name=model_name,
        json_schema=json_schema,
        open_router_api_key=open_router_api_key,
        url=url,
        temperature=temperature,
        timeout=timeout,
        max_retries=max_retries,
        retry_delay=retry_delay,
        prompt_strategy=prompt_strategy,
        skill_library_path=skill_library_path,
        max_skills_per_question=max_skills_per_question,
        skill_statuses=skill_statuses,
    )


def analyze_floorplan_prompt_based(
    image_path: str,
    model_name: str,
    json_schema: Dict = None,
    open_router_api_key: str = None,
    url: str = None,
    temperature: float = 0.0,
    timeout: int = None,
    max_retries: int = None,
    retry_delay: int = None,
    prompt_strategy: str = "one_shot",
    skill_library_path: str = "",
    max_skills_per_question: int = 4,
    skill_statuses=None,
) -> Union[str, Dict]:
    """
    Sends a floor-plan image to an OpenAI-compatible chat-completions endpoint.
    Uses prompt-based JSON extraction instead of strict schema enforcement.

    Use this for models that don't support structured output via response_format.

    Parameters:
    - image_path: Path to the floor-plan image file.
    - model_name: The model identifier to use.
    - json_schema: A dict defining the JSON schema expected in the response.
    - open_router_api_key: API key override. If omitted, OPENAI_API_KEY/API_KEY is used.
    - url: Base URL override. If omitted, OPENAI_BASE_URL/BASE_URL is used.
    - temperature: Sampling temperature (default 0.0 for deterministic).

    Returns:
    - The parsed JSON content from the API response.

    Raises:
    - RuntimeError if the API request fails.
    """
    return _prompt_based_analyzer.analyze_floorplan(
        image_path=image_path,
        model_name=model_name,
        json_schema=json_schema,
        open_router_api_key=open_router_api_key,
        url=url,
        temperature=temperature,
        timeout=timeout,
        max_retries=max_retries,
        retry_delay=retry_delay,
        prompt_strategy=prompt_strategy,
        skill_library_path=skill_library_path,
        max_skills_per_question=max_skills_per_question,
        skill_statuses=skill_statuses,
    )

