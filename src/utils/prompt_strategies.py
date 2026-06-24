"""Prompt strategy helpers for AECV benchmark experiments."""
import json
import re
from typing import Any


DEFAULT_PROMPT_STRATEGY = "one_shot"
SUPPORTED_PROMPT_STRATEGIES = {
    "one_shot",
    "step_by_step",
    "self_refine",
    "two_pass_reflection",
    "skill_guided",
}


def normalize_prompt_strategy(strategy: str = None) -> str:
    """Normalize and validate a prompt strategy name."""
    normalized = (strategy or DEFAULT_PROMPT_STRATEGY).strip().lower().replace("-", "_")
    if normalized not in SUPPORTED_PROMPT_STRATEGIES:
        raise ValueError(
            f"Unsupported prompt_strategy '{strategy}'. "
            f"Supported values: {', '.join(sorted(SUPPORTED_PROMPT_STRATEGIES))}"
        )
    return normalized


def prompt_strategy_suffix(strategy: str = None) -> str:
    """Return a filename suffix for non-default prompt strategies."""
    normalized = normalize_prompt_strategy(strategy)
    return "" if normalized == DEFAULT_PROMPT_STRATEGY else normalized


def make_safe_name(value: str) -> str:
    """Create a stable filesystem-safe identifier."""
    safe = (value or "").lower().replace(" ", "_").replace(".", "").replace("-", "_")
    return re.sub(r"[^a-z0-9_]+", "_", safe).strip("_")


def _append_skill_context(instruction: str, skill_context: str = "") -> str:
    """Append learned skill context to a prompt instruction."""
    if not skill_context or not str(skill_context).strip():
        return instruction
    return (
        instruction
        + "\n\nLearned skill guidance for this task:\n"
        + str(skill_context).strip()
        + "\n\nUse the skill guidance privately. Return only the requested final answer."
    )


def build_qa_prompt(question: str, strategy: str = None, skill_context: str = "") -> str:
    """Build the QA prompt for a given strategy."""
    strategy = normalize_prompt_strategy(strategy)
    base = (
        "Please analyze the engineering/architectural drawing attached and provide "
        "a short and precise answer to the following question. Avoid extended explanations."
    )

    if strategy == "one_shot":
        instruction = base
    elif strategy == "step_by_step":
        instruction = (
            base
            + "\n\nUse this internal process before answering: identify the relevant drawing "
            "region, read any needed labels/symbols/dimensions, reason about spatial or counting "
            "relationships, then verify the answer against the image. Do not show the reasoning; "
            "return only the final concise answer."
        )
    elif strategy in {"self_refine", "two_pass_reflection"}:
        instruction = (
            base
            + "\n\nBefore finalizing, draft an answer internally, then re-check the image for "
            "common mistakes: missed small labels, confusing adjacent rooms, wrong units, duplicated "
            "objects, left/right confusion, and contradictions with visible annotations. Do not show "
            "the draft or critique; return only the corrected final answer."
        )
    elif strategy == "skill_guided":
        instruction = (
            base
            + "\n\nBefore answering, select any applicable learned skills, gather visual evidence, "
            "apply the skill actions, and run the skill validators. Do not reveal the reasoning or "
            "the skill IDs; return only the final concise answer."
        )
    else:
        raise AssertionError(f"Unhandled prompt strategy: {strategy}")

    instruction = _append_skill_context(instruction, skill_context)
    return f"{instruction}\n\n{question}"


def build_qa_reflection_prompt(question: str, previous_answer: str) -> str:
    """Build the second-pass QA reflection prompt."""
    return (
        "Review your previous answer against the attached engineering/architectural drawing.\n\n"
        f"Question: {question}\n\n"
        f"Previous answer: {previous_answer}\n\n"
        "Re-check the relevant region, labels, units, counts, and spatial relationships. If the "
        "previous answer is correct, repeat it exactly or with a cleaner concise phrasing. If it is "
        "wrong or incomplete, correct it. Return only the final short answer."
    )


def build_counting_prompt(base_prompt: str, strategy: str = None, skill_context: str = "") -> str:
    """Build the object-counting prompt for a given strategy."""
    strategy = normalize_prompt_strategy(strategy)
    if strategy == "one_shot":
        return _append_skill_context(base_prompt, skill_context)

    if strategy == "step_by_step":
        strategy_text = (
            "\n\nInternal procedure before returning JSON:\n"
            "1. Survey the entire floor plan before counting.\n"
            "2. Separately enumerate candidate doors, windows, spaces, bedrooms, and toilets.\n"
            "3. Resolve ambiguous symbols using the counting rules.\n"
            "4. Re-check for tiny toilet windows, double-leaf doors, unlabeled spaces, and excluded garage entrances.\n"
            "5. Return ONLY the final JSON object. Do not include your reasoning."
        )
    elif strategy in {"self_refine", "two_pass_reflection"}:
        strategy_text = (
            "\n\nInternal self-refinement before returning JSON:\n"
            "First make a draft count. Then re-scan the drawing and actively look for likely mistakes: "
            "missed small symbols, double-counted adjacent windows, double-leaf doors, sliding openings, "
            "unlabeled enclosed spaces, bedrooms in other languages, and WC/Bath/Shower labels. Correct "
            "the draft if needed. Return ONLY the final JSON object."
        )
    elif strategy == "skill_guided":
        strategy_text = (
            "\n\nLearned-skill counting procedure before returning JSON:\n"
            "Select applicable learned skills, enumerate visible candidates, apply inclusion/exclusion "
            "rules, deduplicate grouped symbols, run validators, and return ONLY the final JSON object."
        )
    else:
        raise AssertionError(f"Unhandled prompt strategy: {strategy}")

    return _append_skill_context(base_prompt + strategy_text, skill_context)


def build_counting_reflection_prompt(base_prompt: str, previous_result: Any) -> str:
    """Build the second-pass object-counting reflection prompt."""
    if isinstance(previous_result, str):
        previous_text = previous_result
    else:
        previous_text = json.dumps(previous_result, ensure_ascii=False, separators=(",", ":"))

    return (
        base_prompt
        + "\n\nNow review this previous count against the attached floor plan:\n"
        + previous_text
        + "\n\nRe-check the full drawing for missed or double-counted doors, windows, spaces, bedrooms, "
        "and toilets. Pay special attention to small toilet windows, double-leaf doors, sliding openings, "
        "unlabeled enclosed spaces, and labels in non-English languages. Return ONLY the corrected final "
        "JSON object following the schema."
    )
