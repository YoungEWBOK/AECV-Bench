"""Fixed skill categories for AEC drawing QA self-evolution.

The concrete skills are expected to be generated and validated from benchmark
traces. This module only defines the fixed category ontology and lightweight
routing hints used to attach cases and prompts to those categories.
"""
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple


VISUAL_EVIDENCE_ACQUISITION = "visual_evidence_acquisition"
TEXT_OCR_GROUNDING = "text_ocr_grounding"
SYMBOL_GEOMETRY_GROUNDING = "symbol_geometry_grounding"
SPATIAL_RELATION_REASONING = "spatial_relation_reasoning"
COUNTING_ENUMERATION = "counting_enumeration"
ANSWER_SYNTHESIS = "answer_synthesis"
VERIFICATION_REFLECTION = "verification_reflection"
SKILL_LIBRARY_MANAGEMENT = "skill_library_management"


@dataclass(frozen=True)
class SkillCategory:
    """A fixed skill category."""

    category_id: str
    title: str
    purpose: str
    default_triggers: Tuple[str, ...]


SKILL_CATEGORIES: Tuple[SkillCategory, ...] = (
    SkillCategory(
        VISUAL_EVIDENCE_ACQUISITION,
        "Visual Evidence Acquisition",
        "Acquire reliable visual evidence before answering: overview, zoom, crop, rotate, and inspect local regions.",
        ("small text", "tiny symbol", "hard to see", "local region", "image unclear"),
    ),
    SkillCategory(
        TEXT_OCR_GROUNDING,
        "Text / OCR Grounding",
        "Read and normalize text, labels, dimensions, room names, legends, and area annotations from the drawing.",
        ("text", "label", "area", "size", "dimension", "name", "ocr"),
    ),
    SkillCategory(
        SYMBOL_GEOMETRY_GROUNDING,
        "Symbol / Geometry Grounding",
        "Ground floor-plan symbols and geometry such as doors, windows, walls, rooms, fixtures, and boundaries.",
        ("door", "window", "wall", "room", "symbol", "fixture", "boundary"),
    ),
    SkillCategory(
        SPATIAL_RELATION_REASONING,
        "Spatial Relation Reasoning",
        "Reason over adjacency, containment, left/right, above/below, access, connection, and paths.",
        ("adjacent", "connected", "between", "left", "right", "above", "below", "access"),
    ),
    SkillCategory(
        COUNTING_ENUMERATION,
        "Counting / Enumeration",
        "Enumerate candidate objects, remove duplicates, apply counting rules, and produce exact counts.",
        ("how many", "count", "number of", "doors", "windows", "bedrooms", "toilets"),
    ),
    SkillCategory(
        ANSWER_SYNTHESIS,
        "Answer Synthesis",
        "Convert grounded evidence into the required concise answer format without unsupported explanation.",
        ("answer", "final", "format", "concise"),
    ),
    SkillCategory(
        VERIFICATION_REFLECTION,
        "Verification / Reflection",
        "Check whether the proposed answer is supported by visible evidence and catch common failure modes.",
        ("verify", "check", "reflect", "mistake", "contradiction"),
    ),
    SkillCategory(
        SKILL_LIBRARY_MANAGEMENT,
        "Skill Library Management",
        "Maintain, validate, merge, prune, and replay skills as reusable benchmark assets.",
        ("merge", "prune", "validate", "replay", "library", "utility"),
    ),
)

FIXED_CATEGORY_IDS: Tuple[str, ...] = tuple(category.category_id for category in SKILL_CATEGORIES)
CATEGORY_BY_ID: Dict[str, SkillCategory] = {category.category_id: category for category in SKILL_CATEGORIES}

_CATEGORY_ALIASES = {
    category.category_id: category.category_id for category in SKILL_CATEGORIES
}
_CATEGORY_ALIASES.update(
    {
        "visual": VISUAL_EVIDENCE_ACQUISITION,
        "visual_evidence": VISUAL_EVIDENCE_ACQUISITION,
        "ocr": TEXT_OCR_GROUNDING,
        "text": TEXT_OCR_GROUNDING,
        "text_ocr": TEXT_OCR_GROUNDING,
        "symbol": SYMBOL_GEOMETRY_GROUNDING,
        "geometry": SYMBOL_GEOMETRY_GROUNDING,
        "spatial": SPATIAL_RELATION_REASONING,
        "spatial_reasoning": SPATIAL_RELATION_REASONING,
        "counting": COUNTING_ENUMERATION,
        "enumeration": COUNTING_ENUMERATION,
        "answer": ANSWER_SYNTHESIS,
        "synthesis": ANSWER_SYNTHESIS,
        "verification": VERIFICATION_REFLECTION,
        "reflection": VERIFICATION_REFLECTION,
        "management": SKILL_LIBRARY_MANAGEMENT,
        "library_management": SKILL_LIBRARY_MANAGEMENT,
    }
)

QA_TYPE_CATEGORY_HINTS: Dict[str, Tuple[str, ...]] = {
    "ocr_qa": (VISUAL_EVIDENCE_ACQUISITION, TEXT_OCR_GROUNDING, ANSWER_SYNTHESIS, VERIFICATION_REFLECTION),
    "spatial_qa": (
        VISUAL_EVIDENCE_ACQUISITION,
        SYMBOL_GEOMETRY_GROUNDING,
        SPATIAL_RELATION_REASONING,
        ANSWER_SYNTHESIS,
        VERIFICATION_REFLECTION,
    ),
    "counting_qa": (
        VISUAL_EVIDENCE_ACQUISITION,
        SYMBOL_GEOMETRY_GROUNDING,
        COUNTING_ENUMERATION,
        ANSWER_SYNTHESIS,
        VERIFICATION_REFLECTION,
    ),
    "comparison_qa": (
        VISUAL_EVIDENCE_ACQUISITION,
        TEXT_OCR_GROUNDING,
        SPATIAL_RELATION_REASONING,
        ANSWER_SYNTHESIS,
        VERIFICATION_REFLECTION,
    ),
    "object_counting": (
        VISUAL_EVIDENCE_ACQUISITION,
        SYMBOL_GEOMETRY_GROUNDING,
        COUNTING_ENUMERATION,
        VERIFICATION_REFLECTION,
    ),
}

TASK_KEYWORD_HINTS: Tuple[Tuple[Tuple[str, ...], Tuple[str, ...]], ...] = (
    (("read", "area", "text", "dimension", "label", "ocr"), (TEXT_OCR_GROUNDING, VISUAL_EVIDENCE_ACQUISITION)),
    (("adjacency", "access", "connect", "left", "right", "nearest"), (SPATIAL_RELATION_REASONING, SYMBOL_GEOMETRY_GROUNDING)),
    (("largest", "smallest", "compare", "more", "less"), (TEXT_OCR_GROUNDING, SPATIAL_RELATION_REASONING)),
    (("count", "number", "enumerate"), (COUNTING_ENUMERATION, SYMBOL_GEOMETRY_GROUNDING)),
)

QUESTION_KEYWORD_HINTS: Tuple[Tuple[Tuple[str, ...], Tuple[str, ...]], ...] = (
    (("how many", "number of", "count"), (COUNTING_ENUMERATION, SYMBOL_GEOMETRY_GROUNDING)),
    (("area", "size", "m2", "square", "label", "text"), (TEXT_OCR_GROUNDING,)),
    (("which room", "connected", "adjacent", "directly", "left", "right", "above", "below"), (SPATIAL_RELATION_REASONING,)),
    (("door", "window", "wall", "toilet", "bedroom", "space"), (SYMBOL_GEOMETRY_GROUNDING,)),
    (("largest", "smallest", "bigger", "smaller", "more", "less"), (TEXT_OCR_GROUNDING, SPATIAL_RELATION_REASONING)),
)


def normalize_category(category: str) -> str:
    """Normalize a category or alias into one of the fixed category IDs."""
    key = (category or "").strip().lower().replace("-", "_").replace("/", "_").replace(" ", "_")
    normalized = _CATEGORY_ALIASES.get(key)
    if not normalized:
        raise ValueError(
            "Unsupported skill category '{}'. Supported categories: {}".format(
                category,
                ", ".join(FIXED_CATEGORY_IDS),
            )
        )
    return normalized


def category_title(category_id: str) -> str:
    """Return the display title for a category."""
    return CATEGORY_BY_ID[normalize_category(category_id)].title


def _add_unique(target: List[str], values: Sequence[str]) -> None:
    for value in values:
        normalized = normalize_category(value)
        if normalized not in target:
            target.append(normalized)


def infer_categories(
    qa_type: str = "",
    task: str = "",
    question: str = "",
    max_categories: int = 5,
    include_answer_and_verification: bool = True,
) -> List[str]:
    """Infer likely skill categories for a benchmark row or prompt.

    This is intentionally conservative. It routes examples and skills but does
    not define the concrete skills themselves.
    """
    inferred: List[str] = []
    qa_type_key = (qa_type or "").strip().lower()
    if qa_type_key in QA_TYPE_CATEGORY_HINTS:
        _add_unique(inferred, QA_TYPE_CATEGORY_HINTS[qa_type_key])

    searchable = " ".join([task or "", question or ""]).lower()
    for keywords, categories in TASK_KEYWORD_HINTS:
        if any(keyword in searchable for keyword in keywords):
            _add_unique(inferred, categories)
    for keywords, categories in QUESTION_KEYWORD_HINTS:
        if any(keyword in searchable for keyword in keywords):
            _add_unique(inferred, categories)

    if not inferred:
        _add_unique(inferred, (VISUAL_EVIDENCE_ACQUISITION,))

    if include_answer_and_verification:
        _add_unique(inferred, (ANSWER_SYNTHESIS, VERIFICATION_REFLECTION))

    return inferred[:max_categories]


def format_category_reference() -> str:
    """Return a compact category reference for generator prompts and reports."""
    lines = []
    for category in SKILL_CATEGORIES:
        lines.append("- {} ({}): {}".format(category.category_id, category.title, category.purpose))
    return "\n".join(lines)
