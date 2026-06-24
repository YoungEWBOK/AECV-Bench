"""Fixed, mostly orthogonal skill categories for AEC drawing QA self-evolution.

The ontology is organized by intermediate representation rather than by a
mixed processing narrative. Each category owns one transformation boundary, so a
concrete skill should have exactly one primary category. Larger procedures
should be decomposed into multiple skills that compose through evidence views,
grounded entities, relation graphs, sets, answers, and verification traces.
"""
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple


VIEW_CONTROL = "view_control"
TEXT_ANNOTATION_GROUNDING = "text_annotation_grounding"
GRAPHIC_SYMBOL_GROUNDING = "graphic_symbol_grounding"
REGION_BOUNDARY_GROUNDING = "region_boundary_grounding"
SPATIAL_TOPOLOGY_MODELING = "spatial_topology_modeling"
QUANTITATIVE_SET_REASONING = "quantitative_set_reasoning"
QUERY_ANSWER_BINDING = "query_answer_binding"
EVIDENCE_VERIFICATION = "evidence_verification"


@dataclass(frozen=True)
class SkillCategory:
    """A fixed skill category."""

    category_id: str
    title: str
    purpose: str
    input_artifact: str
    output_artifact: str
    default_triggers: Tuple[str, ...]


SKILL_CATEGORIES: Tuple[SkillCategory, ...] = (
    SkillCategory(
        VIEW_CONTROL,
        "View Control",
        "Choose how to inspect the drawing: overview, zoom, crop, rotate, and local high-resolution evidence views.",
        "raw image or current view",
        "targeted evidence view",
        ("overview", "zoom", "crop", "rotate", "small text", "tiny symbol", "unclear"),
    ),
    SkillCategory(
        TEXT_ANNOTATION_GROUNDING,
        "Text Annotation Grounding",
        "Read and normalize written annotations such as room names, dimensions, legends, identifiers, and labels.",
        "targeted evidence view",
        "positioned text spans and normalized annotations",
        ("text", "label", "area", "size", "dimension", "name", "legend", "number"),
    ),
    SkillCategory(
        GRAPHIC_SYMBOL_GROUNDING,
        "Graphic Symbol Grounding",
        "Identify typed graphical instances such as doors, windows, fixtures, stairs, furniture, and legend symbols.",
        "targeted evidence view",
        "typed symbol instances",
        ("door", "window", "fixture", "toilet", "sink", "stair", "symbol", "furniture"),
    ),
    SkillCategory(
        REGION_BOUNDARY_GROUNDING,
        "Region Boundary Grounding",
        "Recover spatial regions and boundaries such as walls, room outlines, openings, enclosed spaces, and floor zones.",
        "visual primitives and grounded annotations",
        "rooms, regions, boundaries, and openings",
        ("room", "wall", "boundary", "space", "region", "enclosed", "opening", "floor"),
    ),
    SkillCategory(
        SPATIAL_TOPOLOGY_MODELING,
        "Spatial Topology Modeling",
        "Build relation graphs over grounded entities: adjacency, containment, connection, direction, paths, and ownership.",
        "grounded entities and regions",
        "spatial relation graph",
        ("adjacent", "connected", "between", "left", "right", "above", "below", "inside", "access"),
    ),
    SkillCategory(
        QUANTITATIVE_SET_REASONING,
        "Quantitative Set Reasoning",
        "Define candidate sets, deduplicate, group, count by region, and compare quantities or measurements.",
        "entities, annotations, and relation graph",
        "counts, comparisons, and selected sets",
        ("how many", "count", "number of", "enumerate", "largest", "smallest", "more", "less"),
    ),
    SkillCategory(
        QUERY_ANSWER_BINDING,
        "Query Answer Binding",
        "Bind the natural-language question to the needed evidence, answer type, unit, and concise response format.",
        "question and evidence graph",
        "answer candidate with evidence requirements",
        ("question", "answer", "format", "which", "what", "final", "unit"),
    ),
    SkillCategory(
        EVIDENCE_VERIFICATION,
        "Evidence Verification",
        "Check that the answer is supported by visible evidence and catch omissions, misreads, duplicates, and contradictions.",
        "answer candidate and evidence trace",
        "verified or corrected answer",
        ("verify", "check", "reflect", "mistake", "contradiction", "missed", "duplicate"),
    ),
)

FIXED_CATEGORY_IDS: Tuple[str, ...] = tuple(category.category_id for category in SKILL_CATEGORIES)
CATEGORY_BY_ID: Dict[str, SkillCategory] = {category.category_id: category for category in SKILL_CATEGORIES}

_CATEGORY_ALIASES = {category.category_id: category.category_id for category in SKILL_CATEGORIES}
_CATEGORY_ALIASES.update(
    {
        "view": VIEW_CONTROL,
        "visual": VIEW_CONTROL,
        "visual_evidence": VIEW_CONTROL,
        "visual_evidence_acquisition": VIEW_CONTROL,
        "evidence_acquisition": VIEW_CONTROL,
        "ocr": TEXT_ANNOTATION_GROUNDING,
        "text": TEXT_ANNOTATION_GROUNDING,
        "text_ocr": TEXT_ANNOTATION_GROUNDING,
        "text_ocr_grounding": TEXT_ANNOTATION_GROUNDING,
        "annotation": TEXT_ANNOTATION_GROUNDING,
        "text_grounding": TEXT_ANNOTATION_GROUNDING,
        "symbol": GRAPHIC_SYMBOL_GROUNDING,
        "graphic_symbol": GRAPHIC_SYMBOL_GROUNDING,
        "geometry": GRAPHIC_SYMBOL_GROUNDING,
        "symbol_geometry": GRAPHIC_SYMBOL_GROUNDING,
        "symbol_geometry_grounding": GRAPHIC_SYMBOL_GROUNDING,
        "boundary": REGION_BOUNDARY_GROUNDING,
        "region": REGION_BOUNDARY_GROUNDING,
        "room_boundary": REGION_BOUNDARY_GROUNDING,
        "spatial": SPATIAL_TOPOLOGY_MODELING,
        "spatial_reasoning": SPATIAL_TOPOLOGY_MODELING,
        "spatial_relation": SPATIAL_TOPOLOGY_MODELING,
        "spatial_relation_reasoning": SPATIAL_TOPOLOGY_MODELING,
        "topology": SPATIAL_TOPOLOGY_MODELING,
        "counting": QUANTITATIVE_SET_REASONING,
        "enumeration": QUANTITATIVE_SET_REASONING,
        "counting_enumeration": QUANTITATIVE_SET_REASONING,
        "quantitative": QUANTITATIVE_SET_REASONING,
        "set_reasoning": QUANTITATIVE_SET_REASONING,
        "answer": QUERY_ANSWER_BINDING,
        "synthesis": QUERY_ANSWER_BINDING,
        "answer_synthesis": QUERY_ANSWER_BINDING,
        "query": QUERY_ANSWER_BINDING,
        "query_binding": QUERY_ANSWER_BINDING,
        "verification": EVIDENCE_VERIFICATION,
        "reflection": EVIDENCE_VERIFICATION,
        "verification_reflection": EVIDENCE_VERIFICATION,
        "evidence_check": EVIDENCE_VERIFICATION,
        # Library management is intentionally not a skill category anymore. Map
        # legacy libraries to evidence verification so old artifacts remain
        # loadable without reintroducing a non-inference category.
        "management": EVIDENCE_VERIFICATION,
        "library_management": EVIDENCE_VERIFICATION,
        "skill_library_management": EVIDENCE_VERIFICATION,
    }
)

QA_TYPE_CATEGORY_HINTS: Dict[str, Tuple[str, ...]] = {
    "ocr_qa": (
        VIEW_CONTROL,
        TEXT_ANNOTATION_GROUNDING,
        QUERY_ANSWER_BINDING,
        EVIDENCE_VERIFICATION,
    ),
    "spatial_qa": (
        VIEW_CONTROL,
        REGION_BOUNDARY_GROUNDING,
        SPATIAL_TOPOLOGY_MODELING,
        QUERY_ANSWER_BINDING,
        EVIDENCE_VERIFICATION,
    ),
    "counting_qa": (
        VIEW_CONTROL,
        GRAPHIC_SYMBOL_GROUNDING,
        REGION_BOUNDARY_GROUNDING,
        QUANTITATIVE_SET_REASONING,
        EVIDENCE_VERIFICATION,
    ),
    "comparison_qa": (
        VIEW_CONTROL,
        TEXT_ANNOTATION_GROUNDING,
        SPATIAL_TOPOLOGY_MODELING,
        QUANTITATIVE_SET_REASONING,
        QUERY_ANSWER_BINDING,
        EVIDENCE_VERIFICATION,
    ),
    "object_counting": (
        VIEW_CONTROL,
        GRAPHIC_SYMBOL_GROUNDING,
        REGION_BOUNDARY_GROUNDING,
        QUANTITATIVE_SET_REASONING,
        EVIDENCE_VERIFICATION,
    ),
}

TASK_KEYWORD_HINTS: Tuple[Tuple[Tuple[str, ...], Tuple[str, ...]], ...] = (
    (("read", "area", "text", "dimension", "label", "ocr", "legend"), (TEXT_ANNOTATION_GROUNDING, VIEW_CONTROL)),
    (("door", "window", "toilet", "fixture", "symbol", "stair"), (GRAPHIC_SYMBOL_GROUNDING, VIEW_CONTROL)),
    (("wall", "room", "space", "boundary", "region", "enclosed"), (REGION_BOUNDARY_GROUNDING, VIEW_CONTROL)),
    (("adjacency", "access", "connect", "left", "right", "nearest", "inside"), (SPATIAL_TOPOLOGY_MODELING, REGION_BOUNDARY_GROUNDING)),
    (("largest", "smallest", "compare", "more", "less"), (QUANTITATIVE_SET_REASONING, TEXT_ANNOTATION_GROUNDING)),
    (("count", "number", "enumerate"), (QUANTITATIVE_SET_REASONING, GRAPHIC_SYMBOL_GROUNDING, REGION_BOUNDARY_GROUNDING)),
)

QUESTION_KEYWORD_HINTS: Tuple[Tuple[Tuple[str, ...], Tuple[str, ...]], ...] = (
    (("how many", "number of", "count"), (QUANTITATIVE_SET_REASONING, GRAPHIC_SYMBOL_GROUNDING)),
    (("area", "size", "m2", "square", "label", "text", "dimension"), (TEXT_ANNOTATION_GROUNDING,)),
    (("which room", "connected", "adjacent", "directly", "left", "right", "above", "below", "inside"), (SPATIAL_TOPOLOGY_MODELING, REGION_BOUNDARY_GROUNDING)),
    (("door", "window", "wall", "toilet", "bedroom", "space", "room"), (GRAPHIC_SYMBOL_GROUNDING, REGION_BOUNDARY_GROUNDING)),
    (("largest", "smallest", "bigger", "smaller", "more", "less"), (QUANTITATIVE_SET_REASONING, TEXT_ANNOTATION_GROUNDING)),
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

    The result is a compact routing hint, not a claim that all listed skills
    should be merged. Concrete skills should still have one primary category.
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
        _add_unique(inferred, (VIEW_CONTROL,))

    if include_answer_and_verification:
        _add_unique(inferred, (QUERY_ANSWER_BINDING, EVIDENCE_VERIFICATION))

    return inferred[:max_categories]


def format_category_reference() -> str:
    """Return a compact category reference for generator prompts and reports."""
    lines = []
    for category in SKILL_CATEGORIES:
        lines.append(
            "- {} ({}): {} Input: {}. Output: {}.".format(
                category.category_id,
                category.title,
                category.purpose,
                category.input_artifact,
                category.output_artifact,
            )
        )
    lines.append(
        "\nGovernance note: skill-library merge, pruning, validation, replay, and failure-mode logging are framework governance operations, not primary inference skill categories."
    )
    return "\n".join(lines)
