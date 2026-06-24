"""Generate candidate skills from evolution cases."""
import json
import re
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional, Sequence

from src.utils.openai_compatible import chat_completion_content

from .categories import (
    EVIDENCE_VERIFICATION,
    GRAPHIC_SYMBOL_GROUNDING,
    QUANTITATIVE_SET_REASONING,
    QUERY_ANSWER_BINDING,
    REGION_BOUNDARY_GROUNDING,
    SPATIAL_TOPOLOGY_MODELING,
    TEXT_ANNOTATION_GROUNDING,
    VIEW_CONTROL,
    format_category_reference,
)
from .cases import EvolutionCase
from .contracts import SkillContract, SkillLibrary, SkillUtility, make_skill_id, utc_now_iso
from .validator import SkillContractValidator


class SkillEvolutionGenerator:
    """Generate self-evolution skill candidates.

    LLM generation is the primary path. The heuristic path is intentionally
    conservative and useful for dry runs and smoke tests without spending API
    tokens.
    """

    def __init__(
        self,
        generator_model: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: float = 0.0,
        timeout: int = 90,
        max_retries: int = 3,
        extra_body: Optional[Dict[str, Any]] = None,
    ):
        self.generator_model = generator_model
        self.api_key = api_key
        self.base_url = base_url
        self.temperature = temperature
        self.timeout = timeout
        self.max_retries = max_retries
        self.extra_body = extra_body
        self.validator = SkillContractValidator()

    def generate(
        self,
        cases: Sequence[EvolutionCase],
        existing_library: Optional[SkillLibrary] = None,
        max_skills: int = 8,
        use_llm: bool = True,
    ) -> SkillLibrary:
        if use_llm and self.generator_model:
            skills = self.generate_with_llm(cases, existing_library=existing_library, max_skills=max_skills)
        else:
            skills = self.generate_heuristic_candidates(cases, max_skills=max_skills)

        library = SkillLibrary.empty()
        library.metadata.update(
            {
                "created_by": "skill_evolution_generator",
                "created_at": utc_now_iso(),
                "generation_mode": "llm" if use_llm and self.generator_model else "heuristic_dry_run",
                "case_count": len(cases),
            }
        )
        for skill in skills:
            result = self.validator.validate_skill(skill, cases=cases)
            skill.provenance["validation"] = result.to_dict()
            if not result.passed:
                skill.status = "rejected"
            library.add_or_replace(skill)
        return library

    def generate_with_llm(
        self,
        cases: Sequence[EvolutionCase],
        existing_library: Optional[SkillLibrary] = None,
        max_skills: int = 8,
    ) -> List[SkillContract]:
        prompt = build_generation_prompt(cases, existing_library=existing_library, max_skills=max_skills)
        content = chat_completion_content(
            model=self.generator_model,
            messages=[{"role": "user", "content": prompt}],
            api_key=self.api_key,
            base_url=self.base_url,
            temperature=self.temperature,
            response_format={"type": "json_object"},
            extra_body=self.extra_body,
            timeout=self.timeout,
            max_retries=self.max_retries,
            request_label="Skill evolution candidate generation",
        )
        data = _extract_json_object(content)
        skills_data = data.get("skills", [])
        if not isinstance(skills_data, list):
            raise ValueError("Skill generator response must contain a JSON array field named 'skills'")
        skills = []
        for item in skills_data[:max_skills]:
            if isinstance(item, dict):
                skills.append(SkillContract.from_dict(item))
        return skills

    def generate_heuristic_candidates(
        self,
        cases: Sequence[EvolutionCase],
        max_skills: int = 8,
    ) -> List[SkillContract]:
        grouped = _case_utility_by_category(cases)
        skills: List[SkillContract] = []
        for category, stats in sorted(
            grouped.items(),
            key=lambda item: (item[1]["fixed"] - item[1]["regressed"], item[1]["fixed"]),
            reverse=True,
        ):
            if len(skills) >= max_skills:
                break
            if category not in HEURISTIC_SKILL_BLUEPRINTS:
                continue
            blueprint = HEURISTIC_SKILL_BLUEPRINTS[category]
            utility = SkillUtility(
                fixed=stats["fixed"],
                regressed=stats["regressed"],
                support=stats["support"],
                confidence=_confidence(stats["fixed"], stats["regressed"], stats["support"]),
                notes="Heuristic dry-run candidate derived from fixed/regressed case clusters.",
            )
            source_cases = [case.case_id for case in stats["examples"][:12]]
            skill = SkillContract(
                skill_id=make_skill_id(category, blueprint["trigger"]),
                category=category,
                level=blueprint["level"],
                trigger=blueprint["trigger"],
                preconditions=blueprint["preconditions"],
                observations=blueprint["observations"],
                actions=blueprint["actions"],
                validator=blueprint["validator"],
                failure_modes=blueprint["failure_modes"],
                source_cases=source_cases,
                utility=utility,
                risk=blueprint.get("risk", "medium"),
                status="candidate",
                provenance={
                    "method": "heuristic_dry_run",
                    "category_case_stats": {
                        "fixed": stats["fixed"],
                        "regressed": stats["regressed"],
                        "support": stats["support"],
                    },
                },
            )
            skills.append(skill)
        return skills


def build_generation_prompt(
    cases: Sequence[EvolutionCase],
    existing_library: Optional[SkillLibrary] = None,
    max_skills: int = 8,
) -> str:
    selected_cases = _select_compact_cases(cases, limit=80)
    case_payload = []
    for case in selected_cases:
        case_payload.append(
            {
                "case_id": case.case_id,
                "task_family": case.task_family,
                "qa_type": case.qa_type,
                "task": case.task,
                "question": case.question,
                "baseline_answer": _truncate(case.baseline_answer, 360),
                "candidate_answer": _truncate(case.candidate_answer, 360),
                "baseline_score": case.baseline_score,
                "candidate_score": case.candidate_score,
                "outcome": case.outcome,
                "categories": case.categories,
            }
        )

    existing_payload: List[Dict[str, Any]] = []
    if existing_library:
        for skill in existing_library.skills[:20]:
            existing_payload.append(
                {
                    "skill_id": skill.skill_id,
                    "category": skill.category,
                    "trigger": skill.trigger,
                    "status": skill.status,
                    "utility": skill.utility.to_dict(),
                }
            )

    schema = {
        "skills": [
            {
                "skill_id": "category__short_descriptive_slug",
                "category": "one of the fixed category IDs",
                "level": "strategic|functional|atomic",
                "trigger": "when this skill should fire",
                "preconditions": ["conditions that must be true before using it"],
                "observations": ["evidence to inspect"],
                "actions": ["procedural steps; no case-specific answers"],
                "validator": ["checks that can reject or revise the answer"],
                "failure_modes": ["known ways this skill fails"],
                "source_cases": ["case ids that motivated the skill"],
                "utility": {"fixed": 0, "regressed": 0, "support": 0, "confidence": 0.0, "notes": ""},
                "risk": "low|medium|high",
                "status": "candidate",
                "parents": [],
                "children": [],
                "provenance": {"method": "contrastive_trace_extraction"},
            }
        ]
    }

    return """You are designing a self-evolving skill library for architectural/engineering drawing QA.

Important design rule:
- Humans define only the fixed category ontology and schema.
- Concrete skills must be inferred from success/failure contrast in benchmark traces.
- Each concrete skill must have exactly one primary category. If a procedure spans multiple categories, split it into smaller skills.
- Categories are orthogonal intermediate-representation transforms; do not create broad end-to-end "read then count then answer" skills.
- Do not copy exact ground-truth answers, candidate answers, image ids, or QA ids into trigger/actions/validator text.
- You may put motivating case ids only in source_cases.
- Prefer general reusable procedures that can survive held-out replay.

Fixed categories:
{categories}

Existing library summary:
{existing}

Benchmark comparison cases:
{cases}

Return ONLY valid JSON matching this shape:
{schema}

Generate at most {max_skills} high-precision candidate skills. Favor skills that explain fixed cases while avoiding regressed cases. Use source_cases to cite the cases that motivated each skill.""".format(
        categories=format_category_reference(),
        existing=json.dumps(existing_payload, ensure_ascii=False, indent=2),
        cases=json.dumps(case_payload, ensure_ascii=False, indent=2),
        schema=json.dumps(schema, ensure_ascii=False, indent=2),
        max_skills=max_skills,
    )


def _case_utility_by_category(cases: Sequence[EvolutionCase]) -> Dict[str, Dict[str, Any]]:
    grouped: Dict[str, Dict[str, Any]] = defaultdict(lambda: {"fixed": 0, "regressed": 0, "support": 0, "examples": []})
    for case in cases:
        for category in case.categories:
            grouped[category]["support"] += 1
            if case.outcome == "fixed":
                grouped[category]["fixed"] += 1
            elif case.outcome == "regressed":
                grouped[category]["regressed"] += 1
            if case.outcome in {"fixed", "regressed"} and len(grouped[category]["examples"]) < 20:
                grouped[category]["examples"].append(case)
    return grouped


def _select_compact_cases(cases: Sequence[EvolutionCase], limit: int) -> List[EvolutionCase]:
    priority = {"fixed": 0, "regressed": 1, "still_wrong": 2, "still_correct": 3}
    return sorted(cases, key=lambda case: (priority.get(case.outcome, 9), case.case_id))[:limit]


def _confidence(fixed: int, regressed: int, support: int) -> float:
    if support <= 0:
        return 0.0
    return max(0.0, min(1.0, (fixed - regressed) / float(support)))


def _truncate(value: str, limit: int) -> str:
    value = str(value or "")
    if len(value) <= limit:
        return value
    return value[: limit - 3] + "..."


def _extract_json_object(content: str) -> Dict[str, Any]:
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", content or "", re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(0))


HEURISTIC_SKILL_BLUEPRINTS: Dict[str, Dict[str, Any]] = {
    VIEW_CONTROL: {
        "level": "functional",
        "trigger": "the evidence is small, blurry, rotated, visually dense, or spread across overview and local detail",
        "preconditions": ["A drawing image is available and the next step needs a clearer evidence view."],
        "observations": ["global layout", "candidate local region", "orientation cues", "nearby context"],
        "actions": [
            "Start with a full-plan overview to locate the relevant zone.",
            "Select the smallest local view that preserves enough surrounding context.",
            "Mentally rotate or re-orient the view when labels or symbols are angled.",
            "If local evidence is ambiguous, inspect adjacent context before passing the view onward.",
        ],
        "validator": [
            "The selected view must contain the target evidence and enough context to avoid adjacent-room confusion.",
            "Do not infer text, symbols, or relations in this skill; only decide where and how to inspect.",
        ],
        "failure_modes": ["over-cropping away context", "missing tiny symbols", "wrong orientation"],
        "risk": "medium",
    },
    TEXT_ANNOTATION_GROUNDING: {
        "level": "functional",
        "trigger": "the answer depends on room names, dimensions, legends, numbers, identifiers, or written labels",
        "preconditions": ["A targeted evidence view contains readable or partially readable annotation text."],
        "observations": ["target text span", "unit markers", "room names", "legend entries", "adjacent labels"],
        "actions": [
            "Locate the exact annotation span tied to the queried entity.",
            "Read the text together with units, decimal points, abbreviations, and nearby qualifiers.",
            "Normalize common room-name and unit variants without changing numeric values.",
            "Keep the text linked to its visible position for downstream region or answer binding.",
        ],
        "validator": [
            "Reject the read if another nearby label is spatially closer to the queried region.",
            "Preserve numeric values, units, and sign/decimal punctuation exactly unless normalization is explicit.",
        ],
        "failure_modes": ["digit transcription error", "wrong adjacent label", "dropping units", "legend-label mismatch"],
        "risk": "medium",
    },
    GRAPHIC_SYMBOL_GROUNDING: {
        "level": "functional",
        "trigger": "the answer depends on identifying doors, windows, fixtures, stairs, furniture, or other graphic symbols",
        "preconditions": ["A targeted evidence view contains candidate graphic symbols."],
        "observations": ["symbol shape", "wall attachment", "opening arc/line", "fixture geometry", "legend cue"],
        "actions": [
            "Identify symbol candidates by visual form before applying task-specific counting or relation rules.",
            "Separate visually similar classes such as doors, windows, sliding openings, fixtures, and furniture.",
            "Record each symbol as a typed instance with its approximate location or owning boundary.",
            "Leave room-region ownership and counting decisions to downstream boundary or set reasoning skills.",
        ],
        "validator": [
            "Do not count or answer directly in this skill; output typed symbol candidates only.",
            "Reject a symbol type if its attachment or shape better matches a different class.",
        ],
        "failure_modes": ["door/window confusion", "fixture mistaken for text", "furniture counted as architectural symbol"],
        "risk": "medium",
    },
    REGION_BOUNDARY_GROUNDING: {
        "level": "functional",
        "trigger": "the answer depends on walls, room outlines, openings, enclosed spaces, zones, or object ownership by room",
        "preconditions": ["Visual primitives, symbols, or labels suggest room or region boundaries."],
        "observations": ["walls", "room outlines", "openings", "enclosed regions", "labels inside regions"],
        "actions": [
            "Trace walls and openings to recover candidate enclosed regions or floor zones.",
            "Attach labels and symbols to regions only when they fall inside or on the relevant boundary.",
            "Distinguish a true room/space boundary from furniture lines, dimension lines, and decorative graphics.",
            "Output regions, boundaries, openings, and ownership links for downstream topology or counting.",
        ],
        "validator": [
            "A region must be supported by visible boundary evidence or a clearly labeled zone.",
            "Do not infer adjacency, counts, or final answers here; output boundary-grounded regions.",
        ],
        "failure_modes": ["dimension line treated as wall", "open area treated as closed room", "wrong symbol ownership"],
        "risk": "medium",
    },
    SPATIAL_TOPOLOGY_MODELING: {
        "level": "functional",
        "trigger": "the question asks about adjacency, containment, direct connection, direction, paths, or which region owns an object",
        "preconditions": ["Grounded regions, symbols, labels, or boundaries are available."],
        "observations": ["regions", "shared boundaries", "openings", "directional frame", "object ownership links"],
        "actions": [
            "Build the minimal relation graph needed by the query.",
            "Separate visual adjacency from direct access through a door or opening.",
            "Resolve left/right/above/below from the drawing orientation and spatial layout.",
            "Represent containment and ownership explicitly before selecting a related room or object.",
        ],
        "validator": [
            "Reject a relation if no shared boundary, containment cue, or required opening supports it.",
            "Do not perform counting or final answer formatting inside this skill.",
        ],
        "failure_modes": ["proximity mistaken for connection", "left/right flip", "ownership assigned to neighboring region"],
        "risk": "medium",
    },
    QUANTITATIVE_SET_REASONING: {
        "level": "functional",
        "trigger": "the question asks for counts, candidate enumeration, deduplication, grouping, or quantitative comparison",
        "preconditions": ["Grounded symbols, regions, annotations, or relation graph nodes are available."],
        "observations": ["candidate set", "exclusion rules", "duplicates", "groups", "numeric annotations"],
        "actions": [
            "Define the candidate set implied by the question before counting or comparing.",
            "Apply inclusion and exclusion rules to each candidate.",
            "Deduplicate grouped or multi-part objects without deleting distinct instances.",
            "Compute the count, selected set, or comparison result from the retained candidates only.",
        ],
        "validator": [
            "The final number or comparison must match the retained candidate set.",
            "Re-check tiny symbols, grouped openings, repeated labels, and region-scoped exclusions.",
        ],
        "failure_modes": ["missed candidate", "double count", "wrong denominator", "comparing labels from different units"],
        "risk": "medium",
    },
    QUERY_ANSWER_BINDING: {
        "level": "atomic",
        "trigger": "grounded evidence exists and the question must be mapped to a concise answer type and format",
        "preconditions": ["A question and at least one grounded evidence artifact are available."],
        "observations": ["question intent", "requested entity", "answer type", "required unit or format"],
        "actions": [
            "Identify whether the question asks for text, number, object class, room name, relation, or comparison.",
            "Select the evidence artifact that directly answers that intent.",
            "Return the minimal answer with required unit, qualifier, or normalized label.",
            "Avoid adding reasoning or unsupported context to the final answer.",
        ],
        "validator": [
            "The answer must directly satisfy the question type.",
            "Reject answers that include extra claims not represented in the selected evidence artifact.",
        ],
        "failure_modes": ["answering a related question", "dropping units", "over-explaining", "using wrong evidence type"],
        "risk": "low",
    },
    EVIDENCE_VERIFICATION: {
        "level": "functional",
        "trigger": "a draft answer exists and needs support checking before finalization",
        "preconditions": ["A draft answer and its evidence trace are available."],
        "observations": ["draft answer", "evidence trace", "nearby alternatives", "common error modes"],
        "actions": [
            "Check that every answer element is supported by the cited visual, text, symbol, region, relation, or set evidence.",
            "Look for one plausible missed or confused alternative before finalizing.",
            "Correct the draft only when evidence contradicts it or a required candidate was omitted.",
            "If evidence is insufficient, prefer a conservative answer over unsupported specificity.",
        ],
        "validator": [
            "Reject the draft if the evidence trace supports a different label, symbol, relation, or count.",
            "Do not change a supported answer based on speculation without visible evidence.",
        ],
        "failure_modes": ["confirmation bias", "unsupported correction", "reflection drift", "missed contradiction"],
        "risk": "medium",
    },
}
