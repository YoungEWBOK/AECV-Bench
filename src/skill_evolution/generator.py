"""Generate candidate skills from evolution cases."""
import json
import re
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional, Sequence

from src.utils.openai_compatible import chat_completion_content

from .categories import (
    ANSWER_SYNTHESIS,
    COUNTING_ENUMERATION,
    SPATIAL_RELATION_REASONING,
    SKILL_CATEGORIES,
    SYMBOL_GEOMETRY_GROUNDING,
    TEXT_OCR_GROUNDING,
    VERIFICATION_REFLECTION,
    VISUAL_EVIDENCE_ACQUISITION,
    category_title,
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
    VISUAL_EVIDENCE_ACQUISITION: {
        "level": "functional",
        "trigger": "the question depends on small symbols, local labels, or visually dense drawing regions",
        "preconditions": ["A drawing image is available and the answer depends on visible evidence."],
        "observations": ["global layout first", "target room or symbol region", "nearby labels and boundaries"],
        "actions": [
            "Form an overview of the full plan before focusing on the target region.",
            "Identify the smallest region that can answer the question without losing context.",
            "If evidence is ambiguous, re-inspect neighboring labels, symbols, and room boundaries before answering.",
        ],
        "validator": [
            "Reject an answer if no visible region supports it.",
            "Reject an answer if a neighboring room or symbol could plausibly change the result.",
        ],
        "failure_modes": ["over-focusing on one crop", "missing tiny symbols", "using a nearby label from the wrong room"],
        "risk": "medium",
    },
    TEXT_OCR_GROUNDING: {
        "level": "functional",
        "trigger": "the question asks for room names, area text, labels, dimensions, or written annotations",
        "preconditions": ["The required answer is written or implied by text in the drawing."],
        "observations": ["target text", "unit markers", "room name", "adjacent labels that may be confused"],
        "actions": [
            "Locate the exact text region tied to the question entity.",
            "Read the label with its unit or abbreviation, preserving decimal values and room-name variants.",
            "Compare nearby labels to avoid copying text from an adjacent room.",
        ],
        "validator": [
            "The final answer must preserve the numeric value and unit when the question asks for a measurement.",
            "If two nearby labels are similar, choose the one spatially inside the target room boundary.",
        ],
        "failure_modes": ["digit transcription error", "wrong adjacent room label", "dropping measurement units"],
        "risk": "medium",
    },
    SYMBOL_GEOMETRY_GROUNDING: {
        "level": "functional",
        "trigger": "the question depends on doors, windows, walls, rooms, fixtures, boundaries, or floor-plan symbols",
        "preconditions": ["The answer depends on interpreting floor-plan geometry or architectural symbols."],
        "observations": ["symbol type", "room boundary", "wall openings", "fixture labels"],
        "actions": [
            "Identify the target symbol class and distinguish it from visually similar classes.",
            "Use walls and room boundaries to decide which room owns each symbol.",
            "Apply the benchmark's symbol rules before converting symbols into an answer.",
        ],
        "validator": [
            "Reject counts or relations that mix doors, sliding openings, and windows without applying the rules.",
            "Check that each symbol is attached to the intended room or boundary.",
        ],
        "failure_modes": ["door/window confusion", "fixture counted as room", "boundary ownership error"],
        "risk": "medium",
    },
    SPATIAL_RELATION_REASONING: {
        "level": "functional",
        "trigger": "the question asks which room is connected, adjacent, accessible, left/right, above/below, or spatially related",
        "preconditions": ["The answer depends on a spatial relation between rooms, walls, doors, openings, or labeled regions."],
        "observations": ["target room", "neighboring rooms", "doors/openings", "shared boundaries", "directional cues"],
        "actions": [
            "Locate the named anchor room or region first.",
            "Trace the relevant boundary, door, or opening before naming the related room.",
            "Separate direct access from visual proximity or adjacency without a passable opening.",
            "Resolve direction words from the drawing orientation, not from text order in the prompt.",
        ],
        "validator": [
            "Reject a room name if no visible shared boundary or required opening supports the relation.",
            "Reject answers that confuse adjacent rooms with directly connected rooms.",
        ],
        "failure_modes": ["proximity mistaken for connection", "left/right flip", "using a neighboring label from the wrong side"],
        "risk": "medium",
    },
    COUNTING_ENUMERATION: {
        "level": "functional",
        "trigger": "the question asks for an exact number of objects, rooms, windows, doors, bedrooms, toilets, or spaces",
        "preconditions": ["The answer is a count and candidate objects can be enumerated from the drawing."],
        "observations": ["candidate objects", "deduplication boundaries", "exclusion rules"],
        "actions": [
            "Enumerate candidates before deciding the final number.",
            "Apply inclusion and exclusion rules to each candidate.",
            "Remove duplicates caused by adjacent symbols or multi-leaf/grouped openings.",
            "Only then produce the concise count.",
        ],
        "validator": [
            "The final count must equal the number of retained candidates after exclusions.",
            "Re-check tiny bathroom/toilet windows and adjacent grouped openings.",
        ],
        "failure_modes": ["missed small object", "double count", "wrong inclusion rule"],
        "risk": "medium",
    },
    ANSWER_SYNTHESIS: {
        "level": "atomic",
        "trigger": "after evidence has been grounded and the answer must be returned concisely",
        "preconditions": ["The supporting evidence has been identified."],
        "observations": ["requested output type", "answer unit", "grounded entity"],
        "actions": [
            "Return the exact requested entity, number, or text span.",
            "Keep the final answer short and avoid unsupported explanation.",
            "Preserve units and qualifiers that are required by the question.",
        ],
        "validator": ["The answer directly addresses the question and contains no extra unsupported claims."],
        "failure_modes": ["over-explaining", "dropping units", "answering a related but different question"],
        "risk": "low",
    },
    VERIFICATION_REFLECTION: {
        "level": "functional",
        "trigger": "before finalizing any answer that depends on OCR, counting, symbol interpretation, or spatial relation",
        "preconditions": ["A draft answer exists."],
        "observations": ["draft answer", "supporting evidence", "common error modes"],
        "actions": [
            "Check whether the draft answer is supported by the same region used to infer it.",
            "Actively look for one plausible alternative answer or overlooked object.",
            "Revise only when the visual evidence contradicts the draft.",
        ],
        "validator": [
            "Reject the draft if the evidence region supports a different room, label, symbol, or count.",
            "Do not change the draft based on speculation without visible evidence.",
        ],
        "failure_modes": ["self-reflection changes a correct answer", "confirmation bias", "unsupported correction"],
        "risk": "medium",
    },
}

# These categories usually emerge as library-level controls rather than direct QA
# prompt skills, so the dry-run generator does not emit them by default.
HEURISTIC_SKILL_BLUEPRINTS.setdefault(
    "skill_library_management",
    {
        "level": "strategic",
        "trigger": "when merging or pruning skill candidates after replay validation",
        "preconditions": ["Replay results and validation issues are available."],
        "observations": ["net gain", "regressions", "validation errors", "source case coverage"],
        "actions": [
            "Accept only skills with positive replay utility and no hard validation errors.",
            "Reject skills that copy case answers or mention case ids outside source_cases.",
            "Merge duplicate triggers only when the combined validator remains precise.",
        ],
        "validator": ["Accepted skills must improve held-out replay without excessive regressions."],
        "failure_modes": ["overfitting to discovery cases", "trigger too broad", "duplicate skill drift"],
        "risk": "low",
    },
)
