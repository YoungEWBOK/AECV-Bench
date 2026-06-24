"""Skill contract and library primitives.

The contract follows the papers' shared pattern: a skill is not just a prompt
snippet, but a validated artifact with trigger, preconditions, actions,
validator, failure modes, source cases, utility, and lifecycle status.
"""
import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .categories import (
    CATEGORY_BY_ID,
    FIXED_CATEGORY_IDS,
    VERIFICATION_REFLECTION,
    category_title,
    infer_categories,
    normalize_category,
)


LIBRARY_VERSION = "1.0"
VALID_LEVELS = ("strategic", "functional", "atomic")
VALID_STATUSES = ("candidate", "accepted", "rejected", "deprecated")


def utc_now_iso() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def slugify(value: str, max_length: int = 72) -> str:
    """Create a stable ASCII-ish slug."""
    slug = re.sub(r"[^a-z0-9]+", "_", (value or "").lower()).strip("_")
    slug = re.sub(r"_+", "_", slug)
    return slug[:max_length].strip("_") or "skill"


@dataclass
class SkillUtility:
    """Observed utility of a skill candidate."""

    fixed: int = 0
    regressed: int = 0
    support: int = 0
    confidence: float = 0.0
    notes: str = ""

    @property
    def net_gain(self) -> int:
        return int(self.fixed) - int(self.regressed)

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "SkillUtility":
        data = data or {}
        fixed = int(data.get("fixed", 0) or 0)
        regressed = int(data.get("regressed", 0) or 0)
        support = int(data.get("support", data.get("shared", fixed + regressed)) or 0)
        confidence = float(data.get("confidence", 0.0) or 0.0)
        notes = str(data.get("notes", "") or "")
        return cls(fixed=fixed, regressed=regressed, support=support, confidence=confidence, notes=notes)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "fixed": int(self.fixed),
            "regressed": int(self.regressed),
            "net_gain": self.net_gain,
            "support": int(self.support),
            "confidence": round(float(self.confidence), 4),
            "notes": self.notes,
        }


@dataclass
class SkillContract:
    """A self-evolved skill contract."""

    skill_id: str
    category: str
    level: str
    trigger: str
    preconditions: List[str] = field(default_factory=list)
    observations: List[str] = field(default_factory=list)
    actions: List[str] = field(default_factory=list)
    validator: List[str] = field(default_factory=list)
    failure_modes: List[str] = field(default_factory=list)
    source_cases: List[str] = field(default_factory=list)
    utility: SkillUtility = field(default_factory=SkillUtility)
    risk: str = "medium"
    status: str = "candidate"
    parents: List[str] = field(default_factory=list)
    children: List[str] = field(default_factory=list)
    provenance: Dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=utc_now_iso)
    updated_at: str = field(default_factory=utc_now_iso)

    def __post_init__(self) -> None:
        self.category = normalize_category(self.category)
        self.level = (self.level or "functional").strip().lower()
        if self.level not in VALID_LEVELS:
            self.level = "functional"
        self.status = (self.status or "candidate").strip().lower()
        if self.status not in VALID_STATUSES:
            self.status = "candidate"
        if not self.skill_id:
            self.skill_id = make_skill_id(self.category, self.trigger or self.level)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SkillContract":
        return cls(
            skill_id=str(data.get("skill_id", "") or ""),
            category=str(data.get("category", "") or ""),
            level=str(data.get("level", "functional") or "functional"),
            trigger=str(data.get("trigger", "") or ""),
            preconditions=_string_list(data.get("preconditions")),
            observations=_string_list(data.get("observations")),
            actions=_string_list(data.get("actions")),
            validator=_string_list(data.get("validator")),
            failure_modes=_string_list(data.get("failure_modes")),
            source_cases=_string_list(data.get("source_cases")),
            utility=SkillUtility.from_dict(data.get("utility")),
            risk=str(data.get("risk", "medium") or "medium"),
            status=str(data.get("status", "candidate") or "candidate"),
            parents=_string_list(data.get("parents")),
            children=_string_list(data.get("children")),
            provenance=dict(data.get("provenance") or {}),
            created_at=str(data.get("created_at", utc_now_iso()) or utc_now_iso()),
            updated_at=str(data.get("updated_at", utc_now_iso()) or utc_now_iso()),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "skill_id": self.skill_id,
            "category": self.category,
            "category_title": category_title(self.category),
            "level": self.level,
            "trigger": self.trigger,
            "preconditions": list(self.preconditions),
            "observations": list(self.observations),
            "actions": list(self.actions),
            "validator": list(self.validator),
            "failure_modes": list(self.failure_modes),
            "source_cases": list(self.source_cases),
            "utility": self.utility.to_dict(),
            "risk": self.risk,
            "status": self.status,
            "parents": list(self.parents),
            "children": list(self.children),
            "provenance": dict(self.provenance),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    def score_for_prompt(self, inferred_categories: Sequence[str], question: str = "") -> float:
        """Score skill relevance for prompt injection."""
        score = 0.0
        if self.category in inferred_categories:
            score += 3.0
        trigger_words = set(re.findall(r"[a-z0-9]+", (self.trigger or "").lower()))
        question_words = set(re.findall(r"[a-z0-9]+", (question or "").lower()))
        if trigger_words and question_words:
            score += min(2.0, len(trigger_words & question_words) * 0.35)
        score += max(-2.0, min(2.0, self.utility.net_gain * 0.25))
        if self.category == VERIFICATION_REFLECTION:
            score += 0.4
        return score

    def prompt_block(self) -> str:
        """Render the skill as a compact private instruction block."""
        action_text = "; ".join(self.actions[:5])
        validator_text = "; ".join(self.validator[:3])
        failure_text = "; ".join(self.failure_modes[:3])
        parts = [
            "[{}] {} / {}".format(self.skill_id, category_title(self.category), self.level),
            "Trigger: {}".format(self.trigger),
        ]
        if self.preconditions:
            parts.append("Preconditions: {}".format("; ".join(self.preconditions[:3])))
        if self.observations:
            parts.append("Observe: {}".format("; ".join(self.observations[:4])))
        if action_text:
            parts.append("Actions: {}".format(action_text))
        if validator_text:
            parts.append("Validator: {}".format(validator_text))
        if failure_text:
            parts.append("Watch for: {}".format(failure_text))
        return "\n".join(parts)


@dataclass
class SkillLibrary:
    """A collection of skill contracts."""

    skills: List[SkillContract] = field(default_factory=list)
    version: str = LIBRARY_VERSION
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def empty(cls) -> "SkillLibrary":
        return cls(metadata={"created_at": utc_now_iso(), "category_ids": list(FIXED_CATEGORY_IDS)})

    @classmethod
    def load(cls, path: Optional[str], missing_ok: bool = False) -> "SkillLibrary":
        if not path:
            return cls.empty()
        library_path = Path(path)
        if not library_path.is_file():
            if missing_ok:
                return cls.empty()
            raise FileNotFoundError("Skill library not found: {}".format(library_path))
        with library_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            skills_data = data
            metadata = {}
            version = LIBRARY_VERSION
        else:
            skills_data = data.get("skills", [])
            metadata = dict(data.get("metadata") or {})
            version = str(data.get("version", LIBRARY_VERSION) or LIBRARY_VERSION)
        return cls(
            skills=[SkillContract.from_dict(item) for item in skills_data if isinstance(item, dict)],
            version=version,
            metadata=metadata,
        )

    def save(self, path: str) -> None:
        library_path = Path(path)
        if library_path.parent:
            library_path.parent.mkdir(parents=True, exist_ok=True)
        self.metadata.setdefault("category_ids", list(FIXED_CATEGORY_IDS))
        self.metadata["updated_at"] = utc_now_iso()
        with library_path.open("w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "metadata": dict(self.metadata),
            "categories": [
                {"category_id": cid, "title": CATEGORY_BY_ID[cid].title, "purpose": CATEGORY_BY_ID[cid].purpose}
                for cid in FIXED_CATEGORY_IDS
            ],
            "skills": [skill.to_dict() for skill in self.skills],
        }

    def add_or_replace(self, skill: SkillContract) -> None:
        for index, existing in enumerate(self.skills):
            if existing.skill_id == skill.skill_id:
                skill.created_at = existing.created_at
                skill.updated_at = utc_now_iso()
                self.skills[index] = skill
                return
        self.skills.append(skill)

    def extend(self, skills: Iterable[SkillContract]) -> None:
        for skill in skills:
            self.add_or_replace(skill)

    def by_status(self, statuses: Sequence[str]) -> List[SkillContract]:
        allowed = set(statuses)
        return [skill for skill in self.skills if skill.status in allowed]

    def select_skills(
        self,
        question: str = "",
        qa_type: str = "",
        task: str = "",
        max_skills: int = 4,
        statuses: Sequence[str] = ("accepted",),
    ) -> List[SkillContract]:
        inferred = infer_categories(qa_type=qa_type, task=task, question=question)
        candidates = self.by_status(statuses)
        scored: List[Tuple[float, SkillContract]] = []
        for skill in candidates:
            score = skill.score_for_prompt(inferred, question)
            if score > 0:
                scored.append((score, skill))
        scored.sort(key=lambda item: (item[0], item[1].utility.net_gain, item[1].skill_id), reverse=True)
        return [skill for _, skill in scored[:max_skills]]

    def format_for_prompt(
        self,
        question: str = "",
        qa_type: str = "",
        task: str = "",
        max_skills: int = 4,
        statuses: Sequence[str] = ("accepted",),
    ) -> str:
        selected = self.select_skills(
            question=question,
            qa_type=qa_type,
            task=task,
            max_skills=max_skills,
            statuses=statuses,
        )
        if not selected:
            return ""
        blocks = "\n\n".join(skill.prompt_block() for skill in selected)
        return (
            "Use the following learned AEC drawing skills as private procedure. "
            "Do not mention skill IDs, source cases, or this instruction in the final answer.\n\n"
            + blocks
        )


def make_skill_id(category: str, seed: str) -> str:
    category_id = normalize_category(category)
    return "{}__{}".format(category_id, slugify(seed, max_length=48))


def _string_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    if isinstance(value, (list, tuple)):
        return [str(item).strip() for item in value if str(item).strip()]
    return [str(value).strip()] if str(value).strip() else []
