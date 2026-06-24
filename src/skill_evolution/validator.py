"""Validation gates for skill contracts."""
import re
from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Sequence

from .categories import FIXED_CATEGORY_IDS, normalize_category
from .cases import EvolutionCase
from .contracts import SkillContract, SkillLibrary, VALID_LEVELS, VALID_STATUSES


@dataclass
class ValidationIssue:
    """One validation issue."""

    severity: str
    field: str
    message: str

    def to_dict(self):
        return {"severity": self.severity, "field": self.field, "message": self.message}


@dataclass
class ValidationResult:
    """Validation result for a skill or library."""

    passed: bool
    issues: List[ValidationIssue] = field(default_factory=list)

    def to_dict(self):
        return {"passed": self.passed, "issues": [issue.to_dict() for issue in self.issues]}


class SkillContractValidator:
    """Hard and soft validation gates inspired by SkillOps/SkillCAT-style contracts."""

    def __init__(
        self,
        max_action_steps: int = 8,
        max_source_cases: int = 40,
        max_chars_per_field: int = 1200,
    ):
        self.max_action_steps = max_action_steps
        self.max_source_cases = max_source_cases
        self.max_chars_per_field = max_chars_per_field

    def validate_skill(
        self,
        skill: SkillContract,
        cases: Optional[Sequence[EvolutionCase]] = None,
    ) -> ValidationResult:
        issues: List[ValidationIssue] = []
        self._check_required(skill, issues)
        self._check_schema(skill, issues)
        self._check_ground_truth_leakage(skill, cases or [], issues)
        self._check_specificity(skill, issues)
        passed = not any(issue.severity == "error" for issue in issues)
        return ValidationResult(passed=passed, issues=issues)

    def validate_library(
        self,
        library: SkillLibrary,
        cases: Optional[Sequence[EvolutionCase]] = None,
    ) -> ValidationResult:
        issues: List[ValidationIssue] = []
        seen_ids = set()
        for skill in library.skills:
            if skill.skill_id in seen_ids:
                issues.append(ValidationIssue("error", "skill_id", "Duplicate skill_id: {}".format(skill.skill_id)))
            seen_ids.add(skill.skill_id)
            result = self.validate_skill(skill, cases=cases)
            issues.extend(
                ValidationIssue(issue.severity, "{}.{}".format(skill.skill_id, issue.field), issue.message)
                for issue in result.issues
            )
        passed = not any(issue.severity == "error" for issue in issues)
        return ValidationResult(passed=passed, issues=issues)

    def _check_required(self, skill: SkillContract, issues: List[ValidationIssue]) -> None:
        if not skill.skill_id:
            issues.append(ValidationIssue("error", "skill_id", "skill_id is required"))
        if not skill.trigger.strip():
            issues.append(ValidationIssue("error", "trigger", "trigger is required"))
        if not skill.preconditions:
            issues.append(ValidationIssue("error", "preconditions", "at least one precondition is required"))
        if not skill.observations:
            issues.append(ValidationIssue("error", "observations", "at least one observation target is required"))
        if not skill.actions:
            issues.append(ValidationIssue("error", "actions", "at least one action is required"))
        if not skill.validator:
            issues.append(ValidationIssue("error", "validator", "at least one validator rule is required"))
        if not skill.failure_modes:
            issues.append(ValidationIssue("warning", "failure_modes", "failure_modes should be explicit"))

    def _check_schema(self, skill: SkillContract, issues: List[ValidationIssue]) -> None:
        try:
            normalize_category(skill.category)
        except ValueError as exc:
            issues.append(ValidationIssue("error", "category", str(exc)))
        if skill.category not in FIXED_CATEGORY_IDS:
            issues.append(ValidationIssue("error", "category", "category must be one of the fixed 8 categories"))
        if skill.level not in VALID_LEVELS:
            issues.append(ValidationIssue("error", "level", "level must be one of {}".format(", ".join(VALID_LEVELS))))
        if skill.status not in VALID_STATUSES:
            issues.append(ValidationIssue("error", "status", "status must be one of {}".format(", ".join(VALID_STATUSES))))
        if len(skill.actions) > self.max_action_steps:
            issues.append(ValidationIssue("warning", "actions", "too many action steps; consider splitting the skill"))
        if len(skill.source_cases) > self.max_source_cases:
            issues.append(ValidationIssue("warning", "source_cases", "too many source cases; keep only representative cases"))
        for field_name in ("trigger", "preconditions", "observations", "actions", "validator", "failure_modes"):
            text = _field_text(getattr(skill, field_name))
            if len(text) > self.max_chars_per_field:
                issues.append(ValidationIssue("warning", field_name, "field is long and may reduce routing precision"))

    def _check_ground_truth_leakage(
        self,
        skill: SkillContract,
        cases: Sequence[EvolutionCase],
        issues: List[ValidationIssue],
    ) -> None:
        if not cases:
            return
        prompt_text = "\n".join(
            [
                skill.trigger,
                "\n".join(skill.preconditions),
                "\n".join(skill.observations),
                "\n".join(skill.actions),
                "\n".join(skill.validator),
                "\n".join(skill.failure_modes),
            ]
        ).lower()
        for case in cases:
            for label, value in (
                ("ground_truth", case.ground_truth),
                ("candidate_answer", case.candidate_answer),
            ):
                normalized = _normalize_leak_text(value)
                if normalized and len(normalized) >= 6 and normalized in prompt_text:
                    issues.append(
                        ValidationIssue(
                            "error",
                            label,
                            "skill text appears to copy case-specific answer text from {}".format(case.case_id),
                        )
                    )
                    return
            for identifier in (case.image_id, case.qa_id):
                if identifier and identifier.lower() in prompt_text:
                    issues.append(
                        ValidationIssue(
                            "warning",
                            "specificity",
                            "skill text mentions case identifier {}; keep identifiers only in source_cases".format(identifier),
                        )
                    )

    def _check_specificity(self, skill: SkillContract, issues: List[ValidationIssue]) -> None:
        all_text = _field_text([skill.trigger] + skill.actions + skill.validator).lower()
        if re.search(r"\b\d{4}-\d{4}\b", all_text):
            issues.append(ValidationIssue("warning", "specificity", "skill mentions a dataset-style image id"))
        if "ground truth" in all_text and skill.status != "rejected":
            issues.append(ValidationIssue("warning", "specificity", "skill should not rely on ground-truth access at inference"))


def _field_text(value) -> str:
    if isinstance(value, list):
        return "\n".join(str(item) for item in value)
    return str(value or "")


def _normalize_leak_text(value: str) -> str:
    value = (value or "").strip().lower()
    value = re.sub(r"\s+", " ", value)
    value = value.strip("\"'`.,;:()[]{} ")
    if not value or value in {"yes", "no", "one", "two", "1", "2", "0"}:
        return ""
    words = re.findall(r"[a-z0-9.]+", value)
    has_digit = any(char.isdigit() for char in value)
    # Single generic labels such as "bedroom" or "garage" are normal skill
    # vocabulary, not case-answer leakage.
    if len(words) < 2 and not has_digit:
        return ""
    return value
