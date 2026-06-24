"""Replay and acceptance gates for evolved skills."""
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Sequence, Tuple

from .cases import EvolutionCase, build_qa_evolution_cases
from .contracts import SkillContract, SkillLibrary, SkillUtility, utc_now_iso
from .validator import SkillContractValidator


@dataclass
class AcceptanceConfig:
    """Thresholds for accepting a candidate skill."""

    min_net_gain: int = 1
    max_regression_rate: float = 0.35
    min_support: int = 1
    require_validation_pass: bool = True


@dataclass
class ReplaySummary:
    """Comparison summary between baseline and candidate evaluation CSVs."""

    shared: int = 0
    fixed: int = 0
    regressed: int = 0
    still_correct: int = 0
    still_wrong: int = 0
    baseline_mean: float = 0.0
    candidate_mean: float = 0.0
    by_category: Dict[str, Dict[str, int]] = field(default_factory=dict)
    by_qa_type: Dict[str, Dict[str, int]] = field(default_factory=dict)

    @property
    def net_gain(self) -> int:
        return self.fixed - self.regressed

    @property
    def delta(self) -> float:
        return self.candidate_mean - self.baseline_mean

    def to_dict(self) -> Dict[str, Any]:
        return {
            "shared": self.shared,
            "fixed": self.fixed,
            "regressed": self.regressed,
            "net_gain": self.net_gain,
            "still_correct": self.still_correct,
            "still_wrong": self.still_wrong,
            "baseline_mean": round(self.baseline_mean, 6),
            "candidate_mean": round(self.candidate_mean, 6),
            "delta": round(self.delta, 6),
            "by_category": self.by_category,
            "by_qa_type": self.by_qa_type,
        }


def summarize_replay(cases: Sequence[EvolutionCase]) -> ReplaySummary:
    summary = ReplaySummary()
    base_scores = []
    cand_scores = []
    by_category = defaultdict(lambda: defaultdict(int))
    by_qa_type = defaultdict(lambda: defaultdict(int))
    for case in cases:
        summary.shared += 1
        base_scores.append(case.baseline_score)
        cand_scores.append(case.candidate_score)
        if case.outcome == "fixed":
            summary.fixed += 1
        elif case.outcome == "regressed":
            summary.regressed += 1
        elif case.outcome == "still_correct":
            summary.still_correct += 1
        elif case.outcome == "still_wrong":
            summary.still_wrong += 1
        by_qa_type[case.qa_type][case.outcome] += 1
        for category in case.categories:
            by_category[category][case.outcome] += 1
    if base_scores:
        summary.baseline_mean = sum(base_scores) / len(base_scores)
    if cand_scores:
        summary.candidate_mean = sum(cand_scores) / len(cand_scores)
    summary.by_category = {key: dict(value) for key, value in by_category.items()}
    summary.by_qa_type = {key: dict(value) for key, value in by_qa_type.items()}
    return summary


def replay_qa_evaluations(
    baseline_eval_csv: str,
    candidate_eval_csv: str,
    images_dir: str = "",
    max_cases_per_outcome: int = 1000000,
) -> Tuple[List[EvolutionCase], ReplaySummary]:
    cases = build_qa_evolution_cases(
        baseline_eval_csv=baseline_eval_csv,
        candidate_eval_csv=candidate_eval_csv,
        images_dir=images_dir,
        max_cases_per_outcome=max_cases_per_outcome,
    )
    return cases, summarize_replay(cases)


def apply_acceptance_gate(
    candidate_library: SkillLibrary,
    replay_cases: Sequence[EvolutionCase],
    config: AcceptanceConfig = AcceptanceConfig(),
) -> Tuple[SkillLibrary, Dict[str, Any]]:
    """Accept or reject skills according to replay utility and validation."""
    validator = SkillContractValidator()
    accepted = SkillLibrary.empty()
    accepted.metadata.update(
        {
            "created_by": "skill_replay_gate",
            "created_at": utc_now_iso(),
            "acceptance_config": {
                "min_net_gain": config.min_net_gain,
                "max_regression_rate": config.max_regression_rate,
                "min_support": config.min_support,
                "require_validation_pass": config.require_validation_pass,
            },
        }
    )
    report = {"skills": [], "summary": summarize_replay(replay_cases).to_dict()}

    category_stats = summarize_replay(replay_cases).by_category
    for skill in candidate_library.skills:
        stats = category_stats.get(skill.category, {})
        fixed = int(stats.get("fixed", 0))
        regressed = int(stats.get("regressed", 0))
        support = sum(int(value) for value in stats.values())
        regression_rate = float(regressed) / float(max(1, fixed + regressed))
        utility = SkillUtility(
            fixed=fixed,
            regressed=regressed,
            support=support,
            confidence=max(0.0, min(1.0, (fixed - regressed) / float(max(1, support)))),
            notes="Computed from replay cases sharing the skill category.",
        )
        skill.utility = utility
        validation = validator.validate_skill(skill, cases=replay_cases)
        passes_validation = validation.passed or not config.require_validation_pass
        accepted_by_gate = (
            passes_validation
            and support >= config.min_support
            and utility.net_gain >= config.min_net_gain
            and regression_rate <= config.max_regression_rate
        )
        skill.status = "accepted" if accepted_by_gate else "rejected"
        skill.provenance.setdefault("replay_gate", {})
        skill.provenance["replay_gate"].update(
            {
                "accepted": accepted_by_gate,
                "fixed": fixed,
                "regressed": regressed,
                "support": support,
                "regression_rate": round(regression_rate, 4),
                "validation": validation.to_dict(),
            }
        )
        accepted.add_or_replace(skill)
        report["skills"].append(
            {
                "skill_id": skill.skill_id,
                "category": skill.category,
                "status": skill.status,
                "utility": skill.utility.to_dict(),
                "validation": validation.to_dict(),
                "regression_rate": round(regression_rate, 4),
            }
        )
    return accepted, report
