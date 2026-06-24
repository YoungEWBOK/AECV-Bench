"""Build skill-evolution cases from benchmark outputs."""
import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .categories import (
    EVIDENCE_VERIFICATION,
    GRAPHIC_SYMBOL_GROUNDING,
    QUANTITATIVE_SET_REASONING,
    REGION_BOUNDARY_GROUNDING,
    infer_categories,
)


@dataclass
class EvolutionCase:
    """A single comparison case for skill evolution."""

    case_id: str
    task_family: str
    image_id: str
    qa_id: str = ""
    qa_type: str = ""
    task: str = ""
    question: str = ""
    ground_truth: str = ""
    baseline_answer: str = ""
    candidate_answer: str = ""
    baseline_score: float = 0.0
    candidate_score: float = 0.0
    outcome: str = "unknown"
    categories: List[str] = field(default_factory=list)
    image_path: str = ""
    source_files: Dict[str, str] = field(default_factory=dict)
    notes: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "case_id": self.case_id,
            "task_family": self.task_family,
            "image_id": self.image_id,
            "qa_id": self.qa_id,
            "qa_type": self.qa_type,
            "task": self.task,
            "question": self.question,
            "ground_truth": self.ground_truth,
            "baseline_answer": self.baseline_answer,
            "candidate_answer": self.candidate_answer,
            "baseline_score": self.baseline_score,
            "candidate_score": self.candidate_score,
            "outcome": self.outcome,
            "categories": list(self.categories),
            "image_path": self.image_path,
            "source_files": dict(self.source_files),
            "notes": dict(self.notes),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EvolutionCase":
        return cls(
            case_id=str(data.get("case_id", "")),
            task_family=str(data.get("task_family", "")),
            image_id=str(data.get("image_id", "")),
            qa_id=str(data.get("qa_id", "")),
            qa_type=str(data.get("qa_type", "")),
            task=str(data.get("task", "")),
            question=str(data.get("question", "")),
            ground_truth=str(data.get("ground_truth", "")),
            baseline_answer=str(data.get("baseline_answer", "")),
            candidate_answer=str(data.get("candidate_answer", "")),
            baseline_score=_safe_float(data.get("baseline_score")),
            candidate_score=_safe_float(data.get("candidate_score")),
            outcome=str(data.get("outcome", "unknown")),
            categories=[str(item) for item in data.get("categories", [])],
            image_path=str(data.get("image_path", "")),
            source_files=dict(data.get("source_files") or {}),
            notes=dict(data.get("notes") or {}),
        )


def read_csv_rows(csv_path: str) -> List[Dict[str, str]]:
    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_cases_jsonl(cases: Sequence[EvolutionCase], output_path: str) -> None:
    path = Path(output_path)
    if path.parent:
        path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for case in cases:
            f.write(json.dumps(case.to_dict(), ensure_ascii=False) + "\n")


def read_cases_jsonl(path: str) -> List[EvolutionCase]:
    cases = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                cases.append(EvolutionCase.from_dict(json.loads(line)))
    return cases


def build_qa_evolution_cases(
    baseline_eval_csv: str,
    candidate_eval_csv: str,
    images_dir: str = "",
    max_cases_per_outcome: int = 80,
) -> List[EvolutionCase]:
    """Build fixed/regressed/comparison cases from two QA judge CSVs."""
    baseline_rows = {_qa_key(row): row for row in read_csv_rows(baseline_eval_csv)}
    candidate_rows = {_qa_key(row): row for row in read_csv_rows(candidate_eval_csv)}
    shared_keys = sorted(set(baseline_rows) & set(candidate_rows))
    cases: List[EvolutionCase] = []
    counts_by_outcome: Dict[str, int] = {}
    candidate_name = Path(candidate_eval_csv).stem.replace("_evaluation_results", "")

    for key in shared_keys:
        base = baseline_rows[key]
        cand = candidate_rows[key]
        base_score = _safe_float(base.get("overall"))
        cand_score = _safe_float(cand.get("overall"))
        outcome = _score_outcome(base_score, cand_score)
        counts_by_outcome[outcome] = counts_by_outcome.get(outcome, 0) + 1
        if counts_by_outcome[outcome] > max_cases_per_outcome:
            continue

        image_id, qa_id, qa_type = key
        task = cand.get("task") or base.get("task") or ""
        question = cand.get("question") or base.get("question") or ""
        case_id = "{}:{}:{}:{}".format(candidate_name, image_id, qa_id, qa_type)
        categories = infer_categories(qa_type=qa_type, task=task, question=question)
        cases.append(
            EvolutionCase(
                case_id=case_id,
                task_family="qa",
                image_id=image_id,
                qa_id=qa_id,
                qa_type=qa_type,
                task=task,
                question=question,
                ground_truth=cand.get("ground_truth") or base.get("ground_truth") or "",
                baseline_answer=base.get("predicted") or base.get("model_answer") or "",
                candidate_answer=cand.get("predicted") or cand.get("model_answer") or "",
                baseline_score=base_score,
                candidate_score=cand_score,
                outcome=outcome,
                categories=categories,
                image_path=_resolve_image_path(images_dir, image_id),
                source_files={"baseline": baseline_eval_csv, "candidate": candidate_eval_csv},
            )
        )
    return cases


def build_qa_evolution_cases_many(
    baseline_eval_csv: str,
    candidate_eval_csvs: Sequence[str],
    images_dir: str = "",
    max_cases_per_outcome: int = 80,
) -> List[EvolutionCase]:
    """Build QA evolution cases from multiple candidate/strategy CSVs."""
    all_cases: List[EvolutionCase] = []
    seen = set()
    for candidate_csv in candidate_eval_csvs:
        for case in build_qa_evolution_cases(
            baseline_eval_csv=baseline_eval_csv,
            candidate_eval_csv=candidate_csv,
            images_dir=images_dir,
            max_cases_per_outcome=max_cases_per_outcome,
        ):
            if case.case_id in seen:
                continue
            seen.add(case.case_id)
            all_cases.append(case)
    return all_cases


def build_object_counting_evolution_cases(
    baseline_csv: str,
    candidate_csv: str,
    max_cases_per_outcome: int = 80,
) -> List[EvolutionCase]:
    """Build object-counting comparison cases from two result CSVs."""
    baseline_rows = {row.get("name", ""): row for row in read_csv_rows(baseline_csv)}
    candidate_rows = {row.get("name", ""): row for row in read_csv_rows(candidate_csv)}
    shared = sorted(set(baseline_rows) & set(candidate_rows))
    candidate_name = Path(candidate_csv).stem
    cases: List[EvolutionCase] = []
    counts_by_outcome: Dict[str, int] = {}

    for name in shared:
        base = baseline_rows[name]
        cand = candidate_rows[name]
        original = _loads_json(base.get("original") or cand.get("original"))
        base_extracted = _loads_json(base.get("extracted"))
        cand_extracted = _loads_json(cand.get("extracted"))
        if not original or not base_extracted or not cand_extracted:
            continue
        base_score = _object_count_score(original, base_extracted)
        cand_score = _object_count_score(original, cand_extracted)
        outcome = _score_outcome(base_score, cand_score)
        counts_by_outcome[outcome] = counts_by_outcome.get(outcome, 0) + 1
        if counts_by_outcome[outcome] > max_cases_per_outcome:
            continue
        cases.append(
            EvolutionCase(
                case_id="{}:{}:object_counting".format(candidate_name, name),
                task_family="object_counting",
                image_id=name,
                qa_type="object_counting",
                task="object_counting",
                question="Count doors, windows, spaces, bedrooms, and toilets in the full floor plan.",
                ground_truth=json.dumps(original, ensure_ascii=False, separators=(",", ":")),
                baseline_answer=json.dumps(base_extracted, ensure_ascii=False, separators=(",", ":")),
                candidate_answer=json.dumps(cand_extracted, ensure_ascii=False, separators=(",", ":")),
                baseline_score=base_score,
                candidate_score=cand_score,
                outcome=outcome,
                categories=[
                    QUANTITATIVE_SET_REASONING,
                    GRAPHIC_SYMBOL_GROUNDING,
                    REGION_BOUNDARY_GROUNDING,
                    EVIDENCE_VERIFICATION,
                ],
                source_files={"baseline": baseline_csv, "candidate": candidate_csv},
            )
        )
    return cases


def summarize_cases(cases: Iterable[EvolutionCase]) -> Dict[str, Any]:
    summary: Dict[str, Any] = {
        "total": 0,
        "by_outcome": {},
        "by_category": {},
        "by_qa_type": {},
    }
    for case in cases:
        summary["total"] += 1
        summary["by_outcome"][case.outcome] = summary["by_outcome"].get(case.outcome, 0) + 1
        summary["by_qa_type"][case.qa_type] = summary["by_qa_type"].get(case.qa_type, 0) + 1
        for category in case.categories:
            summary["by_category"][category] = summary["by_category"].get(category, 0) + 1
    return summary


def _qa_key(row: Dict[str, Any]) -> Tuple[str, str, str]:
    return (
        str(row.get("image_id", "") or ""),
        str(row.get("qa_id", "") or ""),
        str(row.get("qa_type", "") or ""),
    )


def _score_outcome(base_score: float, cand_score: float) -> str:
    base_ok = base_score >= 0.5
    cand_ok = cand_score >= 0.5
    if not base_ok and cand_ok:
        return "fixed"
    if base_ok and not cand_ok:
        return "regressed"
    if base_ok and cand_ok:
        return "still_correct"
    return "still_wrong"


def _resolve_image_path(images_dir: str, image_id: str) -> str:
    if not images_dir or not image_id:
        return ""
    root = Path(images_dir)
    for ext in (".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG"):
        path = root / "{}{}".format(image_id, ext)
        if path.is_file():
            return str(path)
    return ""


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _loads_json(value: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _object_count_score(original: Dict[str, Any], extracted: Dict[str, Any]) -> float:
    keys = ["Door", "Window", "Space", "Bedroom", "Toilet"]
    correct = 0
    total = 0
    for key in keys:
        if key in original:
            total += 1
            if _coerce_count(original.get(key)) == _coerce_count(extracted.get(key)):
                correct += 1
    return float(correct) / float(total or 1)


def _coerce_count(value: Any) -> Optional[int]:
    if value in (None, "", "-"):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
