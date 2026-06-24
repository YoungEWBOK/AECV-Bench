"""Independent skill self-evolution loop for AECV QA.

This runner does not depend on baseline-vs-exploration CSVs. It uses label
files as supervised feedback:

1. Split labels into evolution train/dev subsets.
2. Predict train with the current accepted library.
3. Evaluate train predictions against labels with the configured evaluator.
4. Generate candidate skills from training failures.
5. Predict dev with current library and proposal library.
6. Accept/reject candidate skills by dev replay improvement.

External baselines remain only final evaluation controls, not evolution inputs.
"""
import argparse
import csv
import json
import os
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

from src.benchmark.qa_evaluator import QAEvaluator
from src.skill_evolution.cases import EvolutionCase, read_csv_rows, summarize_cases
from src.skill_evolution.categories import infer_categories
from src.skill_evolution.contracts import SkillContract, SkillLibrary, utc_now_iso
from src.skill_evolution.generator import SkillEvolutionGenerator
from src.skill_evolution.replay import AcceptanceConfig, apply_acceptance_gate, replay_qa_evaluations
from src.skill_evolution.validator import SkillContractValidator
from src.utils.benchmark_config import load_benchmark_config
from src.utils.config import get_llm_api_key, get_llm_base_url, require_llm_api_key, require_llm_base_url
from src.utils.prompt_strategies import make_safe_name


DEFAULT_SELF_EVOLUTION_CONFIG: Dict[str, Any] = {
    "output_dir": "results/self_evolution",
    "images_dir": "data/Use Case 2 - Drawing Understanding/01 - Full Dataset/images",
    "labels_dir": "data/Use Case 2 - Drawing Understanding/01 - Full Dataset/labels",
    "train_fraction": 0.7,
    "split_seed": 0,
    "max_train_files": 0,
    "max_dev_files": 0,
    "iterations": 2,
    "force": False,
    "model": {
        "name": "Qwen3.7 Plus",
        "model_id": "qwen3.7-plus",
        "temperature": 0.0,
        "extra_body": {"enable_thinking": False},
    },
    "judge": {
        "model_id": "qwen3.7-plus",
        "temperature": 0.0,
        "extra_body": {"enable_thinking": False},
    },
    "initial_library_path": "",
    "accepted_library_path": "results/self_evolution/accepted_skill_library.json",
    "max_skills": 8,
    "max_skills_per_question": 4,
    "generator_model": "",
    "generator_temperature": 0.0,
    "generator_timeout": 90,
    "generator_max_retries": 3,
    "generator_extra_body": None,
    "feedback": {
        "score_threshold": 0.5,
        "max_wrong_cases": 120,
        "max_correct_cases": 40,
    },
    "acceptance": {
        "min_net_gain": 1,
        "max_regression_rate": 0.35,
        "min_support": 1,
        "require_validation_pass": True,
        "preserve_existing_accepted": True,
    },
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run independent AECV skill self-evolution.")
    parser.add_argument("--config", default="configs/self_evolution_qwen37_plus.json")
    parser.add_argument("command", nargs="?", choices=["run", "prepare-splits"], default="run")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Use heuristic skill generation. Predictions and judge calls still use the configured APIs.",
    )
    parser.add_argument("--force", action="store_true", help="Overwrite prediction/evaluation CSVs for this run.")
    args = parser.parse_args()

    config = load_self_evolution_config(args.config)
    config["force"] = bool(args.force or config.get("force", False))
    splits = prepare_label_splits(config)
    print("Prepared splits:")
    print("  train labels: {} ({})".format(splits["train_dir"], len(splits["train_files"])))
    print("  dev labels: {} ({})".format(splits["dev_dir"], len(splits["dev_files"])))
    if args.command == "prepare-splits":
        return

    run_self_evolution(config, splits, dry_run=args.dry_run)


def load_self_evolution_config(config_path: str) -> Dict[str, Any]:
    path = Path(config_path)
    loaded = load_benchmark_config(str(path)) if path.is_file() else {}
    section = loaded.get("self_evolution", loaded)
    config = json.loads(json.dumps(DEFAULT_SELF_EVOLUTION_CONFIG))
    _deep_update(config, section)
    Path(config["output_dir"]).mkdir(parents=True, exist_ok=True)
    return config


def prepare_label_splits(config: Dict[str, Any]) -> Dict[str, Any]:
    labels_dir = Path(config["labels_dir"])
    if not labels_dir.is_dir():
        raise FileNotFoundError("Labels directory not found: {}".format(labels_dir))
    label_files = sorted(labels_dir.glob("*.json"))
    if not label_files:
        raise FileNotFoundError("No label JSON files found in {}".format(labels_dir))

    train_ids = set(str(item) for item in config.get("train_label_ids", []) or [])
    dev_ids = set(str(item) for item in config.get("dev_label_ids", []) or [])
    if train_ids or dev_ids:
        train_files = [path for path in label_files if path.stem in train_ids]
        dev_files = [path for path in label_files if path.stem in dev_ids]
    else:
        train_count = int(round(len(label_files) * float(config.get("train_fraction", 0.7))))
        train_count = max(1, min(len(label_files) - 1, train_count))
        train_files = label_files[:train_count]
        dev_files = label_files[train_count:]

    max_train = int(config.get("max_train_files", 0) or 0)
    max_dev = int(config.get("max_dev_files", 0) or 0)
    if max_train > 0:
        train_files = train_files[:max_train]
    if max_dev > 0:
        dev_files = dev_files[:max_dev]
    if not train_files or not dev_files:
        raise ValueError("Both train and dev splits must contain at least one label file.")

    split_root = Path(config["output_dir"]) / "splits"
    train_dir = split_root / "train_labels"
    dev_dir = split_root / "dev_labels"
    _sync_label_dir(train_dir, train_files)
    _sync_label_dir(dev_dir, dev_files)
    split_manifest = {
        "created_at": utc_now_iso(),
        "source_labels_dir": str(labels_dir),
        "train_files": [path.name for path in train_files],
        "dev_files": [path.name for path in dev_files],
    }
    with (split_root / "split_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(split_manifest, f, indent=2, ensure_ascii=False)
    return {
        "train_dir": str(train_dir),
        "dev_dir": str(dev_dir),
        "train_files": train_files,
        "dev_files": dev_files,
    }


def _sync_label_dir(target_dir: Path, source_files: Sequence[Path]) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    keep = {path.name for path in source_files}
    for existing in target_dir.glob("*.json"):
        if existing.name not in keep:
            existing.unlink()
    for source in source_files:
        shutil.copy2(source, target_dir / source.name)


def run_self_evolution(config: Dict[str, Any], splits: Dict[str, Any], dry_run: bool = False) -> None:
    output_dir = Path(config["output_dir"])
    model = config["model"]
    current_library_path = _initialize_library(config)
    reports = []

    for iteration in range(1, int(config.get("iterations", 1)) + 1):
        iter_dir = output_dir / "iterations" / "iter_{:02d}".format(iteration)
        iter_dir.mkdir(parents=True, exist_ok=True)
        print("\n" + "=" * 80)
        print("SELF-EVOLUTION ITERATION {}".format(iteration))
        print("=" * 80)
        print("Current accepted library: {}".format(current_library_path or "(empty)"))

        current_train_pred = iter_dir / "train_current_predictions.csv"
        current_train_eval = iter_dir / "train_current_evaluation.csv"
        run_prediction_split(
            config=config,
            labels_dir=splits["train_dir"],
            output_csv=str(current_train_pred),
            library_path=current_library_path,
            library_statuses=["accepted"],
            prompt_strategy=_strategy_for_library(current_library_path),
            phase_label="train/current",
        )
        evaluate_prediction_csv(config, str(current_train_pred), str(current_train_eval))

        feedback_cases = build_feedback_cases_from_evaluation(
            eval_csv=str(current_train_eval),
            images_dir=config["images_dir"],
            score_threshold=float(config.get("feedback", {}).get("score_threshold", 0.5)),
            max_wrong_cases=int(config.get("feedback", {}).get("max_wrong_cases", 120)),
            max_correct_cases=int(config.get("feedback", {}).get("max_correct_cases", 40)),
        )
        cases_path = iter_dir / "train_feedback_cases.jsonl"
        _write_cases_jsonl(feedback_cases, cases_path)
        with (iter_dir / "train_feedback_summary.json").open("w", encoding="utf-8") as f:
            json.dump(summarize_cases(feedback_cases), f, indent=2, ensure_ascii=False)

        candidate_path = iter_dir / "candidate_skill_library.json"
        generate_candidate_library(config, feedback_cases, str(candidate_path), dry_run=dry_run)

        proposal_path = iter_dir / "proposal_skill_library.json"
        proposal_library = merge_current_and_candidates(current_library_path, str(candidate_path))
        proposal_library.save(str(proposal_path))

        current_dev_pred = iter_dir / "dev_current_predictions.csv"
        current_dev_eval = iter_dir / "dev_current_evaluation.csv"
        proposal_dev_pred = iter_dir / "dev_proposal_predictions.csv"
        proposal_dev_eval = iter_dir / "dev_proposal_evaluation.csv"

        run_prediction_split(
            config=config,
            labels_dir=splits["dev_dir"],
            output_csv=str(current_dev_pred),
            library_path=current_library_path,
            library_statuses=["accepted"],
            prompt_strategy=_strategy_for_library(current_library_path),
            phase_label="dev/current",
        )
        evaluate_prediction_csv(config, str(current_dev_pred), str(current_dev_eval))

        run_prediction_split(
            config=config,
            labels_dir=splits["dev_dir"],
            output_csv=str(proposal_dev_pred),
            library_path=str(proposal_path),
            library_statuses=["candidate", "accepted"],
            prompt_strategy="skill_guided",
            phase_label="dev/proposal",
        )
        evaluate_prediction_csv(config, str(proposal_dev_pred), str(proposal_dev_eval))

        replay_cases, replay_summary = replay_qa_evaluations(
            baseline_eval_csv=str(current_dev_eval),
            candidate_eval_csv=str(proposal_dev_eval),
            images_dir=config["images_dir"],
        )
        accepted_library, gate_report = accept_candidates_preserving_current(
            current_library_path=current_library_path,
            proposal_library_path=str(proposal_path),
            replay_cases=replay_cases,
            config=config,
        )
        accepted_path = iter_dir / "accepted_skill_library.json"
        accepted_library.save(str(accepted_path))
        final_path = Path(config["accepted_library_path"])
        accepted_library.save(str(final_path))
        current_library_path = str(final_path)

        report = {
            "iteration": iteration,
            "train_feedback_summary": summarize_cases(feedback_cases),
            "dev_replay_summary": replay_summary.to_dict(),
            "candidate_library_path": str(candidate_path),
            "proposal_library_path": str(proposal_path),
            "iteration_accepted_library_path": str(accepted_path),
            "accepted_library_path": str(final_path),
            "gate_report": gate_report,
        }
        reports.append(report)
        with (iter_dir / "iteration_report.json").open("w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print("Iteration {} replay: {}".format(iteration, replay_summary.to_dict()))
        print("Accepted library saved to: {}".format(final_path))

    with (output_dir / "self_evolution_report.json").open("w", encoding="utf-8") as f:
        json.dump({"created_at": utc_now_iso(), "iterations": reports}, f, indent=2, ensure_ascii=False)
    print("\nSelf-evolution complete. Final accepted library: {}".format(config["accepted_library_path"]))


def _initialize_library(config: Dict[str, Any]) -> str:
    final_path = Path(config["accepted_library_path"])
    if final_path.is_file():
        return str(final_path)
    initial = config.get("initial_library_path", "")
    if initial and Path(initial).is_file():
        library = SkillLibrary.load(initial)
    else:
        library = SkillLibrary.empty()
        library.metadata.update({"created_by": "self_evolution", "created_at": utc_now_iso()})
    library.save(str(final_path))
    return str(final_path)


def _strategy_for_library(library_path: str) -> str:
    if not library_path or not Path(library_path).is_file():
        return "one_shot"
    library = SkillLibrary.load(library_path, missing_ok=True)
    return "skill_guided" if library.skills else "one_shot"


def run_prediction_split(
    config: Dict[str, Any],
    labels_dir: str,
    output_csv: str,
    library_path: str,
    library_statuses: Sequence[str],
    prompt_strategy: str,
    phase_label: str,
) -> None:
    from run_qa_benchmark import process_qa_benchmark

    model = config["model"]
    print("\n[{}] Predicting with model {} ({})".format(phase_label, model["name"], model["model_id"]))
    if config.get("force") and Path(output_csv).is_file():
        Path(output_csv).unlink()
    library_for_prompt = library_path if prompt_strategy == "skill_guided" else ""
    process_qa_benchmark(
        labels_dir=labels_dir,
        images_dir=config["images_dir"],
        output_csv=output_csv,
        model_name=model["model_id"],
        open_router_api_key=require_llm_api_key(),
        url=require_llm_base_url(),
        temperature=float(model.get("temperature", 0.0)),
        prompt_strategy=prompt_strategy,
        skill_library_path=library_for_prompt,
        max_skills_per_question=int(config.get("max_skills_per_question", 4)),
        skill_statuses=list(library_statuses),
        extra_body=model.get("extra_body"),
        stream=bool(model.get("stream", False)),
        stream_options=model.get("stream_options"),
    )


def evaluate_prediction_csv(config: Dict[str, Any], prediction_csv: str, output_csv: str) -> Dict[str, Any]:
    judge = config["judge"]
    print("[judge] Evaluating {}".format(prediction_csv))
    evaluator = QAEvaluator(
        judge_model=judge["model_id"],
        open_router_api_key=require_llm_api_key(),
        url=require_llm_base_url(),
        temperature=float(judge.get("temperature", 0.0)),
        extra_body=judge.get("extra_body"),
        stream=bool(judge.get("stream", False)),
        stream_options=judge.get("stream_options"),
    )
    return evaluator.evaluate_csv(prediction_csv, output_csv=output_csv, force=bool(config.get("force", False)))


def build_feedback_cases_from_evaluation(
    eval_csv: str,
    images_dir: str = "",
    score_threshold: float = 0.5,
    max_wrong_cases: int = 120,
    max_correct_cases: int = 40,
) -> List[EvolutionCase]:
    wrong_count = 0
    correct_count = 0
    cases: List[EvolutionCase] = []
    for row in read_csv_rows(eval_csv):
        score = _safe_float(row.get("overall"))
        is_wrong = score < score_threshold
        if is_wrong:
            if wrong_count >= max_wrong_cases:
                continue
            wrong_count += 1
            outcome = "train_wrong"
        else:
            if correct_count >= max_correct_cases:
                continue
            correct_count += 1
            outcome = "train_correct"
        image_id = row.get("image_id", "")
        qa_id = row.get("qa_id", "")
        qa_type = row.get("qa_type", "")
        task = row.get("task", "")
        question = row.get("question", "")
        cases.append(
            EvolutionCase(
                case_id="feedback:{}:{}:{}:{}".format(Path(eval_csv).stem, image_id, qa_id, qa_type),
                task_family="qa",
                image_id=image_id,
                qa_id=qa_id,
                qa_type=qa_type,
                task=task,
                question=question,
                ground_truth=row.get("ground_truth", ""),
                baseline_answer="",
                candidate_answer=row.get("predicted", ""),
                baseline_score=1.0,
                candidate_score=score,
                outcome=outcome,
                categories=infer_categories(qa_type=qa_type, task=task, question=question),
                image_path=_resolve_image_path(images_dir, image_id),
                source_files={"evaluation": eval_csv},
                notes={"feedback_source": "label_supervision", "evaluation_status": row.get("evaluation_status", "")},
            )
        )
    if not cases:
        raise ValueError("No feedback cases built from {}".format(eval_csv))
    return cases


def generate_candidate_library(
    config: Dict[str, Any],
    cases: Sequence[EvolutionCase],
    output_path: str,
    dry_run: bool = False,
) -> None:
    generator_model = config.get("generator_model") or config.get("model", {}).get("model_id", "")
    generator = SkillEvolutionGenerator(
        generator_model=generator_model,
        api_key=get_llm_api_key(),
        base_url=get_llm_base_url(),
        temperature=float(config.get("generator_temperature", 0.0)),
        timeout=int(config.get("generator_timeout", 90)),
        max_retries=int(config.get("generator_max_retries", 3)),
        extra_body=config.get("generator_extra_body", config.get("model", {}).get("extra_body")),
        stream=bool(config.get("generator_stream", config.get("model", {}).get("stream", False))),
        stream_options=config.get("generator_stream_options", config.get("model", {}).get("stream_options")),
    )
    library = generator.generate(
        cases,
        existing_library=SkillLibrary.load(config["accepted_library_path"], missing_ok=True),
        max_skills=int(config.get("max_skills", 8)),
        use_llm=(not dry_run),
    )
    library.metadata.update(
        {
            "created_by": "self_evolution_feedback_generator",
            "feedback_case_count": len(cases),
        }
    )
    library.save(output_path)
    validation = SkillContractValidator().validate_library(library, cases=cases).to_dict()
    with open(str(Path(output_path).with_suffix(".validation.json")), "w", encoding="utf-8") as f:
        json.dump(validation, f, indent=2, ensure_ascii=False)


def merge_current_and_candidates(current_library_path: str, candidate_library_path: str) -> SkillLibrary:
    merged = SkillLibrary.empty()
    if current_library_path and Path(current_library_path).is_file():
        merged.extend(SkillLibrary.load(current_library_path).skills)
    merged.extend(SkillLibrary.load(candidate_library_path).skills)
    merged.metadata.update(
        {
            "created_by": "self_evolution_proposal_merge",
            "current_library_path": current_library_path,
            "candidate_library_path": candidate_library_path,
        }
    )
    return merged


def accept_candidates_preserving_current(
    current_library_path: str,
    proposal_library_path: str,
    replay_cases: Sequence[EvolutionCase],
    config: Dict[str, Any],
) -> Tuple[SkillLibrary, Dict[str, Any]]:
    acceptance_data = config.get("acceptance", {})
    gate_config = AcceptanceConfig(
        min_net_gain=int(acceptance_data.get("min_net_gain", 1)),
        max_regression_rate=float(acceptance_data.get("max_regression_rate", 0.35)),
        min_support=int(acceptance_data.get("min_support", 1)),
        require_validation_pass=bool(acceptance_data.get("require_validation_pass", True)),
    )
    proposal = SkillLibrary.load(proposal_library_path)
    gated, report = apply_acceptance_gate(proposal, replay_cases=replay_cases, config=gate_config)
    if not acceptance_data.get("preserve_existing_accepted", True):
        return gated, report

    existing_ids = set()
    accepted = SkillLibrary.empty()
    if current_library_path and Path(current_library_path).is_file():
        current = SkillLibrary.load(current_library_path)
        for skill in current.skills:
            if skill.status == "accepted":
                skill.status = "accepted"
                accepted.add_or_replace(skill)
                existing_ids.add(skill.skill_id)

    for skill in gated.skills:
        if skill.skill_id in existing_ids:
            continue
        accepted.add_or_replace(skill)
    accepted.metadata.update(
        {
            "created_by": "self_evolution_acceptance_gate",
            "created_at": utc_now_iso(),
            "preserved_existing_accepted": True,
            "current_library_path": current_library_path,
            "proposal_library_path": proposal_library_path,
        }
    )
    return accepted, report


def _write_cases_jsonl(cases: Sequence[EvolutionCase], output_path: Path) -> None:
    with output_path.open("w", encoding="utf-8") as f:
        for case in cases:
            f.write(json.dumps(case.to_dict(), ensure_ascii=False) + "\n")


def _resolve_image_path(images_dir: str, image_id: str) -> str:
    if not images_dir or not image_id:
        return ""
    for suffix in (".png", ".jpg", ".jpeg", ".webp"):
        candidate = Path(images_dir) / "{}{}".format(image_id, suffix)
        if candidate.is_file():
            return str(candidate)
    return ""


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _deep_update(target: Dict[str, Any], source: Dict[str, Any]) -> None:
    for key, value in source.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _deep_update(target[key], value)
        else:
            target[key] = value


if __name__ == "__main__":
    main()
