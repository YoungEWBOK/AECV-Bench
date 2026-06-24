"""Skill-evolution pipeline for AECV-Bench.

Typical loop:
1. Run baseline QA/object-counting benchmark.
2. Run one or more exploratory strategies and judge their outputs.
3. Build fixed/regressed evolution cases from judge CSVs.
4. Generate candidate skill contracts.
5. Run skill_guided benchmark with the candidate/accepted library.
6. Replay judge results and accept only skills that improve held-out results.
"""
import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List

from src.skill_evolution.cases import (
    build_object_counting_evolution_cases,
    build_qa_evolution_cases_many,
    read_cases_jsonl,
    summarize_cases,
    write_cases_jsonl,
)
from src.skill_evolution.contracts import SkillLibrary
from src.skill_evolution.generator import SkillEvolutionGenerator
from src.skill_evolution.replay import AcceptanceConfig, apply_acceptance_gate, replay_qa_evaluations
from src.skill_evolution.validator import SkillContractValidator
from src.utils.benchmark_config import DEFAULT_CONFIG_PATH, load_benchmark_config
from src.utils.config import get_llm_api_key, get_llm_base_url
from src.utils.prompt_strategies import make_safe_name, prompt_strategy_suffix


DEFAULT_SKILL_EVOLUTION_CONFIG = {
    "output_dir": "results/skill_evolution",
    "cases_path": "results/skill_evolution/evolution_cases.jsonl",
    "candidate_library_path": "results/skill_evolution/candidate_skill_library.json",
    "accepted_library_path": "results/skill_evolution/accepted_skill_library.json",
    "skill_guided_config_path": "configs/qa_skill_guided_generated.json",
    "max_cases_per_outcome": 80,
    "max_skills": 8,
    "max_skills_per_question": 4,
    "generator_model": "",
    "qa_images_dir": "data/Use Case 2 - Drawing Understanding/01 - Full Dataset/images",
    "benchmark_config_template": DEFAULT_CONFIG_PATH,
    "acceptance": {
        "min_net_gain": 1,
        "max_regression_rate": 0.35,
        "min_support": 1,
        "require_validation_pass": True,
    },
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run AECV skill-evolution utilities")
    parser.add_argument("--config", default="configs/skill_evolution.json", help="Skill-evolution config JSON path.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_cases = subparsers.add_parser("build-cases", help="Build evolution cases from judge/object CSVs.")
    build_cases.add_argument("--output", default="", help="Override cases JSONL output path.")

    generate = subparsers.add_parser("generate", help="Generate candidate skill contracts from cases.")
    generate.add_argument("--cases", default="", help="Override cases JSONL path.")
    generate.add_argument("--output", default="", help="Override candidate library path.")
    generate.add_argument("--dry-run", action="store_true", help="Use heuristic generator; no API call.")
    generate.add_argument("--model", default="", help="Override generator model.")

    validate = subparsers.add_parser("validate", help="Validate a skill library.")
    validate.add_argument("--library", default="", help="Library JSON path to validate.")
    validate.add_argument("--cases", default="", help="Optional cases JSONL path for leakage checks.")

    replay = subparsers.add_parser("replay", help="Replay a skill-guided evaluation and accept/reject skills.")
    replay.add_argument("--baseline-eval-csv", default="", help="Baseline QA judge CSV.")
    replay.add_argument("--candidate-eval-csv", default="", help="Skill-guided QA judge CSV.")
    replay.add_argument("--candidate-library", default="", help="Candidate library JSON.")
    replay.add_argument("--accepted-library", default="", help="Accepted library JSON output.")

    evolve = subparsers.add_parser("evolve", help="Build cases, generate candidates, validate, and write config.")
    evolve.add_argument("--dry-run", action="store_true", help="Use heuristic generator; no API call.")
    evolve.add_argument("--model", default="", help="Override generator model.")

    args = parser.parse_args()
    config = load_skill_evolution_config(args.config)

    if args.command == "build-cases":
        path = build_cases_from_config(config, output_override=args.output)
        print("Cases saved to: {}".format(path))
    elif args.command == "generate":
        path = generate_candidates_from_config(
            config,
            cases_override=args.cases,
            output_override=args.output,
            dry_run=args.dry_run,
            model_override=args.model,
        )
        print("Candidate skill library saved to: {}".format(path))
    elif args.command == "validate":
        result = validate_library(
            library_path=args.library or config["candidate_library_path"],
            cases_path=args.cases or config.get("cases_path", ""),
        )
        print(json.dumps(result, indent=2, ensure_ascii=False))
    elif args.command == "replay":
        report_path = replay_from_args(config, args)
        print("Replay report saved to: {}".format(report_path))
    elif args.command == "evolve":
        run_evolve(config, dry_run=args.dry_run, model_override=args.model)


def load_skill_evolution_config(config_path: str) -> Dict[str, Any]:
    path = Path(config_path)
    if path.is_file():
        loaded = load_benchmark_config(str(path))
        section = loaded.get("skill_evolution", loaded)
    else:
        section = {}
    config = json.loads(json.dumps(DEFAULT_SKILL_EVOLUTION_CONFIG))
    _deep_update(config, section)
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    return config


def build_cases_from_config(config: Dict[str, Any], output_override: str = "") -> str:
    cases = []
    baseline = config.get("baseline_eval_csv", "")
    candidates = config.get("candidate_eval_csvs", [])
    if baseline and candidates:
        cases.extend(
            build_qa_evolution_cases_many(
                baseline_eval_csv=baseline,
                candidate_eval_csvs=candidates,
                images_dir=config.get("qa_images_dir", ""),
                max_cases_per_outcome=int(config.get("max_cases_per_outcome", 80)),
            )
        )

    object_baseline = config.get("object_counting_baseline_csv", "")
    object_candidates = config.get("object_counting_candidate_csvs", [])
    if object_baseline and object_candidates:
        for candidate_csv in object_candidates:
            cases.extend(
                build_object_counting_evolution_cases(
                    baseline_csv=object_baseline,
                    candidate_csv=candidate_csv,
                    max_cases_per_outcome=int(config.get("max_cases_per_outcome", 80)),
                )
            )

    if not cases:
        raise ValueError(
            "No evolution cases built. Configure baseline_eval_csv + candidate_eval_csvs "
            "or object_counting_baseline_csv + object_counting_candidate_csvs."
        )

    output_path = output_override or config["cases_path"]
    write_cases_jsonl(cases, output_path)
    summary_path = str(Path(output_path).with_suffix(".summary.json"))
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summarize_cases(cases), f, indent=2, ensure_ascii=False)
    print("Case summary saved to: {}".format(summary_path))
    return output_path


def generate_candidates_from_config(
    config: Dict[str, Any],
    cases_override: str = "",
    output_override: str = "",
    dry_run: bool = False,
    model_override: str = "",
) -> str:
    cases_path = cases_override or config["cases_path"]
    cases = read_cases_jsonl(cases_path)
    existing_library = None
    existing_path = config.get("existing_library_path", "")
    if existing_path and Path(existing_path).is_file():
        existing_library = SkillLibrary.load(existing_path)

    generator_model = model_override or config.get("generator_model", "")
    generator = SkillEvolutionGenerator(
        generator_model=generator_model,
        api_key=get_llm_api_key(),
        base_url=get_llm_base_url(),
        temperature=float(config.get("generator_temperature", 0.0)),
        timeout=int(config.get("generator_timeout", 90)),
        max_retries=int(config.get("generator_max_retries", 3)),
    )
    library = generator.generate(
        cases,
        existing_library=existing_library,
        max_skills=int(config.get("max_skills", 8)),
        use_llm=(not dry_run),
    )
    output_path = output_override or config["candidate_library_path"]
    library.save(output_path)
    return output_path


def validate_library(library_path: str, cases_path: str = "") -> Dict[str, Any]:
    library = SkillLibrary.load(library_path)
    cases = read_cases_jsonl(cases_path) if cases_path and Path(cases_path).is_file() else []
    result = SkillContractValidator().validate_library(library, cases=cases)
    return result.to_dict()


def replay_from_args(config: Dict[str, Any], args: argparse.Namespace) -> str:
    baseline_csv = args.baseline_eval_csv or config.get("baseline_eval_csv", "")
    candidate_csv = args.candidate_eval_csv or config.get("skill_guided_eval_csv", "")
    if not baseline_csv or not candidate_csv:
        raise ValueError("Replay requires baseline_eval_csv and candidate_eval_csv/skill_guided_eval_csv.")
    candidate_library_path = args.candidate_library or config["candidate_library_path"]
    accepted_library_path = args.accepted_library or config["accepted_library_path"]

    cases, summary = replay_qa_evaluations(
        baseline_eval_csv=baseline_csv,
        candidate_eval_csv=candidate_csv,
        images_dir=config.get("qa_images_dir", ""),
    )
    candidate_library = SkillLibrary.load(candidate_library_path)
    accepted_library, gate_report = apply_acceptance_gate(
        candidate_library,
        replay_cases=cases,
        config=_acceptance_config(config),
    )
    accepted_library.save(accepted_library_path)

    report = {
        "baseline_eval_csv": baseline_csv,
        "candidate_eval_csv": candidate_csv,
        "candidate_library_path": candidate_library_path,
        "accepted_library_path": accepted_library_path,
        "replay_summary": summary.to_dict(),
        "acceptance_report": gate_report,
    }
    report_path = str(Path(config["output_dir"]) / "replay_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    return report_path


def run_evolve(config: Dict[str, Any], dry_run: bool = False, model_override: str = "") -> None:
    cases_path = build_cases_from_config(config)
    candidate_path = generate_candidates_from_config(
        config,
        cases_override=cases_path,
        dry_run=dry_run,
        model_override=model_override,
    )
    validation = validate_library(candidate_path, cases_path=cases_path)
    validation_path = str(Path(config["output_dir"]) / "candidate_validation.json")
    with open(validation_path, "w", encoding="utf-8") as f:
        json.dump(validation, f, indent=2, ensure_ascii=False)
    guided_config = write_skill_guided_benchmark_config(config, skill_library_path=candidate_path)
    print("Candidate validation saved to: {}".format(validation_path))
    print("Skill-guided benchmark config saved to: {}".format(guided_config))
    print("Next: run QA with `python run_qa_benchmark.py --config {}` and judge its CSV, then run `python run_skill_evolution.py --config {} replay`.".format(guided_config, "configs/skill_evolution.json"))


def write_skill_guided_benchmark_config(config: Dict[str, Any], skill_library_path: str) -> str:
    template_path = config.get("benchmark_config_template", DEFAULT_CONFIG_PATH)
    benchmark_config = load_benchmark_config(template_path)
    qa_config = benchmark_config.get("qa", {})
    qa_config["skill_library_path"] = skill_library_path
    qa_config["max_skills_per_question"] = int(config.get("max_skills_per_question", 4))
    qa_config["skill_statuses"] = ["candidate", "accepted"]
    options = qa_config.get("_prompt_strategy_options")
    if isinstance(options, list) and "skill_guided" not in options:
        options.append("skill_guided")
    expected_csvs = []
    qa_output_dir = qa_config.get("output_dir", "benchmark_result_qa")
    for model in qa_config.get("models", []):
        if model.get("enabled", True):
            model["prompt_strategy"] = "skill_guided"
            model["skill_library_path"] = skill_library_path
            model["max_skills_per_question"] = int(config.get("max_skills_per_question", 4))
            model["skill_statuses"] = ["candidate", "accepted"]
            suffix = prompt_strategy_suffix("skill_guided")
            safe_name = make_safe_name(model.get("name", "model"))
            if suffix:
                safe_name = "{}_{}".format(safe_name, suffix)
            expected_csvs.append(str(Path(qa_output_dir) / "qa_results_{}.csv".format(safe_name)))
    benchmark_config["qa"] = qa_config
    if expected_csvs:
        judge_config = benchmark_config.get("judge", {})
        judge_config["csv_files"] = expected_csvs
        benchmark_config["judge"] = judge_config
    output_path = config.get("skill_guided_config_path", "configs/qa_skill_guided_generated.json")
    path = Path(output_path)
    if path.parent:
        path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(benchmark_config, f, indent=2, ensure_ascii=False)
    return output_path


def _acceptance_config(config: Dict[str, Any]) -> AcceptanceConfig:
    data = config.get("acceptance", {})
    return AcceptanceConfig(
        min_net_gain=int(data.get("min_net_gain", 1)),
        max_regression_rate=float(data.get("max_regression_rate", 0.35)),
        min_support=int(data.get("min_support", 1)),
        require_validation_pass=bool(data.get("require_validation_pass", True)),
    )


def _deep_update(target: Dict[str, Any], source: Dict[str, Any]) -> None:
    for key, value in source.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _deep_update(target[key], value)
        else:
            target[key] = value


if __name__ == "__main__":
    main()
