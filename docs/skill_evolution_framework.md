# Skill Evolution Framework

This repo now has a lightweight skill-evolution loop for AEC drawing QA and object counting. It follows the design pattern used by recent self-evolving skill papers without adding DSPy/GEPA dependencies:

- Fixed category ontology, not hand-written concrete skills.
- Trace/case extraction from benchmark results.
- Candidate skill contracts with trigger, preconditions, observations, actions, validator, failure modes, source cases, utility, and lifecycle status.
- Validation gates for category integrity, required fields, case-answer leakage, and over-specific skills.
- Replay acceptance gates that accept skills only after a later skill-guided run improves judged results without excessive regressions.

## Fixed Categories

The eight fixed categories live in `src/skill_evolution/categories.py`:

1. `visual_evidence_acquisition`
2. `text_ocr_grounding`
3. `symbol_geometry_grounding`
4. `spatial_relation_reasoning`
5. `counting_enumeration`
6. `answer_synthesis`
7. `verification_reflection`
8. `skill_library_management`

Concrete skills should be generated from fixed/regressed cases, not manually defined.

## Main Files

- `run_skill_evolution.py`: CLI for building cases, generating candidates, validating, and replay acceptance.
- `src/skill_evolution/contracts.py`: `SkillContract`, `SkillUtility`, and `SkillLibrary`.
- `src/skill_evolution/cases.py`: fixed/regressed case builders for QA judge CSVs and object-counting CSVs.
- `src/skill_evolution/generator.py`: LLM generator plus dry-run heuristic generator.
- `src/skill_evolution/validator.py`: hard validation gates.
- `src/skill_evolution/replay.py`: replay summary and acceptance gate.
- `src/utils/prompt_strategies.py`: adds `skill_guided`.
- `run_qa_benchmark.py` and `run_object_counting_benchmark.py`: can consume `skill_library_path`.

## Recommended Loop

Build evolution cases from existing baseline and strategy judge CSVs:

```bash
conda run -n exe python run_skill_evolution.py --config configs/skill_evolution.json build-cases
```

Generate candidate skills without spending API tokens:

```bash
conda run -n exe python run_skill_evolution.py --config configs/skill_evolution.json generate --dry-run
```

Generate candidate skills with your configured OpenAI-compatible API:

```bash
conda run -n exe python run_skill_evolution.py --config configs/skill_evolution.json generate --model gpt-5.5
```

Or run build-cases + generate + validation + skill-guided config export:

```bash
conda run -n exe python run_skill_evolution.py --config configs/skill_evolution.json evolve --dry-run
```

Then run the generated benchmark config:

```bash
conda run -n exe python run_qa_benchmark.py --config configs/qa_skill_guided_generated.json
conda run -n exe python run_qa_llm_judge_evaluation.py --config configs/qa_skill_guided_generated.json
```

After the skill-guided CSV has been judged, set `skill_guided_eval_csv` in `configs/skill_evolution.json`, then accept/reject skills by replay:

```bash
conda run -n exe python run_skill_evolution.py --config configs/skill_evolution.json replay
```

The accepted library is written to `results/skill_evolution/accepted_skill_library.json`.

## Benchmark Config Fields

QA models can consume a skill library with:

```json
{
  "prompt_strategy": "skill_guided",
  "skill_library_path": "results/skill_evolution/accepted_skill_library.json",
  "max_skills_per_question": 4,
  "skill_statuses": ["accepted"]
}
```

Object-counting models support the same two library fields. The runner injects the most relevant accepted skills according to question/task/category routing.
For candidate experiments before replay, use `["candidate", "accepted"]`.

## Notes

- `--dry-run` is only for plumbing validation. It emits conservative candidate contracts from fixed/regressed clusters without calling an LLM.
- For real self-evolution, use `generate --model <your model_id>` so candidate skills are inferred from contrastive cases.
- The generator prompt forbids copying case-specific answers into skill text; the validator checks for this again.
- Skills are accepted only after replay. Discovery cases are useful for proposing skills, not final proof.
