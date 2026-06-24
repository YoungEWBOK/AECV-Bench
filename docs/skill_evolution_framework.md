# Skill Evolution Framework

This repo now has a lightweight skill-evolution loop for AEC drawing QA and object counting. It follows the design pattern used by recent self-evolving skill papers without adding DSPy/GEPA dependencies:

- Fixed category ontology, not hand-written concrete skills.
- Trace/case extraction from benchmark results.
- Candidate skill contracts with trigger, preconditions, observations, actions, validator, failure modes, source cases, utility, and lifecycle status.
- Validation gates for category integrity, required fields, case-answer leakage, and over-specific skills.
- Replay acceptance gates that accept skills only after a later skill-guided run improves judged results without excessive regressions.

## Fixed Categories

The eight fixed categories live in `src/skill_evolution/categories.py`. They
are organized by intermediate representation, not by a broad end-to-end
reasoning script. A concrete skill should have one primary category; if it spans
multiple categories, split it into smaller composable skills.

1. `view_control`: raw image/current view -> targeted evidence view.
2. `text_annotation_grounding`: evidence view -> positioned text spans and normalized annotations.
3. `graphic_symbol_grounding`: evidence view -> typed symbol instances.
4. `region_boundary_grounding`: visual primitives/annotations -> rooms, regions, boundaries, and openings.
5. `spatial_topology_modeling`: grounded entities/regions -> spatial relation graph.
6. `quantitative_set_reasoning`: entities/annotations/relation graph -> counts, comparisons, and selected sets.
7. `query_answer_binding`: question/evidence graph -> concise answer candidate.
8. `evidence_verification`: answer/evidence trace -> verified or corrected answer.

Concrete skills should be generated from fixed/regressed cases, not manually
defined. Legacy category names such as `text_ocr_grounding` and
`counting_enumeration` are accepted as aliases for loading older libraries, but
newly generated libraries should use the category IDs above.

`Skill Library Management` is no longer a primary skill category. Merge, prune,
accept/reject, replay validation, provenance, and failure-mode tracking are
framework governance operations handled by the library, validator, and replay
modules.

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
