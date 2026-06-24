# Prompt Strategy Experiments

This note documents the small ablation stage before building a full skill-evolution
framework.

## Why This Stage Exists

The current benchmark originally used direct one-shot prompts for QA and object
counting. Manual web UI tests suggested that asking the same model to re-check or
reflect can fix some errors. Before adding a larger skill-evolution system, run a
small controlled matrix so the extra complexity has evidence behind it.

The `hermes-agent-self-evolution` repo is most useful here as an evaluation-loop
reference: define a baseline, evaluate variants, keep constraint gates, and only
then evolve reusable skills. Do not start by importing DSPy/GEPA into this repo.

## Supported Strategies

The strategies are configured with `prompt_strategy` in `benchmark_config.json`.

| Strategy | API Calls | Behavior |
| --- | ---: | --- |
| `one_shot` | 1 | Existing direct prompt behavior. |
| `step_by_step` | 1 | Asks the model to internally inspect, reason, verify, then output only the final answer. |
| `self_refine` | 1 | Asks the model to internally draft, check likely mistakes, and return the corrected final answer. |
| `two_pass_reflection` | 2 | First gets an answer/count, then sends the image plus prior output for a second correction pass. |

`two_pass_reflection` is closest to the manual "answer once, then reflect" workflow,
but it roughly doubles model-call cost.

## Recommended Mini Matrix

Start small before running the full benchmark. Prefer using the presets in
`configs/` instead of repeatedly editing `benchmark_config.json`.

1. QA: run `configs/qa_prompt_matrix_mini.json`.
2. Object counting: run `configs/object_counting_prompt_matrix_mini.json`.
3. Judge QA results with the same QA config; `judge.csv_files` lists the expected strategy outputs.
4. Compare:
   - QA mean score and QA-type breakdown.
   - Object counting per-field accuracy/error.
   - API failure rate, latency, and cost.

Commands:

```powershell
conda run -n exe python run_qa_benchmark.py --config configs\qa_prompt_matrix_mini.json
conda run -n exe python run_qa_llm_judge_evaluation.py --config configs\qa_prompt_matrix_mini.json
conda run -n exe python run_object_counting_benchmark.py --config configs\object_counting_prompt_matrix_mini.json
```

## Output Naming

Default strategy keeps the old filenames:

- `benchmark_result_qa/qa_results_<model>.csv`
- `benchmark_result_object_counting/<model>.csv`

Non-default strategies append a suffix:

- `qa_results_<model>_step_by_step.csv`
- `<model>_self_refine.csv`
- `<model>_two_pass_reflection.csv`

This makes reruns and judge evaluation strategy-specific.

## Suggested Evolution Gate

Only build a larger skill-evolution layer after this ablation shows one of:

- A strategy improves QA or object counting by a meaningful margin on held-out samples.
- Improvements concentrate in interpretable failure classes, such as spatial QA,
  room counting, door/window ambiguity, or small OCR labels.
- Error reports show reusable procedural fixes that can become domain skills.

Candidate future skills should be specific, for example:

- Door/window symbol discrimination.
- Room and enclosed-space enumeration.
- Small label/OCR verification.
- Left/right and adjacency verification.
- Count-draft and count-audit procedure.
