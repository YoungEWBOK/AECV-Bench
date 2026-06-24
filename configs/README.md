# Benchmark Config Presets

All benchmark entrypoints accept `--config <json-file>`.

## API Environment

For the XMAPI streamed Responses gateway:

```powershell
set XMAPI_BASE_URL=https://www.cst9.cn
set XMAPI_API_KEY=sk-your-api-key
```

`XMAPI_BASE_URL` automatically selects `/v1/responses` streaming mode. For a standard OpenAI-compatible chat-completions endpoint, use:

```powershell
set OPENAI_BASE_URL=https://your-endpoint/v1
set OPENAI_API_KEY=sk-your-api-key
```

Models are normally selected in each config file via `model_id`. If you want to select the model from the environment, set `XMAPI_MODEL` and put `"model_id": "$XMAPI_MODEL"` in the JSON config.

For DashScope OpenAI-compatible mode, use chat-completions mode and do not leave `XMAPI_BASE_URL` set:

```powershell
set OPENAI_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
set OPENAI_API_KEY=sk-your-api-key
set LLM_API_MODE=chat_completions
```

Thinking-mode controls are passed through `extra_body`. The DashScope presets set `{"enable_thinking": false}` by default.

All QA judge presets use the same text-only judge model, `qwen3.6-flash`, and write to the same ranking directory:

```text
results/qa_llm_judge_results_unified_qwen36_flash
```

This keeps baseline and skill-guided results from different tested models in one comparable leaderboard.

## Presets

| File | Use |
| --- | --- |
| `baseline_one_shot.json` | Full one-shot baseline for QA and object counting. |
| `qa_gpt54_baseline.json` | QA-only GPT-5.4 one-shot baseline. |
| `qa_gpt54_self_evolution.json` | QA-only GPT-5.4 skill-guided/self-evolution run. |
| `qa_gpt54_exploration.json` | QA-only GPT-5.4 step-by-step and self-refine traces for skill evolution. |
| `skill_evolution_gpt54.json` | GPT-5.4-specific skill-evolution config using GPT-5.4 judge CSVs. |
| `qa_qwen37_plus_baseline.json` | QA-only Qwen3.7 Plus one-shot baseline. |
| `self_evolution_qwen37_plus.json` | Independent Qwen3.7 Plus supervised skill self-evolution from labels and predictions. |
| `qa_qwen37_plus_exploration.json` | Optional QA-only Qwen3.7 Plus step-by-step and self-refine traces for prompt ablation diagnostics. |
| `qa_qwen37_plus_self_evolution.json` | QA-only Qwen3.7 Plus skill-guided/self-evolution run. |
| `qa_deepseek_v4_pro_baseline.json` | QA-only DeepSeek V4 Pro one-shot baseline. |
| `self_evolution_deepseek_v4_pro.json` | Independent DeepSeek V4 Pro supervised skill self-evolution from labels and predictions. |
| `qa_deepseek_v4_pro_self_evolution.json` | QA-only DeepSeek V4 Pro skill-guided/self-evolution run. |
| `qa_kimi_k26_baseline.json` | QA-only Kimi K2.6 one-shot baseline. |
| `self_evolution_kimi_k26.json` | Independent Kimi K2.6 supervised skill self-evolution from labels and predictions. |
| `qa_kimi_k26_self_evolution.json` | QA-only Kimi K2.6 skill-guided/self-evolution run. |
| `skill_evolution_qwen37_plus.json` | Legacy/optional contrastive skill-mining config using Qwen judge CSVs. |
| `qa_prompt_matrix_mini.json` | QA-only mini matrix: `one_shot`, `step_by_step`, `self_refine`. |
| `qa_prompt_matrix_with_reflection.json` | QA-only matrix including `two_pass_reflection`. |
| `object_counting_prompt_matrix_mini.json` | Object-counting mini matrix: `one_shot`, `step_by_step`, `self_refine`, 20 folders. |
| `object_counting_prompt_matrix_with_reflection.json` | Object-counting matrix including `two_pass_reflection`, 20 folders. |
| `prompt_matrix_mini.json` | QA + object-counting mini matrix without two-pass reflection. |
| `skill_evolution.json` | Builds fixed/regressed cases, candidate skills, and skill-guided configs. |

`two_pass_reflection` makes about two LLM calls per item for that strategy.
`skill_guided` injects accepted skills from a JSON skill library; use it after running `run_skill_evolution.py`.

## Common Commands

Run QA:

```powershell
conda run -n exe python run_qa_benchmark.py --config configs\qa_prompt_matrix_mini.json
```

Judge QA using `judge.csv_files` from the same config:

```powershell
conda run -n exe python run_qa_llm_judge_evaluation.py --config configs\qa_prompt_matrix_mini.json
```

Run object counting:

```powershell
conda run -n exe python run_object_counting_benchmark.py --config configs\object_counting_prompt_matrix_mini.json
```

Run both mini matrices:

```powershell
conda run -n exe python run_qa_benchmark.py --config configs\prompt_matrix_mini.json
conda run -n exe python run_qa_llm_judge_evaluation.py --config configs\prompt_matrix_mini.json
conda run -n exe python run_object_counting_benchmark.py --config configs\prompt_matrix_mini.json
```

Build skill-evolution cases and dry-run candidate skills without API calls:

```powershell
conda run -n exe python run_skill_evolution.py --config configs\skill_evolution.json evolve --dry-run
```

Generate candidate skills with your OpenAI-compatible endpoint:

```powershell
conda run -n exe python run_skill_evolution.py --config configs\skill_evolution.json generate --model gpt-5.5
```

Compare GPT-5.4 one-shot baseline against skill-guided/self-evolution QA:

```powershell
conda run -n exe python run_qa_benchmark.py --config configs\qa_gpt54_baseline.json
conda run -n exe python run_qa_benchmark.py --config configs\qa_gpt54_self_evolution.json
conda run -n exe python run_qa_llm_judge_evaluation.py --config configs\qa_gpt54_baseline.json
conda run -n exe python run_qa_llm_judge_evaluation.py --config configs\qa_gpt54_self_evolution.json
conda run -n exe python generate_strategy_analysis_report.py --qa-eval-dir results\qa_llm_judge_results_unified_qwen36_flash --object-dir results\object_counting_unused_for_gpt54 --output-dir results\strategy_analysis_gpt54
```

Build a GPT-5.4 self-evolution skill library from GPT-5.4 traces:

```powershell
conda run -n exe python run_qa_benchmark.py --config configs\qa_gpt54_exploration.json
conda run -n exe python run_qa_llm_judge_evaluation.py --config configs\qa_gpt54_exploration.json
conda run -n exe python run_skill_evolution.py --config configs\skill_evolution_gpt54.json evolve --dry-run
conda run -n exe python run_qa_benchmark.py --config configs\qa_gpt54_self_evolution.json
```

Run independent Qwen3.7 Plus self-evolution, then compare one-shot baseline against evolved skill-guided QA:

```powershell
conda run -n exe python run_self_evolution.py --config configs\self_evolution_qwen37_plus.json --dry-run
conda run -n exe python run_qa_benchmark.py --config configs\qa_qwen37_plus_baseline.json
conda run -n exe python run_qa_llm_judge_evaluation.py --config configs\qa_qwen37_plus_baseline.json
conda run -n exe python run_qa_benchmark.py --config configs\qa_qwen37_plus_self_evolution.json
conda run -n exe python run_qa_llm_judge_evaluation.py --config configs\qa_qwen37_plus_self_evolution.json
conda run -n exe python generate_strategy_analysis_report.py --qa-eval-dir results\qa_llm_judge_results_unified_qwen36_flash --object-dir results\object_counting_unused_for_qwen37_plus --output-dir results\strategy_analysis_qwen37_plus
```

Run the same independent self-evolution comparison for DeepSeek V4 Pro:

```powershell
conda run -n exe python run_self_evolution.py --config configs\self_evolution_deepseek_v4_pro.json
conda run -n exe python run_qa_benchmark.py --config configs\qa_deepseek_v4_pro_baseline.json
conda run -n exe python run_qa_benchmark.py --config configs\qa_deepseek_v4_pro_self_evolution.json
conda run -n exe python run_qa_llm_judge_evaluation.py --config configs\qa_deepseek_v4_pro_baseline.json
conda run -n exe python run_qa_llm_judge_evaluation.py --config configs\qa_deepseek_v4_pro_self_evolution.json
```

Run the same independent self-evolution comparison for Kimi K2.6:

```powershell
conda run -n exe python run_self_evolution.py --config configs\self_evolution_kimi_k26.json
conda run -n exe python run_qa_benchmark.py --config configs\qa_kimi_k26_baseline.json
conda run -n exe python run_qa_benchmark.py --config configs\qa_kimi_k26_self_evolution.json
conda run -n exe python run_qa_llm_judge_evaluation.py --config configs\qa_kimi_k26_baseline.json
conda run -n exe python run_qa_llm_judge_evaluation.py --config configs\qa_kimi_k26_self_evolution.json
```
