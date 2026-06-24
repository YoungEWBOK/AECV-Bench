"""
Script to test QA pairs from ground truth labels.

This script:
1. Reads label JSON files with QA pairs (qa_pairs, ocr_qa, spatial_qa, counting_qa, comparison_qa)
2. Sends questions to the model with the corresponding image
3. Saves results: question, model answer, ground truth answer, task type
"""
import json
import os
import csv
import time
import argparse
from pathlib import Path
from src.utils.image_utils import encode_image_to_base64
from src.utils.config import require_llm_api_key, require_llm_base_url
from src.utils.benchmark_config import (
    DEFAULT_CONFIG_PATH,
    get_enabled_models,
    get_required_value,
    get_section,
    load_benchmark_config,
)
from src.utils.openai_compatible import chat_completion_content
from src.utils.prompt_strategies import (
    build_qa_prompt,
    build_qa_reflection_prompt,
    make_safe_name,
    normalize_prompt_strategy,
    prompt_strategy_suffix,
)
from src.skill_evolution.contracts import SkillLibrary

# Single OpenAI-compatible API configuration
llm_api_key = require_llm_api_key()
llm_base_url = require_llm_base_url()

# Configuration
images_dir = "data/Use Case 2 - Drawing Understanding/01 - Full Dataset/images"
labels_dir = "data/Use Case 2 - Drawing Understanding/01 - Full Dataset/labels"
output_dir = "benchmark_result_qa"
os.makedirs(output_dir, exist_ok=True)
url = llm_base_url
temperature = 0.0

# Model configurations - based on run_all_models_benchmark.py
# Uncomment models you want to test
models = [
    # {
    #     "name": "Gemini 3 Pro Preview",
    #     "model_id": "google/gemini-3-pro-preview",
    #     "note": "Latest flagship model, high-precision multimodal reasoning"
    # },
    # {
    #     "name": "Claude Opus 4.5",
    #     "model_id": "anthropic/claude-opus-4.5",
    #     "note": "Most advanced Opus model, optimized for complex reasoning tasks"
    # },
    # {
    #     "name": "Claude Sonnet 4.5",
    #     "model_id": "anthropic/claude-sonnet-4.5",
    #     "note": "Most advanced Sonnet, optimized for real-world agents"
    # },
    # {
    #     "name": "Gemini 3.1 Pro",
    #     "model_id": "google/gemini-3.1-pro-preview",
    #     "note": "Gemini 3.1 Pro preview model"
    # },
    # {
    #     "name": "Claude Sonnet 4.6",
    #     "model_id": "anthropic/claude-sonnet-4.6",
    #     "note": "Anthropic Claude Sonnet 4.6, latest Sonnet model"
    # },
    # {
    #     "name": "Qwen 3.5 Plus",
    #     "model_id": "qwen/qwen3.5-plus-02-15",
    #     "note": "Qwen 3.5 Plus model from February 2025"
    # },
    # {
    #     "name": "Claude Opus 4.6",
    #     "model_id": "anthropic/claude-opus-4.6",
    #     "note": "Anthropic Claude Opus 4.6, most advanced Opus model"
    # },
    # {
    #     "name": "Qwen3-VL 8B Instruct",
    #     "model_id": "qwen/qwen3-vl-8b-instruct",
    #     "note": "8B Qwen3 vision-language model – efficient and fast"
    # },
    # {
    #     "name": "Qwen3-VL 8B Thinking",
    #     "model_id": "qwen/qwen3-vl-8b-thinking",
    #     "note": "8B Qwen3 thinking model – better reasoning, slower throughput"
    # },
    # {
    #     "name": "Mistral Large 2512",
    #     "model_id": "mistralai/mistral-large-2512",
    #     "note": "Mistral Large model from December 2025"
    # },
    # {
    #     "name": "OpenAI GPT-5.2",
    #     "model_id": "openai/gpt-5.2",
    #     "note": "OpenAI GPT-5.2 model"
    # },
    # {
    #     "name": "OpenAI GPT-5.3",
    #     "model_id": "openai/gpt-5.3-chat",
    #     "note": "OpenAI GPT-5.3 Chat model"
    # },
    {
        "name": "OpenAI GPT-5.4",
        "model_id": "openai/gpt-5.4",
        "note": "OpenAI GPT-5.4 model"
    },
    # {
    #     "name": "Amazon Nova 2 Lite v1",
    #     "model_id": "amazon/nova-2-lite-v1",
    #     "note": "Amazon Nova 2 Lite vision-language model"
    # },
    # {
    #     "name": "Grok 4.1 Fast",
    #     "model_id": "x-ai/grok-4.1-fast",
    #     "note": "Best agentic tool calling model, 2M context"
    # },
    # {
    #     "name": "OpenAI GPT-4 Vision",
    #     "model_id": "openai/gpt-4o",  # GPT-4o provides the latest GPT-4 vision features
    #     "note": "GPT-4o multimodal model (uses prompt-based JSON extraction via OpenRouter)"
    # },
    # {
    #     "name": "Nvidia Nemotron Nano 12B V2 VL",
    #     "model_id": "nvidia/nemotron-nano-12b-v2-vl",
    #     "note": "Nvidia Nemotron Nano 12B V2 vision-language model"
    # },
    # {
    #     "name": "Llama Nemotron Embed VL 1B V2",
    #     "model_id": "nvidia/llama-nemotron-embed-vl-1b-v2:free",
    #     "note": "Nvidia Llama Nemotron Embed VL 1B V2 vision-language model (free)"
    # },
    # {
    #     "name": "GLM-4.6V",
    #     "model_id": "z-ai/glm-4.6v",
    #     "note": "Z-AI vision-language model - newer version (uses prompt-based JSON extraction)"
    # },
    # {
    #     "name": "Cohere Command A Vision",
    #     "model_id": "command-a-vision-07-2025",  # Cohere's first commercial multimodal vision model
    #     "use_cohere_api": True,  # Set this flag to use Cohere API instead of OpenRouter
    #     "note": "Cohere Command A Vision - multimodal model for document analysis, chart interpretation, and OCR. 128K context, supports up to 20 images per request."
    # },
]


def ask_question_with_image(
    image_path: str,
    question: str,
    model_name: str,
    open_router_api_key: str,
    url: str,
    temperature: float = 0.0,
    prompt_strategy: str = "one_shot",
    skill_library: SkillLibrary = None,
    qa_type: str = "",
    task: str = "",
    max_skills_per_question: int = 4,
    skill_statuses=None,
    extra_body=None,
) -> str:
    """
    Send a question with an image to the model and get the answer.
    
    Args:
        image_path: Path to the image file
        question: The question to ask
        model_name: Model identifier
        open_router_api_key: API key override
        url: OpenAI-compatible base URL
        temperature: Sampling temperature
        
    Returns:
        The model's answer as a string
    """
    # Read and encode image
    base64_image = encode_image_to_base64(image_path)
    from src.utils.image_utils import get_image_mime_type
    mime_type = get_image_mime_type(image_path)
    data_url = f"data:{mime_type};base64,{base64_image}"

    prompt_strategy = normalize_prompt_strategy(prompt_strategy)
    skill_context = ""
    if skill_library is not None:
        skill_context = skill_library.format_for_prompt(
            question=question,
            qa_type=qa_type,
            task=task,
            max_skills=max_skills_per_question,
            statuses=skill_statuses or ("accepted",),
        )

    def build_messages(prompt_text: str):
        return [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt_text
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": data_url
                        }
                    }
                ]
            }
        ]

    first_answer = chat_completion_content(
        model=model_name,
        messages=build_messages(build_qa_prompt(question, prompt_strategy, skill_context=skill_context)),
        api_key=open_router_api_key,
        base_url=url,
        temperature=temperature,
        extra_body=extra_body,
        timeout=60,
        max_retries=3,
        request_label=f"QA image question for '{image_path}'",
    )

    if prompt_strategy != "two_pass_reflection":
        return first_answer

    return chat_completion_content(
        model=model_name,
        messages=build_messages(build_qa_reflection_prompt(question, first_answer)),
        api_key=open_router_api_key,
        base_url=url,
        temperature=temperature,
        extra_body=extra_body,
        timeout=60,
        max_retries=3,
        request_label=f"QA reflection for '{image_path}'",
    )

def ask_question_with_image_cohere(
    image_path: str,
    question: str,
    model_name: str,
    cohere_api_key: str,
    url: str = None,
    temperature: float = 0.0,
    prompt_strategy: str = "one_shot",
    skill_library: SkillLibrary = None,
    qa_type: str = "",
    task: str = "",
    max_skills_per_question: int = 4,
    skill_statuses=None,
    extra_body=None,
) -> str:
    """
    Backward-compatible wrapper for older Cohere-configured models.

    All LLM calls now use the configured OpenAI-compatible API key and base URL.
    """
    return ask_question_with_image(
        image_path=image_path,
        question=question,
        model_name=model_name,
        open_router_api_key=cohere_api_key,
        url=url or llm_base_url,
        temperature=temperature,
        prompt_strategy=prompt_strategy,
        skill_library=skill_library,
        qa_type=qa_type,
        task=task,
        max_skills_per_question=max_skills_per_question,
        skill_statuses=skill_statuses,
        extra_body=extra_body,
    )


def process_qa_benchmark(labels_dir: str, images_dir: str, output_csv: str, model_name: str, open_router_api_key: str, url: str, temperature: float = 0.0, use_cohere_api: bool = False, cohere_api_key: str = None, prompt_strategy: str = "one_shot", skill_library_path: str = "", max_skills_per_question: int = 4, skill_statuses=None, extra_body=None):
    """
    Process QA pairs from label files and save results.
    
    Args:
        labels_dir: Directory containing label JSON files
        images_dir: Directory containing image files
        output_csv: Path to output CSV file
        model_name: Model identifier
        open_router_api_key: API key override
        url: OpenAI-compatible base URL
        temperature: Sampling temperature
        use_cohere_api: Backward-compatible flag; still uses the OpenAI-compatible endpoint
        cohere_api_key: API key override for backward-compatible Cohere configs
        prompt_strategy: Prompt strategy for this run
    """
    prompt_strategy = normalize_prompt_strategy(prompt_strategy)
    skill_library = None
    if skill_library_path:
        skill_library = SkillLibrary.load(skill_library_path)
        accepted_count = len([skill for skill in skill_library.skills if skill.status == "accepted"])
        candidate_count = len([skill for skill in skill_library.skills if skill.status == "candidate"])
        print(
            f"Skill library: {skill_library_path} "
            f"({accepted_count} accepted, {candidate_count} candidate; "
            f"statuses={skill_statuses or ('accepted',)}; max {max_skills_per_question}/question)"
        )
    # Get all label files
    label_files = sorted(Path(labels_dir).glob("*.json"))
    
    if not label_files:
        print(f"Error: No label files found in {labels_dir}")
        return

    output_parent = os.path.dirname(output_csv)
    if output_parent:
        os.makedirs(output_parent, exist_ok=True)
    
    print(f"Found {len(label_files)} label files")
    print(f"Model: {model_name}")
    print(f"Prompt strategy: {prompt_strategy}")
    print(f"Processing QA pairs...\n")
    
    results = []
    total_questions = 0
    processed_questions = 0
    fieldnames = ["image_id", "qa_id", "qa_type", "task", "question", "ground_truth", "model_answer"]
    processed_keys = set()

    def result_key(image_id: str, qa_id: str, qa_type: str):
        return (image_id, qa_id, qa_type)

    def record_result(result: dict):
        """Keep in-memory summary and stream each completed QA result to disk."""
        results.append(result)
        processed_keys.add(result_key(result["image_id"], result["qa_id"], result["qa_type"]))
        with open(output_csv, 'a', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writerow(result)
            csvfile.flush()

    if os.path.isfile(output_csv):
        completed_rows = []
        failed_rows = 0
        with open(output_csv, 'r', newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                if row.get("image_id") and row.get("qa_id") and row.get("qa_type"):
                    if row.get("model_answer", "").startswith("[ERROR:"):
                        failed_rows += 1
                        continue
                    results.append(row)
                    completed_rows.append(row)
                    processed_keys.add(result_key(row["image_id"], row["qa_id"], row["qa_type"]))

        with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for row in completed_rows:
                writer.writerow(row)

        print(f"Resuming from existing CSV: {output_csv} ({len(processed_keys)} completed rows)")
        if failed_rows:
            print(f"  Retrying {failed_rows} previous failed row(s)")
    else:
        with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
    
    for label_file in label_files:
        try:
            with open(label_file, 'r', encoding='utf-8') as f:
                label_data = json.load(f)
        except Exception as e:
            print(f"[ERROR] Could not read {label_file}: {e}")
            continue
        
        image_id = label_data.get("image_id", label_file.stem)
        image_path_from_label = label_data.get("image_path", "")
        
        # Find the actual image file
        # Try different possible paths/extensions
        image_path = None
        possible_names = [
            f"{image_id}.png",
            f"{image_id}.jpg",
            f"{image_id}.jpeg",
            os.path.basename(image_path_from_label) if image_path_from_label else None
        ]
        
        for name in possible_names:
            if name:
                potential_path = os.path.join(images_dir, name)
                if os.path.isfile(potential_path):
                    image_path = potential_path
                    break
        
        if not image_path or not os.path.isfile(image_path):
            print(f"[WARN] Skipping {image_id}: image file not found")
            continue
        
        # Process qa_pairs (skip if not present)
        qa_pairs = label_data.get("qa_pairs", [])
        if not isinstance(qa_pairs, list):
            qa_pairs = []
        
        for qa in qa_pairs:
            total_questions += 1
            question = qa.get("question", "")
            ground_truth = qa.get("answer", "")
            task = qa.get("task", "unknown")
            qa_id = qa.get("id", f"{image_id}_q{total_questions}")
            
            if not question:
                continue

            key = result_key(image_id, qa_id, "qa_pairs")
            if key in processed_keys:
                processed_questions += 1
                print(f"Skipping already processed {qa_id}")
                continue
            
            try:
                print(f"Processing {qa_id}: {question[:60]}...")
                if use_cohere_api:
                    model_answer = ask_question_with_image_cohere(
                        image_path=image_path,
                        question=question,
                        model_name=model_name,
                        cohere_api_key=cohere_api_key,
                        temperature=temperature,
                        prompt_strategy=prompt_strategy,
                        skill_library=skill_library,
                        qa_type="qa_pairs",
                        task=task,
                        max_skills_per_question=max_skills_per_question,
                        skill_statuses=skill_statuses,
                        extra_body=extra_body,
                    )
                else:
                    model_answer = ask_question_with_image(
                        image_path=image_path,
                        question=question,
                        model_name=model_name,
                        open_router_api_key=open_router_api_key,
                        url=url,
                        temperature=temperature,
                        prompt_strategy=prompt_strategy,
                        skill_library=skill_library,
                        qa_type="qa_pairs",
                        task=task,
                        max_skills_per_question=max_skills_per_question,
                        skill_statuses=skill_statuses,
                        extra_body=extra_body,
                    )
                
                record_result({
                    "image_id": image_id,
                    "qa_id": qa_id,
                    "qa_type": "qa_pairs",
                    "task": task,
                    "question": question,
                    "ground_truth": ground_truth,
                    "model_answer": model_answer
                })
                processed_questions += 1
                print(f"  Done ({processed_questions}/{total_questions})")
                
                # Small delay to avoid rate limiting
                time.sleep(0.5)
                
            except Exception as e:
                print(f"[ERROR] Failed to process {qa_id}: {e}")
                record_result({
                    "image_id": image_id,
                    "qa_id": qa_id,
                    "qa_type": "qa_pairs",
                    "task": task,
                    "question": question,
                    "ground_truth": ground_truth,
                    "model_answer": f"[ERROR: {str(e)}]"
                })
        
        # Process ocr_qa (skip if not present)
        ocr_qa = label_data.get("ocr_qa", [])
        if not isinstance(ocr_qa, list):
            ocr_qa = []
        
        for qa in ocr_qa:
            total_questions += 1
            question = qa.get("question", "")
            ground_truth = qa.get("answer", "")
            task = qa.get("task", "unknown")
            qa_id = qa.get("id", f"{image_id}_ocr{total_questions}")
            
            if not question:
                continue

            key = result_key(image_id, qa_id, "ocr_qa")
            if key in processed_keys:
                processed_questions += 1
                print(f"Skipping already processed {qa_id}")
                continue
            
            try:
                print(f"Processing {qa_id}: {question[:60]}...")
                if use_cohere_api:
                    model_answer = ask_question_with_image_cohere(
                        image_path=image_path,
                        question=question,
                        model_name=model_name,
                        cohere_api_key=cohere_api_key,
                        temperature=temperature,
                        prompt_strategy=prompt_strategy,
                        skill_library=skill_library,
                        qa_type="ocr_qa",
                        task=task,
                        max_skills_per_question=max_skills_per_question,
                        skill_statuses=skill_statuses,
                        extra_body=extra_body,
                    )
                else:
                    model_answer = ask_question_with_image(
                        image_path=image_path,
                        question=question,
                        model_name=model_name,
                        open_router_api_key=open_router_api_key,
                        url=url,
                        temperature=temperature,
                        prompt_strategy=prompt_strategy,
                        skill_library=skill_library,
                        qa_type="ocr_qa",
                        task=task,
                        max_skills_per_question=max_skills_per_question,
                        skill_statuses=skill_statuses,
                        extra_body=extra_body,
                    )
                
                record_result({
                    "image_id": image_id,
                    "qa_id": qa_id,
                    "qa_type": "ocr_qa",
                    "task": task,
                    "question": question,
                    "ground_truth": ground_truth,
                    "model_answer": model_answer
                })
                processed_questions += 1
                print(f"  Done ({processed_questions}/{total_questions})")
                
                time.sleep(0.5)
                
            except Exception as e:
                print(f"[ERROR] Failed to process {qa_id}: {e}")
                record_result({
                    "image_id": image_id,
                    "qa_id": qa_id,
                    "qa_type": "ocr_qa",
                    "task": task,
                    "question": question,
                    "ground_truth": ground_truth,
                    "model_answer": f"[ERROR: {str(e)}]"
                })
        
        # Process spatial_qa (skip if not present)
        spatial_qa = label_data.get("spatial_qa", [])
        if not isinstance(spatial_qa, list):
            spatial_qa = []
        
        for qa in spatial_qa:
            total_questions += 1
            question = qa.get("question", "")
            ground_truth = qa.get("answer", "")
            task = qa.get("task", "unknown")
            qa_id = qa.get("id", f"{image_id}_sp{total_questions}")
            
            if not question:
                continue

            key = result_key(image_id, qa_id, "spatial_qa")
            if key in processed_keys:
                processed_questions += 1
                print(f"Skipping already processed {qa_id}")
                continue
            
            try:
                print(f"Processing {qa_id}: {question[:60]}...")
                if use_cohere_api:
                    model_answer = ask_question_with_image_cohere(
                        image_path=image_path,
                        question=question,
                        model_name=model_name,
                        cohere_api_key=cohere_api_key,
                        temperature=temperature,
                        prompt_strategy=prompt_strategy,
                        skill_library=skill_library,
                        qa_type="spatial_qa",
                        task=task,
                        max_skills_per_question=max_skills_per_question,
                        skill_statuses=skill_statuses,
                        extra_body=extra_body,
                    )
                else:
                    model_answer = ask_question_with_image(
                        image_path=image_path,
                        question=question,
                        model_name=model_name,
                        open_router_api_key=open_router_api_key,
                        url=url,
                        temperature=temperature,
                        prompt_strategy=prompt_strategy,
                        skill_library=skill_library,
                        qa_type="spatial_qa",
                        task=task,
                        max_skills_per_question=max_skills_per_question,
                        skill_statuses=skill_statuses,
                        extra_body=extra_body,
                    )
                
                record_result({
                    "image_id": image_id,
                    "qa_id": qa_id,
                    "qa_type": "spatial_qa",
                    "task": task,
                    "question": question,
                    "ground_truth": ground_truth,
                    "model_answer": model_answer
                })
                processed_questions += 1
                print(f"  Done ({processed_questions}/{total_questions})")
                
                time.sleep(0.5)
                
            except Exception as e:
                print(f"[ERROR] Failed to process {qa_id}: {e}")
                record_result({
                    "image_id": image_id,
                    "qa_id": qa_id,
                    "qa_type": "spatial_qa",
                    "task": task,
                    "question": question,
                    "ground_truth": ground_truth,
                    "model_answer": f"[ERROR: {str(e)}]"
                })

        # Process counting_qa (skip if not present)
        counting_qa = label_data.get("counting_qa", [])
        if not isinstance(counting_qa, list):
            counting_qa = []

        for qa in counting_qa:
            total_questions += 1
            question = qa.get("question", "")
            ground_truth = qa.get("answer", "")
            task = qa.get("task", "unknown")
            qa_id = qa.get("id", f"{image_id}_cnt{total_questions}")

            if not question:
                continue

            key = result_key(image_id, qa_id, "counting_qa")
            if key in processed_keys:
                processed_questions += 1
                print(f"Skipping already processed {qa_id}")
                continue

            try:
                print(f"Processing {qa_id}: {question[:60]}...")
                if use_cohere_api:
                    model_answer = ask_question_with_image_cohere(
                        image_path=image_path,
                        question=question,
                        model_name=model_name,
                        cohere_api_key=cohere_api_key,
                        temperature=temperature,
                        prompt_strategy=prompt_strategy,
                        skill_library=skill_library,
                        qa_type="counting_qa",
                        task=task,
                        max_skills_per_question=max_skills_per_question,
                        skill_statuses=skill_statuses,
                        extra_body=extra_body,
                    )
                else:
                    model_answer = ask_question_with_image(
                        image_path=image_path,
                        question=question,
                        model_name=model_name,
                        open_router_api_key=open_router_api_key,
                        url=url,
                        temperature=temperature,
                        prompt_strategy=prompt_strategy,
                        skill_library=skill_library,
                        qa_type="counting_qa",
                        task=task,
                        max_skills_per_question=max_skills_per_question,
                        skill_statuses=skill_statuses,
                        extra_body=extra_body,
                    )

                record_result({
                    "image_id": image_id,
                    "qa_id": qa_id,
                    "qa_type": "counting_qa",
                    "task": task,
                    "question": question,
                    "ground_truth": ground_truth,
                    "model_answer": model_answer
                })
                processed_questions += 1
                print(f"  Done ({processed_questions}/{total_questions})")

                time.sleep(0.5)

            except Exception as e:
                print(f"[ERROR] Failed to process {qa_id}: {e}")
                record_result({
                    "image_id": image_id,
                    "qa_id": qa_id,
                    "qa_type": "counting_qa",
                    "task": task,
                    "question": question,
                    "ground_truth": ground_truth,
                    "model_answer": f"[ERROR: {str(e)}]"
                })

        # Process comparison_qa (skip if not present)
        comparison_qa = label_data.get("comparison_qa", [])
        if not isinstance(comparison_qa, list):
            comparison_qa = []

        for qa in comparison_qa:
            total_questions += 1
            question = qa.get("question", "")
            ground_truth = qa.get("answer", "")
            task = qa.get("task", "unknown")
            qa_id = qa.get("id", f"{image_id}_cmp{total_questions}")

            if not question:
                continue

            key = result_key(image_id, qa_id, "comparison_qa")
            if key in processed_keys:
                processed_questions += 1
                print(f"Skipping already processed {qa_id}")
                continue

            try:
                print(f"Processing {qa_id}: {question[:60]}...")
                if use_cohere_api:
                    model_answer = ask_question_with_image_cohere(
                        image_path=image_path,
                        question=question,
                        model_name=model_name,
                        cohere_api_key=cohere_api_key,
                        temperature=temperature,
                        prompt_strategy=prompt_strategy,
                        skill_library=skill_library,
                        qa_type="comparison_qa",
                        task=task,
                        max_skills_per_question=max_skills_per_question,
                        skill_statuses=skill_statuses,
                        extra_body=extra_body,
                    )
                else:
                    model_answer = ask_question_with_image(
                        image_path=image_path,
                        question=question,
                        model_name=model_name,
                        open_router_api_key=open_router_api_key,
                        url=url,
                        temperature=temperature,
                        prompt_strategy=prompt_strategy,
                        skill_library=skill_library,
                        qa_type="comparison_qa",
                        task=task,
                        max_skills_per_question=max_skills_per_question,
                        skill_statuses=skill_statuses,
                        extra_body=extra_body,
                    )

                record_result({
                    "image_id": image_id,
                    "qa_id": qa_id,
                    "qa_type": "comparison_qa",
                    "task": task,
                    "question": question,
                    "ground_truth": ground_truth,
                    "model_answer": model_answer
                })
                processed_questions += 1
                print(f"  Done ({processed_questions}/{total_questions})")

                time.sleep(0.5)

            except Exception as e:
                print(f"[ERROR] Failed to process {qa_id}: {e}")
                record_result({
                    "image_id": image_id,
                    "qa_id": qa_id,
                    "qa_type": "comparison_qa",
                    "task": task,
                    "question": question,
                    "ground_truth": ground_truth,
                    "model_answer": f"[ERROR: {str(e)}]"
                })

    # Results are streamed to CSV as each question completes.
    if results:
        print(f"\n[SUCCESS] Results saved to {output_csv}")
        print(f"Total questions processed: {processed_questions}/{total_questions}")
    else:
        print("\n[WARN] No results to save")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run QA benchmark")
    parser.add_argument(
        "--config",
        type=str,
        default=DEFAULT_CONFIG_PATH,
        help=f"Benchmark config JSON path (default: {DEFAULT_CONFIG_PATH})",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing QA result CSVs and rerun all questions for enabled models.",
    )
    args = parser.parse_args()

    config = load_benchmark_config(args.config)
    qa_config = get_section(config, "qa")
    images_dir = get_required_value(qa_config, "images_dir", "qa")
    labels_dir = get_required_value(qa_config, "labels_dir", "qa")
    output_dir = get_required_value(qa_config, "output_dir", "qa")
    models = get_enabled_models(qa_config, "qa")
    os.makedirs(output_dir, exist_ok=True)
    force = args.force or bool(qa_config.get("force", False))

    print("="*60)
    print("QA BENCHMARK - Testing Question/Answer Pairs")
    print("="*60)
    print(f"Labels directory: {labels_dir}")
    print(f"Images directory: {images_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Number of models: {len(models)}\n")
    
    # Run benchmark for each model
    results_summary = []
    for i, model_config in enumerate(models, 1):
        model_name = model_config["name"]
        model_id = model_config["model_id"]
        prompt_strategy = normalize_prompt_strategy(model_config.get("prompt_strategy", qa_config.get("prompt_strategy", "one_shot")))
        skill_library_path = model_config.get("skill_library_path", qa_config.get("skill_library_path", ""))
        max_skills_per_question = int(model_config.get("max_skills_per_question", qa_config.get("max_skills_per_question", 4)))
        skill_statuses = model_config.get("skill_statuses", qa_config.get("skill_statuses", ["accepted"]))
        extra_body = model_config.get("extra_body", qa_config.get("extra_body"))
        if isinstance(skill_statuses, str):
            skill_statuses = [item.strip() for item in skill_statuses.split(",") if item.strip()]
        
        # Create output filename
        safe_name = make_safe_name(model_name)
        strategy_suffix = prompt_strategy_suffix(prompt_strategy)
        display_name = model_name
        if strategy_suffix:
            safe_name = f"{safe_name}_{strategy_suffix}"
            display_name = f"{model_name} ({prompt_strategy})"
        output_csv = os.path.join(output_dir, f"qa_results_{safe_name}.csv")
        if force and os.path.isfile(output_csv):
            os.remove(output_csv)
        
        print(f"\n[{i}/{len(models)}] Processing: {model_name}")
        print(f"  Model ID: {model_id}")
        print(f"  Prompt strategy: {prompt_strategy}")
        if skill_library_path:
            print(
                f"  Skill library: {skill_library_path} "
                f"(statuses={skill_statuses}, max {max_skills_per_question}/question)"
            )
        if extra_body:
            print(f"  Extra body: {extra_body}")
        if "note" in model_config:
            print(f"  Note: {model_config['note']}")
        print(f"  Output: {output_csv}")
        print("-" * 60)
        
        try:
            # Backward-compatible handling for older Cohere-configured entries.
            use_cohere = model_config.get("use_cohere_api", False)
            cohere_key = None
            if use_cohere:
                cohere_key = llm_api_key
            
            process_qa_benchmark(
                labels_dir=labels_dir,
                images_dir=images_dir,
                output_csv=output_csv,
                model_name=model_id,
                open_router_api_key=llm_api_key,
                url=url,
                temperature=temperature,
                use_cohere_api=use_cohere,
                cohere_api_key=cohere_key,
                prompt_strategy=prompt_strategy,
                skill_library_path=skill_library_path,
                max_skills_per_question=max_skills_per_question,
                skill_statuses=skill_statuses,
                extra_body=extra_body,
            )
            results_summary.append({
                "name": display_name,
                "csv": output_csv,
                "status": "success"
            })
            print(f"[SUCCESS] {model_name} completed")
        except Exception as e:
            print(f"[ERROR] {model_name} failed: {e}")
            results_summary.append({
                "name": display_name,
                "csv": output_csv,
                "status": "failed",
                "error": str(e)
            })
    
    print("\n" + "="*60)
    print("QA BENCHMARK COMPLETE")
    print("="*60)
    print("\nResults Summary:")
    for result in results_summary:
        status_icon = "[OK]" if result["status"] == "success" else "[FAIL]"
        print(f"  {status_icon} {result['name']}: {result['csv']}")
        if result["status"] == "failed" and "error" in result:
            print(f"      Error: {result['error'][:100]}")

