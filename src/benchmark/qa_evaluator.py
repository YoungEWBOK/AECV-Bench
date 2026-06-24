"""
LLM-based evaluation system for QA benchmark results.

Uses an OpenAI-compatible LLM judge to evaluate if model answers are correct
compared to ground truth. Returns binary scores: 1.0 for correct, 0.0 for incorrect.
"""
import csv
import os
from typing import Dict, Optional
from collections import defaultdict
import numpy as np

from ..utils.config import require_llm_api_key, require_llm_base_url
from ..utils.openai_compatible import chat_completion_content


class QAEvaluator:
    """Evaluates QA answers using LLM-as-Judge via an OpenAI-compatible endpoint."""
    
    def __init__(
        self,
        judge_model: str = "openai/gpt-4o",
        open_router_api_key: Optional[str] = None,
        url: str = None,
        temperature: float = 0.0,
        extra_body: Optional[Dict] = None,
        stream: bool = False,
        stream_options: Optional[Dict] = None,
    ):
        """
        Initialize the QA evaluator.
        
        Args:
            judge_model: Model identifier for the judge LLM (default: openai/gpt-4o)
            open_router_api_key: API key override (if None, uses OPENAI_API_KEY/API_KEY)
            url: Base URL override (if None, uses OPENAI_BASE_URL/BASE_URL)
            temperature: Temperature for judge model (default 0.0 for deterministic)
        """
        self.judge_model = judge_model
        self.base_url = url or require_llm_base_url()
        self.temperature = temperature
        self.extra_body = extra_body
        self.stream = stream
        self.stream_options = stream_options
        
        # Get API key
        if open_router_api_key is None or not open_router_api_key.strip():
            self.api_key = require_llm_api_key()
        else:
            self.api_key = open_router_api_key.strip()
    
    def judge_answer(self, question: str, ground_truth: str, model_answer: str) -> Optional[float]:
        """
        Use LLM to judge if the model answer is correct compared to ground truth.
        
        Args:
            question: The question asked
            ground_truth: The correct answer
            model_answer: The model's answer
            
        Returns:
            1.0 if correct, 0.0 if incorrect, or None if the judge call itself failed
        """
        # Prepare the evaluation prompt
        prompt = f"""You are an expert evaluator for question-answering tasks. Your task is to determine if a model's answer is correct compared to the ground truth answer.

Question: {question}

Ground Truth Answer: {ground_truth}

Model Answer: {model_answer}

Evaluate whether the model's answer is correct and contextually similar to the ground truth. Consider:
- If the model answer correctly addresses the question
- If the key information matches the ground truth (even if phrased differently)
- If the answer is factually correct and not contradictory

Respond with ONLY a single number:
- 1 if the answer is correct/contextually similar
- 0 if the answer is incorrect, contradictory, or completely different

Your response (just the number, nothing else):"""

        # Build the message payload
        messages = [
            {
                "role": "user",
                "content": prompt
            }
        ]

        # Parse response
        try:
            content = chat_completion_content(
                model=self.judge_model,
                messages=messages,
                api_key=self.api_key,
                base_url=self.base_url,
                temperature=self.temperature,
                extra_body=self.extra_body,
                stream=self.stream,
                stream_options=self.stream_options,
                timeout=60,
                max_retries=3,
                request_label="QA judge evaluation",
            ).strip()
            
            # Extract score (look for 1 or 0 in the response)
            if not content:
                print(f"[WARN] Empty response from judge model")
                return None
            
            # Try to extract 1 or 0 from response
            content_lower = content.lower()
            if '1' in content or 'correct' in content_lower or 'yes' in content_lower:
                # Check if it's explicitly 0 or incorrect
                if '0' in content or 'incorrect' in content_lower or 'wrong' in content_lower or 'no' in content_lower:
                    # If both present, check which comes first or is more explicit
                    if content.strip() == '0' or content.strip().startswith('0'):
                        return 0.0
                    elif content.strip() == '1' or content.strip().startswith('1'):
                        return 1.0
                    # Default to checking if incorrect keywords are present
                    if any(word in content_lower for word in ['incorrect', 'wrong', 'no', 'different']):
                        return 0.0
                return 1.0
            elif '0' in content or 'incorrect' in content_lower or 'wrong' in content_lower:
                return 0.0
            else:
                # Default: try to parse as float/int
                try:
                    score = float(content.strip())
                    return 1.0 if score >= 0.5 else 0.0
                except:
                    print(f"[WARN] Could not parse judge response: {content}. Will retry on a future run.")
                    return None
                    
        except Exception as e:
            print(f"[ERROR] Failed to parse judge response: {e}")
            return None
    
    def evaluate_answer(self, question: str, ground_truth: str, model_answer: str, task: str = "", qa_type: str = "") -> Dict[str, float]:
        """
        Evaluate a single answer using LLM judge.
        
        Args:
            question: The question asked
            ground_truth: Ground truth answer
            model_answer: Model's answer
            task: Task type (optional, for statistics)
            qa_type: QA type (optional, for statistics)
            
        Returns:
            Dictionary with evaluation score (binary: 1.0 or 0.0)
        """
        # Source QA errors count as incorrect without spending judge tokens.
        if model_answer.startswith('[ERROR:'):
            return {
                'score': 0.0,
                'overall': 0.0,
                'evaluation_status': 'source_error',
            }
        else:
            score = self.judge_answer(question, ground_truth, model_answer)

        if score is None:
            return {
                'score': None,
                'overall': None,
                'evaluation_status': 'judge_error',
            }
        
        return {
            'score': score,
            'overall': score,  # For compatibility with existing code
            'evaluation_status': 'ok',
        }
    
    def evaluate_csv(self, csv_path: str, output_csv: Optional[str] = None, force: bool = False) -> Dict:
        """
        Evaluate all answers in a QA results CSV file.
        
        Args:
            csv_path: Path to the CSV file with QA results
            output_csv: Optional path to stream individual evaluation rows as they finish
            
        Returns:
            Dictionary with evaluation results and statistics
        """
        results = []
        task_stats = defaultdict(lambda: {'total': 0, 'scores': []})
        qa_type_stats = defaultdict(lambda: {'total': 0, 'scores': []})
        fieldnames = [
            'image_id',
            'qa_id',
            'task',
            'qa_type',
            'question',
            'ground_truth',
            'predicted',
            'score',
            'overall',
            'evaluation_status',
        ]

        def result_key(result: Dict):
            return (result.get('image_id', ''), result.get('qa_id', ''), result.get('qa_type', ''))

        def add_result(result: Dict):
            results.append(result)
            task = result.get('task', 'unknown')
            qa_type = result.get('qa_type', 'unknown')
            overall = float(result['overall'])
            task_stats[task]['total'] += 1
            task_stats[task]['scores'].append(overall)
            qa_type_stats[qa_type]['total'] += 1
            qa_type_stats[qa_type]['scores'].append(overall)

        completed_keys = set()
        if output_csv:
            output_dir = os.path.dirname(output_csv)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
            if force and os.path.isfile(output_csv):
                os.remove(output_csv)

            existing_rows = []
            resume_sources = [output_csv]
            legacy_tmp_csv = f"{output_csv}.tmp"
            if os.path.isfile(legacy_tmp_csv):
                resume_sources.append(legacy_tmp_csv)

            for resume_csv in resume_sources:
                if not os.path.isfile(resume_csv):
                    continue
                with open(resume_csv, 'r', newline='', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    for existing in reader:
                        if not existing.get('score') or not existing.get('overall'):
                            continue
                        try:
                            existing['score'] = float(existing['score'])
                            existing['overall'] = float(existing['overall'])
                        except (TypeError, ValueError):
                            continue
                        existing.setdefault('evaluation_status', 'ok')
                        key = result_key(existing)
                        if key in completed_keys:
                            continue
                        completed_keys.add(key)
                        existing_rows.append(existing)
                        add_result(existing)

            # Normalize headers so future appends include evaluation_status.
            with open(output_csv, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for existing in existing_rows:
                    writer.writerow(existing)

            if completed_keys:
                print(f"Resuming judge evaluation from {output_csv} ({len(completed_keys)} completed rows)")
                if os.path.isfile(legacy_tmp_csv):
                    print(f"  Recovered rows from legacy temp file: {legacy_tmp_csv}")
        
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            total_rows = len(rows)
            
            print(f"Evaluating {total_rows} question-answer pairs...")
            
            for idx, row in enumerate(rows, 1):
                question = row.get('question', '').strip()
                predicted = row.get('model_answer', '').strip()
                ground_truth = row.get('ground_truth', '').strip()
                task = row.get('task', 'unknown')
                qa_type = row.get('qa_type', 'unknown')
                qa_id = row.get('qa_id', '')
                image_id = row.get('image_id', '')
                key = (image_id, qa_id, qa_type)
                if key in completed_keys:
                    continue
                
                # Evaluate using LLM judge
                if idx % 10 == 0:
                    print(f"  Progress: {idx}/{total_rows} ({100*idx/total_rows:.1f}%)")
                
                scores = self.evaluate_answer(question, ground_truth, predicted, task, qa_type)
                if scores.get('overall') is None:
                    status = scores.get('evaluation_status', 'judge_error')
                    print(f"[WARN] Skipping {qa_id or idx}: {status}. It will be retried on a future run.")
                    continue
                
                result = {
                    'image_id': image_id,
                    'qa_id': qa_id,
                    'task': task,
                    'qa_type': qa_type,
                    'question': question,
                    'ground_truth': ground_truth,
                    'predicted': predicted,
                    'score': scores['score'],
                    'overall': scores['overall'],
                    'evaluation_status': scores.get('evaluation_status', 'ok')
                }
                add_result(result)
                completed_keys.add(key)

                if output_csv:
                    with open(output_csv, 'a', newline='', encoding='utf-8') as f:
                        writer = csv.DictWriter(f, fieldnames=fieldnames)
                        writer.writerow(result)
                        f.flush()
                

        # Compute aggregate statistics
        if results:
            overall_scores = [r['overall'] for r in results]
            summary = {
                'total_questions': len(results),
                'mean_overall_score': np.mean(overall_scores),
                'std_overall_score': np.std(overall_scores),
                'task_breakdown': {
                    task: {
                        'total': stats['total'],
                        'mean_score': np.mean(stats['scores']) if stats['scores'] else 0.0,
                        'std_score': np.std(stats['scores']) if stats['scores'] else 0.0
                    }
                    for task, stats in task_stats.items()
                },
                'qa_type_breakdown': {
                    qa_type: {
                        'total': stats['total'],
                        'mean_score': np.mean(stats['scores']) if stats['scores'] else 0.0,
                        'std_score': np.std(stats['scores']) if stats['scores'] else 0.0
                    }
                    for qa_type, stats in qa_type_stats.items()
                }
            }
        else:
            summary = {'total_questions': 0}
        
        return {
            'results': results,
            'summary': summary
        }
