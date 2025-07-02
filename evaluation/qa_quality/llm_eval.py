"""
llm_eval.py

Author: Lijo Raju
Purpose: LLM based evaluation for QA dataset quality.
"""
import json
import logging
from typing import List, Dict, Tuple
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from tqdm import tqdm
import statistics
import re
import random

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class QAEvaluator:
    """
    Evaluates generated QA pairs using multiple criteria and open-source LLMs
    """
    
    def __init__(self, model_name: str = "meta-llama/Llama-3.1-70B-Instruct"):
        """
        Initialize the evaluator with a specified model
        
        Args:
            model_name: HuggingFace model name for evaluation
        """
        self.model_name = model_name
        self.generator = None
        self.tokenizer = None
        self._init_model()
        
    def _init_model(self):
        """Initialize the evaluation model"""
        try:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True
            )

            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                device_map="auto",
                quantization_config=quantization_config,
                trust_remote_code=True
            )

            self.generator = pipeline(
                "text-generation",
                model=model,
                tokenizer=self.tokenizer,
                device_map="auto",
                torch_dtype=torch.float16,
                max_new_tokens=512,
                do_sample=False,  # Deterministic for evaluation
                temperature=0.1,   # Low temperature for consistent evaluation
                pad_token_id=self.tokenizer.pad_token_id
            )
            logger.info(f"Successfully initialized evaluation model: {self.model_name}")
            
        except Exception as e:
            logger.error(f"Failed to initialize evaluation model: {e}")
            raise

    def create_evaluation_prompt(self, qa_pair: Dict[str, str], evaluation_type: str = "comprehensive") -> str:
        """
        Create evaluation prompt for different evaluation types
        
        Args:
            qa_pair: Dictionary with 'context', 'question', 'answer'
            evaluation_type: Type of evaluation ('comprehensive', 'accuracy', 'difficulty', 'clarity')
        """
        context = qa_pair.get('context', '')
        question = qa_pair.get('question', '')
        answer = qa_pair.get('answer', '')
        
        if evaluation_type == "comprehensive":
            return f"""<|system|>
You are an expert educational evaluator assessing QA pairs for high school social science students. Evaluate the following question-answer pair across multiple dimensions.

<|user|>
CONTEXT: {context}

QUESTION: {question}

ANSWER: {answer}

Evaluate this QA pair on a scale of 1-10 (where 10 is excellent) across these dimensions:

1. FACTUAL ACCURACY: Is the answer factually correct based on the context?
2. COMPLETENESS: Does the answer fully address the question?
3. CLARITY: Is the question clear and unambiguous?
4. EDUCATIONAL VALUE: Does this QA pair help students learn key concepts?
5. APPROPRIATE DIFFICULTY: Is this suitable for high school level?
6. CONTEXT ALIGNMENT: Is the answer well-supported by the given context?

Provide your evaluation in this exact JSON format:
{{
    "factual_accuracy": [score 1-10],
    "completeness": [score 1-10],
    "clarity": [score 1-10],
    "educational_value": [score 1-10],
    "appropriate_difficulty": [score 1-10],
    "context_alignment": [score 1-10],
    "overall_score": [average score],
    "feedback": "Brief explanation of strengths and areas for improvement",
    "recommendation": "keep" or "revise" or "discard"
}}

<|assistant|>
"""
        
        elif evaluation_type == "accuracy":
            return f"""<|system|>
You are a fact-checker evaluating the accuracy of educational content.

<|user|>
CONTEXT: {context}

QUESTION: {question}

ANSWER: {answer}

Is this answer factually accurate based on the provided context? 

Respond in JSON format:
{{
    "is_accurate": true/false,
    "accuracy_score": [1-10],
    "explanation": "Brief explanation of your assessment",
    "errors_found": ["list of any factual errors"]
}}

<|assistant|>
"""
        
        elif evaluation_type == "difficulty":
            return f"""<|system|>
You are an educational expert assessing question difficulty for high school students.

<|user|>
CONTEXT: {context}

QUESTION: {question}

ANSWER: {answer}

Assess the cognitive difficulty of this question for high school social science students.

Respond in JSON format:
{{
    "difficulty_level": "basic|intermediate|advanced|expert",
    "bloom_taxonomy_level": "remember|understand|apply|analyze|evaluate|create",
    "difficulty_score": [1-10, where 1=too easy, 5-6=appropriate, 10=too hard],
    "explanation": "Why you assigned this difficulty level",
    "suggestions": "How to adjust difficulty if needed"
}}

<|assistant|>
"""
        
        elif evaluation_type == "clarity":
            return f"""<|system|>
You are a language expert evaluating question clarity and answer quality.

<|user|>
QUESTION: {question}

ANSWER: {answer}

Evaluate the clarity and quality of this question-answer pair.

Respond in JSON format:
{{
    "question_clarity": [1-10],
    "answer_clarity": [1-10],
    "language_quality": [1-10],
    "issues_found": ["list of any language or clarity issues"],
    "improvements": ["suggested improvements"]
}}

<|assistant|>
"""

    def evaluate_qa_pair(self, qa_pair: Dict[str, str], evaluation_type: str = "comprehensive") -> Dict:
        """
        Evaluate a single QA pair
        
        Args:
            qa_pair: QA pair to evaluate
            evaluation_type: Type of evaluation to perform
            
        Returns:
            Dictionary with evaluation results
        """
        try:
            prompt = self.create_evaluation_prompt(qa_pair, evaluation_type)
            
            response = self.generator(
                prompt,
                max_new_tokens=512,
                return_full_text=False,
                truncation=True
            )
            
            if response and len(response) > 0:
                response_text = response[0]["generated_text"]
                return self._extract_evaluation_json(response_text, evaluation_type)
            else:
                return {"error": "No response generated"}
                
        except Exception as e:
            logger.error(f"Error evaluating QA pair: {e}")
            return {"error": str(e)}

    def _extract_evaluation_json(self, text: str, evaluation_type: str) -> Dict:
        """
        Extract JSON evaluation from generated text
        """
        try:
            # Find JSON in the response
            json_match = re.search(r'\{.*\}', text, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                # Clean up the JSON string
                json_str = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', json_str)
                evaluation = json.loads(json_str)
                evaluation['evaluation_type'] = evaluation_type
                return evaluation
            else:
                return {"error": "No JSON found in response", "raw_text": text}
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse evaluation JSON: {e}")
            return {"error": f"JSON parsing failed: {e}", "raw_text": text}

    def evaluate_dataset(self, qa_pairs: List[Dict[str, str]], 
                        evaluation_types: List[str] = ["comprehensive"],
                        sample_size: int = None) -> Dict:
        """
        Evaluate a dataset of QA pairs
        
        Args:
            qa_pairs: List of QA pairs to evaluate
            evaluation_types: Types of evaluations to perform
            sample_size: If specified, randomly sample this many pairs
            
        Returns:
            Comprehensive evaluation results
        """
        if sample_size and sample_size < len(qa_pairs):
            qa_pairs = random.sample(qa_pairs, sample_size)
            logger.info(f"Evaluating random sample of {sample_size} QA pairs")
        
        results = {
            'total_pairs': len(qa_pairs),
            'evaluation_types': evaluation_types,
            'individual_results': [],
            'summary_statistics': {},
            'recommendations': {
                'keep': 0,
                'revise': 0,
                'discard': 0
            }
        }
        
        for i, qa_pair in enumerate(tqdm(qa_pairs, desc="Evaluating QA pairs")):
            pair_results = {'pair_index': i, 'evaluations': {}}
            
            for eval_type in evaluation_types:
                evaluation = self.evaluate_qa_pair(qa_pair, eval_type)
                pair_results['evaluations'][eval_type] = evaluation
                
                # Track recommendations for comprehensive evaluations
                if eval_type == "comprehensive" and 'recommendation' in evaluation:
                    rec = evaluation['recommendation']
                    if rec in results['recommendations']:
                        results['recommendations'][rec] += 1
            
            results['individual_results'].append(pair_results)
        
        # Calculate summary statistics
        results['summary_statistics'] = self._calculate_summary_stats(results['individual_results'])
        
        return results

    def _calculate_summary_stats(self, individual_results: List[Dict]) -> Dict:
        """Calculate summary statistics from individual evaluation results"""
        stats = {}
        
        # Collect scores for comprehensive evaluations
        comprehensive_scores = []
        accuracy_scores = []
        difficulty_scores = []
        clarity_scores = []
        
        for result in individual_results:
            evaluations = result.get('evaluations', {})
            
            if 'comprehensive' in evaluations:
                comp_eval = evaluations['comprehensive']
                if 'overall_score' in comp_eval and isinstance(comp_eval['overall_score'], (int, float)):
                    comprehensive_scores.append(comp_eval['overall_score'])
            
            if 'accuracy' in evaluations:
                acc_eval = evaluations['accuracy']
                if 'accuracy_score' in acc_eval and isinstance(acc_eval['accuracy_score'], (int, float)):
                    accuracy_scores.append(acc_eval['accuracy_score'])
            
            if 'difficulty' in evaluations:
                diff_eval = evaluations['difficulty']
                if 'difficulty_score' in diff_eval and isinstance(diff_eval['difficulty_score'], (int, float)):
                    difficulty_scores.append(diff_eval['difficulty_score'])
        
        # Calculate statistics for each score type
        for score_type, scores in [
            ('comprehensive', comprehensive_scores),
            ('accuracy', accuracy_scores),
            ('difficulty', difficulty_scores)
        ]:
            if scores:
                stats[f'{score_type}_mean'] = statistics.mean(scores)
                stats[f'{score_type}_median'] = statistics.median(scores)
                stats[f'{score_type}_stdev'] = statistics.stdev(scores) if len(scores) > 1 else 0
                stats[f'{score_type}_min'] = min(scores)
                stats[f'{score_type}_max'] = max(scores)
                stats[f'{score_type}_count'] = len(scores)
        
        return stats

    def generate_evaluation_report(self, results: Dict, output_path: str = None):
        """
        Generate a comprehensive evaluation report
        
        Args:
            results: Results from evaluate_dataset
            output_path: Optional path to save the report
        """
        report = {
            'evaluation_summary': {
                'total_pairs_evaluated': results['total_pairs'],
                'evaluation_types': results['evaluation_types'],
                'summary_statistics': results['summary_statistics'],
                'recommendations': results['recommendations']
            },
            'detailed_results': results['individual_results']
        }
        
        # Add quality insights
        stats = results['summary_statistics']
        insights = []
        
        if 'comprehensive_mean' in stats:
            mean_score = stats['comprehensive_mean']
            if mean_score >= 8:
                insights.append("Overall quality is excellent")
            elif mean_score >= 6:
                insights.append("Overall quality is good with room for improvement")
            else:
                insights.append("Overall quality needs significant improvement")
        
        if 'accuracy_mean' in stats:
            acc_mean = stats['accuracy_mean']
            if acc_mean < 7:
                insights.append("Factual accuracy concerns - review generation process")
        
        report['evaluation_summary']['quality_insights'] = insights
        
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            logger.info(f"Evaluation report saved to {output_path}")
        
        return report


def run_llm_evalaution(path: str, model: str = "meta-llama/Llama-3.1-70B-Instruct"):
    """
    Main function to run LLM based QA evaluation

    Args:
        path (str): Path the QA JSON file 
        model (str): LLM model to use for evaluation
    """
    evaluator = QAEvaluator(model)
    
    with open(path, "r", encoding='utf-8') as f:
        qa_pairs = json.load(f)
    
    results = evaluator.evaluate_dataset(
        qa_pairs,
        evaluation_types=["comprehensive", "accuracy", "difficulty"]
    )
    
    report = evaluator.generate_evaluation_report(results, "evaluation_report.json")
    
    logger.info(f"Evaluated {results['total_pairs']} QA pairs")
    logger.info(f"Recommendations: {results['recommendations']}")
    if 'comprehensive_mean' in results['summary_statistics']:
        logger.info(f"Average quality score: {results['summary_statistics']['comprehensive_mean']:.2f}")