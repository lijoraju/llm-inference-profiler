"""
llm_eval_ollama.py

Author: Lijo Raju
Purpose: LLM based evaluation for QA dataset quality using Ollama.
"""
import json
import logging
from typing import List, Dict, Tuple
import requests
import re
import random
import statistics
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class QAEvaluator:
    """
    Evaluates generated QA pairs using multiple criteria and Ollama models
    """
    
    def __init__(self, model_name: str = "llama3.1:70b", ollama_url: str = "http://localhost:11434"):
        """
        Initialize the evaluator with a specified Ollama model
        
        Args:
            model_name: Ollama model name for evaluation (e.g., "llama3.1:70b", "mistral", "qwen2.5:72b")
            ollama_url: Ollama server URL
        """
        self.model_name = model_name
        self.ollama_url = ollama_url
        self.api_url = f"{ollama_url}/api/generate"
        self._check_ollama_connection()
        
    def _check_ollama_connection(self):
        """Check if Ollama server is running and model is available"""
        try:
            response = requests.get(f"{self.ollama_url}/api/tags", timeout=10)
            if response.status_code == 200:
                models = response.json()
                available_models = [model['name'] for model in models.get('models', [])]
                
                if self.model_name not in available_models:
                    logger.warning(f"Model {self.model_name} not found in available models: {available_models}")
                    logger.info("You may need to pull the model first: ollama pull {self.model_name}")
                else:
                    logger.info(f"Successfully connected to Ollama with model: {self.model_name}")
            else:
                raise Exception(f"Ollama server responded with status {response.status_code}")
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to connect to Ollama server at {self.ollama_url}: {e}")
            logger.info("Make sure Ollama is running: ollama serve")
            raise

    def _generate_response(self, prompt: str) -> str:
        """
        Generate response using Ollama API
        
        Args:
            prompt: Input prompt for the model
            
        Returns:
            Generated response text
        """
        try:
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.1,  # Low temperature for consistent evaluation
                    "top_p": 0.9,
                    "num_predict": 256,  # Maximum tokens to generate
                    "stop": ["<|user|>", "<|system|>"]  # Stop sequences
                }
            }
            
            response = requests.post(
                self.api_url,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=120  # Increased timeout for large models
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get("response", "")
            else:
                logger.error(f"Ollama API error: {response.status_code} - {response.text}")
                return ""
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Error calling Ollama API: {e}")
            return ""

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
            return f"""You are an expert educational evaluator assessing QA pairs for high school social science students. Evaluate the following question-answer pair across multiple dimensions.

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

Response:"""
        
        elif evaluation_type == "accuracy":
            return f"""You are a fact-checker evaluating the accuracy of educational content.

CONTEXT: {context}

QUESTION: {question}

ANSWER: {answer}

Is this answer factually accurate based on the provided context? 

Respond in JSON format:
{{
    "is_accurate": true or false,
    "accuracy_score": [1-10],
    "explanation": "Brief explanation of your assessment",
    "errors_found": ["list of any factual errors"]
}}

Response:"""
        
        elif evaluation_type == "difficulty":
            return f"""You are an educational expert assessing question difficulty for high school students.

CONTEXT: {context}

QUESTION: {question}

ANSWER: {answer}

Assess the cognitive difficulty of this question for high school social science students.

Respond in JSON format:
{{
    "difficulty_level": "basic" or "intermediate" or "advanced" or "expert",
    "bloom_taxonomy_level": "remember" or "understand" or "apply" or "analyze" or "evaluate" or "create",
    "difficulty_score": [1-10, where 1=too easy, 5-6=appropriate, 10=too hard],
    "explanation": "Why you assigned this difficulty level",
    "suggestions": "How to adjust difficulty if needed"
}}

Response:"""
        
        elif evaluation_type == "clarity":
            return f"""You are a language expert evaluating question clarity and answer quality.

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

Response:"""

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
            response_text = self._generate_response(prompt)
            
            if response_text:
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
            json_patterns = [
                r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}',  # Nested JSON pattern
                r'\{.*?\}',  # Simple JSON pattern
            ]
            
            json_match = None
            for pattern in json_patterns:
                json_match = re.search(pattern, text, re.DOTALL)
                if json_match:
                    break
            
            if json_match:
                json_str = json_match.group(0)
                json_str = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', json_str)
                # Replace any single quotes with double quotes for JSON compliance
                json_str = re.sub(r"'([^']*)':", r'"\1":', json_str)
                json_str = re.sub(r":\s*'([^']*)'", r': "\1"', json_str)
                
                evaluation = json.loads(json_str)
                evaluation['evaluation_type'] = evaluation_type
                return evaluation
            else:
                return self._extract_fallback_evaluation(text, evaluation_type)
                
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse evaluation JSON: {e}")
            return self._extract_fallback_evaluation(text, evaluation_type)

    def _extract_fallback_evaluation(self, text: str, evaluation_type: str) -> Dict:
        """
        Fallback method to extract evaluation when JSON parsing fails
        """
        evaluation = {"evaluation_type": evaluation_type, "parsing_method": "fallback"}
        
        if evaluation_type == "comprehensive":
            score_patterns = {
                "factual_accuracy": r"factual[_\s]*accuracy[\":\s]*(\d+)",
                "completeness": r"completeness[\":\s]*(\d+)",
                "clarity": r"clarity[\":\s]*(\d+)",
                "educational_value": r"educational[_\s]*value[\":\s]*(\d+)",
                "appropriate_difficulty": r"appropriate[_\s]*difficulty[\":\s]*(\d+)",
                "context_alignment": r"context[_\s]*alignment[\":\s]*(\d+)"
            }
            
            scores = []
            for key, pattern in score_patterns.items():
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    score = int(match.group(1))
                    evaluation[key] = score
                    scores.append(score)
            
            if scores:
                evaluation["overall_score"] = sum(scores) / len(scores)
            
            # Try to extract recommendation
            if "keep" in text.lower():
                evaluation["recommendation"] = "keep"
            elif "revise" in text.lower():
                evaluation["recommendation"] = "revise"
            elif "discard" in text.lower():
                evaluation["recommendation"] = "discard"
            
            evaluation["feedback"] = "Extracted using fallback parsing"
        
        evaluation["raw_text"] = text
        return evaluation

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
        
        # Collect scores for different evaluation types
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
                'recommendations': results['recommendations'],
                'model_used': self.model_name
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


def run_llm_evaluation(path: str, model: str = "llama3.1:70b", ollama_url: str = "http://localhost:11434"):
    """
    Main function to run LLM based QA evaluation using Ollama

    Args:
        path (str): Path to the QA JSON file 
        model (str): Ollama model to use for evaluation
        ollama_url (str): Ollama server URL
    """
    evaluator = QAEvaluator(model, ollama_url)
    
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