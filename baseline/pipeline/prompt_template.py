"""
prompt_template.py

Author: Lijo Raju
Purpose: Prompt template formatter for EduRAG LLM inference.
"""
import random
from typing import List


def format_prompt(question: str, context: str) -> str:
    """
    Format prompt string with question and retrieved context.

    Args:
        question (str): User's query.
        context (str): Retrieved document context (top-k concatenated)

    Returns:
        str: Full prompt string for the model.
    """
    prompt = f"""### Instruction:
Answer the question based on the provided context.

### Context:
{context}

### Question:
{question}

### Response:"""
    return prompt


class EducationalPromptGenerator:
    """
    Generates diverse educational prompts based on Bloom's Taxonomy
    and social science pedagogical principles.
    """
    
    def __init__(self):
        self.question_types = {
            'factual': {
                'starters': ['What', 'When', 'Where', 'Who', 'Which'],
                'templates': [
                    "What {concept} is mentioned in the passage?",
                    "When did {event} occur according to the text?",
                    "Who was responsible for {action}?",
                    "Which {element} is described in the passage?"
                ]
            },
            'comprehension': {
                'starters': ['How', 'Why', 'What was the purpose', 'What does', 'Explain'],
                'templates': [
                    "How did {concept} influence {outcome}?",
                    "Why was {event} significant?",
                    "What was the purpose of {action}?",
                    "Explain the meaning of {concept} in the context."
                ]
            },
            'analytical': {
                'starters': ['Analyze', 'Compare', 'What caused', 'What were the effects', 'How did'],
                'templates': [
                    "What caused {event} to happen?",
                    "What were the effects of {action}?",
                    "How did {factor} contribute to {outcome}?",
                    "What is the relationship between {concept1} and {concept2}?"
                ]
            },
            'evaluative': {
                'starters': ['Why do you think', 'What impact did', 'How significant', 'To what extent'],
                'templates': [
                    "What impact did {event} have on {society/group}?",
                    "How significant was {person/event} in {context}?",
                    "Why was {concept} important for {outcome}?",
                    "What role did {factor} play in {development}?"
                ]
            }
        }
        
        self.social_science_keywords = [
            'revolution', 'nationalism', 'democracy', 'republic', 'constitution',
            'sovereignty', 'liberty', 'citizenship', 'political', 'social',
            'economic', 'cultural', 'reform', 'movement', 'ideology',
            'government', 'society', 'institution', 'power', 'authority'
        ]

    def create_diverse_prompt(self, chunk: str, num_pairs: int = 2) -> str:
        """
        Create a prompt that encourages diverse question types.

        Args:
            chunks (str): Context for QA to embed in the prompt.
            num_pairs (int): Number of QA pairs to embed in the prompt.
        
        Returns:
            str: Prompt generated
        """
        # Select diverse question types for this chunk
        selected_types = self._select_question_types(num_pairs)
        type_descriptions = self._get_type_descriptions(selected_types)
        
        prompt = f"""<|system|>
You are an expert social science educator creating diverse questions for high school students studying history. Generate exactly {num_pairs} question-answer pairs that test different cognitive skills.

QUESTION TYPES TO GENERATE:
{type_descriptions}

QUALITY GUIDELINES:
- Questions must be directly answerable from the provided text
- Use varied question starters (avoid repetitive "What is..." patterns)  
- Focus on key historical concepts: nationalism, political change, social movements, economies and societies, culture and politics, industrialisation
- Answers should be 2-3 sentences with specific details from the text
- Maintain academic vocabulary appropriate for high school level

<|user|>
Based on this passage about history, create {num_pairs} diverse question-answer pairs:

PASSAGE: {chunk}

Return your response as a valid JSON array:
[{{"question": "Your question here", "answer": "Your detailed answer here"}}]

Remember: Generate {len(selected_types)} different types of questions as specified above.
<|assistant|>
"""
        return prompt

    def create_topic_focused_prompt(self, chunk: str, num_pairs: int = 2) -> str:
        """
        Create a prompt focused on identifying and exploring key topics.
        """
        prompt = f"""<|system|>
You are an expert social science educator. Create {num_pairs} questions that explore different aspects of the key topics in this passage. Focus on the most important historical concepts and their significance.

<|user|>
Analyze this passage and create {num_pairs} questions that cover:
1. KEY CONCEPTS: Main ideas, definitions, or principles
2. HISTORICAL SIGNIFICANCE: Why events/people/ideas matter
3. CAUSE AND EFFECT: How events influenced each other
4. CONNECTIONS: Links between ideas, people, or events

PASSAGE: {chunk}

Generate questions that use these varied approaches:
- Factual understanding: "What was..." "Who were..." "When did..."
- Conceptual analysis: "How did..." "Why was... significant..." "What role did..."
- Critical evaluation: "What impact did..." "How effectively did..." "To what extent..."

Return as JSON: [{{"question": "...", "answer": "..."}}]
<|assistant|>
"""
        return prompt

    def create_bloom_taxonomy_prompt(self, chunk: str, num_pairs: int = 2) -> str:
        """
        Create a prompt explicitly based on Bloom's Taxonomy levels.
        """
        bloom_levels = [
            "REMEMBER (recall facts): Ask about specific dates, names, events, or definitions",
            "UNDERSTAND (explain concepts): Ask students to explain meanings, describe processes, or summarize ideas", 
            "APPLY (use knowledge): Ask how concepts apply to situations or connect to other contexts",
            "ANALYZE (examine relationships): Ask about causes, effects, comparisons, or component parts",
            "EVALUATE (make judgments): Ask about significance, importance, effectiveness, or impact"
        ]
        
        selected_levels = random.sample(bloom_levels, min(num_pairs, len(bloom_levels)))
        
        prompt = f"""<|system|>
You are an educational expert creating questions at different cognitive levels. Generate {num_pairs} questions covering these Bloom's Taxonomy levels:

{chr(10).join(f"{i+1}. {level}" for i, level in enumerate(selected_levels))}

<|user|>
Create {num_pairs} questions from this passage, with each question targeting a different cognitive level as specified above:

PASSAGE: {chunk}

REQUIREMENTS:
- Each question should clearly target its assigned Bloom's level
- Questions must be answerable from the passage content
- Use appropriate question starters for each level
- Provide comprehensive answers with evidence from the text

Return as JSON array: [{{"question": "...", "answer": "..."}}]
<|assistant|>
"""
        return prompt

    def create_scenario_based_prompt(self, chunk: str, num_pairs: int = 2) -> str:
        """
        Create prompts that encourage scenario-based and application questions.
        """
        prompt = f"""<|system|>
You are creating educational questions that help students think critically about historical events and their broader implications. Generate {num_pairs} questions that encourage deeper analysis.

<|user|>
Based on this historical passage, create {num_pairs} questions that explore:

TYPE 1 - CAUSE & CONSEQUENCE: "What led to..." "What were the results of..." "How did X influence Y?"
TYPE 2 - HISTORICAL THINKING: "Why was this significant?" "How does this connect to..." "What does this reveal about..."

PASSAGE: {chunk}

Create questions that:
- Go beyond simple factual recall
- Encourage students to think about WHY things happened
- Connect events to broader historical patterns
- Ask students to evaluate importance and significance

Return as JSON: [{{"question": "...", "answer": "..."}}]
<|assistant|>
"""
        return prompt

    def _select_question_types(self, num_pairs: int) -> List[str]:
        """Select diverse question types for the given number of pairs."""
        if num_pairs <= 2:
            return random.sample(['factual', 'analytical'], min(num_pairs, 2))
        elif num_pairs == 3:
            return ['factual', 'comprehension', 'analytical']
        else:
            # For 4+ questions, ensure good variety
            base_types = ['factual', 'comprehension', 'analytical', 'evaluative']
            selected = base_types[:num_pairs]
            if num_pairs > 4:
                # Add more variety by repeating some types
                additional = random.choices(base_types, k=num_pairs-4)
                selected.extend(additional)
            return selected

    def _get_type_descriptions(self, selected_types: List[str]) -> str:
        """Generate descriptions for selected question types."""
        descriptions = []
        for i, qtype in enumerate(selected_types, 1):
            if qtype == 'factual':
                descriptions.append(f"{i}. FACTUAL RECALL: Ask for specific facts, dates, names, or definitions from the text")
            elif qtype == 'comprehension':
                descriptions.append(f"{i}. COMPREHENSION: Ask students to explain concepts, describe processes, or summarize ideas")
            elif qtype == 'analytical':
                descriptions.append(f"{i}. ANALYTICAL: Ask about causes, effects, relationships, or comparisons between concepts")
            elif qtype == 'evaluative':
                descriptions.append(f"{i}. EVALUATIVE: Ask about significance, impact, importance, or effectiveness of events/ideas")
        
        return '\n'.join(descriptions)