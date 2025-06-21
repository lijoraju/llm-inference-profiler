"""
prompt_template.py

Author: Lijo Raju
Purpose: Prompt template formatter for EduRAG LLM inference.
"""


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

