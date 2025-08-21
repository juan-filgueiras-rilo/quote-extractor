from typing import List, Tuple, Dict, Any


QUOTE_HOTPOT_QA_SCHEMA = {
    "type": "object",
    "properties": {
        "answer": {
            "type": "string",
            "description": "The answer to the question based on the provided context"
        },
        "supporting_quotes": {
            "type": "array",
            "items": {
                "type": "string",
                "description": "Exact sentence from the context that supports the answer"
            },
            "description": "List of exact quotes from the context that support the answer"
        },
        "reasoning": {
            "type": "string",
            "description": "Brief explanation of how the answer was derived"
        }
    },
    "required": ["answer", "supporting_quotes"],
    "additionalProperties": False
}


class QuoteHotpotQAPromptTemplate:
    
    def __init__(self):
        self.base_instruction = """You are an expert at multi-hop question answering. Given a question and context paragraphs, you must:
1. Answer the question accurately using ONLY information from the provided context
2. Identify the exact sentences that support your answer by quoting them verbatim"""
    
    def render(self, 
               question: str, 
               context: List[Tuple[str, List[str]]], 
               include_structured_instruction: bool = True,
               template_style: str = "detailed") -> str:
        if template_style == "detailed":
            return self._render_detailed_template(question, context, include_structured_instruction)
        elif template_style == "concise":
            return self._render_concise_template(question, context, include_structured_instruction)
        elif template_style == "chain_of_thought":
            return self._render_cot_template(question, context, include_structured_instruction)
        else:
            raise ValueError(f"Unknown template style: {template_style}")
    
    def _render_detailed_template(self, 
                                  question: str, 
                                  context: List[Tuple[str, List[str]]], 
                                  include_structured_instruction: bool) -> str:
        prompt = f"""{self.base_instruction}

CONTEXT:
"""
        for doc_idx, (title, sentences) in enumerate(context):
            prompt += f"\nDocument {doc_idx}: {title}\n"
            for sent_idx, sentence in enumerate(sentences):
                prompt += f"  {sentence}\n"
        
        prompt += f"""
QUESTION: {question}

INSTRUCTIONS:
1. Read all context carefully
2. Identify which sentences contain information needed to answer the question
3. Formulate a clear, concise answer based ONLY on the context provided
4. Quote the exact sentences that support your answer (copy them word-for-word)"""
        
        if include_structured_instruction:
            prompt += f"""

You MUST respond with a valid JSON object in this exact format:
{{
    "answer": "Your answer here",
    "supporting_quotes": [
        "Exact sentence from context that supports the answer",
        "Another exact sentence from context that supports the answer"
    ],
    "reasoning": "Brief explanation of how you arrived at the answer"
}}

CRITICAL REQUIREMENTS:
- Use ONLY information from the context provided
- The supporting_quotes MUST be EXACT copies of sentences from the context (word-for-word)
- Do NOT paraphrase or modify the quotes in any way
- If the question cannot be answered from the context, say so clearly
- Ensure your JSON is properly formatted

RESPONSE:"""
        else:
            prompt += """

Provide your answer followed by exact quotes from the context that support your answer.

CRITICAL: The quotes must be EXACT copies of sentences from the context.

RESPONSE:"""
        
        return prompt
    
    def _render_concise_template(self, 
                                 question: str, 
                                 context: List[Tuple[str, List[str]]], 
                                 include_structured_instruction: bool) -> str:

        prompt = "Answer the question using only the provided context. Quote exact sentences that support your answer.\n\nCONTEXT:\n"
        
        for doc_idx, (title, sentences) in enumerate(context):
            prompt += f"\n{doc_idx}: {title}\n"
            for sent_idx, sentence in enumerate(sentences):
                prompt += f"  {sentence}\n"
        
        prompt += f"\nQUESTION: {question}\n"
        
        if include_structured_instruction:
            prompt += """
Respond in JSON format with EXACT quotes from the context:
{"answer": "your answer", "supporting_quotes": ["exact sentence 1", "exact sentence 2"], "reasoning": "brief explanation"}

IMPORTANT: supporting_quotes must be word-for-word copies from the context.

RESPONSE:"""
        else:
            prompt += "\nProvide your answer and quote the exact supporting sentences.\n\nANSWER:"
        
        return prompt
    
    def _render_cot_template(self, 
                           question: str, 
                           context: List[Tuple[str, List[str]]], 
                           include_structured_instruction: bool) -> str:
        
        prompt = f"""{self.base_instruction}

Let's work through this step by step.

CONTEXT:
"""
        
        for doc_idx, (title, sentences) in enumerate(context):
            prompt += f"\nDocument {doc_idx}: {title}\n"
            for sent_idx, sentence in enumerate(sentences):
                prompt += f"  {sentence}\n"
        
        prompt += f"""
QUESTION: {question}

Let me think through this step by step:

1. First, I'll identify what information I need to answer this question
2. Then, I'll find the relevant sentences in the context
3. Finally, I'll formulate my answer and quote the exact supporting sentences

STEP-BY-STEP REASONING:"""
        
        if include_structured_instruction:
            prompt += """

After your reasoning, provide the final answer in JSON format with EXACT quotes:
{
    "answer": "your final answer",
    "supporting_quotes": ["exact sentence 1", "exact sentence 2"],
    "reasoning": "summary of your reasoning above"
}

CRITICAL: supporting_quotes must be word-for-word copies from the context.

RESPONSE:"""
        else:
            prompt += """

After your reasoning, provide your final answer and quote the exact supporting sentences from the context.

RESPONSE:"""
        
        return prompt
    
    def get_schema(self) -> Dict[str, Any]:
        return QUOTE_HOTPOT_QA_SCHEMA.copy()
    
    def get_example_output(self) -> Dict[str, Any]:
        return {
            "answer": "Kinnairdy Castle",
            "supporting_quotes": [
                "David Gregory inherited Kinnairdy Castle in 1664.",
                "Kinnairdy Castle is a tower house, located 0.5 mi south of Aberchirder, Aberdeenshire, Scotland."
            ],
            "reasoning": "The context mentions that David Gregory inherited Kinnairdy Castle, and another sentence confirms that Kinnairdy Castle is a tower house in Scotland."
        }


def create_quote_hotpot_prompt_template(style: str = "detailed") -> QuoteHotpotQAPromptTemplate:
    template = QuoteHotpotQAPromptTemplate()
    return template


def render_quote_hotpot_prompt(question: str, 
                              context: List[Tuple[str, List[str]]], 
                              structured_output: bool = True,
                              style: str = "detailed") -> str:
                         
    template = create_quote_hotpot_prompt_template(style)
    return template.render(
        question=question,
        context=context,
        include_structured_instruction=structured_output,
        template_style=style
    )