from typing import List, Tuple, Dict, Any


HOTPOT_QA_SCHEMA = {
    "type": "object",
    "properties": {
        "answer": {
            "type": "string",
            "description": "The answer to the question based on the provided context"
        },
        "supporting_facts": {
            "type": "array",
            "items": {
                "type": "array",
                "items": {"type": "integer"},
                "minItems": 2,
                "maxItems": 2,
                "description": "[document_index, sentence_index] pairs"
            },
            "description": "List of supporting facts as [doc_idx, sent_idx] pairs"
        },
        "reasoning": {
            "type": "string",
            "description": "Brief explanation of how the answer was derived"
        }
    },
    "required": ["answer", "supporting_facts"],
    "additionalProperties": False
}


class HotpotQAPromptTemplate:
    
    def __init__(self):
        self.base_instruction = """You are an expert at multi-hop question answering. Given a question and context paragraphs, you must:
1. Answer the question accurately using ONLY information from the provided context
2. Identify the specific sentences that support your answer"""
    
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
                prompt += f"  [{doc_idx}-{sent_idx}] {sentence}\n"
        
        prompt += f"""
QUESTION: {question}

INSTRUCTIONS:
1. Read all context carefully
2. Identify which sentences contain information needed to answer the question
3. Formulate a clear, concise answer based ONLY on the context provided
4. List the supporting facts as [document_index, sentence_index] pairs"""
        
        if include_structured_instruction:
            prompt += f"""

You MUST respond with a valid JSON object in this exact format:
{{
    "answer": "Your answer here",
    "supporting_facts": [
        [document_index, sentence_index],
        [document_index, sentence_index]
    ],
    "reasoning": "Brief explanation of how you arrived at the answer"
}}

Remember:
- Use ONLY information from the context provided
- The supporting_facts must reference actual sentences from the context using their indices
- If the question cannot be answered from the context, say so clearly
- Ensure your JSON is properly formatted

RESPONSE:"""
        else:
            prompt += """

Provide your answer followed by the supporting facts as [document_index, sentence_index] pairs.

RESPONSE:"""
        
        return prompt
    
    def _render_concise_template(self, 
                                 question: str, 
                                 context: List[Tuple[str, List[str]]], 
                                 include_structured_instruction: bool) -> str:

        prompt = "Answer the question using only the provided context.\n\nCONTEXT:\n"
        
        for doc_idx, (title, sentences) in enumerate(context):
            prompt += f"\n{doc_idx}: {title}\n"
            for sent_idx, sentence in enumerate(sentences):
                prompt += f"  {doc_idx}.{sent_idx}: {sentence}\n"
        
        prompt += f"\nQUESTION: {question}\n"
        
        if include_structured_instruction:
            prompt += """
Respond in JSON format:
{"answer": "your answer", "supporting_facts": [[doc_idx, sent_idx], ...], "reasoning": "brief explanation"}

RESPONSE:"""
        else:
            prompt += "\nANSWER:"
        
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
                prompt += f"  [{doc_idx}-{sent_idx}] {sentence}\n"
        
        prompt += f"""
QUESTION: {question}

Let me think through this step by step:

1. First, I'll identify what information I need to answer this question
2. Then, I'll find the relevant sentences in the context
3. Finally, I'll formulate my answer based on those sentences

STEP-BY-STEP REASONING:"""
        
        if include_structured_instruction:
            prompt += """

After your reasoning, provide the final answer in JSON format:
{
    "answer": "your final answer",
    "supporting_facts": [[doc_idx, sent_idx], ...],
    "reasoning": "summary of your reasoning above"
}

RESPONSE:"""
        else:
            prompt += """

After your reasoning, provide your final answer and the supporting sentence references.

RESPONSE:"""
        
        return prompt
    
    def get_schema(self) -> Dict[str, Any]:
        return HOTPOT_QA_SCHEMA.copy()
    
    def get_example_output(self) -> Dict[str, Any]:
        return {
            "answer": "Kinnairdy Castle",
            "supporting_facts": [[0, 1], [1, 0]],
            "reasoning": "Document 0 mentions that David Gregory inherited Kinnairdy Castle in 1664, and Document 1 confirms that Kinnairdy Castle is a tower house in Scotland."
        }


def create_hotpot_prompt_template(style: str = "detailed") -> HotpotQAPromptTemplate:
    template = HotpotQAPromptTemplate()
    return template


def render_hotpot_prompt(question: str, 
                         context: List[Tuple[str, List[str]]], 
                         structured_output: bool = True,
                         style: str = "detailed") -> str:
                         
    template = create_hotpot_prompt_template(style)
    return template.render(
        question=question,
        context=context,
        include_structured_instruction=structured_output,
        template_style=style
    )