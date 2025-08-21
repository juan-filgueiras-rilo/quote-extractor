import logging
from typing import List, Tuple, Dict, Any

from src.core.model import QAResponse, SupportingFact
from src.providers.base import LLMProvider
from src.templates.hotpot_qa import HotpotQAPromptTemplate, HOTPOT_QA_SCHEMA

logger = logging.getLogger(__name__)


class HotpotQAProcessor:
    
    def __init__(self, 
                 llm_provider: LLMProvider, 
                 prompt_style: str = "detailed",
                 use_structured_output: bool = None):

        self.llm_provider = llm_provider
        self.prompt_template = HotpotQAPromptTemplate()
        self.prompt_style = prompt_style
        
        # Auto-detect structured output capability if not specified
        if use_structured_output is None:
            self.use_structured_output = llm_provider.supports_structured_output()
        else:
            self.use_structured_output = use_structured_output
        
        logger.info(f"Initialized HotpotQA processor with:")
        logger.info(f"  - Provider: {type(llm_provider).__name__}")
        logger.info(f"  - Prompt style: {prompt_style}")
        logger.info(f"  - Structured output: {self.use_structured_output}")
    
    def create_prompt(self, question: str, context: List[Tuple[str, List[str]]]) -> str:
        return self.prompt_template.render(
            question=question,
            context=context,
            include_structured_instruction=self.use_structured_output,
            template_style=self.prompt_style
        )
    
    def process_question(self, question_data: Dict[str, Any]) -> QAResponse:
        question = question_data['question']
        context = question_data['context']
        
        prompt = self.create_prompt(question, context)
        
        logger.info(f"Processing question: {question[:100]}...")
        raw_response = self.llm_provider.generate(prompt)
        
        parsed = self._parse_hotpot_response(raw_response)
        
        supporting_facts = self._extract_supporting_facts(parsed, context)
        
        return QAResponse(
            answer=parsed.get('answer', 'No answer found'),
            supporting_facts=supporting_facts,
            raw_response=raw_response
        )
    
    def _parse_hotpot_response(self, response: str) -> Dict[str, Any]:
        if self.use_structured_output:
            try:
                parsed = self.llm_provider.parse_structured_output(
                    response=response,
                    schema=HOTPOT_QA_SCHEMA,
                    format_type="json"
                )
                
                self._validate_hotpot_structure(parsed)
                return parsed
                
            except Exception as e:
                logger.warning(f"Structured parsing failed: {e}")
                # Fall back to manual parsing
                return self._manual_parse_hotpot_response(response)
        else:
            return self._manual_parse_hotpot_response(response)
    
    def _manual_parse_hotpot_response(self, response: str) -> Dict[str, Any]:
        import json
        import re
        
        response = response.strip()
        
        # Try to find JSON in the response
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            try:
                parsed = json.loads(json_match.group())
                if isinstance(parsed, dict) and 'answer' in parsed:
                    return self._normalize_hotpot_structure(parsed)
            except json.JSONDecodeError:
                pass
        
        # This should not happen easily but... just in case
        logger.warning("Using manual answer extraction")
        
        lines = response.split('\n')
        answer = ""
        supporting_facts = []
        
        for line in lines:
            line = line.strip()
            if line.startswith('Answer:') or line.startswith('ANSWER:'):
                answer = line.split(':', 1)[1].strip()
            elif '[' in line and ']' in line:
                # Try to get supporting facts: [0, 1] or [0-1]
                fact_matches = re.findall(r'\[(\d+)[\s,\-]+(\d+)\]', line)
                for match in fact_matches:
                    try:
                        supporting_facts.append([int(match[0]), int(match[1])])
                    except ValueError:
                        continue
        
        if not answer:
            # if i cannot find answer, assumme first big line as answer
            for line in lines:
                if len(line.strip()) > 10 and not line.strip().startswith('['):
                    answer = line.strip()
                    break
        
        return {
            "answer": answer or response[:200],
            "supporting_facts": supporting_facts,
            "reasoning": "Manual extraction used"
        }
    
    def _validate_hotpot_structure(self, parsed: Dict[str, Any]) -> None:
        if 'answer' not in parsed:
            parsed['answer'] = "No answer provided"
        
        if 'supporting_facts' not in parsed:
            parsed['supporting_facts'] = []
        
        if not isinstance(parsed['supporting_facts'], list):
            logger.warning("supporting_facts is not a list, converting")
            parsed['supporting_facts'] = []
        
        valid_facts = []
        for fact in parsed['supporting_facts']:
            if isinstance(fact, list) and len(fact) >= 2:
                try:
                    doc_idx = int(fact[0])
                    sent_idx = int(fact[1])
                    valid_facts.append([doc_idx, sent_idx])
                except (ValueError, TypeError):
                    logger.warning(f"Invalid supporting fact format: {fact}")
            else:
                logger.warning(f"Invalid supporting fact structure: {fact}")
        
        parsed['supporting_facts'] = valid_facts
    
    def _normalize_hotpot_structure(self, parsed: Dict[str, Any]) -> Dict[str, Any]:
        normalized = {
            "answer": parsed.get('answer', ''),
            "supporting_facts": parsed.get('supporting_facts', []),
            "reasoning": parsed.get('reasoning', parsed.get('explanation', ''))
        }
        
        self._validate_hotpot_structure(normalized)
        return normalized
    
    def _extract_supporting_facts(self, 
                                parsed: Dict[str, Any], 
                                context: List[Tuple[str, List[str]]]) -> List[SupportingFact]:
        supporting_facts = []
        
        for fact in parsed.get('supporting_facts', []):
            if len(fact) == 2:
                doc_idx, sent_idx = fact
                try:
                    # Validate indices
                    if 0 <= doc_idx < len(context) and 0 <= sent_idx < len(context[doc_idx][1]):
                        title = context[doc_idx][0]
                        sentence = context[doc_idx][1][sent_idx]
                        supporting_facts.append(
                            SupportingFact(title=title, sentence_idx=sent_idx, sentence=sentence)
                        )
                    else:
                        logger.warning(f"Supporting fact indices out of range: [{doc_idx}, {sent_idx}]")
                except (IndexError, TypeError) as e:
                    logger.warning(f"Error extracting supporting fact {fact}: {e}")
        
        return supporting_facts
    
    def evaluate_response(self, response: QAResponse, ground_truth: Dict[str, Any]) -> Dict[str, Any]:
        metrics = {}
        
        correct_answer = ground_truth.get('answer', '').lower().strip()
        predicted_answer = response.answer.lower().strip()
        metrics['exact_match'] = self._compute_exact_match(predicted_answer, correct_answer)
        
        f1, precision, recall = self._compute_f1_score(predicted_answer, correct_answer)
        metrics['answer_f1'] = f1
        metrics['answer_precision'] = precision
        metrics['answer_recall'] = recall
        
        # This is HotpotQA-specific
        if 'supporting_facts' in ground_truth:
            sf_metrics = self._evaluate_supporting_facts(response, ground_truth)
            metrics.update(sf_metrics)
        
        return metrics
    
    def _compute_exact_match(self, prediction: str, ground_truth: str) -> bool:
        return self._normalize_answer(prediction) == self._normalize_answer(ground_truth)
    
    def _compute_f1_score(self, prediction: str, ground_truth: str) -> Tuple[float, float, float]:
        pred_tokens = self._normalize_answer(prediction).split()
        gold_tokens = self._normalize_answer(ground_truth).split()
        
        if not pred_tokens and not gold_tokens:
            return 1.0, 1.0, 1.0
        if not pred_tokens or not gold_tokens:
            return 0.0, 0.0, 0.0
        
        common_tokens = set(pred_tokens) & set(gold_tokens)
        
        if not common_tokens:
            return 0.0, 0.0, 0.0
        
        precision = len(common_tokens) / len(pred_tokens)
        recall = len(common_tokens) / len(gold_tokens)
        f1 = (2 * precision * recall) / (precision + recall)
        
        return f1, precision, recall
    
    def _normalize_answer(self, answer: str) -> str:
        import re
        import string
        
        def remove_articles(text):
            return re.sub(r'\b(a|an|the)\b', ' ', text)
        
        def white_space_fix(text):
            return ' '.join(text.split())
        
        def remove_punc(text):
            exclude = set(string.punctuation)
            return ''.join(ch for ch in text if ch not in exclude)
        
        def lower(text):
            return text.lower()
        
        return white_space_fix(remove_articles(remove_punc(lower(answer))))
    
    def _evaluate_supporting_facts(self, 
                                   response: QAResponse, 
                                   ground_truth: Dict[str, Any]) -> Dict[str, Any]:
        true_facts = set(tuple(fact) for fact in ground_truth['supporting_facts'])
        pred_facts = set()
        
        for sf in response.supporting_facts:
            pred_facts.add((sf.title, sf.sentence_idx))
        
        tp = len(true_facts & pred_facts)
        fp = len(pred_facts - true_facts)
        fn = len(true_facts - pred_facts)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        em = 1.0 if fp + fn == 0 else 0.0
        
        return {
            'supporting_facts_em': em,
            'supporting_facts_f1': f1,
            'supporting_facts_precision': precision,
            'supporting_facts_recall': recall
        }
    
    def get_task_schema(self) -> Dict[str, Any]:
        return HOTPOT_QA_SCHEMA.copy()
    
    def update_prompt_style(self, style: str) -> None:
        if style not in ["detailed", "concise", "chain_of_thought"]:
            raise ValueError(f"Unknown prompt style: {style}")
        
        self.prompt_style = style
        logger.info(f"Updated prompt style to: {style}")
    
    def update_structured_output(self, enabled: bool) -> None:
        if enabled and not self.llm_provider.supports_structured_output():
            logger.warning("Provider doesn't support structured output, enabling may not work well")
        
        self.use_structured_output = enabled
        logger.info(f"Updated structured output to: {enabled}")