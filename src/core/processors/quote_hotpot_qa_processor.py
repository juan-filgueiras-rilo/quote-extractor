import logging
from typing import List, Tuple, Dict, Any

from src.core.processors.hotpot_qa_processor import HotpotQAProcessor
from src.core.model import QAResponse, SupportingFact
from src.providers.base import LLMProvider
import Levenshtein

from src.templates.quote_hotpot_qa import (
    QuoteHotpotQAPromptTemplate,
    QUOTE_HOTPOT_QA_SCHEMA,
)

logger = logging.getLogger(__name__)


class QuoteHotpotQAProcessor(HotpotQAProcessor):

    def __init__(
        self,
        llm_provider: LLMProvider,
        prompt_style: str = "detailed",
        use_structured_output: bool = None,
        levenshtein_threshold: float = 0.9,
    ):

        super().__init__(llm_provider, prompt_style, use_structured_output)

        # Override prompt template with quote-based version
        self.prompt_template = QuoteHotpotQAPromptTemplate()
        self.levenshtein_threshold = levenshtein_threshold

        logger.info(f"Initialized Quote HotpotQA processor with:")
        logger.info(f"  - Provider: {type(llm_provider).__name__}")
        logger.info(f"  - Prompt style: {prompt_style}")
        logger.info(f"  - Structured output: {self.use_structured_output}")
        logger.info(f"  - Levenshtein threshold: {levenshtein_threshold}")

    def process_question(self, question_data: Dict[str, Any]) -> QAResponse:
        question = question_data["question"]
        context = question_data["context"]

        prompt = self.create_prompt(question, context)

        logger.info(f"Processing question: {question[:100]}...")
        raw_response = self.llm_provider.generate(prompt)

        parsed = self._parse_quote_response(raw_response)

        supporting_facts = self._extract_supporting_facts_from_quotes(parsed, context)

        return QAResponse(
            answer=parsed.get("answer", "No answer found"),
            supporting_facts=supporting_facts,
            raw_response=raw_response,
        )

    def _parse_quote_response(self, response: str) -> Dict[str, Any]:
        if self.use_structured_output:
            try:
                parsed = self.llm_provider.parse_structured_output(
                    response=response, schema=QUOTE_HOTPOT_QA_SCHEMA, format_type="json"
                )

                self._validate_quote_structure(parsed)
                return parsed

            except Exception as e:
                logger.warning(f"Structured parsing failed: {e}")
                # Fall back to manual parsing
                return self._manual_parse_quote_response(response)
        else:
            return self._manual_parse_quote_response(response)

    def _manual_parse_quote_response(self, response: str) -> Dict[str, Any]:
        import json
        import re

        response = response.strip()

        # Try to find JSON in the response
        json_match = re.search(r"\{.*\}", response, re.DOTALL)
        if json_match:
            try:
                parsed = json.loads(json_match.group())
                if isinstance(parsed, dict) and "answer" in parsed:
                    return self._normalize_quote_structure(parsed)
            except json.JSONDecodeError:
                pass

        logger.warning("Using manual quote extraction")

        lines = response.split("\n")
        answer = ""
        supporting_quotes = []

        for line in lines:
            line = line.strip()
            if line.startswith("Answer:") or line.startswith("ANSWER:"):
                answer = line.split(":", 1)[1].strip()
            elif line.startswith('"') and line.endswith('"'):
                quote = line.strip('"')
                if len(quote) > 10:
                    supporting_quotes.append(quote)
            elif "Quote:" in line or "Supporting quote:" in line:
                quote_part = line.split(":", 1)[1].strip().strip('"')
                if len(quote_part) > 10:
                    supporting_quotes.append(quote_part)

        if not answer:
            # If no answer found, assume first substantial line is the answer
            for line in lines:
                if len(line.strip()) > 10 and not line.strip().startswith('"'):
                    answer = line.strip()
                    break

        return {
            "answer": answer or response[:200],
            "supporting_quotes": supporting_quotes,
            "reasoning": "Manual extraction used",
        }

    def _validate_quote_structure(self, parsed: Dict[str, Any]) -> None:
        if "answer" not in parsed:
            parsed["answer"] = "No answer provided"

        if "supporting_quotes" not in parsed:
            parsed["supporting_quotes"] = []

        if not isinstance(parsed["supporting_quotes"], list):
            logger.warning("supporting_quotes is not a list, converting")
            parsed["supporting_quotes"] = []

        # Ensure all quotes are strings and filter out empty ones
        valid_quotes = []
        for quote in parsed["supporting_quotes"]:
            if isinstance(quote, str) and len(quote.strip()) > 0:
                valid_quotes.append(quote.strip())
            else:
                logger.warning(f"Invalid quote format: {quote}")

        parsed["supporting_quotes"] = valid_quotes

    def _normalize_quote_structure(self, parsed: Dict[str, Any]) -> Dict[str, Any]:
        normalized = {
            "answer": parsed.get("answer", ""),
            "supporting_quotes": parsed.get("supporting_quotes", []),
            "reasoning": parsed.get("reasoning", parsed.get("explanation", "")),
        }

        self._validate_quote_structure(normalized)
        return normalized

    def _extract_supporting_facts_from_quotes(
        self, parsed: Dict[str, Any], context: List[Tuple[str, List[str]]]
    ) -> List[SupportingFact]:
        supporting_facts = []
        quotes = parsed.get("supporting_quotes", [])

        for quote in quotes:
            match_found = False
            best_match = None
            best_similarity = 0

            # Search for the quote in the context using Levenshtein distance
            for doc_idx, (title, sentences) in enumerate(context):
                for sent_idx, sentence in enumerate(sentences):
                    similarity = self._levenshtein_similarity(quote, sentence)

                    if similarity >= self.levenshtein_threshold:
                        if similarity > best_similarity:
                            best_similarity = similarity
                            best_match = (doc_idx, sent_idx, title, sentence)
                        match_found = True

            if match_found and best_match:
                doc_idx, sent_idx, title, sentence = best_match
                supporting_facts.append(
                    SupportingFact(
                        title=title, sentence_idx=sent_idx, sentence=sentence
                    )
                )
                logger.debug(
                    f"Matched quote '{quote[:50]}...' to [{doc_idx}, {sent_idx}] with similarity {best_similarity:.3f}"
                )
            else:
                logger.warning(
                    f"Could not find matching sentence for quote: '{quote[:100]}...'"
                )

        return supporting_facts

    def _levenshtein_similarity(self, s1: str, s2: str) -> float:
        """Calculate Levenshtein similarity (1 - normalized distance)"""
        try:
            distance = Levenshtein.distance(s1.lower().strip(), s2.lower().strip())
            max_len = max(len(s1), len(s2))
            if max_len == 0:
                return 1.0
            similarity = 1 - (distance / max_len)
            return similarity
        except ImportError:
            # Fallback to simple implementation if python-Levenshtein not available
            return self._simple_levenshtein_similarity(s1, s2)

    def _simple_levenshtein_similarity(self, s1: str, s2: str) -> float:
        """Simple Levenshtein distance implementation as fallback"""
        s1, s2 = s1.lower().strip(), s2.lower().strip()

        if s1 == s2:
            return 1.0

        len1, len2 = len(s1), len(s2)
        if len1 == 0 or len2 == 0:
            return 0.0

        # Create distance matrix
        matrix = [[0] * (len2 + 1) for _ in range(len1 + 1)]

        # Initialize first row and column
        for i in range(len1 + 1):
            matrix[i][0] = i
        for j in range(len2 + 1):
            matrix[0][j] = j

        # Fill the matrix
        for i in range(1, len1 + 1):
            for j in range(1, len2 + 1):
                if s1[i - 1] == s2[j - 1]:
                    cost = 0
                else:
                    cost = 1

                matrix[i][j] = min(
                    matrix[i - 1][j] + 1,  # deletion
                    matrix[i][j - 1] + 1,  # insertion
                    matrix[i - 1][j - 1] + cost,  # substitution
                )

        distance = matrix[len1][len2]
        max_len = max(len1, len2)
        similarity = 1 - (distance / max_len)
        return similarity

    def get_task_schema(self) -> Dict[str, Any]:
        """Override parent method to return quote schema"""
        return QUOTE_HOTPOT_QA_SCHEMA.copy()

    def update_levenshtein_threshold(self, threshold: float) -> None:
        """Update the Levenshtein similarity threshold for quote matching"""
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("Threshold must be between 0.0 and 1.0")

        self.levenshtein_threshold = threshold
        logger.info(f"Updated Levenshtein threshold to: {threshold}")

    def get_quote_matching_stats(
        self, question_data: Dict[str, Any], parsed_response: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Get statistics about quote matching quality"""
        context = question_data["context"]
        quotes = parsed_response.get("supporting_quotes", [])

        stats = {
            "total_quotes": len(quotes),
            "matched_quotes": 0,
            "unmatched_quotes": [],
            "match_similarities": [],
        }

        for quote in quotes:
            best_similarity = 0
            match_found = False

            for doc_idx, (title, sentences) in enumerate(context):
                for sent_idx, sentence in enumerate(sentences):
                    similarity = self._levenshtein_similarity(quote, sentence)
                    best_similarity = max(best_similarity, similarity)

                    if similarity >= self.levenshtein_threshold:
                        match_found = True

            if match_found:
                stats["matched_quotes"] += 1
                stats["match_similarities"].append(best_similarity)
            else:
                stats["unmatched_quotes"].append(
                    {"quote": quote, "best_similarity": best_similarity}
                )

        if stats["match_similarities"]:
            stats["avg_similarity"] = sum(stats["match_similarities"]) / len(
                stats["match_similarities"]
            )
            stats["min_similarity"] = min(stats["match_similarities"])
        else:
            stats["avg_similarity"] = 0.0
            stats["min_similarity"] = 0.0

        return stats
