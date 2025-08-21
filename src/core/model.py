from dataclasses import dataclass
from typing import List, Optional


@dataclass
class SupportingFact:
    title: str
    sentence_idx: int
    sentence: str


@dataclass
class QAResponse:
    answer: str
    supporting_facts: List[SupportingFact]
    raw_response: Optional[str] = None