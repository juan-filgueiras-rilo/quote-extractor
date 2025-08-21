import json
import logging
from typing import Optional, Dict, Any

from .base import LLMProvider

logger = logging.getLogger(__name__)


class OpenAIProvider(LLMProvider):
    
    def __init__(self, model_name: str = "gpt-4", api_key: Optional[str] = None):
        self.model_name = model_name
        try:
            import openai
            self.client = openai.OpenAI(api_key=api_key)
            self.available = True
        except ImportError:
            logger.warning("OpenAI not installed. Install with: pip install openai")
            self.available = False
    
    def generate(self, prompt: str) -> str:
        if not self.available:
            raise RuntimeError("OpenAI client not available")
        
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            response_format={"type": "json_object"}
        )
        return response.choices[0].message.content
    
    def parse_structured_output(self, response: str) -> Dict[str, Any]:
        return json.loads(response)