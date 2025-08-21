import json
import logging
from typing import Optional, Dict, Any

from src.providers.base import LLMProvider
# from src.templates.hotpot_qa import create_hotpot_qa_prompt

logger = logging.getLogger(__name__)


class Llama3Provider(LLMProvider):
    
    def __init__(self, model_name: str = "llama3", api_url: Optional[str] = None):
        self.model_name = model_name
        self.api_url = api_url or "http://localhost:11434"
        
        try:
            import ollama
            self.ollama_client = ollama.Client(host=self.api_url)
        except ImportError:
            raise RuntimeError("Ollama client not available")
    
    def generate(self, prompt: str) -> str:
        messages = ""#create_hotpot_qa_prompt(prompt, False)
        try:
            response = self.ollama_client.chat(
                model=self.model_name,
                messages=messages,
                options={"temperature": 0.1}
            )
            return response['message']['content']
        except Exception as e:
            logger.error(f"Error calling Ollama: {e}")
            raise
    
    def parse_structured_output(self, response: str) -> Dict[str, Any]:
        response = response.strip()
        if response.startswith("```json"):
            response = response[7:]
        if response.startswith("```"):
            response = response[3:]
        if response.endswith("```"):
            response = response[:-3]
        
        try:
            return json.loads(response.strip())
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON: {e}")
            logger.error(f"Response was: {response}")
            return {"answer": response, "supporting_facts": []}