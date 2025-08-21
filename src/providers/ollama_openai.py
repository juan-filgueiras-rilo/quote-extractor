import json
import logging
from typing import Dict, Any, Optional

from src.providers.base import LLMProvider

logger = logging.getLogger(__name__)


class OllamaOpenAIProvider(LLMProvider):
    
    def __init__(self, 
                 model_name: str = "llama3.1:8b",
                 base_url: str = "http://localhost:11434/v1",
                 api_key: str = "ollama",  # Ollama doesn't need a real key
                 timeout: int = 120,
                 temperature: float = 0.1,
                 max_tokens: int = 1024,
                 enable_structured_output: bool = True):

        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.enable_structured_output = enable_structured_output
        
        try:
            import openai
            self.client = openai.OpenAI(
                base_url=base_url,
                api_key=api_key,
                timeout=timeout
            )
            logger.info(f"Initialized Ollama OpenAI provider with model: {model_name}")
            logger.info(f"Base URL: {base_url}")
            
            self._test_connection()
            
        except ImportError:
            logger.error("OpenAI package not installed. Install with: pip install openai")
            raise RuntimeError("Ollama OpenAI provider not available")
        except Exception as e:
            logger.error(f"Error initializing Ollama OpenAI provider: {e}")
            raise RuntimeError("Ollama OpenAI provider not available")
    
    def _test_connection(self):
        try:
            response = self.client.models.list()
            available_models = [model.id for model in response.data]
            logger.info(f"Available Ollama models: {available_models}")
            
            if self.model_name not in available_models:
                logger.warning(f"Model '{self.model_name}' not found. Available: {available_models}")
                raise RuntimeError("Ollama OpenAI provider not available")
                
        except Exception as e:
            logger.warning(f"Could not list models (this is normal for some Ollama versions): {e}")
            raise
    
    def generate(self, prompt: str) -> str:
        try:
            messages = [
                {
                    "role": "system", 
                    "content": "You are a helpful assistant that follows instructions precisely."
                },
                {
                    "role": "user", 
                    "content": prompt
                }
            ]
            
            logger.debug(f"Sending request to Ollama (model: {self.model_name})...")
            
            if self.enable_structured_output:
                try:
                    response = self.client.chat.completions.create(
                        model=self.model_name,
                        messages=messages,
                        temperature=self.temperature,
                        max_tokens=self.max_tokens,
                        response_format={"type": "json_object"},
                        stream=False
                    )
                except Exception as e:
                    # Fallback to standard request if structured output call failed
                    logger.debug("response_format not supported, using standard request")
                    response = self.client.chat.completions.create(
                        model=self.model_name,
                        messages=messages,
                        temperature=self.temperature,
                        max_tokens=self.max_tokens,
                        stream=False
                    )
            else:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    stream=False
                )
            
            content = response.choices[0].message.content
            logger.debug(f"Raw response: {content[:200]}...")
            
            return content
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            raise
    
    def parse_structured_output(self, 
                                response: str, 
                                schema: Optional[Dict[str, Any]] = None,
                                format_type: str = "json") -> Dict[str, Any]:
        if format_type != "json":
            raise ValueError(f"Format type '{format_type}' not supported. Only 'json' is supported.")
        
        response = response.strip()
        
        # Remove markdown code blocks if present
        if "```json" in response:
            start = response.find("```json") + 7
            end = response.rfind("```")
            if end > start:
                response = response[start:end].strip()
        elif "```" in response:
            parts = response.split("```")
            if len(parts) >= 3:
                response = parts[1].strip()
        
        # Find JSON content more aggressively
        start_idx = response.find('{')
        end_idx = response.rfind('}')
        
        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            response = response[start_idx:end_idx + 1]
        
        try:
            parsed = json.loads(response)
            
            if not isinstance(parsed, dict):
                raise ValueError("Response is not a JSON object")

            if schema:
                self._validate_against_schema(parsed, schema)
            
            return parsed
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON: {e}")
            logger.error(f"Response was: {response[:500]}...")
            
            return {
                "raw_response": response,
                "parse_error": str(e)
            }
    
    def _validate_against_schema(self, data: Dict[str, Any], schema: Dict[str, Any]) -> None:
        try:
            import jsonschema
            jsonschema.validate(data, schema)
        except ImportError:
            if "required" in schema:
                for field in schema["required"]:
                    if field not in data:
                        raise ValueError(f"Required field '{field}' missing from response")
        except Exception as e:
            logger.warning(f"Schema validation failed: {e}")
            # Don't fail hard on schema validation - let processor handle it
    
    def supports_structured_output(self) -> bool:
        return self.enable_structured_output
    
    def get_generation_config(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "timeout": self.timeout,
            "enable_structured_output": self.enable_structured_output
        }
    
    def update_generation_config(self, **kwargs) -> None:
        if "temperature" in kwargs:
            self.temperature = kwargs["temperature"]
        if "max_tokens" in kwargs:
            self.max_tokens = kwargs["max_tokens"]
        if "enable_structured_output" in kwargs:
            self.enable_structured_output = kwargs["enable_structured_output"]
        
        logger.info(f"Updated generation config: {kwargs}")