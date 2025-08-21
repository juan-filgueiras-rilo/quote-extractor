import json
import logging
from typing import Optional, Dict, Any

from src.providers.base import LLMProvider

logger = logging.getLogger(__name__)


class HuggingFaceLlamaProvider(LLMProvider):
    
    def __init__(self, 
                 model_name: str = "meta-llama/Llama-3.2-3B-Instruct",
                 device: str = "auto",
                 load_in_8bit: bool = False,
                 load_in_4bit: bool = False,
                 max_new_tokens: int = 512,
                 temperature: float = 0.1,
                 auth_token: Optional[str] = None):
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.device = device
        self.temperature = temperature
        self.load_in_8bit = load_in_8bit
        self.load_in_4bit = load_in_4bit
        
        try:
            import torch
            from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
            
            logger.info(f"Loading model {model_name} from Hugging Face...")
            
            # Quantization
            bnb_config = None
            if load_in_4bit:
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                )
            elif load_in_8bit:
                bnb_config = BitsAndBytesConfig(load_in_8bit=True)
            
            # Tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                token=auth_token,
                trust_remote_code=True
            )
            
            # Set padding token if not set
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Model
            model_kwargs = {
                "token": auth_token,
                "trust_remote_code": True,
                "torch_dtype": torch.float16 if torch.cuda.is_available() else torch.float32,
            }
            
            if bnb_config:
                model_kwargs["quantization_config"] = bnb_config
            
            if device == "auto":
                model_kwargs["device_map"] = "auto"
            
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                **model_kwargs
            )
            
            # Move to device if not using device_map
            if device != "auto" and not (load_in_4bit or load_in_8bit):
                self.model = self.model.to(device)
            
            self.available = True
            logger.info(f"Model {model_name} loaded successfully!")
            
        except ImportError as e:
            logger.error(f"Required packages not installed: {e}")
            logger.error("Install with: pip install transformers torch accelerate bitsandbytes")
            raise RuntimeError("Hugging Face model not available")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise RuntimeError("Hugging Face model not available")
    
    def generate(self, prompt: str) -> str:
        if not self.available:
            raise RuntimeError("Hugging Face model not available")
        
        try:
            import torch
            
            messages = [
                {"role": "system", "content": "You are a helpful assistant that follows instructions precisely."},
                {"role": "user", "content": prompt}
            ]
            
            if hasattr(self.tokenizer, 'apply_chat_template'):
                formatted_prompt = self.tokenizer.apply_chat_template(
                    messages, 
                    tokenize=False,
                    add_generation_prompt=True
                )
            else:
                formatted_prompt = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
            
            # Tokenize
            inputs = self.tokenizer(
                formatted_prompt,
                return_tensors="pt",
                truncation=True,
                max_length=4096
            )
            
            # Inputs to model device
            if hasattr(self.model, 'device'):
                inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            
            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    temperature=self.temperature,
                    do_sample=True if self.temperature > 0 else False,
                    top_p=0.95,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
            
            # Decode
            response = self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
            return response.strip()
            
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
        
        json_start = response.find('{')
        json_end = response.rfind('}') + 1
        
        if json_start != -1 and json_end > json_start:
            response = response[json_start:json_end]
        
        # Clean up markdown
        if "```json" in response:
            response = response.split("```json")[1].split("```")[0]
        elif "```" in response:
            response = response.split("```")[1].split("```")[0]
        
        try:
            parsed = json.loads(response.strip())
            
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
    
    def supports_structured_output(self) -> bool:
        return False
    
    def get_generation_config(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "device": self.device,
            "load_in_4bit": self.load_in_4bit,
            "load_in_8bit": self.load_in_8bit,
        }
    
    def update_generation_config(self, **kwargs) -> None:
        if "max_new_tokens" in kwargs:
            self.max_new_tokens = kwargs["max_new_tokens"]
        if "temperature" in kwargs:
            self.temperature = kwargs["temperature"]
        
        logger.info(f"Updated generation config: {kwargs}")