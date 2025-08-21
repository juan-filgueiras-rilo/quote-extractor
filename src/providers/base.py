from abc import ABC, abstractmethod
from typing import Dict, Any, Optional


class LLMProvider(ABC):

    @abstractmethod
    def generate(self, prompt: str) -> str:
        pass
    
    @abstractmethod
    def parse_structured_output(self, 
                                response: str, 
                                schema: Optional[Dict[str, Any]] = None,
                                format_type: str = "json") -> Dict[str, Any]:
        pass
    
    @abstractmethod
    def supports_structured_output(self) -> bool:
        pass
    
    def get_generation_config(self) -> Dict[str, Any]:
        return {}
    
    def update_generation_config(self, **kwargs) -> None:
        pass