""" 
LLM Implementation of Huggingface CausalLM and Text Generation Models
"""

from .settings import HF_API_KEY
from declarai.operators import BaseLLM, BaseLLMParams, LLMResponse, Message
from declarai.operators.registry import register_llm

from transformers import pipeline, Conversation
from typing import Union, List, Optional

class HuggingFaceError(Exception):
    """Generic HuggingFace error class when working with HF provider."""

    pass

class BaseHuggingFaceLLM(BaseLLM):
    provider = "hf"

    def __init__(self, api_key: str,
                 model_name: str,
                 stream: bool = False,
                 **kwargs) -> None:
        
        self.pipeline = pipeline(task="conversational", model=model_name,
                                 token=api_key)
        
        self.api_key = api_key
        self.stream = stream

        self.model = model_name

        if self.stream:
            raise HuggingFaceError("Currently HuggingFace LLM not supporting streaming output")
        
    @property
    def streaming(self) -> bool:
        """
        Returns whether the LLM is streaming or not
        Returns:
            bool: True if the LLM is streaming, False otherwise
        """
        return self.stream
    
    def predict(self, messages: List[Message],
                temperature: float = 0,
                top_p: float = 1,
                stream: bool = False):
        
        if stream:
            raise HuggingFaceError("Currently HuggingFace LLM not supporting streaming output")
        
        hf_messages = [{"role": m.role, "content": m.message} for m in messages]
        conversation = Conversation(hf_messages)

        res = self.pipeline(conversation, top_p = top_p, temperature = temperature)
        answer = res.generated_responses[-1]

        return LLMResponse(
            response=answer
        )

@register_llm(provider="hf")
class HuggingFaceLLM(BaseHuggingFaceLLM):

    def __init__(self,
                 model_name: str,
                 api_key: str = None, 
                 stream: bool = False, 
                 **kwargs) -> None:
        
        api_key = api_key or HF_API_KEY
        if not api_key:
            raise HuggingFaceError(
                "Missing an Huggingface API Token"
                "In order to work with HuggingFace, you will need to provide an API key"
                "either by setting the HF_API_KEY or by providing"
                "the API key via the init interface."
            )
        
        super().__init__(api_key, model_name, stream, **kwargs)

class HuggingFaceLLMParams(BaseLLMParams):

    temperature: Optional[float]
    top_p: Optional[float]

