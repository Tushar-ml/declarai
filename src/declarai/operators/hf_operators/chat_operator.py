"""
Chat implementation of HuggingFace operator.
"""
import logging
from declarai.operators.hf_operators.hf_llm import HuggingFaceLLM
from declarai.operators.operator import BaseChatOperator
from declarai.operators.registry import register_operator

logger = logging.getLogger("HuggingFaceChatOperator")


@register_operator(provider="hf", operator_type="chat")
class HuggingFaceChatOperator(BaseChatOperator):
    """
    Chat implementation of HuggingFace operator. This is a child of the BaseChatOperator class. See the BaseChatOperator class for further documentation.

    Attributes:
        llm: HuggingFaceLLM
    """

    llm: HuggingFaceLLM

