"""
language_models.py - Language Model Factory

This module provides a factory function for creating Large Language Model (LLM) instances
based on the system configuration. It supports both Azure OpenAI and OpenAI chat models,
allowing for flexible deployment options.

The language models are used for:
- Generating answers from retrieved documents
- Evaluating answer quality
- Detecting potential hallucinations
- Processing and expanding search queries

Available Models:
- Azure OpenAI Chat (GPT-4)
- OpenAI Chat (GPT-4)
"""

from langchain_openai import ChatOpenAI, AzureChatOpenAI
from rag_config import LLM_TYPE, AZURE_OPENAI_CONFIG

def get_llm(temperature=0):
    """
    Factory function to create and configure LLM instances.
    
    Args:
        temperature (float, optional): Controls randomness in the model's output.
                                     Lower values (like 0) make the output more focused and deterministic.
                                     Higher values (up to 1) make the output more creative and diverse.
                                     Defaults to 0 for maximum consistency.
    
    Returns:
        ChatModel: An instance of either AzureChatOpenAI or ChatOpenAI
                  configured according to the system settings.
    
    The function checks LLM_TYPE from rag_config to determine which chat model to use:
    - For "azure": Returns AzureChatOpenAI with Azure-specific configuration
    - For "openai": Returns ChatOpenAI with specified temperature
    
    Raises:
        ValueError: If LLM_TYPE is not supported
    """
    if LLM_TYPE.lower() == "azure":
        return AzureChatOpenAI(
            temperature=temperature,
            **AZURE_OPENAI_CONFIG
        )
    elif LLM_TYPE.lower() == "openai":
        return ChatOpenAI(temperature=temperature)
    else:
        raise ValueError(f"Unsupported LLM type: {LLM_TYPE}")
