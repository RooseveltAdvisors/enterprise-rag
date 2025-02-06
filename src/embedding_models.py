"""
embedding_models.py - Embedding Model Factory

This module provides a factory function for creating text embedding models based on the system configuration.
It supports both Azure OpenAI and OpenAI embedding models, allowing for flexible deployment options.

The embedding models are used to convert text into high-dimensional vectors for semantic search
and document retrieval operations in the RAG pipeline.

Available Models:
- Azure OpenAI Embeddings (text-embedding-ada-002)
- OpenAI Embeddings (text-embedding-ada-002)
"""

from langchain_openai import OpenAIEmbeddings, AzureOpenAIEmbeddings
from rag_config import EMBEDDING_TYPE, AZURE_OPENAI_CONFIG

def get_embeddings():
    """
    Factory function to create and configure embedding model instances.
    
    Returns:
        EmbeddingModel: An instance of either AzureOpenAIEmbeddings or OpenAIEmbeddings
                       configured according to the system settings.
    
    The function checks EMBEDDING_TYPE from rag_config to determine which embedding model to use:
    - For "azure": Returns AzureOpenAIEmbeddings with Azure-specific configuration
    - For "openai": Returns OpenAIEmbeddings with default configuration
    
    Raises:
        ValueError: If EMBEDDING_TYPE is not supported
    """
    if EMBEDDING_TYPE.lower() == "azure":
        return AzureOpenAIEmbeddings(**AZURE_OPENAI_CONFIG)
    elif EMBEDDING_TYPE.lower() == "openai":
        return OpenAIEmbeddings()
    else:
        raise ValueError(f"Unsupported embedding type: {EMBEDDING_TYPE}")
