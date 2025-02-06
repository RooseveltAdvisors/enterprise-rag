# rag_config.py - Configuration settings for the RAG system
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set environment variables
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["TAVILY_API_KEY"] = os.getenv('TAVILY_API_KEY')
os.environ["AZURE_EMBEDDING_API_KEY"] = os.getenv("AZURE_EMBEDDING_API_KEY")
os.environ["AZURE_CHAT_API_KEY"] = os.getenv("AZURE_CHAT_API_KEY")

# Model Configuration
EMBEDDING_TYPE = "azure"  # Change to "openai" to use OpenAI instead of Azure
LLM_TYPE = "azure"  # Change to "openai" to use OpenAI instead of Azure

# Azure Base Configuration
AZURE_BASE_URL = os.getenv("AZURE_BASE_URL")

# Azure Model Deployments
AZURE_DEPLOYMENTS = {
    "embedding": {
        "deployment_name": "text-embedding-ada-002",
        "api_version": "2023-05-15",
        "api_key": os.getenv("AZURE_EMBEDDING_API_KEY")
    },
    "chat": {
        "deployment_name": "gpt-4o",
        "api_version": "2024-08-01-preview",
        "api_key": os.getenv("AZURE_CHAT_API_KEY")
    }
}

# Azure OpenAI Embedding Configuration
AZURE_OPENAI_CONFIG = {
    "azure_endpoint": AZURE_BASE_URL,
    "azure_deployment": AZURE_DEPLOYMENTS["embedding"]["deployment_name"],
    "api_version": AZURE_DEPLOYMENTS["embedding"]["api_version"],
    "api_key": AZURE_DEPLOYMENTS["embedding"]["api_key"]
}

# Azure OpenAI Chat Configuration
AZURE_CHAT_CONFIG = {
    "azure_endpoint": AZURE_BASE_URL,
    "azure_deployment": AZURE_DEPLOYMENTS["chat"]["deployment_name"],
    "api_version": AZURE_DEPLOYMENTS["chat"]["api_version"],
    "temperature": 0,
    "api_key": AZURE_DEPLOYMENTS["chat"]["api_key"]
}

# OpenAI Configuration (if needed)
OPENAI_CONFIG = {
    "temperature": 0
    # Add any specific OpenAI configuration here if needed
}

# Directory Configuration
PROJECT_ROOT = os.path.abspath(os.getcwd())
RUNTIME_DIR = os.path.join(PROJECT_ROOT, 'runtime/document_store')
DATA_DIR = os.path.join(PROJECT_ROOT, 'data/documents')

# Processing Configuration
BATCH_SIZE = 1000

# Grader Configuration
ENABLE_ANSWER_GRADER = False  # Set to False to disable answer grading
ENABLE_HALLUCINATION_GRADER = False  # Set to False to disable hallucination checking

def validate_config():
    """Validate required environment variables and configurations."""
    required_vars = ["TAVILY_API_KEY"]
    
    if EMBEDDING_TYPE.lower() == "azure":
        required_vars.extend([
            "AZURE_EMBEDDING_API_KEY"
        ])
    elif EMBEDDING_TYPE.lower() == "openai":
        required_vars.extend([
            "OPENAI_API_KEY"
        ])
    
    if LLM_TYPE.lower() == "azure":
        required_vars.extend([
            "AZURE_CHAT_API_KEY"
        ])
    elif LLM_TYPE.lower() == "openai":
        if "OPENAI_API_KEY" not in required_vars:
            required_vars.extend([
                "OPENAI_API_KEY"
            ])
    
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    if missing_vars:
        raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")

    # Validate Azure configurations
    if EMBEDDING_TYPE.lower() == "azure":
        if not AZURE_DEPLOYMENTS["embedding"]["api_key"]:
            raise ValueError("Azure OpenAI embedding API key not configured")

    if LLM_TYPE.lower() == "azure":
        if not AZURE_DEPLOYMENTS["chat"]["api_key"]:
            raise ValueError("Azure OpenAI chat API key not configured")

# Validate configuration on import
validate_config()

def get_llm_config():
    """Get the appropriate LLM configuration based on the selected type."""
    if LLM_TYPE.lower() == "azure":
        return AZURE_CHAT_CONFIG
    elif LLM_TYPE.lower() == "openai":
        return OPENAI_CONFIG
    else:
        raise ValueError(f"Unsupported LLM type: {LLM_TYPE}")

def get_embedding_config():
    """Get the appropriate embedding configuration based on the selected type."""
    if EMBEDDING_TYPE.lower() == "azure":
        return AZURE_OPENAI_CONFIG
    elif EMBEDDING_TYPE.lower() == "openai":
        return OPENAI_CONFIG
    else:
        raise ValueError(f"Unsupported embedding type: {EMBEDDING_TYPE}")
