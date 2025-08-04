import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Settings:
    """Application settings and configuration"""
    
    # API Keys
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    
    # Model configurations
    CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
    LLM_MODEL_NAME = "gpt-4o"
    
    # Text processing
    CHUNK_SIZE = 500
    CHUNK_OVERLAP = 100
    MAX_TEXT_LENGTH = 77  # CLIP's max token length
    
    # Vector store
    DEFAULT_K = 5  # Default number of results to retrieve
    
    # Image processing
    IMAGE_FORMAT = "PNG"
    MAX_IMAGE_SIZE = (1024, 1024)  # Max dimensions for images
    
    # Streamlit
    PAGE_TITLE = "Multimodal RAG Assistant"
    PAGE_ICON = "🖼️"
    LAYOUT = "wide"
    
    @classmethod
    def validate_api_key(cls):
        """Validate that required API keys are set"""
        if not cls.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        return True

# Create global settings instance
settings = Settings()