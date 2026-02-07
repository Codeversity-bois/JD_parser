import os
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables from .env file
env_path = Path(__file__).parent / '.env'
load_dotenv(dotenv_path=env_path)


class Config:
    """Configuration class for JD Parser application"""
    
    # MongoDB Configuration
    MONGODB_URL = os.getenv('MONGODB_URL')
    MONGODB_DB_NAME = os.getenv('MONGODB_DB_NAME', 'jd_parser_db')
    COLLECTION_NAME = os.getenv('COLLECTION_NAME', 'job_descriptions')
    
    # OpenRouter Configuration
    OPENROUTER_API_KEY = os.getenv('OPENROUTER_API_KEY')
    OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
    
    # FAISS Configuration
    FAISS_INDEX_PATH = os.path.join(os.path.dirname(__file__), 'faiss_index')
    EMBEDDING_DIMENSION = 1536  # Default embedding dimension
    
    @classmethod
    def validate(cls):
        """Validate required configuration"""
        if not cls.MONGODB_URL:
            raise ValueError("MONGODB_URL is not set in environment variables")
        if not cls.OPENROUTER_API_KEY:
            raise ValueError("OPENROUTER_API_KEY is not set in environment variables")


config = Config()
