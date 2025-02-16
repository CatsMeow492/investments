from pydantic import BaseSettings
from typing import Optional
import os
from dotenv import load_dotenv

load_dotenv()

class Settings(BaseSettings):
    # API Keys
    POLYGON_API_KEY: str = os.getenv("POLYGON_API_KEY", "")
    ALPHA_VANTAGE_API_KEY: Optional[str] = os.getenv("ALPHA_VANTAGE_API_KEY")
    OPENAI_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY")
    ANTHROPIC_API_KEY: Optional[str] = os.getenv("ANTHROPIC_API_KEY")
    OPENROUTER_API_KEY: Optional[str] = os.getenv("OPENROUTER_API_KEY")
    DEEPSEEK_API_KEY: Optional[str] = os.getenv("DEEPSEEK_API_KEY")

    # Redis Configuration
    REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379")
    
    # Cache TTL Configuration (in seconds)
    CACHE_TTL_MARKET_DATA: int = 24 * 60 * 60  # 24 hours
    CACHE_TTL_NEWS: int = 12 * 60 * 60  # 12 hours
    CACHE_TTL_ANALYST_RATINGS: int = 7 * 24 * 60 * 60  # 7 days
    CACHE_TTL_PORTFOLIO_RECOMMENDATIONS: int = 24 * 60 * 60  # 24 hours

    # API Rate Limiting
    POLYGON_CALLS_PER_MINUTE: int = 30
    
    # Model Configuration
    MODEL_TIMEOUT: float = 60.0
    DEEPSEEK_MODEL: str = "deepseek-chat"
    DEEPSEEK_TIMEOUT: float = 60.0
    DEEPSEEK_API_URL: str = "https://api.deepseek.com/v1/chat/completions"

    # CORS Configuration
    CORS_ORIGINS: list = ["http://localhost:3000", "http://127.0.0.1:3000"]

    class Config:
        case_sensitive = True

settings = Settings()
