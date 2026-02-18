"""Configuration management for the application."""
import os
from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # Application
    app_name: str = "Real Estate Doc Intelligence"
    debug: bool = False
    use_in_memory: bool = True  # Use in-memory storage (no Docker required)
    
    # Qdrant Vector Database
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_collection: str = "real_estate_docs"
    
    # PostgreSQL Database
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_user: str = "admin"
    postgres_password: str = "password"
    postgres_db: str = "doc_intelligence"
    
    # Redis Cache
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_ttl: int = 3600  # 1 hour
    
    # Embedding Model
    embedding_model: str = "all-mpnet-base-v2"
    embedding_dim: int = 768
    
    # Chunking Settings
    chunk_size: int = 500
    chunk_overlap: int = 50
    
    # Search Settings
    top_k: int = 5
    similarity_threshold: float = 0.7
    
    # File Storage
    upload_dir: str = "data/pdfs"
    
    @property
    def postgres_url(self) -> str:
        return f"postgresql://{self.postgres_user}:{self.postgres_password}@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
