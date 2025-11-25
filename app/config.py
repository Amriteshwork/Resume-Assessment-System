import os
from dotenv import load_dotenv
from pydantic_settings import BaseSettings

load_dotenv()


class Settings(BaseSettings):
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    db_url: str = "sqlite:///./assessments.db"
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

    class Config:
        env_file = ".env"


settings = Settings()