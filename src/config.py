
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    openai_api_key: str
    embedding_model: str = "all-MiniLM-L6-v2"
    persist_dir: str = "./data/chroma"

    class Config:
        env_file = ".env"


settings = Settings()
