from pathlib import Path
from pydantic import DirectoryPath, FilePath
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # Пути
    kb_path: FilePath
    chroma_dir: Path
    chroma_collection: str

    # Embeddings
    embed_model: str
    embed_device: str

    # Ollama
    ollama_model: str
    ollama_base_url: str

    # RAG
    retriever_top_k: int
    temperature: float
    max_tokens: int

    # Gradio
    gradio_server_name: str
    gradio_server_port: int
    gradio_share: bool

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

settings = Settings()
print(settings.ollama_model)