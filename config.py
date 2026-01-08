import os
from dataclasses import dataclass
from dotenv import load_dotenv

# Load secrets from .env.local
load_dotenv(".env.local", override=True)


@dataclass
class Settings:
    # Postgres
    pg_user: str = os.getenv("POSTGRES_USER", "postgres")
    pg_password: str = os.getenv("POSTGRES_PASSWORD", "")
    pg_db: str = os.getenv("POSTGRES_DB", "fastapi_db")
    pg_host: str = os.getenv("POSTGRES_HOST", "localhost")
    pg_port: int = int(os.getenv("POSTGRES_PORT", "5432"))

    # OpenAI (chat LLM – embeddings stay in SentenceTransformers)
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    openai_chat_model: str = os.getenv(
        "OPENAI_CHAT_MODEL",
        "gpt-4.1-mini",  # or your preferred default
    )


    # 👉 NEW: SentenceTransformers model (used for all embeddings)
    sentence_model_name: str = os.getenv("SENTENCE_MODEL_NAME", "all-MiniLM-L6-v2")

    # Chroma
    chroma_persist_dir: str = os.getenv("CHROMA_PERSIST_DIR", "./chroma_property_knowledge")
    chroma_collection_name: str = os.getenv("CHROMA_COLLECTION_NAME", "property_sql_knowledge")

    # MongoDB (for chat history)
    mongo_uri: str = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
    mongo_db: str = os.getenv("MONGODB_DB", "pbchs_chat")
    mongo_history_collection: str = os.getenv(
        "MONGODB_HISTORY_COLLECTION", "chat_history"
    )

    # History
    # history_file: str = os.getenv("HISTORY_FILE", "history.json")


settings = Settings()

def get_database_url() -> str:
    return (
        f"postgresql+psycopg://{settings.pg_user}:{settings.pg_password}"
        f"@{settings.pg_host}:{settings.pg_port}/{settings.pg_db}"
    )
