import os

from dotenv import load_dotenv


load_dotenv()


APP_NAME = os.getenv(
    "APP_NAME",
    "AI-Powered Energy Consumption Optimization Assistant"
)

APP_VERSION = os.getenv(
    "APP_VERSION",
    "1.0.0"
)

RAW_DATA_PATH = os.getenv(
    "RAW_DATA_PATH",
    "data/raw/energy_consumption.csv"
)

PROCESSED_DATA_PATH = os.getenv(
    "PROCESSED_DATA_PATH",
    "data/processed/energy_cases.csv"
)

KNOWLEDGE_BASE_PATH = os.getenv(
    "KNOWLEDGE_BASE_PATH",
    "data/knowledge_base/energy_best_practices.csv"
)

CHROMA_DB_PATH = os.getenv(
    "CHROMA_DB_PATH",
    "chroma_db"
)

EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "all-MiniLM-L6-v2"
)

DEFAULT_TOP_K = int(
    os.getenv("DEFAULT_TOP_K", "5")
)

OPENROUTER_API_KEY = os.getenv(
    "OPENROUTER_API_KEY",
    ""
)

OPENROUTER_MODEL = os.getenv(
    "OPENROUTER_MODEL",
    "openrouter/free"
)

OPENROUTER_BASE_URL = (
    "https://openrouter.ai/api/v1"
)