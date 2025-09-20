import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    PDF_PATH = "data/merck_manual.pdf"
    VECTOR_DB_PATH = "data/merck_vector_db"
    EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200
    RETRIEVER_K = 5
    GROQ_MODEL = "mixtral-8x7b-32768"  # Or "llama3-70b-8192"
    MAX_TOKENS = 500
    TEMPERATURE = 0.7
    TOP_P = 0.9

if not Config.GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found in .env")