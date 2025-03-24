import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Retrieve API key from environment variables
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Embedding model name
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# ChromaDB path
CHROMA_DB_PATH = "./chroma_db"

# AssemblyAI API key
ASSEMBLYAI_API_KEY=os.getenv("ASSEMBLYAI_API_KEY")