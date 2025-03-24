import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings  # ✅ Updated import
from langchain_chroma import Chroma  # ✅ Updated import

# Embedding model name
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# ChromaDB path
CHROMA_DB_PATH = "./chroma_db"


# Load processed subtitle data
df = pd.read_csv("data/cleaned_subtitles.csv")

# Initialize text splitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

# Prepare chunks for embedding
chunks = []
for _, row in df.iterrows():
    num = str(row["num"])  # Convert to string for metadata
    text = row["cleaned_text"]

    if pd.isna(text) or not text.strip():
        continue  # Skip empty texts

    # Split text into smaller chunks
    split_chunks = text_splitter.create_documents([text], metadatas=[{"num": num}])
    chunks.extend(split_chunks)


def store_embeddings(chunks):
    """Generate embeddings and store them in ChromaDB."""
    
    # Load the embedding model
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    
    # Store the embeddings in ChromaDB
    db = Chroma.from_documents(chunks, embeddings, persist_directory=CHROMA_DB_PATH)
    
    print("✅ Embeddings stored successfully in ChromaDB!")
    return db


def load_vector_store():
    """Load the stored ChromaDB vector store."""
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    db = Chroma(persist_directory=CHROMA_DB_PATH, embedding_function=embeddings)
    return db


# Store embeddings in ChromaDB
store_embeddings(chunks)
