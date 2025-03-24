from langchain_huggingface import HuggingFaceEmbeddings  # ✅ Updated import
from langchain_chroma import Chroma  # ✅ Updated import
from config import CHROMA_DB_PATH, EMBEDDING_MODEL

# Load embeddings model
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

# Load ChromaDB
db = Chroma(persist_directory=CHROMA_DB_PATH, embedding_function=embeddings)


def retrieve_similar_chunks(query: str, k=5):
    """Retrieve top-k most relevant document chunks from ChromaDB with metadata."""
    
    results = db.similarity_search_with_score(query, k=k)  # ✅ Retrieves content + score
    
    retrieved_docs = [
        {
            "num": doc.metadata.get("num", "Unknown"),  # ✅ Retrieve stored 'num'
            "content": doc.page_content,
            "score": score  # ✅ Include similarity score
        }
        for doc, score in results
    ]
    
    return retrieved_docs


# Example usage
query = "artificial intelligence in movies"
top_chunks = retrieve_similar_chunks(query, k=5)

# Print results
for chunk in top_chunks:
    print(f"📂 File Num: {chunk['num']}\n🔍 Similarity Score: {chunk['score']:.4f}\n📝 Content: {chunk['content']}\n{'-'*80}")
