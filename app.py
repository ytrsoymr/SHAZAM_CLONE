import streamlit as st
import assemblyai as aai
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from config import CHROMA_DB_PATH, EMBEDDING_MODEL, ASSEMBLYAI_API_KEY
import tempfile

# Initialize AssemblyAI
aai.settings.api_key = ASSEMBLYAI_API_KEY
transcriber = aai.Transcriber()

# Load embeddings model
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

# Load ChromaDB
db = Chroma(persist_directory=CHROMA_DB_PATH, embedding_function=embeddings)

def transcribe_audio(audio_path):
    """Convert audio to text using AssemblyAI."""
    transcript = transcriber.transcribe(audio_path)
    return transcript.text if transcript else ""

def retrieve_similar_chunks(query: str, k=5):
    """Retrieve top-k most relevant document chunks from ChromaDB."""
    results = db.similarity_search(query, k=k)
    return [(doc.metadata['num'], doc.page_content) for doc in results]

# Streamlit UI
st.title("Video Subtitle Search Engine")

uploaded_file = st.file_uploader("Upload an audio/video file", type=["mp3", "wav", "mp4"])
num_results = st.slider("Number of results", 1, 10, 5)

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name
    
    st.write("Transcribing audio...")
    query_text = transcribe_audio(tmp_path)
    st.write("Transcription:", query_text)
    
    if query_text:
        st.write("Searching for relevant subtitles...")
        results = retrieve_similar_chunks(query_text, num_results)
        
        for num, content in results:
            st.markdown(f"**Subtitle ID:** [{num}](https://www.opensubtitles.org/en/subtitles/{num})")
            st.write(content)
            st.markdown("---")
