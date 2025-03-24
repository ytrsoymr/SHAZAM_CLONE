# **Enhancing Search Engine Relevance for Video Subtitles (Cloning Shazam) 🎬🔍**  

## **📌 Overview**  
This project focuses on improving **video subtitle search relevance** by leveraging **Natural Language Processing (NLP)** and **machine learning** techniques. It enables users to **search subtitles using audio input**, converting speech to text and retrieving the most relevant subtitle segments using **semantic search**.  

## **🎯 Objective**  
✅ Develop a **search engine for video subtitles** that retrieves relevant subtitles based on user queries.  
✅ Compare **keyword-based** vs. **semantic search engines** to determine the best approach.  
✅ Implement **cosine similarity** to rank relevant subtitle chunks.  
✅ Use **AssemblyAI** for **audio transcription** and **ChromaDB** for **storing embeddings**.  

---

## **🔬 Comparison: Keyword-Based vs. Semantic Search**  

| Search Type       | Description | Pros | Cons |
|------------------|-------------|------|------|
| **Keyword-Based Search** | Relies on **exact keyword matches** between user queries and indexed subtitles. | Faster, simple implementation. | Cannot capture meaning or context. |
| **Semantic Search** | Uses **word embeddings (BERT/SentenceTransformers)** to understand the **meaning and context** of the query. | More **accurate**, retrieves relevant results even when exact words don’t match. | Requires **vectorization and embedding storage**. |

Semantic search provides **better relevance** and is the preferred method for this project.  

---

## **🧠 Core Logic**  

### **1️⃣ Data Preprocessing**  
- Load the **subtitle database**.  
- Apply **text cleaning** (e.g., remove timestamps).  
- Convert subtitles into **vector representations** using:  
  - **TF-IDF / BoW** (Keyword-Based Search)  
  - **BERT (SentenceTransformers)** (Semantic Search)  
- **Chunk subtitles** for better embedding representation.  

### **2️⃣ Query Processing & Retrieval**  
- Take **audio input** (2-minute clip from a movie/TV series).  
- **Transcribe** the audio into text using **AssemblyAI**.  
- Convert the **query text** into an embedding.  
- Compute **cosine similarity** between **query embedding** and **stored subtitle embeddings**.  
- Return **top-k most relevant** subtitle chunks.  

---

## **🛠️ Setup Instructions**  

### **1️⃣ Install Dependencies**  
Ensure **Python 3.8+** is installed. Then, install the required libraries:  
```bash
pip install streamlit assemblyai langchain_huggingface langchain_chroma sentence-transformers numpy scipy dotenv
```
## **2️⃣ Set Up Environment Variables
### Create a .env file and add your AssemblyAI API key:

```
ASSEMBLYAI_API_KEY=your_api_key_here
```
## 3️⃣ Run the Streamlit App
### Launch the application with:
```
streamlit run app.py
```
📌 Future Improvements
Enhance accuracy with fine-tuned embeddings.

Add YouTube video link support for search.

Implement real-time speech-to-subtitle matching.


