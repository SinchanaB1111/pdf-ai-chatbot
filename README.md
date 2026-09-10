# 📄 PDF AI Chatbot

An AI-powered PDF question-answering application using
Retrieval-Augmented Generation (RAG).

## Features

- Upload PDF documents
- Extract PDF text
- Text chunking
- Gemini embeddings
- FAISS vector database
- Semantic document retrieval
- Gemini-powered answers
- Bring Your Own API Key (BYOK)
- Public Streamlit deployment

## Technologies

- Python
- Streamlit
- LangChain
- Gemini
- FAISS
- pdfplumber

## Architecture

PDF → Text Extraction → Chunking → Gemini Embeddings
→ FAISS → Retrieval → Gemini → Answer

## Privacy

Users provide their own Gemini API key for using the
application. The application does not intentionally
persist user API keys. 

