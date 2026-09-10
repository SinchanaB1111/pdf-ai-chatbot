# 📄 PDF AI Chatbot

An AI-powered PDF question-answering chatbot built using
Retrieval-Augmented Generation (RAG), Gemini, LangChain,
and FAISS.

## 🚀 Features

- Upload PDF documents
- Extract text from PDFs
- Intelligent text chunking
- Gemini embeddings
- FAISS vector database
- Semantic document retrieval
- Gemini-powered question answering
- Bring Your Own API Key (BYOK)
- Public web deployment using Streamlit

## 🧠 Architecture

PDF
↓
Text Extraction
↓
Text Chunking
↓
Gemini Embeddings
↓
FAISS Vector Database
↓
User Question
↓
Relevant Context Retrieval
↓
Gemini LLM
↓
Answer

## 🛠️ Technologies

- Python
- Streamlit
- LangChain
- Google Gemini
- FAISS
- pdfplumber

## 🔑 Bring Your Own API Key

Users provide their own Gemini API key through the
application interface.

The application does not intentionally persist user
API keys.

## ▶️ Run Locally

Install the required packages:

```bash
pip install -r requirements.txt
