# 📚 AI-Powered Document Q&A System (RAG with Flask + LangChain)

---

## 🚀 Project Overview

This project is an **AI-powered Retrieval-Augmented Generation (RAG) system** that allows users to upload PDF documents and ask questions based strictly on the uploaded content.

The system retrieves relevant text chunks from the document using vector search and generates accurate answers using a Large Language Model (LLM).  
It is designed to ensure responses are **strictly grounded in the document**, preventing hallucinations.

---

## 🎯 Objective

To build a web application that:
- Allows PDF upload
- Enables question answering from uploaded documents
- Ensures answers are strictly based only on document content
- Avoids hallucination and external knowledge usage

---

## ✨ Features

### 📄 PDF Upload
- Users can upload PDF documents
- File type validation implemented

### 🧠 Text Processing
- Extracts text from uploaded PDFs
- Splits text into chunks for efficient retrieval
- Generates embeddings for semantic search

### 🔍 AI Question Answering
- Answers questions strictly based on document context
- Prevents hallucination
- Returns:
  "Not available in document."
  when information is missing

### 💬 Chat Interface
- Interactive web-based chat system
- Displays user questions and AI responses
- Simple and user-friendly UI using Flask

---

## 🧠 AI Approach Used

This system uses **Retrieval-Augmented Generation (RAG)**:

1. PDF is uploaded and text is extracted
2. Text is split into chunks
3. Embeddings are generated using Sentence Transformers
4. ChromaDB stores vector embeddings
5. User query is converted into vector form
6. Most relevant document chunks are retrieved
7. LLM generates response using only retrieved context

---

## 🧾 Prompt Design

The system uses a strict prompt to ensure grounded answers:

- The AI is instructed to use ONLY the provided context
- External knowledge is strictly disabled
- If answer is not found, it responds:
  "Not available in document."
- No assumptions or hallucinated answers are allowed

This ensures factual and document-based responses only.

---

## ⚠️ Hallucination Handling

To prevent incorrect or made-up answers:

- The model is restricted to document context only
- Explicit instruction to avoid external knowledge
- Mandatory fallback response:
  "Not available in document."
- If context does not contain answer, system refuses to guess

---

## 🛠️ Tech Stack

- Python  
- Flask  
- LangChain  
- Mistral AI (via LangChain Mistralai)  
- ChromaDB (Vector Database)  
- Sentence Transformers  
- PyPDF  
- HTML/CSS (Frontend Templates)  

---

## 📁 Project Structure
RAG-Deploy/
│

├── app.py # Flask backend

├── database.py # RAG pipeline logic

├── requirements.txt # Dependencies

├── .gitignore

├── templates/

│ └── index.html # Chat UI

├── chroma_db/ # Vector database storage

## ⚙️ Setup Instructions

```bash
1. git clone https://github.com/JayalakshmiT-2708/RAG-Deploy.git
2. cd RAG-Deploy
3. python -m venv venv
4. venv\Scripts\activate   # Windows
5. pip install -r requirements.txt
6. python app.py
7. Open: http://127.0.0.1:5000
