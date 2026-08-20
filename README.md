# 🎓 RAG Academic Assistant

An AI-powered academic assistant that allows students to upload academic PDFs, ask questions, generate notes, create MCQs, and practice quizzes using Retrieval-Augmented Generation (RAG).

## 🚀 Features

- 📄 Multiple PDF upload
- 🔎 Semantic document search
- 🧠 RAG-based question answering
- 🤖 Google Gemini integration
- 🗄️ ChromaDB vector database
- 📚 Source and page references
- 📝 Automatic note generation
- 📌 Document summarization
- ❓ MCQ generation
- 🧠 Simple concept explanation
- 🎯 Interactive quiz mode
- 📊 Automatic quiz scoring
- 💬 Chat history

## 🛠️ Tech Stack

- Python
- Streamlit
- LangChain
- Google Gemini
- ChromaDB
- HuggingFace Sentence Transformers
- PyPDF

## 🏗️ Architecture

PDF Documents
↓
PDF Text Extraction
↓
Text Chunking
↓
HuggingFace Embeddings
↓
ChromaDB
↓
Semantic Retrieval
↓
Relevant Context
↓
Google Gemini
↓
Answer + Sources

## ⚙️ Installation

Clone the repository:

```bash
git clone YOUR_GITHUB_REPOSITORY_URL