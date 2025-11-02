# Retrieval-Augmented Generation (RAG) Project

This project implements a **Retrieval-Augmented Generation (RAG)** system that combines a language model with vector database retrieval to provide more accurate and context-aware responses.  
It uses **Pinecone** as a vector store and integrates with **GROQ API** for LLM-powered reasoning.

---

## 🚀 Features

- **RAG Pipeline**: Enhances answers with retrieved knowledge from a vector database.  
- **Pinecone Integration**: Stores and retrieves embeddings efficiently.  
- **SQLite Support**: Local database (`1_multistep_rag.sqlite`) for multi-step workflows.  
- **Environment-Based Config**: API keys and configs managed via `.env`.  
- **Extensible**: Easily adaptable for different datasets and tasks.  

---
## Requirements

- Python 3.12
- Virtual environment recommended
- `GROQ_API_KEY` — required to use Groq's ChatGroq. Put it in a `.env` file or export it in your environment.

---

## 🛠️ Setup

### 1️⃣ Clone the repository
```bash
git clone https://github.com/keshabshrestha007/langgraph-multistep-rag.git
```
```bash
cd langgraph-multistep-rag
```


### 2️⃣ Create and activate a virtual environment
```bash
python -m venv venv
```
On Linux/Mac
```bash
source venv/bin/activate
```
On Windows
```bash    
venv\Scripts\activate 
```       


### 3️⃣ Install dependencies
```bash
pip install -r requirements.txt
```

### 4️⃣ Add your Groq API key.

```bash
copy .env.example .env
# edit .env to set GROQ_API_KEY (no surrounding quotes preferred)
```
### 5️⃣ Run the Streamlit app:

```bash
streamlit run streamlit_app.py
```

---

## 📂 Project Structure
```bash
.
├── models
|   └── llms.py           
├── schema
|    └── validator.py
├── requirements.txt
├── streamlit_app.py            
├── multistep_rag_system1.py
├── tools.py
├── venv         
├── .env
├── .env.example 
├── 1_multistep_rag.sqlite 
├── .gitignore            
└── README.md
```
---

## ⚙️ Development Notes
- Temporary files, cache, and local environment are ignored via .gitignore.

- SQLite database (1_multistep_rag.sqlite) supports multi-step conversation memory.

- API keys must not be committed to GitHub.



## 📝 License
This project is open-source. Use at your own risk, and keep your API keys safe.


