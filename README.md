# RAG-Chatbot
This project is a **Retrieval-Augmented Generation (RAG)** chatbot built with:
- [LangChain](https://www.langchain.com/) for orchestration
- [Chroma](https://www.trychroma.com/) as the vector store
- [Gradio](https://www.gradio.app/) for the web UI

It allows you to **ask questions about your own PDFs**, using embeddings + retrieval for grounded answers.


## Features
- Upload PDFs into a local knowledge base (`data/` folder).
- Split documents into overlapping chunks for context retention.
- Store embeddings in a persistent Chroma database (`chroma_db/`).
- Query via a chat UI with **streaming responses**.


## Setup

### 1. Clone the repo
```bash
git clone https://github.com/your-username/RAG-Chatbot.git
cd RAG-Chatbot
```

### 2. Create and activate a virtual environment
```bash
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Environment Variables
Create a .env file in the project root:
```bash
OPENAI_API_KEY=xxxxxxxxxxxxxxxxxxxx
OPENAI_API_BASE=xxxxxxxxxxxxxxxxxxxx
```


### Project Structure
```bash
├── data/                # Place your PDFs here
├── chroma_db/           # Auto-generated vector DB (created after ingestion)
├── ingest_database.py   # Script to process PDFs into embeddings
├── chatbot.py           # Main chatbot app (Gradio UI)
├── requirements.txt
└── README.md
```

### Steps to run 
1. Put PDFs into the data/ folder.
2. Run the ingestion script:
```bash
python ingest_database.py
```
3. Run the chatbot
```bash
python chatbot.py
```

