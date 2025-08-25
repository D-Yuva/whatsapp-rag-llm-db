# whatsapp-rag-llm-db

A local, privacy-first **WhatsApp RAG** assistant.  
It embeds your WhatsApp `.txt` exports, lets an LLM plan **multi-queries** from a topic file (e.g., “ACC” portions), retrieves the most relevant messages via **vector search**, and answers **strictly** from those lines.

- **Local-first**: uses **Ollama** by default (no API key).
- **Simple storage**: SQLite + Sentence-Transformers embeddings.
- **Topic-aware planning**: LLM reads `topics/topics.json` to expand/focus queries.
- **Verbatim receipts**: includes exact message lines (with timestamps/senders).

---

## 1) Requirements

- macOS or Linux (Apple Silicon works great)
- **Python 3.10+**
- **Ollama** (for local LLMs)
- WhatsApp **.txt** chat exports

## 2) Install (virtualenv + requirements)

    git clone <your-repo-url> whatsapp-rag-llm-db
    cd whatsapp-rag-llm-db

    python -m venv .venv
    source .venv/bin/activate

# install core dependencies
    pip install -U pip
    pip install -r requirements.txt


## 3) Folder Arch

├── requirements.txt
├── src/
│   ├── chat_llm.py               # main app (LLM planning + retrieval + answer)
│   ├── emb_store.py              # SQLite vector store (embeddings + message meta)
│   └── whatsapp_chat_cleaning.py # parses WhatsApp .txt exports → rows
├── topics/
│   └── topics.json               # your course/subject portions (editable)
├── vectors/                      # auto-created SQLite DB (embeddings.sqlite)
├── exports/                      # put WhatsApp .txt exports here 
└── .env

## 4) Ollama (local LLM) — install & use

brew install ollama
ollama serve     # if it says "address already in use", it's already running
ollama pull qwen2.5:14b-instruct

## 5) ENV File
    # LLM (Ollama) — local default
    LLM_BACKEND=ollama
    OLLAMA_BASE=http://localhost:11434
    OLLAMA_MODEL=qwen2.5:14b-instruct
    OLLAMA_THINK=false            # important: disable hidden "thinking" channel
    OLLAMA_TIMEOUT_SEC=600        # allow slow first token if model is large

    # Embeddings + storage
    VECTORS_DIR=vectors
    EMB_MODEL=intfloat/multilingual-e5-small

    # Topics file (see next section)
    TOPICS_PATH=topics/topics.json

## Topics
topics.json
    {
    "ACC": {
        "items": [
        "Simple Sieve",
        "Segmented Incremental Sieve",
        "Euler's phi Algorithm",
        "Complexity analysis",
        "Algorithm definitions",
        "Code snippets",
        "Rounding decimals",
        "Pattern printing",
        "Brute force vs optimized"
        ]
    },
    "Compiler Design": {
        "items": ["Lexical analysis", "Parsing", "LR(0)/SLR(1)", "Code generation", "Optimization"]
    }
    }

## 6) Run
    source .venv/bin/activate
    python -m src.chat_llm --data "exports/_chat.txt"