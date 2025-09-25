# 🚀 ElasticSearch RAG Project

An AI-powered Retrieval-Augmented Generation (RAG) pipeline using Elasticsearch for semantic search, query handling, and grievance dataset processing.

---

## 📂 Project Structure

```text
es_rag/
  ├── config/
  │   ├── __init__.py
  │   └── settings.py
  ├── modules/
  │   ├── __init__.py
  │   ├── es_client.py
  │   ├── formatter.py
  │   ├── interactive.py
  │   ├── parser.py
  │   └── query_builder.py
  ├── resources/
  │   ├── complaints/
  │   │   ├── __init__.py
  │   │   ├── dataset3.json
  │   │   └── schema3.json
  │   ├── consumer/
  │   │   └── __init__.py
  │   └── grivance/
  │       └── __init__.py
  ├── .env
  ├── app2.py
  ├── main.py
  └── requirements.txt
```

---

## ⚙️ Setup

Clone the repo:
```bash
git clone https://github.com/your-username/ElasticSearch-project.git
```

Navigate into the project:
```bash
cd ElasticSearch-project
```

Create a virtual environment (optional but recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Linux/Mac
# venv\Scripts\activate   # On Windows
```

Install dependencies:
```bash
pip install -r requirements.txt
```

---

## ▶️ Usage

Run in **interactive mode**:
```bash
python main.py --interactive
```

Run with a dataset:
```bash
python main.py --json-file resources/complaints/dataset3.json
```

---

## 🛠️ Tech Stack

- **Python** – Core development  
- **Elasticsearch** – Document indexing and search  
- **RAG (Retrieval-Augmented Generation)** – Query answering  
- **OCR & NLP** – Text extraction and parsing  
- **LLMs** – Response generation  

---

## 📌 Notes

- Create a `.env` file with your Elasticsearch and API configs (example keys: `ES_HOST`, `ES_USER`, `ES_PASS`, `OPENAI_API_KEY`, etc.).  
- Update `resources/` to add datasets and schemas.  
- If you want the project tree generated automatically, consider using `tree` or `find` commands in a script.

---


---

## 📜 License

MIT License © 2025 
