# ElasticSearch-project
Modular version of es-rag project
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

Setup

Clone the repo
git clone https://github.com/your-username/ElasticSearch-project.git

Install dependencies
pip install -r requirements.txt

Set environment variables
Copy the example file and update values:
cp env.example .env
Edit .env with your Elasticsearch and LLM settings.

Dataset & Schema
Default dataset: resources/complaints/dataset3.json

Default schema: resources/complaints/schema3.json

You can replace these files with your own dataset/schema if needed.

Running the Project

Interactive mode:
python main.py --interactive

Run with dataset:
python main.py --json-file resources/complaints/dataset3.json

Run with schema:
python main.py --schema-file resources/complaints/schema3.json








