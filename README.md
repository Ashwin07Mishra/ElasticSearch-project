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
