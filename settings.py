import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Elasticsearch Configuration
ES_HOST = os.getenv("ELASTICSEARCH_HOST", "https://localhost:9200")
ES_USER = os.getenv("ELASTICSEARCH_USER", "elastic")
ES_PASS = os.getenv("ELASTICSEARCH_PASSWORD", "change_me")

# Index Configuration
INDEX_NAME = os.getenv("INDEX_NAME", "complaints")
QUERY_LOGS_INDEX = os.getenv("QUERY_LOGS_INDEX", "query_logs")
EXAMPLES_INDEX = os.getenv("EXAMPLES_INDEX", "es_examples")

# LLM Configuration
LLAMA_API_URL = os.getenv("LLAMA_API_URL", "https://openrouter.ai/api/v1/chat/completions")
LLAMA_API_KEY = os.getenv("LLAMA_API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME", "meta-llama/llama-3-70b-instruct")

# Embedding Configuration
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

# Similarity Threshold
RAG_SIMILARITY_THRESHOLD = float(os.getenv("RAG_SIMILARITY_THRESHOLD", "0.80"))

# Visualization Directory
VISUALIZATION_DIR = os.getenv("VISUALIZATION_DIR", "visualizations")