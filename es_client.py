import urllib3
from elasticsearch8 import Elasticsearch
from elasticsearch8.helpers import bulk
from sentence_transformers import SentenceTransformer
from typing import Optional, Tuple, Dict, Any
import json
import os
import time
import pandas as pd
from datetime import datetime
import json
import os
from config.settings import INDEX_NAME

from config.settings import ES_HOST, ES_USER, ES_PASS, INDEX_NAME, EXAMPLES_INDEX, EMBEDDING_MODEL, VISUALIZATION_DIR

# Disable SSL warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Global client variables
es_client: Optional[Elasticsearch] = None
embedding_model: Optional[SentenceTransformer] = None


def initialize_clients() -> bool:
    """Initialize Elasticsearch and embedding model clients."""
    global es_client, embedding_model
    try:
        print("🔄 Initializing Elasticsearch client...")
        es_client = Elasticsearch(
            ES_HOST,
            basic_auth=(ES_USER, ES_PASS),
            verify_certs=False,
            request_timeout=60
        )
        
        if not es_client.ping():
            raise ConnectionError("Could not connect to Elasticsearch.")
        
        print(f"✅ Connected to Elasticsearch cluster: {es_client.info()['cluster_name']}")
        
        print(f"🔄 Loading embedding model ({EMBEDDING_MODEL})...")
        embedding_model = SentenceTransformer(EMBEDDING_MODEL)
        print("✅ Embedding model loaded.")
        
        return True
    except Exception as e:
        print(f"❌ Initialization Failed: {e}")
        return False


def get_client() -> Elasticsearch:
    """Get the initialized Elasticsearch client."""
    if es_client is None:
        raise RuntimeError("Elasticsearch client not initialized. Call initialize_clients() first.")
    return es_client


def get_embedding_model() -> SentenceTransformer:
    """Get the initialized embedding model."""
    if embedding_model is None:
        raise RuntimeError("Embedding model not initialized. Call initialize_clients() first.")
    return embedding_model


def execute_elasticsearch_query(es_query: Dict[str, Any]) -> Tuple[bool, Any]:
    """Execute Elasticsearch query with error handling."""
    try:
        if not es_client:
            return False, "Elasticsearch client not initialized"
        
        response = es_client.search(index=INDEX_NAME, body=es_query, request_timeout=30)
        return True, response
    except Exception as e:
        print(f"❌ Elasticsearch query execution failed: {e}")
        return False, str(e)


def setup_indices(json_file_path: str) -> bool:
    """Set up Elasticsearch indices with mappings for schema3."""
    if not es_client:
        return False
    
    print("--- Setting up Elasticsearch Indices ---")
    try:
        # Delete existing index if it exists
        if es_client.indices.exists(index=INDEX_NAME):
            es_client.indices.delete(index=INDEX_NAME)
            print(f"🗑️ Deleted existing data index: {INDEX_NAME}.")
        
        # Define mappings with case-insensitive normalizer
        mappings = {
            "properties": {
                "id": {"type": "keyword"},
                "complaintDetails": {
                    "type": "text",
                    "analyzer": "standard"
                },
                "complaintDetails_embedding": {
                    "type": "dense_vector",
                    "dims": 384,
                    "index": True,
                    "similarity": "cosine"
                },
                "userId": {"type": "integer"},
                "fullName": {
                    "type": "text",
                    "analyzer": "standard",
                    "fields": {"keyword": {"type": "keyword"}}
                },
                "CityName": {
                    "type": "text",
                    "analyzer": "standard",
                    "fields": {"keyword": {"type": "keyword"}}
                },
                "stateName": {
                    "type": "text",
                    "analyzer": "standard",
                    "fields": {"keyword": {"type": "keyword"}}
                },
                "country": {
                    "type": "keyword",
                    "normalizer": "lowercase_normalizer"
                },
                "userType": {
                    "type": "keyword",
                    "normalizer": "lowercase_normalizer"
                },
                "status": {"type": "integer"},
                "complaintRegDate": {"type": "date"},
                "updationDate": {"type": "date"},
                "complaintType": {
                    "type": "keyword",
                    "normalizer": "lowercase_normalizer"
                },
                "complaintMode": {
                    "type": "keyword",
                    "normalizer": "lowercase_normalizer"
                },
                "categoryCode": {"type": "integer"},
                "companyName": {
                    "type": "text",
                    "analyzer": "standard",
                    "fields": {"keyword": {"type": "keyword"}}
                },
                "complaintStatus": {
                    "type": "keyword",
                    "normalizer": "lowercase_normalizer"
                },
                "companyStatus": {
                    "type": "keyword",
                    "normalizer": "lowercase_normalizer"
                },
                "lastUpdationDate": {"type": "keyword"},
                "processing_time_days": {"type": "integer"}
            }
        }
        
        # Settings with lowercase normalizer
        settings = {
            "index.max_ngram_diff": 8,
            "analysis": {
                "normalizer": {
                    "lowercase_normalizer": {
                        "type": "custom",
                        "filter": ["lowercase", "trim"]
                    }
                },
                "analyzer": {
                    "ngram_analyzer": {
                        "type": "custom",
                        "tokenizer": "standard",
                        "filter": ["lowercase", "ngram_filter"]
                    }
                },
                "filter": {
                    "ngram_filter": {
                        "type": "ngram",
                        "min_gram": 2,
                        "max_gram": 10
                    }
                }
            }
        }
        
        index_config = {
            "settings": settings,
            "mappings": mappings
        }
        
        es_client.indices.create(index=INDEX_NAME, body=index_config)
        print(f"✅ Created data index: {INDEX_NAME} with enhanced mappings.")
        
        # Ingest data
        ingest_data_from_json(json_file_path)
        
        # Setup examples index
        examples_mapping = {
            "properties": {
                "original_query": {"type": "text"},
                "es_query": {"type": "keyword", "index": False},
                "embedding": {
                    "type": "dense_vector",
                    "dims": 384,
                    "index": True,
                    "similarity": "cosine"
                },
                "timestamp": {"type": "date"},
                "intent": {"type": "keyword"},
                "entities": {"type": "object"}
            }
        }
        
        if not es_client.indices.exists(index=EXAMPLES_INDEX):
            es_client.indices.create(index=EXAMPLES_INDEX, body={"mappings": examples_mapping})
            print(f"✅ Created RAG examples index: {EXAMPLES_INDEX}.")
        else:
            print(f"ℹ️ RAG examples index {EXAMPLES_INDEX} exists.")
        
        # Create visualizations directory
        if not os.path.exists(VISUALIZATION_DIR):
            os.makedirs(VISUALIZATION_DIR)
            print(f"📁 Created visualizations directory: {VISUALIZATION_DIR}")
        
        return True
    except Exception as e:
        print(f"❌ Index setup failed: {e}")
        return False


def ingest_data_from_json(json_file_path: str):
    """Enhanced data ingestion adapted for schema3 structure."""
    if not embedding_model:
        print("❌ Embedding model not loaded. Cannot ingest data.")
        return
    
    if not os.path.exists(json_file_path):
        print(f"❌ File not found: {json_file_path}")
        return
    
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    actions = []
    date_fields = ['complaintRegDate', 'updationDate']
    
    print("🔄 Generating embeddings and preparing documents for ingestion...")
    
    for i, record in enumerate(data):
        try:
            # Process date fields
            for field in date_fields:
                if field in record and record[field]:
                    try:
                        if field == 'updationDate' and isinstance(record[field], str):
                            if ' ' in record[field]:
                                record[field] = record[field].split(' ')[0] + 'T' + record[field].split(' ')[1]
                        record[field] = pd.to_datetime(record[field]).isoformat()
                    except (ValueError, TypeError):
                        record[field] = None
            
            # Calculate processing time
            reg_date, upd_date = record.get('complaintRegDate'), record.get('updationDate')
            if reg_date and upd_date:
                try:
                    reg_dt = pd.to_datetime(reg_date)
                    upd_dt = pd.to_datetime(upd_date)
                    record['processing_time_days'] = int((upd_dt - reg_dt).days)
                except Exception:
                    record['processing_time_days'] = None
            else:
                record['processing_time_days'] = None
            
            # Generate embeddings for complaintDetails
            if 'complaintDetails' in record and record['complaintDetails']:
                try:
                    embedding = embedding_model.encode(record['complaintDetails']).tolist()
                    record['complaintDetails_embedding'] = embedding
                except Exception as e:
                    print(f"⚠️ Could not generate embedding for record {i}: {e}")
            
            actions.append({
                "_index": INDEX_NAME,
                "_source": record
            })
        except Exception as e:
            print(f"❌ Error processing record {i}: {e}")
            continue
    
    # Bulk index documents
    if actions:
        try:
            success, failed = bulk(es_client, actions, chunk_size=100, request_timeout=60)
            print(f"✅ Ingested {success} documents. Failed: {len(failed)}.")
        except Exception as e:
            print(f"❌ Bulk ingestion failed: {e}")


def save_query_example(user_query: str, es_query: Dict[str, Any], intent: str, entities: Dict[str, Any]):
    """Save successful query examples for RAG learning."""
    try:
        if not embedding_model:
            return
        
        embedding = embedding_model.encode(user_query).tolist()
        
        example_doc = {
            "original_query": user_query,
            "es_query": json.dumps(es_query),
            "embedding": embedding,
            "timestamp": datetime.now().isoformat(),
            "intent": intent,
            "entities": entities
        }
        
        es_client.index(index=EXAMPLES_INDEX, body=example_doc)
        print("✅ Query example saved for future learning.")
    except Exception as e:
        print(f"⚠️ Could not save query example: {e}")


def get_database_schema() -> dict:
    schema_path = os.path.join(os.path.dirname(__file__), '..', 'resources/complaints/schema3.json')
    if not os.path.exists(schema_path):
        raise FileNotFoundError(f"Schema file not found: {schema_path}")
    with open(schema_path, "r") as f:
        schema_data = json.load(f)
    return {INDEX_NAME: schema_data.get("complaints", {})}