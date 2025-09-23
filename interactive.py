import time
import json
from datetime import datetime
from modules.parser import parse_query
from modules.query_builder import construct_elasticsearch_query
from modules.formatter import format_results
from modules.es_client import get_client, execute_elasticsearch_query, save_query_example
from config.settings import QUERY_LOGS_INDEX


def process_user_query(user_query: str, save_example: bool = True) -> str:
    """Main function to process user queries end-to-end, adapted for schema3."""
    try:
        print(f"\\n🔍 Processing query: {user_query}")

        # Parse intent and entities
        parsed_query = parse_query(user_query)
        if not parsed_query:
            return "❌ Could not understand your query. Please try rephrasing."

        print(f"📋 Parsed intent: {parsed_query.get('intent')}")
        print(f"🎯 Filters: {len(parsed_query.get('filters', []))}")

        # Construct Elasticsearch query
        es_query = construct_elasticsearch_query(parsed_query)
        print("🔧 Executing Elasticsearch query:")
        print(json.dumps(es_query, indent=2))

        # Execute query
        success, response = execute_elasticsearch_query(es_query)
        if not success:
            return f"❌ Query execution failed: {response}"

        # Log query and ES query JSON into dedicated logs index for analytics
        try:
            log_doc = {
                "user_query": user_query,
                "es_query": es_query,
                "timestamp": datetime.utcnow().isoformat()
            }
            client = get_client()
            client.index(index=QUERY_LOGS_INDEX, body=log_doc)
            print("✅ Logged query and ES JSON for analysis.")
        except Exception as e:
            print(f"⚠️ Failed to log query example: {e}")

        # Format results
        formatted_results = format_results(response, parsed_query["intent"], user_query)

        # Save successful example for RAG learning
        if save_example and success:
            save_query_example(user_query, es_query, parsed_query["intent"], parsed_query.get("entities", {}))

        return formatted_results

    except Exception as e:
        print(f"❌ Error processing query: {e}")
        return f"Error processing your query: {str(e)}"


def interactive_mode():
    """Interactive query mode for testing."""
    print("=" * 60)
    print("🚀 INTERACTIVE QUERY MODE (Schema3 & Dataset3)")
    print("=" * 60)
    print("Enter your queries below. Type 'quit' or 'exit' to stop.")
    print("Examples for schema3:")
    print("• 'show all complaints from Flipkart'")
    print("• 'count complaints by status'") 
    print("• 'show id, fullName for open complaints'")
    print("• 'list unique userType values'")
    print("-" * 60)
    
    while True:
        try:
            user_input = input("\\n📝 Your query: ").strip()
            if user_input.lower() in {"quit", "exit", "q"}:
                print("👋 Goodbye!")
                break
            if not user_input:
                continue
            
            start_time = time.time()
            result = process_user_query(user_input)
            end_time = time.time()
            
            print(f"\\n{result}")
            print(f"\\n⏱️ Query processed in {end_time - start_time:.2f} seconds")
        except KeyboardInterrupt:
            print("\\n👋 Interrupted. Goodbye!")
            break
        except Exception as e:
            print(f"❌ An error occurred: {e}")