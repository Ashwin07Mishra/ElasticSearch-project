import argparse
import json
import os
import time
from modules.es_client import initialize_clients, setup_indices
from modules.parser import parse_query
from modules.query_builder import construct_elasticsearch_query
from modules.formatter import format_results
from modules.interactive import process_user_query
from config.settings import QUERY_LOGS_INDEX
from datetime import datetime

def main():
    """Main function with argument parsing."""
    parser = argparse.ArgumentParser(description="Enhanced Elasticsearch NLQ System")
    parser.add_argument("--json-file", default='resources/complaints/dataset3.json', help="Path to JSON data file")
    parser.add_argument("--setup-indices", action="store_true", help="Set up Elasticsearch indices")
    parser.add_argument("--interactive", action="store_true", help="Start interactive mode")
    parser.add_argument("--query", type=str, help="Single query to execute")

    args = parser.parse_args()

    # Default to interactive if no args provided
    if not any(vars(args).values()):
        args.interactive = True

    # Validate data file exists
    if not os.path.exists(args.json_file):
        print(f"❌ Error: Data file '{args.json_file}' not found.")
        return

    # Initialize clients
    if not initialize_clients():
        print("❌ Failed to initialize. Exiting.")
        return

    # Setup indices if requested
    if args.setup_indices:
        success = setup_indices(args.json_file)
        print("✅ Indices setup completed." if success else "❌ Index setup failed.")
        return

    # Process single query mode
    if args.query:
        result = process_user_query(args.query)
        print(f"\n{result}")
        return

    # Start interactive mode
    if args.interactive:
        print("=" * 60)
        print("🚀 Starting interactive query mode")
        print("Type 'quit' or 'exit' to stop")
        print("=" * 60)

        try:
            while True:
                user_query = input("📝 Your query: ").strip()
                if user_query.lower() in {"quit", "exit"}:
                    print("👋 Goodbye!")
                    break
                if not user_query:
                    continue

                start = time.monotonic()
                output = process_user_query(user_query)
                print(f"\n{output}")
                elapsed = (time.monotonic() - start) * 1000
                print(f"⏱️ Processed in {elapsed:.1f} ms\n")

        except KeyboardInterrupt:
            print("\n👋 Interrupted. Goodbye!")

        return

    # If none matched, show help
    parser.print_help()


if __name__ == "__main__":
    main()
