from typing import Dict, Any
import pandas as pd
from tabulate import tabulate


def format_results(response: Dict[str, Any], intent: str, user_query: str) -> str:
    """Format Elasticsearch results for display, adapted for schema3."""
    try:
        if intent == "aggregate_data":
            if "aggregations" in response and "grouped_data" in response["aggregations"]:
                buckets = response["aggregations"]["grouped_data"]["buckets"]
                if not buckets:
                    return "No data found for the specified criteria."

                headers = ["Value", "Count"]
                rows = []
                for bucket in buckets:
                    rows.append([bucket["key"], bucket["doc_count"]])

                # Add stats if available
                stats_info = ""
                if "stats" in response.get("aggregations", {}):
                    stats = response["aggregations"]["stats"]
                    stats_info = f"\\n\\nStatistics: Min: {stats.get('min', 'N/A')}, Max: {stats.get('max', 'N/A')}, Avg: {stats.get('avg', 'N/A'):.2f}"

                return f"**Aggregation Results:**\\n```\\n{tabulate(rows, headers=headers, tablefmt='grid')}\\n```{stats_info}"
            else:
                return "No aggregation data found."

        elif intent == "list_distinct_values":
            if "aggregations" in response and "distinct_values" in response["aggregations"]:
                buckets = response["aggregations"]["distinct_values"]["buckets"]
                if not buckets:
                    return "No distinct values found."

                values = [bucket["key"] for bucket in buckets]
                return f"**Distinct Values ({len(values)} total):**\\n" + "\\n".join([f"• {value}" for value in values[:50]])
            else:
                return "No distinct values found."

        else:  # filter_data or filter_data_columns
            hits = response.get("hits", {}).get("hits", [])
            if not hits:
                return "No results found for your query."

            total = response.get("hits", {}).get("total", {})
            if isinstance(total, dict):
                total_count = total.get("value", 0)
            else:
                total_count = total

            # Format results based on the data structure
            headers = []
            rows = []

            # Get first hit to determine available fields
            first_hit = hits[0]["_source"]

            # Show all fields from the source for full visibility
            headers = list(first_hit.keys())
            # Remove embedding field from display as it's not human-readable
            headers = [h for h in headers if 'embedding' not in h.lower()]

            for hit in hits[:20]:  # Limit to 20 results for display
                source = hit["_source"]
                row = []
                for header in headers:
                    value = source.get(header, "N/A")
                    # Format specific fields
                    if header in ["complaintRegDate", "updationDate"] and value:
                        try:
                            value = pd.to_datetime(value).strftime("%Y-%m-%d")
                        except:
                            pass
                    elif header == "status":
                        value = "Open" if value == 0 else "Closed" if value == 1 else value
                    elif isinstance(value, str) and len(value) > 50:
                        value = value[:47] + "..."
                    row.append(str(value))
                rows.append(row)

            table = tabulate(rows, headers=headers, tablefmt="grid", maxcolwidths=[None] * len(headers))

            return f"**Query Results** (Showing {len(rows)} of {total_count} total results):\\n```\\n{table}\\n```"

    except Exception as e:
        print(f"❌ Error formatting results: {e}")
        return f"Error formatting results: {str(e)}"