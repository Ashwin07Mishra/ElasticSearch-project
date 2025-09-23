import re
import json
import requests
from typing import Dict, Any, Optional, List, Set, Tuple
from config.settings import LLAMA_API_URL, LLAMA_API_KEY, MODEL_NAME
from modules.es_client import get_database_schema

# Entity keywords for schema3
ENTITY_KEYWORDS = {
    "status": ["0", "1", "0 (Open)", "1 (Closed)"],  # based on dataset context
    "complaintType": [
        "Delayed Service", "Overcharging", "Misleading Advertisement", 
        "Defective Product", "Warranty Issue", "Refund not processed", 
        "Service Denial", "Fraudulent Activity"
    ],
    "complaintMode": ["Email", "Phone", "Website", "NCHAPP"],
    "companyStatus": ["Pending", "Resolved", "Fraud", "Escalated", "Responded"],
    "complaintStatus": ["Open", "Closed", "Escalated", "In Progress"],
    "userType": ["Consumer", "Wholesaler", "Retailer"],
    "country": ["India"],
    "columns": [
        "id", "complaintDetails", "userId", "fullName", "CityName", "stateName", 
        "country", "userType", "status", "complaintRegDate", "updationDate", 
        "complaintType", "complaintMode", "categoryCode", "companyName", 
        "complaintStatus", "companyStatus", "lastUpdationDate"
    ]
}

# Intent patterns
INTENT_PATTERNS = {
    "count": ["count", "how many", "number of", "total", "sum", "min", "max", "avg", "average"],
    "list": ["list", "show", "get", "find", "retrieve", "display"],
    "aggregate": ["group by", "breakdown", "distribution", "by company", "by type"],
    "distinct": ["unique", "distinct", "different", "all possible"],
    "sort": ["sort", "order", "arrange", "ranked", "top", "bottom"],
    "filter": ["from", "in", "where", "with", "having", "contains"],
    "columns": ["show only", "select", "display only", "columns", "fields"]
}


def call_openrouter_api(prompt: str):
    """Call OpenRouter API for LLM processing."""
    headers = {
        "Authorization": f"Bearer {LLAMA_API_KEY}",
        "Content-Type": "application/json",
    }
    
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": "You are an expert query analysis system."},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 1024,
        "temperature": 0.1
    }
    
    try:
        response = requests.post(LLAMA_API_URL, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        
        response_data = response.json()
        if "choices" in response_data and len(response_data["choices"]) > 0:
            return response_data["choices"][0]["message"]["content"]
        else:
            print(f"❌ Unexpected API response structure: {response_data}")
            return None
    except requests.exceptions.HTTPError as http_err:
        print(f"❌ HTTP error occurred: {http_err}")
        try:
            error_details = response.json()
            print(f"API error details: {error_details}")
        except ValueError:
            print(f"Response content: {response.text}")
        return None
    except requests.exceptions.RequestException as req_err:
        print(f"❌ Request error occurred: {req_err}")
        return None


import re
from typing import Dict, Any

def normalize_query(query: str) -> str:
    query = query.lower().strip()

    # Corrected safe patterns without invalid group references
    replacements = [
        (r"from (\w+)", r"from \1"),
        (r"show all complaints", "show all complaints"),
        (r"list all complaints", "list all complaints"),
        # Add more normalized patterns suited to your use case
    ]
    for pattern, repl in replacements:
        query = re.sub(pattern, repl, query)
    return query

def fallback_parse_intent(user_query: str, entities: Dict[str, Any], original_query: str) -> Dict[str, Any]:
    try:
        lower_query = user_query.lower().strip()
        # Your full fallback intent parsing logic from app10.py goes here,
        # including entity handling, filters, columns, grouping etc.
        # This should replicate the fallback parsing you had earlier.
        
        # For example placeholder:
        result = {
            "intent": "filter_data",
            "filters": [],
            "columns": []
        }
        # Implement full original fallback code here

        return result
    except Exception as e:
        print(f"❌ Fallback parsing failed: {e}")
        return {"intent": "filter_data", "filters": [], "columns": []}

def parse_intent_and_entities_enhanced(user_query: str) -> Dict[str, Any]:
    try:
        lower_query = user_query.strip().lower()

        if lower_query.startswith("show all") or lower_query.startswith("list all"):
            norm_query = normalize_query(user_query)
            # Use your entity extraction logic etc.
            # build filters based on entities
            # return parsed dict
            pass
        
        # Use external LLM api call or original parsing logic here

    except Exception as e:
        print(f"❌ Error in parse_intent_and_entities_enhanced: {e}")
        try:
            norm_query = normalize_query(user_query)
            # entities = extract_entities_with_disambiguation(norm_query)
            return fallback_parse_intent(norm_query, {}, user_query)
        except Exception as fallback_error:
            print(f"❌ Fallback parsing failed also: {fallback_error}")
            return None

def parse_query(user_query: str) -> Dict[str, Any]:
    return parse_intent_and_entities_enhanced(user_query)


def extract_entities_with_disambiguation(text: str) -> Dict[str, Any]:
    """Extract entities based on schema3 fields."""
    entities = {}
    lower_text = text.lower()
    
    for field, keywords in ENTITY_KEYWORDS.items():
        if field == "columns":
            continue
        
        found_keywords = []
        for kw in keywords:
            if isinstance(kw, str):
                if re.search(rf"\\b{re.escape(kw.lower())}\\b", lower_text):
                    found_keywords.append(kw)
                elif len(str(kw).split()) > 1 and any(part.lower() in lower_text for part in str(kw).split()):
                    found_keywords.append(kw)
            else:
                if str(kw) in lower_text:
                    found_keywords.append(kw)
        
        if found_keywords:
            entities[field] = list(set(found_keywords))
    
    # Extract dates, numbers, and comparisons
    date_patterns = [r"(\\d{4}-\\d{1,2}-\\d{1,2})", r"(\\d{1,2}/\\d{1,2}/\\d{4})", r"(\\d{1,2}-\\d{1,2}-\\d{4})"]
    dates = []
    for pattern in date_patterns:
        dates.extend(re.findall(pattern, lower_text))
    if dates:
        entities["dates"] = dates
    
    numbers = re.findall(r"\\b(\\d+)\\b", lower_text)
    ranges = re.findall(r"(\\d+-\\d+)", lower_text)
    if numbers:
        entities["numbers"] = numbers
    if ranges:
        entities["ranges"] = ranges
    
    # Extract comparisons
    comparisons = []
    gt_match = re.search(r"(more than|greater than|above|over)\\s+(\\w+)", lower_text)
    if gt_match:
        comparisons.append(("gt", gt_match.group(2)))
    
    lt_match = re.search(r"(less than|under|below)\\s+(\\w+)", lower_text)
    if lt_match:
        comparisons.append(("lt", lt_match.group(2)))
    
    if comparisons:
        entities["comparisons"] = comparisons
    
    # Detect logical operators
    logical_operators = {
        "and": ["and", ",", "plus", "also"],
        "or": ["or", "|", "either"],
        "not": ["not", "except", "excluding", "but not", "without"]
    }
    
    detected_operators = []
    for op_type, keywords in logical_operators.items():
        for keyword in keywords:
            if keyword in lower_text:
                detected_operators.append(op_type)
                break
    
    if detected_operators:
        entities["logical_operators"] = list(set(detected_operators))
    
    return entities


def classify_query_intent(user_query: str) -> str:
    """Classify query intent using pattern matching."""
    lower_query = user_query.lower()
    
    if any(pattern in lower_query for pattern in INTENT_PATTERNS["count"]):
        return "aggregate_data"
    if any(pattern in lower_query for pattern in INTENT_PATTERNS["distinct"]):
        return "list_distinct_values"
    if any(pattern in lower_query for pattern in INTENT_PATTERNS["sort"]):
        return "filter_data_sorted"
    if any(pattern in lower_query for pattern in INTENT_PATTERNS["columns"]):
        return "filter_data_columns"
    
    return "filter_data"


def ensure_conditional_columns(result, query):
    """Helper function to ensure conditional columns are included only when appropriate."""
    filters = result.get("filters", [])
    columns = result.get("columns", [])
    valid_columns = [col["name"] for col in get_database_schema()["complaints"]["columns"] if "embedding" not in col["name"]]
    
    if result.get("intent") == "filter_data_columns":
        for f in filters:
            field = f.get("field")
            if field in valid_columns and field not in columns:
                columns.append(field)
    
    # Handle show all queries
    lower_query = query.lower().strip()
    if lower_query.startswith("show all") or lower_query.startswith("show data") and not any(p in lower_query for p in INTENT_PATTERNS["columns"]):
        columns = []
    
    result["columns"] = columns
    return result




def parse_intent_and_entities_enhanced(user_query: str) -> Optional[Dict[str, Any]]:
    """Enhanced intent parsing adapted for schema3 with column selection and conditional column inclusion."""
    try:
        lower_query = user_query.strip().lower()
        
        # Handle show all queries explicitly
        if lower_query.startswith("show all") or lower_query.startswith("list all"):
            normalized_query = normalize_query(user_query)
            entities = extract_entities_with_disambiguation(normalized_query)
            
            filters = []
            for field, values in entities.items():
                if field in ["status", "complaintType", "complaintMode", "companyStatus", "complaintStatus", "userType"]:
                    for value in values:
                        filters.append({"field": field, "value": value, "operator": "term"})
                elif field == "dates":
                    for date in values:
                        filters.append({"field": "complaintRegDate", "value": date, "operator": "range"})
            
            # Handle content-based filters
            content_indicators = ["regarding", "about", "concerning"]
            for indicator in content_indicators:
                if indicator in lower_query:
                    topic = lower_query.split(indicator, 1)[1].strip()
                    if topic:
                        filters.append({"field": "complaintDetails", "value": topic, "operator": "match"})
            
            result = {
                "intent": "filter_data",
                "filters": filters,
                "columns": []
            }
            return ensure_conditional_columns(result, user_query)
        
        # Handle simple queries first
        if user_query.strip().lower() in ["get all", "fetch everything"]:
            result = {
                "intent": "filter_data",
                "filters": [],
                "columns": []
            }
            return ensure_conditional_columns(result, user_query)
        
        normalized_query = normalize_query(user_query)
        entities = extract_entities_with_disambiguation(normalized_query)
        
        # Build enhanced LLM prompt
        schema_info = get_database_schema()["complaints"]["columns"]
        field_descriptions = {
            "id": "Unique ID for each complaint record",
            "complaintDetails": "Detailed description of the complaint",
            "userId": "User ID of the complainant",
            "fullName": "Full name of the complainant",
            "CityName": "City name of the complainant",
            "stateName": "State name of the complainant",
            "country": "Country of the complainant",
            "userType": "Type of user (Consumer, Wholesaler, Retailer)",
            "status": "Complaint status as integer (0=Open, 1=Closed)",
            "complaintRegDate": "Date when complaint was registered",
            "updationDate": "Date when complaint was last updated",
            "complaintType": "Type/category of the complaint",
            "complaintMode": "Mode of complaint submission (Email, Phone, Website, NCHAPP)",
            "categoryCode": "Numeric category code for the complaint",
            "companyName": "Name of the company against which complaint was filed",
            "complaintStatus": "Current status of complaint processing",
            "companyStatus": "Company response status",
            "lastUpdationDate": "Last update timestamp as string"
        }
        
        field_info = "\\n".join([f"- {col['name']}: {field_descriptions.get(col['name'], 'No description')}" for col in schema_info])
        
        prompt = f"""You are an expert query analysis system that converts natural language queries into structured JSON for Elasticsearch.

Available Fields:
{field_info}

Intent Types: filter_data, filter_data_columns, aggregate_data, list_distinct_values

Output Format: JSON only (no explanations, comments, or additional text)

Critical Rules - Column Selection:
- When phrases like "show only", "select", "display only", "columns", "fields", or singular/plural field names are used, include only those fields in the columns array.
- If no columns are specified or query starts with "show all"/"list all", return an empty columns array for filter_data.
- Invalid column names should be ignored.
- Column names must match schema fields exactly.
- Include the field used in a filter in the columns array only for filter_data_columns intent, not for filter_data

Critical Rules - Intent Detection:
Query Pattern | Intent | Notes
------------------------------
count, how many, total, chart, graph, plot, visualize, min, max, avg, average | aggregate_data | Group by relevant field, include stats if min/max/avg mentioned
list unique, distinct, show all unique | list_distinct_values | Return unique values for the specified field
show, find, get, display, fetch | filter_data | If no columns specified
 | use filter_data_columns if columns are present
show only, select, display only, columns, fields, or field names | filter_data_columns | Filter and return specified columns
top N, top X | aggregate_data | Limit aggregation to top N results

Critical Rules - Field Mapping Rules:
Query Context | Field | Operator | Notes
---------------------------------------
Company name | companyName | match | Use match for full-text search
Date ranges | complaintRegDate/updationDate | range | Use ISO format YYYY-MM-DD
Complaint type | complaintType | match | Use match for full-text search
Complaint mode | complaintMode | match | Use match for full-text search
Complaint details | complaintDetails | match | Use match (full-text search)
Status | status | term | Use integer values (0 or 1)
Complaint status | complaintStatus | term | Exact match
Company status | companyStatus | term | Exact match
User type | userType | term | Exact match
Full name | fullName | match | Use match for full-text search
City/State | CityName/stateName | match | Use match for full-text search

Critical Rules - Date Processing:
- "after 2024 May" → {{"gte": "2024-05-01"}}
- "before March 2024" → {{"lt": "2024-03-01"}}
- "last 30 days" → {{"gte": "now-30d"}}
- "2024" → {{"gte": "2024-01-01", "lt": "2025-01-01"}}

Critical Rules - Sorting & Pagination:
- Add sort_order for explicit sorting requests.
- Default size: 10 for filter queries.
- Use "asc" for chronological/numerical order, "desc" for recent-first.
- For top N in aggregate_data, set size in terms aggregation to N

Output Schema:
{{
  "intent": "filter_data|filter_data_columns|aggregate_data|list_distinct_values",
  "filters": [{{"field": "fieldname", "value": "searchvalue", "operator": "match|term|range"}}],
  "columns": ["fieldname1", "fieldname2"], // Only for filter_data_columns
  "group_by_field": "fieldname", // Only for aggregate_data
  "sort_order": {{"field": "fieldname", "order": "asc|desc"}},
  "size": 10, // For filter queries (for aggregate_data, use in terms aggregation)
  "top_n": N, // Optional for aggregate_data to limit to top N results
  "include_stats": true // Include stats aggregation for min/max/avg
}}

Examples:
Q: "show all complaints by phone"
A: {{"intent": "filter_data", "filters": [{{"field": "complaintMode", "value": "Phone", "operator": "match"}}], "columns": []}}

Q: "show only id and status for email complaints"
A: {{"intent": "filter_data_columns", "filters": [{{"field": "complaintMode", "value": "Email", "operator": "match"}}], "columns": ["id", "status", "complaintMode"]}}

Q: "count complaints by companyStatus"
A: {{"intent": "aggregate_data", "group_by_field": "companyStatus", "columns": []}}

Q: "show complaintDetails, fullName for open complaints"
A: {{"intent": "filter_data_columns", "filters": [{{"field": "status", "value": 0, "operator": "term"}}], "columns": ["complaintDetails", "fullName", "status"]}}

Q: "list unique values of userType"
A: {{"intent": "list_distinct_values", "field": "userType", "columns": []}}

Q: "show all complaints from Flipkart"
A: {{"intent": "filter_data", "filters": [{{"field": "companyName", "value": "Flipkart", "operator": "match"}}], "columns": []}}

Q: "count complaints received after January 2024"
A: {{"intent": "aggregate_data", "filters": [{{"field": "complaintRegDate", "value": {{"gte": "2024-01-01"}}, "operator": "range"}}], "group_by_field": "complaintType", "columns": []}}

Q: "show id, complaintRegDate for closed complaints sorted by date"
A: {{"intent": "filter_data_columns", "filters": [{{"field": "status", "value": 1, "operator": "term"}}], "columns": ["id", "complaintRegDate", "status"], "sort_order": {{"field": "complaintRegDate", "order": "asc"}}}}

Query: {user_query}
Output:"""

        try:
            output = call_openrouter_api(prompt)
            if not output:
                print("❌ LLM returned no valid output. Falling back to rule-based parsing.")
                return fallback_parse_intent(normalized_query, entities, user_query)
            
            try:
                result = json.loads(output.strip())
                if not isinstance(result, dict) or "intent" not in result:
                    raise ValueError("Invalid LLM response structure")
                
                result = ensure_conditional_columns(result, user_query)
                return result
            except (json.JSONDecodeError, ValueError):
                # Try to extract JSON from response
                json_match = re.search(r'{.*}', output, re.DOTALL)
                if json_match:
                    try:
                        result = json.loads(json_match.group(0))
                        if isinstance(result, dict) and "intent" in result:
                            result = ensure_conditional_columns(result, user_query)
                            return result
                    except json.JSONDecodeError:
                        pass
                
                print(f"❌ LLM returned invalid JSON: {output}")
                return fallback_parse_intent(normalized_query, entities, user_query)
        except Exception as e:
            print(f"❌ Error calling LLM: {e}")
            return fallback_parse_intent(normalized_query, entities, user_query)
    except Exception as e:
        print(f"❌ Error in parse_intent_and_entities_enhanced: {e}")
        try:
            normalized_query = normalize_query(user_query)
            entities = extract_entities_with_disambiguation(normalized_query)
            return fallback_parse_intent(normalized_query, entities, user_query)
        except Exception as fallback_error:
            print(f"❌ Fallback parsing also failed: {fallback_error}")
            return None


def parse_query(user_query: str) -> Dict[str, Any]:
    """
    Public API: parse the user's natural language query to structured parsed query dict.
    """
    return parse_intent_and_entities_enhanced(user_query)