from typing import Dict, Any
from config.settings import INDEX_NAME


def construct_elasticsearch_query(parsed_query: Dict[str, Any]) -> Dict[str, Any]:
    """Construct Elasticsearch query adapted for schema3 structure."""
    intent = parsed_query.get("intent", "filter_data")
    filters = parsed_query.get("filters", [])
    group_by_field = parsed_query.get("group_by_field", "companyName")
    size = parsed_query.get("size", 10)

    # Case-insensitive fields
    case_insensitive_fields = {
        "companyStatus", "complaintStatus", "userType", "complaintType", 
        "complaintMode", "country"
    }

    if intent == "aggregate_data":
        agg_size = parsed_query.get("top_n", 20)
        query = {
            "size": 0,
            "query": {"match_all": {}},
            "aggs": {
                "grouped_data": {
                    "terms": {
                        "field": f"{group_by_field}.keyword" if group_by_field in ["companyName", "fullName", "CityName", "stateName", "complaintType", "complaintMode"] else group_by_field,
                        "size": agg_size
                    }
                }
            }
        }
        
        if parsed_query.get("include_stats"):
            query["aggs"]["stats"] = {"stats": {"field": "userId"}}
            
    elif intent == "list_distinct_values":
        field = parsed_query.get("group_by_field", parsed_query.get("field", "companyName"))
        query = {
            "size": 0,
            "query": {"match_all": {}},
            "aggs": {
                "distinct_values": {
                    "terms": {
                        "field": f"{field}.keyword" if field in ["companyName", "fullName", "CityName", "stateName", "complaintType", "complaintMode"] else field,
                        "size": 1000
                    }
                }
            }
        }
    else:  # filter_data or filter_data_columns
        query = {
            "size": size,
            "query": {"match_all": {}},
            "_source": True
        }
        
        if not filters:
            query = {"size": size, "query": {"match_all": {}}, "_source": True}
        else:
            bool_query = {"must": []}
            
            for filter_item in filters:
                field = filter_item["field"]
                value = filter_item["value"]
                operator = filter_item["operator"]
                
                # Normalize value to lowercase if field is in the case-insensitive set
                if field in case_insensitive_fields and isinstance(value, str):
                    value = value.lower()
                
                if operator == "match":
                    bool_query["must"].append({"match": {field: value}})
                elif operator == "term":
                    bool_query["must"].append({"term": {field: value}})
                elif operator == "range":
                    bool_query["must"].append({"range": {field: value}})
            
            query = {"size": size, "query": {"bool": bool_query}, "_source": True}
        
        if intent == "filter_data_columns":
            columns = parsed_query.get("columns", [])
            if columns:
                query["_source"] = columns
        
        if "sort_order" in parsed_query:
            sort_field = parsed_query["sort_order"]["field"]
            sort_order = parsed_query["sort_order"]["order"]
            query["sort"] = [{sort_field: {"order": sort_order}}]
    
    return query