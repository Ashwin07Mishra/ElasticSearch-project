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


"Examples"

show complaints by email where code is 452 from swiggy company status pending with closed status
📝 Your query: show complaints by email where code is 452 from swiggy company status pending with closed status
\n🔍 Processing query: show complaints by email where code is 452 from swiggy company status pending with closed status
📋 Parsed intent: filter_data_columns
🎯 Filters: 5
🔧 Executing Elasticsearch query:
{
  "size": 10,
  "query": {
    "bool": {
      "must": [
        {
          "match": {
            "complaintMode": "email"
          }
        },
        {
          "term": {
            "categoryCode": 452
          }
        },
        {
          "match": {
            "companyName": "Swiggy"
          }
        },
        {
          "term": {
            "companyStatus": "pending"
          }
        },
        {
          "term": {
            "status": 1
          }
        }
      ]
    }
  },
  "_source": [
    "id",
    "complaintDetails",
    "fullName",
    "CityName",
    "stateName",
    "country",
    "userType",
    "status",
    "complaintRegDate",
    "updationDate",
    "complaintType",
    "complaintMode",
    "categoryCode",
    "companyName",
    "complaintStatus",
    "companyStatus",
    "lastUpdationDate",
    "complaintMode",
    "categoryCode",
    "companyName",
    "companyStatus",
    "status"
  ]
}
✅ Query example saved for future learning.

**Query Results** (Showing 1 of 1 total results):\n```\n+---------+------------------------+------------+------------+----------------+-----------+------------+----------+--------------------+----------------+---------------------+-----------------+----------------+---------------+-------------------+-----------------+---------------------+
|      id | complaintDetails       | fullName   | CityName   | stateName      | country   | userType   | status   | complaintRegDate   | updationDate   | complaintType       | complaintMode   |   categoryCode | companyName   | complaintStatus   | companyStatus   | lastUpdationDate    |
+=========+========================+============+============+================+===========+============+==========+====================+================+=====================+=================+================+===============+===================+=================+=====================+
| 5000039 | Late delivery of order | Aarav Kibe | Hazaribagh | Madhya Pradesh | India     | Retailer   | Closed   | 2025-09-07         | 2025-10-20     | Fraudulent Activity | Email           |            452 | Swiggy        | In Progress       | Pending         | 2025-10-20 02:27:57 |
+---------+------------------------+------------+------------+----------------+-----------+------------+----------+--------------------+----------------+---------------------+-----------------+----------------+---------------+-------------------+-----------------+---------------------+\n```
⏱️ Processed in 4750.2 ms


------------------------------------------------------------------------------------------------

📝 Your query: show count 
\n🔍 Processing query: show count
📋 Parsed intent: aggregate_data
🎯 Filters: 0
🔧 Executing Elasticsearch query:
{
  "size": 0,
  "query": {
    "match_all": {}
  },
  "aggs": {
    "grouped_data": {
      "terms": {
        "field": "companyName.keyword",
        "size": 20
      }
    }
  }
}
✅ Logged query and ES JSON for analysis.
✅ Query example saved for future learning.

**Aggregation Results:**\n```\n+---------------------------------+---------+
| Value                           |   Count |
+=================================+=========+
| Dell                            |       8 |
+---------------------------------+---------+
| Ola Cabs                        |       7 |
+---------------------------------+---------+
| Apple India                     |       6 |
+---------------------------------+---------+
| Epson                           |       6 |
+---------------------------------+---------+
| Swiggy                          |       6 |
+---------------------------------+---------+
| ICICI Prudential Life Insurance |       5 |
+---------------------------------+---------+
| Paytm                           |       5 |
+---------------------------------+---------+
| Zomato                          |       5 |
+---------------------------------+---------+
| Flipkart                        |       3 |
+---------------------------------+---------+
| HP                              |       3 |
+---------------------------------+---------+
| Samsung                         |       3 |
+---------------------------------+---------+
| HDFC Bank                       |       2 |
+---------------------------------+---------+
| Sony                            |       1 |
+---------------------------------+---------+\n```
⏱️ Processed in 3566.4 ms


----------------------------------------------------------------------------------------------------


📝 Your query: show data by email ordered by date
\n🔍 Processing query: show data by email ordered by date
📋 Parsed intent: filter_data_columns
🎯 Filters: 1
🔧 Executing Elasticsearch query:
{
  "size": 10,
  "query": {
    "bool": {
      "must": [
        {
          "match": {
            "complaintMode": "email"
          }
        }
      ]
    }
  },
  "_source": true,
  "sort": [
    {
      "complaintRegDate": {
        "order": "asc"
      }
    }
  ]
}
✅ Logged query and ES JSON for analysis.
✅ Query example saved for future learning.

**Query Results** (Showing 10 of 19 total results):\n```\n+---------+---------------------------------+----------+--------------------+------------+------------------+-----------+------------+----------+--------------------+----------------+--------------------------+-----------------+----------------+---------------+-------------------+-----------------+---------------------+------------------------+
|      id | complaintDetails                |   userId | fullName           | CityName   | stateName        | country   | userType   | status   | complaintRegDate   | updationDate   | complaintType            | complaintMode   |   categoryCode | companyName   | complaintStatus   | companyStatus   | lastUpdationDate    |   processing_time_days |
+=========+=================================+==========+====================+============+==================+===========+============+==========+====================+================+==========================+=================+================+===============+===================+=================+=====================+========================+
| 5000009 | Product not working as expected |  1137969 | Arhaan Wason       | Hosur      | Uttar Pradesh    | India     | Wholesaler | Closed   | 2025-06-01         | 2025-07-31     | Warranty Issue           | Email           |            457 | Epson         | Escalated         | Responded       | 2025-07-31 05:13:01 |                     60 |
+---------+---------------------------------+----------+--------------------+------------+------------------+-----------+------------+----------+--------------------+----------------+--------------------------+-----------------+----------------+---------------+-------------------+-----------------+---------------------+------------------------+
| 5000058 | Fraudulent transaction noticed  |  1213740 | Akarsh Jani        | Korba      | Assam            | India     | Consumer   | Open     | 2025-06-10         | 2025-08-07     | Misleading Advertisement | Email           |            455 | Epson         | Escalated         | Resolved        | 2025-08-07 04:39:23 |                     58 |
+---------+---------------------------------+----------+--------------------+------------+------------------+-----------+------------+----------+--------------------+----------------+--------------------------+-----------------+----------------+---------------+-------------------+-----------------+---------------------+------------------------+
| 5000037 | Product not working as expected |  1968784 | Siya Butala        | Saharanpur | Jharkhand        | India     | Wholesaler | Open     | 2025-06-17         | 2025-07-04     | Service Denial           | Email           |            455 | Samsung       | Open              | Responded       | 2025-07-04 03:37:20 |                     17 |
+---------+---------------------------------+----------+--------------------+------------+------------------+-----------+------------+----------+--------------------+----------------+--------------------------+-----------------+----------------+---------------+-------------------+-----------------+---------------------+------------------------+
| 5000022 | Warranty claim denied           |  1660949 | Suhana Subramanian | Fatehpur   | Tripura          | India     | Retailer   | Closed   | 2025-06-20         | 2025-08-06     | Service Denial           | Email           |            456 | Zomato        | In Progress       | Pending         | 2025-08-06 18:24:39 |                     47 |
+---------+---------------------------------+----------+--------------------+------------+------------------+-----------+------------+----------+--------------------+----------------+--------------------------+-----------------+----------------+---------------+-------------------+-----------------+---------------------+------------------------+
| 5000026 | Warranty claim denied           |  1652069 | Arhaan Ghose       | Bhatpara   | Maharashtra      | India     | Retailer   | Open     | 2025-06-21         | 2025-07-14     | Fraudulent Activity      | Email           |            452 | Ola Cabs      | Escalated         | Pending         | 2025-07-14 20:14:16 |                     23 |
+---------+---------------------------------+----------+--------------------+------------+------------------+-----------+------------+----------+--------------------+----------------+--------------------------+-----------------+----------------+---------------+-------------------+-----------------+---------------------+------------------------+
| 5000013 | Company not responding to calls |  1716485 | Mamooty Chaudhary  | Bhilwara   | Tripura          | India     | Retailer   | Closed   | 2025-06-26         | 2025-06-29     | Misleading Advertisement | Email           |            450 | HP            | Closed            | Pending         | 2025-06-29 23:25:03 |                      3 |
+---------+---------------------------------+----------+--------------------+------------+------------------+-----------+------------+----------+--------------------+----------------+--------------------------+-----------------+----------------+---------------+-------------------+-----------------+---------------------+------------------------+
| 5000021 | Company not responding to calls |  1103568 | Zoya Shah          | Dindigul   | Nagaland         | India     | Consumer   | Closed   | 2025-07-15         | 2025-09-01     | Defective Product        | Email           |            455 | Ola Cabs      | Closed            | Resolved        | 2025-09-01 18:19:05 |                     48 |
+---------+---------------------------------+----------+--------------------+------------+------------------+-----------+------------+----------+--------------------+----------------+--------------------------+-----------------+----------------+---------------+-------------------+-----------------+---------------------+------------------------+
| 5000010 | Warranty claim denied           |  1628230 | Anahita Sundaram   | Barasat    | Bihar            | India     | Wholesaler | Open     | 2025-07-26         | 2025-08-31     | Defective Product        | Email           |            457 | Zomato        | In Progress       | Resolved        | 2025-08-31 02:11:08 |                     36 |
+---------+---------------------------------+----------+--------------------+------------+------------------+-----------+------------+----------+--------------------+----------------+--------------------------+-----------------+----------------+---------------+-------------------+-----------------+---------------------+------------------------+
| 5000034 | Service request not attended    |  1069610 | Farhan Chawla      | Ahmednagar | Himachal Pradesh | India     | Retailer   | Open     | 2025-08-05         | 2025-08-11     | Fraudulent Activity      | Email           |            451 | Zomato        | Open              | Resolved        | 2025-08-11 17:29:40 |                      6 |
+---------+---------------------------------+----------+--------------------+------------+------------------+-----------+------------+----------+--------------------+----------------+--------------------------+-----------------+----------------+---------------+-------------------+-----------------+---------------------+------------------------+
| 5000005 | Product not working as expected |  1364351 | Divyansh Krishnan  | Rewa       | Uttarakhand      | India     | Consumer   | Open     | 2025-08-06         | 2025-08-13     | Warranty Issue           | Email           |            452 | HP            | Open              | Pending         | 2025-08-13 10:35:13 |                      7 |
+---------+---------------------------------+----------+--------------------+------------+------------------+-----------+------------+----------+--------------------+----------------+--------------------------+-----------------+----------------+---------------+-------------------+-----------------+---------------------+------------------------+\n```
⏱️ Processed in 6700.3 ms








