# SQL Natural Language Query API Documentation

This document provides detailed information about the API endpoints for the SQL Natural Language Query system.

## Base URL

```
http://localhost:5000
```

## Authentication

The API currently does not require authentication for local development. For production deployment, you should implement an appropriate authentication mechanism.

## Endpoints

### Query Endpoints

#### Process a Natural Language Query

```http
POST /api/query
Content-Type: application/json
```

Process a natural language query against the SQL database.

**Request Body:**

```json
{
  "query": "What are the most popular places in New York?"
}
```

**Response:**

```json
{
  "result": {
    "places": [
      {
        "place_name": "Times Square",
        "popularity": 92,
        "created_date": "2025-03-10"
      },
      {
        "place_name": "Central Park",
        "popularity": 89,
        "created_date": "2025-03-10"
      },
      {
        "place_name": "Empire State Building",
        "popularity": 85,
        "created_date": "2025-03-10"
      }
    ]
  },
  "sourceContext": [
    "Place Times Square in New York had a popularity of 92 on 2025-03-10",
    "Place Central Park in New York had a popularity of 89 on 2025-03-10",
    "Place Empire State Building in New York had a popularity of 85 on 2025-03-10"
  ],
  "processingTimeMs": 245
}
```

**Status Codes:**
- `200 OK`: Query processed successfully
- `400 Bad Request`: Invalid query format
- `500 Internal Server Error`: Server error processing the query

#### Rebuild Vector Index

```http
POST /api/query/rebuild-index
```

Rebuild the vector index for faster query processing.

**Response:**

```
Index rebuilt successfully
```

**Status Codes:**
- `200 OK`: Index rebuilt successfully
- `500 Internal Server Error`: Error rebuilding the index

### Admin Endpoints

#### Retrain Model

```http
POST /api/admin/retrain-model
```

Retrain the AI model using the current data.

**Response:**

```
Model retrained successfully
```

**Status Codes:**
- `200 OK`: Model retrained successfully
- `500 Internal Server Error`: Error retraining the model

#### Get System Status

```http
GET /api/admin/system-status
```

Get current system status including vector database and model status.

**Response:**

```json
{
  "vectorDbStatus": "Initialized",
  "modelStatus": "Initialized",
  "systemHealth": "OK"
}
```

**Status Codes:**
- `200 OK`: Status retrieved successfully

## Example Queries

Here are some example natural language queries to try:

1. "What are the most popular places in New York?"
2. "What is the popularity trend for Times Square over the past week?"
3. "Which city has the highest average place popularity?"
4. "Compare the popularity between Central Park and Times Square"
5. "Show me the top attractions in Berlin"
6. "List places with popularity rating over 80"
7. "What are the most popular tourist attractions in European cities?"
8. "Which place had the highest popularity rating last week?"
9. "Compare weekday vs weekend popularity for Central Park"
10. "What are the top 5 most popular places overall?"

## Error Handling

The API returns standard HTTP status codes to indicate success or failure of requests.

For errors, the response body will include an error message:

```json
{
  "error": "An error occurred while processing your request"
}
```

## Limitations

- The natural language understanding has certain limitations and may not understand extremely complex or ambiguous queries.
- Large result sets may be truncated to improve performance.
- The model requires retraining when the underlying data structure changes significantly.
