# SQL Natural Language Query Implementation Guide

This guide provides step-by-step instructions for implementing the SQL Natural Language Query system in your environment.

## Prerequisites

- .NET 7.0 SDK or later
- SQL Server with your data (containing tables [10654], [10057], and [10051])
- Python 3.8+ with PyTorch, Transformers libraries for model training
- Docker & Docker Compose (optional, for containerized deployment)

## Installation

### 1. Clone the Repository

```bash
git clone <repository-url>
cd SQLQueryAI
```

### 2. Configure the Database Connection

Edit `src/SQLQueryAI.API/appsettings.json` to set your SQL Server connection string:

```json
{
  "ConnectionStrings": {
    "SQLServerDatabase": "Server=yourserver;Database=yourdatabase;User Id=username;Password=password;"
  }
}
```

### 3. Install Python Dependencies

```bash
pip install torch transformers datasets onnx onnxruntime
```

### 4. Build the Solution

```bash
dotnet build
```

### 5. Run the Application

```bash
cd src/SQLQueryAI.API
dotnet run
```

The application will be available at `http://localhost:5000`

## Setup Steps

### 1. Initialize the Vector Database

When you first run the application, you need to initialize the vector database by calling:

```
POST http://localhost:5000/api/query/rebuild-index
```

This will extract data from your SQL Server, create text descriptions, and build embeddings for semantic search.

### 2. Train the Model

Train the model to respond to natural language queries:

```
POST http://localhost:5000/api/admin/retrain-model
```

This will extract training data and fine-tune the model. The process may take several minutes depending on your hardware.

### 3. Start Querying

Now you can query your data with natural language:

```
POST http://localhost:5000/api/query
Content-Type: application/json

{
  "query": "What are the most popular places in New York?"
}
```

Or use the web client at `http://localhost:5000`.

## Docker Deployment

For production deployment, use Docker Compose:

1. Navigate to the deployment directory:
   ```bash
   cd deployment
   ```

2. Customize the `docker-compose.yml` if needed

3. Build and start the services:
   ```bash
   docker-compose up -d
   ```

4. Initialize the system:
   ```bash
   curl -X POST http://localhost:5000/api/query/rebuild-index
   curl -X POST http://localhost:5000/api/admin/retrain-model
   ```

## Data Schema

The system is configured to work with the following database schema:

- Table `[10654]` (PlacePopularity):
  - PlaceId (int)
  - CreatedDate (datetime)
  - CurrentPopularity (int)

- Table `[10057]` (Place):
  - PlaceId (int)
  - CityId (int)
  - PlaceName (nvarchar)

- Table `[10051]` (City):
  - CityId (int)
  - CityName (nvarchar)

If your schema differs, you'll need to modify the data access code in `DataPreparationService.cs`.

## Performance Tuning

For optimal performance with large datasets:

1. **Indexing**: Ensure proper indexes on frequently queried columns, especially:
   - PlacePopularity.PlaceId
   - PlacePopularity.CreatedDate
   - Place.CityId

2. **Sampling**: The system uses a sample of your data for building the vector database. If you need more comprehensive results, increase the sample size in `GetPlaceDataAsync()`.

3. **Memory**: For very large datasets (>100M rows), consider increasing the RAM available to the application.

4. **Scaling**: For high-load production environments, consider deploying multiple instances behind a load balancer.

## Extending the System

### Adding New Query Types

To support new types of queries:

1. Add training examples to `CreateTrainingExamples()` in `ModelService.cs`
2. Retrain the model using the API

### Customizing Output Format

Modify the prompt template in `CreatePrompt()` within `ModelService.cs` to change how responses are formatted.

### Integrating with Other Systems

The API can be called from any system capable of making HTTP requests. Use the endpoints documented in `api-docs.md`.

## Troubleshooting

### Vector Database Issues

If search results are poor:

1. Check that the vector database was built correctly
2. Try increasing the context size (k parameter in Search method)
3. Ensure your data descriptions are informative

### Model Issues

If model responses are inaccurate:

1. Check training examples for quality and diversity
2. Try a different base model in the Python script
3. Increase the number of training epochs

### Database Connection Issues

If you encounter database connection issues:

1. Verify the connection string in `appsettings.json`
2. Ensure the SQL Server is accessible from the application
3. Check that the required tables exist and have the expected structure

## Maintenance

To keep the system performing optimally:

1. Periodically rebuild the vector index to incorporate new data:
   ```
   POST /api/query/rebuild-index
   ```

2. Retrain the model if query quality decreases:
   ```
   POST /api/admin/retrain-model
   ```

3. Monitor system health with:
   ```
   GET /api/admin/system-status
   ```

## Support

For additional support, consult the documentation or contact the development team.
