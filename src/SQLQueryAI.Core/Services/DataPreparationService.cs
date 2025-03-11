using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.Logging;
using SQLQueryAI.Core.Interfaces;
using SQLQueryAI.Core.Models;
using System;
using System.Collections.Generic;
using System.Data;
using System.Data.SqlClient;
using System.Linq;
using System.Threading.Tasks;

namespace SQLQueryAI.Core.Services
{
    /// <summary>
    /// Service for preparing data from SQL Server for use in the AI system
    /// </summary>
    public class DataPreparationService : IDataPreparationService
    {
        private readonly ILogger<DataPreparationService> _logger;
        private readonly string _connectionString;
        private readonly IVectorDatabaseService _vectorDbService;
        private bool _databaseInitialized = false;

        public DataPreparationService(
            ILogger<DataPreparationService> logger,
            IConfiguration configuration,
            IVectorDatabaseService vectorDbService)
        {
            _logger = logger;
            _connectionString = configuration.GetConnectionString("SQLServerDatabase")
                ?? throw new ArgumentException("SQLServerDatabase connection string is not configured");
            _vectorDbService = vectorDbService;

            // Try to load existing vector index if available
            var indexPath = configuration["VectorDatabase:IndexPath"] ?? "vector_index.bin";
            if (System.IO.File.Exists(indexPath))
            {
                _databaseInitialized = _vectorDbService.LoadIndex(indexPath);
                _logger.LogInformation("Vector database loaded: {Status}", _databaseInitialized ? "Success" : "Failed");
            }
        }

        /// <inheritdoc />
        public async Task<DataTable> ExtractTrainingDataAsync()
        {
            var dataTable = new DataTable();

            try
            {
                using (var connection = new SqlConnection(_connectionString))
                {
                    await connection.OpenAsync();

                    // Extract data for training based on the provided query structure
                    string query = @"
                        SELECT TOP 500000 
                            PlacePopularity.CreatedDate, 
                            PlacePopularity.CurrentPopularity, 
                            Place.PlaceId, 
                            Place.CityId, 
                            Place.PlaceName, 
                            City.CityName
                        FROM [10654] PlacePopularity 
                        LEFT JOIN [10057] Place ON Place.PlaceId = PlacePopularity.PlaceId
                        LEFT JOIN [10051] City ON City.CityId = Place.CityId 
                        WHERE PlacePopularity.CurrentPopularity > 0
                        ORDER BY PlacePopularity.CreatedDate DESC";

                    using (var command = new SqlCommand(query, connection))
                    {
                        command.CommandTimeout = 300; // 5 minutes timeout for large query

                        using (var adapter = new SqlDataAdapter(command))
                        {
                            adapter.Fill(dataTable);
                        }
                    }
                }

                _logger.LogInformation("Extracted {RowCount} rows for training data", dataTable.Rows.Count);
                return dataTable;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error extracting training data from SQL Server");
                throw;
            }
        }

        /// <inheritdoc />
        public async Task<List<CompanyData>> GetCompanyDataAsync(int limit = 1000)
        {
            // This method is now replaced by GetPlaceDataAsync
            throw new NotImplementedException("This method is replaced by GetPlaceDataAsync");
        }

        /// <summary>
        /// Gets place data from the database for building the vector database
        /// </summary>
        /// <param name="limit">Maximum number of places to retrieve</param>
        /// <returns>List of place data objects</returns>
        public async Task<List<PlaceData>> GetPlaceDataAsync(int limit = 1000)
        {
            var result = new List<PlaceData>();

            try
            {
                using (var connection = new SqlConnection(_connectionString))
                {
                    await connection.OpenAsync();

                    string query = $@"
                        SELECT TOP {limit}
                            PlacePopularity.CreatedDate, 
                            PlacePopularity.CurrentPopularity, 
                            Place.PlaceId, 
                            Place.CityId, 
                            Place.PlaceName, 
                            City.CityName
                        FROM [10654] PlacePopularity 
                        LEFT JOIN [10057] Place ON Place.PlaceId = PlacePopularity.PlaceId
                        LEFT JOIN [10051] City ON City.CityId = Place.CityId 
                        WHERE PlacePopularity.CurrentPopularity > 0
                        ORDER BY PlacePopularity.CreatedDate DESC";

                    using (var command = new SqlCommand(query, connection))
                    {
                        using (var reader = await command.ExecuteReaderAsync())
                        {
                            while (await reader.ReadAsync())
                            {
                                result.Add(new PlaceData
                                {
                                    CreatedDate = reader.GetDateTime(0),
                                    CurrentPopularity = reader.GetInt32(1),
                                    PlaceId = reader.GetInt32(2),
                                    CityId = reader.GetInt32(3),
                                    PlaceName = reader.GetString(4),
                                    CityName = reader.GetString(5),
                                    Metadata = new Dictionary<string, object>()
                                });
                            }
                        }
                    }
                }

                return result;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error retrieving place data from SQL Server");
                throw;
            }
        }

        /// <inheritdoc />
        public async Task<bool> RebuildVectorDatabaseAsync()
        {
            try
            {
                // Get a sample of data to build vector database
                var data = await GetPlaceDataAsync(limit: 100000);

                // Convert to text descriptions for embedding
                var descriptions = data.Select(p => p.ToDescription()).ToList();

                // Build vector database
                await _vectorDbService.BuildIndexForPlaces(descriptions, data);

                // Save the index to disk
                var indexPath = "vector_index.bin";
                _vectorDbService.SaveIndex(indexPath);

                _databaseInitialized = true;
                _logger.LogInformation("Vector database rebuilt successfully with {Count} entries", descriptions.Count);

                return true;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error rebuilding vector database");
                _databaseInitialized = false;
                throw;
            }
        }

        /// <inheritdoc />
        public string GetDatabaseStatus()
        {
            return _databaseInitialized ? "Initialized" : "Not Initialized";
        }
    }
}