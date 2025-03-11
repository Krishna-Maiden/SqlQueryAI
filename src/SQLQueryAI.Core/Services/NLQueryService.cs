using Microsoft.Extensions.Logging;
using Newtonsoft.Json;
using SQLQueryAI.Core.Interfaces;
using System;
using System.Collections.Generic;
using System.Threading.Tasks;

namespace SQLQueryAI.Core.Services
{
    /// <summary>
    /// Service for processing natural language queries
    /// </summary>
    public class NLQueryService : INLQueryService
    {
        private readonly ILogger<NLQueryService> _logger;
        private readonly IVectorDatabaseService _vectorDbService;
        private readonly IModelService _modelService;
        private readonly IDataPreparationService _dataService;

        public NLQueryService(
            ILogger<NLQueryService> logger,
            IVectorDatabaseService vectorDbService,
            IModelService modelService,
            IDataPreparationService dataService)
        {
            _logger = logger;
            _vectorDbService = vectorDbService;
            _modelService = modelService;
            _dataService = dataService;
        }

        /// <inheritdoc />
        public async Task<(object Result, List<string> Context)> ProcessQueryAsync(string query)
        {
            try
            {
                _logger.LogInformation("Processing query: {Query}", query);

                // Step 1: Search vector database for relevant context
                var (contexts, places) = await _vectorDbService.SearchPlaces(query);

                if (contexts.Count == 0)
                {
                    _logger.LogWarning("No relevant context found for query: {Query}", query);
                    return (new { message = "No relevant data found for your query" }, new List<string>());
                }

                // Step 2: Generate response using the model
                string jsonResponse = await _modelService.GenerateResponseAsync(query, contexts);

                // Step 3: Parse JSON response
                try
                {
                    var result = JsonConvert.DeserializeObject<object>(jsonResponse);
                    if (result == null)
                    {
                        throw new JsonException("Deserialized object is null");
                    }
                    return (result, contexts);
                }
                catch (JsonException ex)
                {
                    _logger.LogError(ex, "Error parsing model response as JSON: {Response}", jsonResponse);
                    return (new { response = jsonResponse }, contexts);
                }
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error processing query: {Query}", query);
                throw;
            }
        }

        /// <inheritdoc />
        public async Task<bool> RebuildIndexAsync()
        {
            try
            {
                return await _dataService.RebuildVectorDatabaseAsync();
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error rebuilding index");
                throw;
            }
        }
    }
}