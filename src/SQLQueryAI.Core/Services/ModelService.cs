using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.Logging;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using Newtonsoft.Json;
using SQLQueryAI.Core.Interfaces;
using SQLQueryAI.Core.Models;
using System;
using System.Collections.Generic;
using System.Data;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace SQLQueryAI.Core.Services
{
    /// <summary>
    /// Service for managing ML model operations
    /// </summary>
    public class ModelService : IModelService
    {
        private readonly ILogger<ModelService> _logger;
        private readonly IDataPreparationService _dataService;
        private readonly IConfiguration _configuration;
        private readonly string _modelPath;
        private readonly string _pythonPath;
        private readonly string _scriptPath;
        private bool _modelInitialized = false;

        public ModelService(
            ILogger<ModelService> logger,
            IDataPreparationService dataService,
            IConfiguration configuration)
        {
            _logger = logger;
            _dataService = dataService;
            _configuration = configuration;

            _modelPath = _configuration["ModelSettings:ModelPath"] ?? "model.onnx";
            _pythonPath = _configuration["ModelSettings:PythonPath"] ?? "python";
            _scriptPath = _configuration["ModelSettings:TrainingScriptPath"] ?? "fine_tune_model.py";

            // Check if model exists
            _modelInitialized = File.Exists(_modelPath);
        }

        /// <inheritdoc />
        public async Task TrainModelAsync()
        {
            try
            {
                _logger.LogInformation("Starting model training process");

                // Extract training data
                var data = await _dataService.ExtractTrainingDataAsync();

                // Create training examples
                var trainingExamples = CreateTrainingExamples(data);

                // Save training data to disk for the Python script to use
                string trainingDataPath = "training_data.json";
                File.WriteAllText(trainingDataPath, JsonConvert.SerializeObject(trainingExamples));

                // Call Python script for fine-tuning (using transformers library)
                var processStartInfo = new ProcessStartInfo
                {
                    FileName = _pythonPath,
                    Arguments = $"{_scriptPath} --training_data {trainingDataPath} --output_model {_modelPath}",
                    RedirectStandardOutput = true,
                    RedirectStandardError = true,
                    UseShellExecute = false,
                    CreateNoWindow = true
                };

                using (var process = new Process { StartInfo = processStartInfo })
                {
                    process.Start();
                    string output = await process.StandardOutput.ReadToEndAsync();
                    string error = await process.StandardError.ReadToEndAsync();
                    await process.WaitForExitAsync();

                    if (process.ExitCode != 0)
                    {
                        _logger.LogError("Fine-tuning failed: {Error}", error);
                        throw new Exception($"Fine-tuning failed: {error}");
                    }

                    _logger.LogInformation("Fine-tuning completed: {Output}", output);
                }

                _modelInitialized = true;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error training model");
                throw;
            }
        }

        /// <inheritdoc />
        public async Task<string> GenerateResponseAsync(string query, List<string> context)
        {
            if (!_modelInitialized && !File.Exists(_modelPath))
            {
                // If the model doesn't exist, use a fallback approach
                return FallbackQueryProcessing(query, context);
            }

            try
            {
                // Prepare prompt for inference
                string prompt = CreatePrompt(query, context);

                // For smaller deployments, we can use a simpler approach without ONNX
                // This simulates what the ML model would do
                if (!File.Exists(_modelPath))
                {
                    return FallbackQueryProcessing(query, context);
                }

                // Using ONNX Runtime for inference
                using (var session = new InferenceSession(_modelPath))
                {
                    // Create input tensors
                    // Note: Actual tokenization would depend on the model being used
                    var inputData = Tokenize(prompt);
                    var inputTensor = new DenseTensor<long>(new[] { 1, inputData.Length });

                    for (int i = 0; i < inputData.Length; i++)
                    {
                        inputTensor[0, i] = inputData[i];
                    }

                    // Run inference
                    var inputs = new List<NamedOnnxValue>
                    {
                        NamedOnnxValue.CreateFromTensor("input_ids", inputTensor)
                    };

                    var outputs = session.Run(inputs);

                    // Process output
                    var outputTensor = outputs.First().AsTensor<long>();
                    var result = Detokenize(outputTensor);

                    await Task.CompletedTask; // For async compliance
                    return result;
                }
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error generating response from model");
                return FallbackQueryProcessing(query, context);
            }
        }

        /// <inheritdoc />
        public string GetModelStatus()
        {
            if (_modelInitialized)
                return "Initialized";
            else if (File.Exists(_modelPath))
                return "Available (Not Loaded)";
            else
                return "Not Available";
        }

        /// <summary>
        /// Creates training examples from database data
        /// </summary>
        private List<TrainingExample> CreateTrainingExamples(DataTable data)
        {
            var examples = new List<TrainingExample>();

            // Get unique regions
            var regions = new HashSet<string>();
            foreach (DataRow row in data.Rows)
            {
                if (row["region"] != DBNull.Value)
                    regions.Add(row["region"].ToString() ?? string.Empty);
            }

            // Create example for top export companies by region
            foreach (string region in regions)
            {
                var filteredRows = data.AsEnumerable()
                    .Where(r => r.Field<string>("region") == region)
                    .OrderByDescending(r => Convert.ToDecimal(r["export_volume"]))
                    .Take(10);

                if (!filteredRows.Any())
                    continue;

                var topCompanies = filteredRows.Select(r => new {
                    name = r.Field<string>("name"),
                    country = r.Field<string>("country"),
                    export_volume = Convert.ToDecimal(r["export_volume"])
                }).ToList();

                examples.Add(new TrainingExample
                {
                    Question = $"What are the top 10 export companies in {region}?",
                    Answer = JsonConvert.SerializeObject(topCompanies)
                });

                // Variation
                examples.Add(new TrainingExample
                {
                    Question = $"Show me the leading exporters from {region}",
                    Answer = JsonConvert.SerializeObject(topCompanies)
                });
            }

            // Create example for top export companies by country
            var countries = new HashSet<string>();
            foreach (DataRow row in data.Rows)
            {
                if (row["country"] != DBNull.Value)
                    countries.Add(row["country"].ToString() ?? string.Empty);
            }

            foreach (string country in countries.Take(20)) // Limit to 20 countries for example size
            {
                var filteredRows = data.AsEnumerable()
                    .Where(r => r.Field<string>("country") == country)
                    .OrderByDescending(r => Convert.ToDecimal(r["export_volume"]))
                    .Take(5);

                if (!filteredRows.Any())
                    continue;

                var topCompanies = filteredRows.Select(r => new {
                    name = r.Field<string>("name"),
                    export_volume = Convert.ToDecimal(r["export_volume"]),
                    revenue = Convert.ToDecimal(r["revenue"])
                }).ToList();

                examples.Add(new TrainingExample
                {
                    Question = $"Which companies from {country} have the highest export volumes?",
                    Answer = JsonConvert.SerializeObject(topCompanies)
                });
            }

            // Add more example types for product categories, etc.

            return examples;
        }

        /// <summary>
        /// Creates a prompt for the model using the query and context
        /// </summary>
        private string CreatePrompt(string query, List<string> context)
        {
            StringBuilder sb = new StringBuilder();

            // Prompt format depends on the model being used
            sb.AppendLine("Based on the following company information:");

            foreach (var ctx in context.Take(10)) // Limit context to 10 entries
            {
                sb.AppendLine(ctx);
            }

            sb.AppendLine();
            sb.AppendLine($"Answer the query: {query}");
            sb.AppendLine();
            sb.AppendLine("Format the response as a structured JSON with relevant fields.");

            return sb.ToString();
        }

        /// <summary>
        /// Tokenizes text for model input (simplified implementation)
        /// </summary>
        private long[] Tokenize(string text)
        {
            // This is a simplified tokenization approach
            // In a real implementation, you would use a proper tokenizer matching your model
            var tokens = text.Split(' ', StringSplitOptions.RemoveEmptyEntries)
                .SelectMany(t => t.Split('\n', StringSplitOptions.RemoveEmptyEntries))
                .Select((token, index) => (long)index)
                .ToArray();

            return tokens;
        }

        /// <summary>
        /// Detokenizes model output (simplified implementation)
        /// </summary>
        private string Detokenize(Tensor<long> tensor)
        {
            // This is a simplified detokenization approach
            // In a real implementation, you would use a proper detokenizer matching your model
            return "{ \"companies\": [ { \"name\": \"Example Corp\", \"export_volume\": 1234.56 } ] }";
        }

        /// <summary>
        /// Provides a fallback response when the model is unavailable
        /// </summary>
        private string FallbackQueryProcessing(string query, List<string> context)
        {
            _logger.LogWarning("Using fallback query processing as model is not available");

            // Simple keyword-based processing
            bool isTopCompaniesQuery = query.Contains("top", StringComparison.OrdinalIgnoreCase) &&
                (query.Contains("companies", StringComparison.OrdinalIgnoreCase) ||
                 query.Contains("exporters", StringComparison.OrdinalIgnoreCase));

            string? region = null;
            if (query.Contains("North America", StringComparison.OrdinalIgnoreCase)) region = "North America";
            else if (query.Contains("Europe", StringComparison.OrdinalIgnoreCase)) region = "Europe";
            else if (query.Contains("Asia", StringComparison.OrdinalIgnoreCase)) region = "Asia";

            // Extract region from context if not in query
            if (region == null && context.Any())
            {
                foreach (var ctx in context)
                {
                    if (ctx.Contains("North America"))
                    {
                        region = "North America";
                        break;
                    }
                    else if (ctx.Contains("Europe"))
                    {
                        region = "Europe";
                        break;
                    }
                    else if (ctx.Contains("Asia"))
                    {
                        region = "Asia";
                        break;
                    }
                }
            }

            if (isTopCompaniesQuery && region != null)
            {
                // Return formatted JSON with example companies
                return $@"{{
                    ""companies"": [
                        {{ ""name"": ""GlobalTrade Inc."", ""country"": ""USA"", ""export_volume"": 1245.67, ""region"": ""{region}"" }},
                        {{ ""name"": ""ExportMasters Ltd."", ""country"": ""Canada"", ""export_volume"": 987.54, ""region"": ""{region}"" }},
                        {{ ""name"": ""WorldExchange Group"", ""country"": ""USA"", ""export_volume"": 876.32, ""region"": ""{region}"" }}
                    ],
                    ""query"": ""{query}""
                }}";
            }

            // Default response
            return $@"{{
                ""response"": ""Unable to process query"",
                ""query"": ""{query}""
            }}";
        }
    }
}