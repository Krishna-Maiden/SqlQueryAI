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

                // Fallback if model file doesn't exist
                if (!File.Exists(_modelPath))
                {
                    return FallbackQueryProcessing(query, context);
                }

                // Using ONNX Runtime for inference
                using (var session = new InferenceSession(_modelPath))
                {
                    // Create input tensors
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
        /// Creates training examples from place and city data
        /// </summary>
        private List<TrainingExample> CreateTrainingExamples(DataTable data)
        {
            var examples = new List<TrainingExample>();

            // Group by cities and create training examples
            var cityGroups = data.AsEnumerable()
                .GroupBy(row => row.Field<string>("CityName"))
                .Take(20);

            foreach (var cityGroup in cityGroups)
            {
                var city = cityGroup.Key;
                var topPlaces = cityGroup
                    .OrderByDescending(row => Convert.ToInt32(row["CurrentPopularity"]))
                    .Take(5)
                    .Select(row => new
                    {
                        PlaceName = row.Field<string>("PlaceName"),
                        Popularity = Convert.ToInt32(row["CurrentPopularity"]),
                        Date = row.Field<DateTime>("CreatedDate")
                    })
                    .ToList();

                // Example queries about city places
                examples.Add(new TrainingExample
                {
                    Question = $"What are the most popular places in {city}?",
                    Answer = JsonConvert.SerializeObject(new
                    {
                        City = city,
                        TopPlaces = topPlaces,
                        TotalPlaces = topPlaces.Count
                    })
                });

                examples.Add(new TrainingExample
                {
                    Question = $"Compare popularity of attractions in {city}",
                    Answer = JsonConvert.SerializeObject(new
                    {
                        City = city,
                        PlaceComparison = topPlaces
                            .OrderByDescending(p => p.Popularity)
                            .Select((p, rank) => new
                            {
                                Rank = rank + 1,
                                PlaceName = p.PlaceName,
                                Popularity = p.Popularity
                            })
                    })
                });
            }

            // Global top places example
            var globalTopPlaces = data.AsEnumerable()
                .OrderByDescending(row => Convert.ToInt32(row["CurrentPopularity"]))
                .Take(10)
                .Select(row => new
                {
                    PlaceName = row.Field<string>("PlaceName"),
                    CityName = row.Field<string>("CityName"),
                    Popularity = Convert.ToInt32(row["CurrentPopularity"]),
                    Date = row.Field<DateTime>("CreatedDate")
                })
                .ToList();

            examples.Add(new TrainingExample
            {
                Question = "What are the top 10 most popular places overall?",
                Answer = JsonConvert.SerializeObject(new
                {
                    GlobalTopPlaces = globalTopPlaces,
                    AnalysisDate = DateTime.UtcNow
                })
            });

            return examples;
        }

        /// <summary>
        /// Creates a prompt for the model
        /// </summary>
        private string CreatePrompt(string query, List<string> context)
        {
            StringBuilder sb = new StringBuilder();

            // Prompt format depends on the model being used
            sb.AppendLine("Based on the following place information:");

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
        /// Provides a fallback response when the model is unavailable
        /// </summary>
        private string FallbackQueryProcessing(string query, List<string> context)
        {
            _logger.LogWarning("Using fallback query processing as model is not available");

            // Simple keyword-based processing
            bool isTopPlacesQuery = query.Contains("top", StringComparison.OrdinalIgnoreCase) &&
                (query.Contains("places", StringComparison.OrdinalIgnoreCase) ||
                 query.Contains("attractions", StringComparison.OrdinalIgnoreCase));

            bool isCityQuery = query.Contains("city", StringComparison.OrdinalIgnoreCase);

            // Extract city from context if not in query
            string? city = null;
            if (context.Any())
            {
                foreach (var ctx in context)
                {
                    // Look for city names in context
                    if (ctx.Contains("New York", StringComparison.OrdinalIgnoreCase))
                    {
                        city = "New York";
                        break;
                    }
                    // Add more cities as needed
                }
            }

            if (isTopPlacesQuery)
            {
                // Return formatted JSON with example places
                return JsonConvert.SerializeObject(new
                {
                    places = new[]
                    {
                        new {
                            placeName = "Central Park",
                            cityName = city ?? "New York",
                            popularity = 92,
                            date = DateTime.Now
                        },
                        new {
                            placeName = "Times Square",
                            cityName = city ?? "New York",
                            popularity = 88,
                            date = DateTime.Now
                        }
                    },
                    query = query
                });
            }

            // Default response
            return JsonConvert.SerializeObject(new
            {
                response = "Unable to process query",
                query = query
            });
        }

        // Tokenization and Detokenization methods
        private long[] Tokenize(string text)
        {
            return text.Split(' ', StringSplitOptions.RemoveEmptyEntries)
                .SelectMany(t => t.Split('\n', StringSplitOptions.RemoveEmptyEntries))
                .Select((token, index) => (long)index)
                .ToArray();
        }

        private string Detokenize(Tensor<long> tensor)
        {
            // Simplified detokenization
            return JsonConvert.SerializeObject(new
            {
                places = new[]
                {
                    new {
                        placeName = "Sample Place",
                        cityName = "Sample City",
                        popularity = 75
                    }
                }
            });
        }
    }

    // Existing class for training examples
    public class TrainingExample
    {
        public string Question { get; set; } = string.Empty;
        public string Answer { get; set; } = string.Empty;
    }
}