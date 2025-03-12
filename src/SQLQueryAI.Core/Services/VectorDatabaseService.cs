using Microsoft.Extensions.Logging;
using Microsoft.ML;
using SQLQueryAI.Core.Interfaces;
using SQLQueryAI.Core.Models;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using Microsoft.ML.Transforms.Text;

namespace SQLQueryAI.Core.Services
{
    /// <summary>
    /// Service for managing vector database operations
    /// </summary>
    public class VectorDatabaseService : IVectorDatabaseService
    {
        private readonly ILogger<VectorDatabaseService> _logger;
        private MLContext _mlContext;
        private PredictionEngine<TextData, TextEmbedding>? _predictionEngine;
        private List<EmbeddingData> _embeddingDataset;
        private List<PlaceEmbeddingData> _placeEmbeddingDataset;
        private bool _isPlaceData = false;

        public VectorDatabaseService(ILogger<VectorDatabaseService> logger)
        {
            _logger = logger;
            _mlContext = new MLContext(seed: 1);
            _embeddingDataset = new List<EmbeddingData>();
            _placeEmbeddingDataset = new List<PlaceEmbeddingData>();
        }

        /// <inheritdoc />
        public async Task BuildIndex(List<string> descriptions, List<CompanyData> data)
        {
            if (descriptions.Count != data.Count)
            {
                throw new ArgumentException("Descriptions and data must have the same length");
            }

            try
            {
                // Create and train the model for text embedding
                var pipeline = _mlContext.Transforms.Text.NormalizeText("NormalizedText", "Text")
                    .Append(_mlContext.Transforms.Text.TokenizeIntoWords("Tokens", "NormalizedText"))
                    .Append(_mlContext.Transforms.Text.RemoveDefaultStopWords("Tokens"))
                    .Append(_mlContext.Transforms.Text.ApplyWordEmbedding("Features", "Tokens", WordEmbeddingEstimator.PretrainedModelKind.GloVe100D));

                // Convert descriptions to IDataView
                var textData = descriptions.Select((text, index) => new TextData { Text = text, Id = index }).ToList();
                var dataView = _mlContext.Data.LoadFromEnumerable(textData);

                // Train the model
                var model = pipeline.Fit(dataView);

                // Create prediction engine
                _predictionEngine = _mlContext.Model.CreatePredictionEngine<TextData, TextEmbedding>(model);

                // Create embeddings for all descriptions
                _embeddingDataset.Clear();
                _isPlaceData = false;

                for (int i = 0; i < descriptions.Count; i++)
                {
                    var prediction = _predictionEngine.Predict(new TextData { Text = descriptions[i], Id = i });
                    _embeddingDataset.Add(new EmbeddingData
                    {
                        Id = i,
                        Description = descriptions[i],
                        Embedding = prediction.Features,
                        CompanyData = data[i]
                    });
                }

                _logger.LogInformation("Vector database built with {Count} company entries", _embeddingDataset.Count);

                // Execute the task asynchronously to avoid blocking
                await Task.CompletedTask;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error building vector index for company data");
                throw;
            }
        }

        /// <inheritdoc />
        public async Task BuildIndexForPlaces(List<string> descriptions, List<PlaceData> data)
        {
            if (descriptions.Count != data.Count)
            {
                throw new ArgumentException("Descriptions and data must have the same length");
            }

            try
            {
                // Create and train the model for text embedding
                var pipeline = _mlContext.Transforms.Text.NormalizeText("NormalizedText", "Text")
                    .Append(_mlContext.Transforms.Text.TokenizeIntoWords("Tokens", "NormalizedText"))
                    .Append(_mlContext.Transforms.Text.RemoveDefaultStopWords("Tokens"))
                    .Append(_mlContext.Transforms.Text.ApplyWordEmbedding("Features", "Tokens", WordEmbeddingEstimator.PretrainedModelKind.GloVe100D));

                // Convert descriptions to IDataView
                var textData = descriptions.Select((text, index) => new TextData { Text = text, Id = index }).ToList();
                var dataView = _mlContext.Data.LoadFromEnumerable(textData);

                // Train the model
                var model = pipeline.Fit(dataView);

                // Create prediction engine
                _predictionEngine = _mlContext.Model.CreatePredictionEngine<TextData, TextEmbedding>(model);

                // Create embeddings for all descriptions
                _placeEmbeddingDataset.Clear();
                _isPlaceData = true;

                for (int i = 0; i < descriptions.Count; i++)
                {
                    var prediction = _predictionEngine.Predict(new TextData { Text = descriptions[i], Id = i });
                    _placeEmbeddingDataset.Add(new PlaceEmbeddingData
                    {
                        Id = i,
                        Description = descriptions[i],
                        Embedding = prediction.Features,
                        PlaceData = data[i]
                    });
                }

                _logger.LogInformation("Vector database built with {Count} place entries", _placeEmbeddingDataset.Count);

                // Execute the task asynchronously to avoid blocking
                await Task.CompletedTask;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error building vector index for place data");
                throw;
            }
        }

        /// <inheritdoc />
        public async Task<(List<string> Contexts, List<CompanyData> Companies)> Search(string query, int k = 50)
        {
            if (_predictionEngine == null)
            {
                _logger.LogWarning("Search attempted but prediction engine is not initialized");
                return (new List<string>(), new List<CompanyData>());
            }

            if (_isPlaceData)
            {
                _logger.LogWarning("Search for company data attempted but place data is loaded");
                return (new List<string>(), new List<CompanyData>());
            }

            try
            {
                // Predict embedding for query
                var queryEmbedding = _predictionEngine.Predict(new TextData { Text = query });

                // Calculate cosine similarity with all embeddings
                var results = _embeddingDataset
                    .Select(item => new
                    {
                        Item = item,
                        Similarity = CosineSimilarity(queryEmbedding.Features, item.Embedding)
                    })
                    .OrderByDescending(r => r.Similarity)
                    .Take(k)
                    .ToList();

                // Return contexts and company data
                var contexts = results.Select(r => r.Item.Description).ToList();
                var companies = results.Select(r => r.Item.CompanyData).ToList();

                // Execute the task asynchronously to avoid blocking
                await Task.CompletedTask;

                return (contexts, companies);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error searching vector database for company data");
                throw;
            }
        }

        /// <inheritdoc />
        public async Task<(List<string> Contexts, List<PlaceData> Places)> SearchPlaces(string query, int k = 50)
        {
            if (_predictionEngine == null)
            {
                _logger.LogWarning("Search attempted but prediction engine is not initialized");
                return (new List<string>(), new List<PlaceData>());
            }

            if (!_isPlaceData)
            {
                _logger.LogWarning("Search for place data attempted but company data is loaded");
                return (new List<string>(), new List<PlaceData>());
            }

            try
            {
                // Predict embedding for query
                var queryEmbedding = _predictionEngine.Predict(new TextData { Text = query });

                // Calculate cosine similarity with all embeddings
                var results = _placeEmbeddingDataset
                    .Select(item => new
                    {
                        Item = item,
                        Similarity = CosineSimilarity(queryEmbedding.Features, item.Embedding)
                    })
                    .OrderByDescending(r => r.Similarity)
                    .Take(k)
                    .ToList();

                // Return contexts and place data
                var contexts = results.Select(r => r.Item.Description).ToList();
                var places = results.Select(r => r.Item.PlaceData).ToList();

                // Execute the task asynchronously to avoid blocking
                await Task.CompletedTask;

                return (contexts, places);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error searching vector database for place data");
                throw;
            }
        }

        /// <inheritdoc />
        public void SaveIndex(string path)
        {
            try
            {
                // Validate the path
                if (string.IsNullOrWhiteSpace(path))
                {
                    // Use a default path if no path is provided
                    path = Path.Combine(
                        Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData),
                        "SQLQueryAI",
                        "VectorIndex",
                        "vector_index.bin"
                    );
                }

                // Ensure directory exists
                Directory.CreateDirectory(Path.GetDirectoryName(path));

                // Serialize _embeddingDataset to disk
                using (var stream = new FileStream(path, FileMode.Create))
                using (var writer = new BinaryWriter(stream))
                {
                    // Write flag to indicate the type of data
                    writer.Write(_isPlaceData);

                    if (_isPlaceData)
                    {
                        // Write number of entries
                        writer.Write(_placeEmbeddingDataset.Count);

                        foreach (var item in _placeEmbeddingDataset)
                        {
                            // Write ID
                            writer.Write(item.Id);

                            // Write description
                            writer.Write(item.Description);

                            // Write embedding vector
                            writer.Write(item.Embedding.Length);
                            foreach (var value in item.Embedding)
                            {
                                writer.Write(value);
                            }

                            // Write place data
                            writer.Write(item.PlaceData.PlaceId.ToString());
                            writer.Write(item.PlaceData.CityId.ToString());
                            writer.Write(item.PlaceData.PlaceName);
                            writer.Write(item.PlaceData.CityName);
                            writer.Write(item.PlaceData.CurrentPopularity);
                            writer.Write(item.PlaceData.CreatedDate.Ticks);
                        }
                    }
                    else
                    {
                        // Write number of entries
                        writer.Write(_embeddingDataset.Count);

                        foreach (var item in _embeddingDataset)
                        {
                            // Write ID
                            writer.Write(item.Id);

                            // Write description
                            writer.Write(item.Description);

                            // Write embedding vector
                            writer.Write(item.Embedding.Length);
                            foreach (var value in item.Embedding)
                            {
                                writer.Write(value);
                            }

                            // Write company data
                            writer.Write(item.CompanyData.Id);
                            writer.Write(item.CompanyData.Name);
                            writer.Write(item.CompanyData.Country);
                            writer.Write(item.CompanyData.Region);
                            writer.Write(item.CompanyData.Revenue);
                            writer.Write(item.CompanyData.ExportVolume);
                        }
                    }
                }

                _logger.LogInformation("Vector index saved to {Path}", path);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error saving vector index");
                throw;
            }
        }

        /// <inheritdoc />
        public bool LoadIndex(string path)
        {
            try
            {
                if (!File.Exists(path))
                {
                    _logger.LogWarning("Cannot load index, file does not exist: {Path}", path);
                    return false;
                }

                // Load serialized dataset from disk
                using (var stream = new FileStream(path, FileMode.Open))
                using (var reader = new BinaryReader(stream))
                {
                    // Read flag indicating the type of data
                    _isPlaceData = reader.ReadBoolean();

                    if (_isPlaceData)
                    {
                        _placeEmbeddingDataset.Clear();

                        // Read number of entries
                        int placeCount = reader.ReadInt32();

                        for (int i = 0; i < placeCount; i++)
                        {
                            // Read ID
                            int id = reader.ReadInt32();

                            // Read description
                            string description = reader.ReadString();

                            // Read embedding vector
                            int vectorLength = reader.ReadInt32();
                            float[] embedding = new float[vectorLength];
                            for (int j = 0; j < vectorLength; j++)
                            {
                                embedding[j] = reader.ReadSingle();
                            }

                            // Read place data
                            var placeData = new PlaceData
                            {
                                PlaceId = Guid.Parse(reader.ReadString()),
                                CityId = Guid.Parse(reader.ReadString()),
                                PlaceName = reader.ReadString(),
                                CityName = reader.ReadString(),
                                CurrentPopularity = reader.ReadInt32(),
                                CreatedDate = new DateTime(reader.ReadInt64()),
                                Metadata = new Dictionary<string, object>()
                            };

                            _placeEmbeddingDataset.Add(new PlaceEmbeddingData
                            {
                                Id = id,
                                Description = description,
                                Embedding = embedding,
                                PlaceData = placeData
                            });
                        }
                    }
                    else
                    {
                        _embeddingDataset.Clear();

                        // Read number of entries
                        int companyCount = reader.ReadInt32();

                        for (int i = 0; i < companyCount; i++)
                        {
                            // Read ID
                            int id = reader.ReadInt32();

                            // Read description
                            string description = reader.ReadString();

                            // Read embedding vector
                            int vectorLength = reader.ReadInt32();
                            float[] embedding = new float[vectorLength];
                            for (int j = 0; j < vectorLength; j++)
                            {
                                embedding[j] = reader.ReadSingle();
                            }

                            // Read company data
                            var companyData = new CompanyData
                            {
                                Id = reader.ReadInt32(),
                                Name = reader.ReadString(),
                                Country = reader.ReadString(),
                                Region = reader.ReadString(),
                                Revenue = reader.ReadDecimal(),
                                ExportVolume = reader.ReadDecimal(),
                                Metadata = new Dictionary<string, object>()
                            };

                            _embeddingDataset.Add(new EmbeddingData
                            {
                                Id = id,
                                Description = description,
                                Embedding = embedding,
                                CompanyData = companyData
                            });
                        }
                    }
                }

                // Recreate prediction engine
                // This assumes you're using the same ML model each time
                var pipeline = _mlContext.Transforms.Text.NormalizeText("NormalizedText", "Text")
                    .Append(_mlContext.Transforms.Text.TokenizeIntoWords("Tokens", "NormalizedText"))
                    .Append(_mlContext.Transforms.Text.RemoveDefaultStopWords("Tokens"))
                    .Append(_mlContext.Transforms.Text.ApplyWordEmbedding("Features", "Tokens", WordEmbeddingEstimator.PretrainedModelKind.GloVe100D));

                // Create a sample dataview to fit the pipeline
                var sampleText = new List<TextData> { new TextData { Text = "Sample text", Id = 0 } };
                var dataView = _mlContext.Data.LoadFromEnumerable(sampleText);
                var model = pipeline.Fit(dataView);

                // Create prediction engine
                _predictionEngine = _mlContext.Model.CreatePredictionEngine<TextData, TextEmbedding>(model);

                string dataType = _isPlaceData ? "place" : "company";
                int entryCount = _isPlaceData ? _placeEmbeddingDataset.Count : _embeddingDataset.Count;
                _logger.LogInformation("Vector index loaded from {Path} with {Count} {Type} entries",
                    path, entryCount, dataType);

                return true;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error loading vector index");
                return false;
            }
        }

        /// <summary>
        /// Calculates cosine similarity between two vectors
        /// </summary>
        private float CosineSimilarity(float[] vectorA, float[] vectorB)
        {
            float dotProduct = 0;
            float normA = 0;
            float normB = 0;

            for (int i = 0; i < vectorA.Length; i++)
            {
                dotProduct += vectorA[i] * vectorB[i];
                normA += vectorA[i] * vectorA[i];
                normB += vectorB[i] * vectorB[i];
            }

            return dotProduct / ((float)Math.Sqrt(normA) * (float)Math.Sqrt(normB));
        }

        // Data classes for ML.NET
        private class TextData
        {
            public string Text { get; set; } = string.Empty;
            public int Id { get; set; }
        }

        private class TextEmbedding
        {
            public float[] Features { get; set; } = Array.Empty<float>();
        }

        private class EmbeddingData
        {
            public int Id { get; set; }
            public string Description { get; set; } = string.Empty;
            public float[] Embedding { get; set; } = Array.Empty<float>();
            public CompanyData CompanyData { get; set; } = new CompanyData();
        }

        private class PlaceEmbeddingData
        {
            public int Id { get; set; }
            public string Description { get; set; } = string.Empty;
            public float[] Embedding { get; set; } = Array.Empty<float>();
            public PlaceData PlaceData { get; set; } = new PlaceData();
        }
    }
}