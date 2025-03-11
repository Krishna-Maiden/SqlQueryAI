using System.Collections.Generic;
using System.Threading.Tasks;

namespace SQLQueryAI.Core.Interfaces
{
    /// <summary>
    /// Interface for ML model operations
    /// </summary>
    public interface IModelService
    {
        /// <summary>
        /// Trains the ML model using examples generated from database data
        /// </summary>
        Task TrainModelAsync();

        /// <summary>
        /// Generates a response to a query using the trained model
        /// </summary>
        /// <param name="query">The natural language query</param>
        /// <param name="context">Context information to consider when generating the response</param>
        /// <returns>A JSON string with the response</returns>
        Task<string> GenerateResponseAsync(string query, List<string> context);

        /// <summary>
        /// Gets the current status of the model
        /// </summary>
        /// <returns>A status string</returns>
        string GetModelStatus();
    }
}