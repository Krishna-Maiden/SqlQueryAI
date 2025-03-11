using System.Collections.Generic;
using System.Threading.Tasks;

namespace SQLQueryAI.Core.Interfaces
{
    /// <summary>
    /// Interface for natural language query processing
    /// </summary>
    public interface INLQueryService
    {
        /// <summary>
        /// Processes a natural language query
        /// </summary>
        /// <param name="query">The query to process</param>
        /// <returns>A tuple with the query result object and context used</returns>
        Task<(object Result, List<string> Context)> ProcessQueryAsync(string query);

        /// <summary>
        /// Rebuilds the search index
        /// </summary>
        /// <returns>True if successful</returns>
        Task<bool> RebuildIndexAsync();
    }
}