using SQLQueryAI.Core.Models;
using System.Collections.Generic;
using System.Threading.Tasks;

namespace SQLQueryAI.Core.Interfaces
{
    /// <summary>
    /// Interface for vector database operations
    /// </summary>
    public interface IVectorDatabaseService
    {
        /// <summary>
        /// Builds a vector index from text descriptions and associated company data
        /// </summary>
        /// <param name="descriptions">List of text descriptions</param>
        /// <param name="data">List of company data corresponding to descriptions</param>
        Task BuildIndex(List<string> descriptions, List<CompanyData> data);

        /// <summary>
        /// Builds a vector index from text descriptions and associated place data
        /// </summary>
        /// <param name="descriptions">List of text descriptions</param>
        /// <param name="data">List of place data corresponding to descriptions</param>
        Task BuildIndexForPlaces(List<string> descriptions, List<PlaceData> data);

        /// <summary>
        /// Searches the vector database for similar content to the query
        /// </summary>
        /// <param name="query">The search query</param>
        /// <param name="k">Number of results to return</param>
        /// <returns>A tuple with the relevant contexts and company data</returns>
        Task<(List<string> Contexts, List<CompanyData> Companies)> Search(string query, int k = 50);

        /// <summary>
        /// Searches the vector database for similar content to the query using place data
        /// </summary>
        /// <param name="query">The search query</param>
        /// <param name="k">Number of results to return</param>
        /// <returns>A tuple with the relevant contexts and place data</returns>
        Task<(List<string> Contexts, List<PlaceData> Places)> SearchPlaces(string query, int k = 50);

        /// <summary>
        /// Saves the vector index to disk
        /// </summary>
        /// <param name="path">Path to save the index</param>
        void SaveIndex(string path);

        /// <summary>
        /// Loads a vector index from disk
        /// </summary>
        /// <param name="path">Path to the index file</param>
        /// <returns>True if successfully loaded</returns>
        bool LoadIndex(string path);
    }
}