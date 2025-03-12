using SQLQueryAI.Core.Models;
using System.Collections.Generic;
using System.Data;
using System.Threading.Tasks;

namespace SQLQueryAI.Core.Interfaces
{
    /// <summary>
    /// Interface for data preparation operations
    /// </summary>
    public interface IDataPreparationService
    {
        /// <summary>
        /// Extracts training data from the SQL database
        /// </summary>
        /// <returns>A data table with the extracted training data</returns>
        Task<DataTable> ExtractTrainingDataAsync();

        /// <summary>
        /// Gets company data for building the vector database
        /// </summary>
        /// <param name="limit">Maximum number of companies to retrieve</param>
        /// <returns>List of company data objects</returns>
        Task<List<PlaceData>> GetPlaceDataAsync(int limit = 1000);

        /// <summary>
        /// Rebuilds the vector database from SQL data
        /// </summary>
        /// <returns>True if successful</returns>
        Task<bool> RebuildVectorDatabaseAsync();

        /// <summary>
        /// Gets the current status of the database operations
        /// </summary>
        /// <returns>A status string</returns>
        string GetDatabaseStatus();
    }
}