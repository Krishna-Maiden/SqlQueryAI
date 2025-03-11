using Microsoft.AspNetCore.Mvc;
using SQLQueryAI.Core.Interfaces;
using SQLQueryAI.Core.Models;
using System.Diagnostics;

namespace SQLQueryAI.API.Controllers
{
    [ApiController]
    [Route("api/[controller]")]
    public class QueryController : ControllerBase
    {
        private readonly ILogger<QueryController> _logger;
        private readonly INLQueryService _queryService;

        public QueryController(ILogger<QueryController> logger, INLQueryService queryService)
        {
            _logger = logger;
            _queryService = queryService;
        }

        /// <summary>
        /// Process a natural language query against the SQL data
        /// </summary>
        /// <param name="request">The query request containing the natural language query</param>
        /// <returns>Query results with source context</returns>
        [HttpPost]
        [ProducesResponseType(typeof(QueryResponse), StatusCodes.Status200OK)]
        [ProducesResponseType(StatusCodes.Status400BadRequest)]
        [ProducesResponseType(StatusCodes.Status500InternalServerError)]
        public async Task<ActionResult<QueryResponse>> Query(QueryRequest request)
        {
            if (string.IsNullOrEmpty(request.Query))
            {
                return BadRequest("Query cannot be empty");
            }

            try
            {
                var stopwatch = Stopwatch.StartNew();
                var (result, context) = await _queryService.ProcessQueryAsync(request.Query);
                stopwatch.Stop();

                var response = new QueryResponse
                {
                    Result = result,
                    SourceContext = context,
                    ProcessingTimeMs = stopwatch.ElapsedMilliseconds
                };

                return Ok(response);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error processing query: {Query}", request.Query);
                return StatusCode(500, "An error occurred while processing your request");
            }
        }

        /// <summary>
        /// Rebuild the vector index for faster query processing
        /// </summary>
        [HttpPost("rebuild-index")]
        [ProducesResponseType(StatusCodes.Status200OK)]
        [ProducesResponseType(StatusCodes.Status500InternalServerError)]
        public async Task<ActionResult> RebuildIndex()
        {
            try
            {
                await _queryService.RebuildIndexAsync();
                return Ok("Index rebuilt successfully");
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error rebuilding index");
                return StatusCode(500, "An error occurred while rebuilding the index");
            }
        }
    }
}