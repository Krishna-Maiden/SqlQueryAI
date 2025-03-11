using Microsoft.AspNetCore.Mvc;
using SQLQueryAI.Core.Interfaces;
using SQLQueryAI.Core.Services;

namespace SQLQueryAI.API.Controllers
{
    [ApiController]
    [Route("api/[controller]")]
    public class AdminController : ControllerBase
    {
        private readonly ILogger<AdminController> _logger;
        private readonly IDataPreparationService _dataService;
        private readonly IModelService _modelService;

        public AdminController(
            ILogger<AdminController> logger,
            IDataPreparationService dataService,
            IModelService modelService)
        {
            _logger = logger;
            _dataService = dataService;
            _modelService = modelService;
        }

        /// <summary>
        /// Retrain the AI model using the current data
        /// </summary>
        [HttpPost("retrain-model")]
        [ProducesResponseType(StatusCodes.Status200OK)]
        [ProducesResponseType(StatusCodes.Status500InternalServerError)]
        public async Task<ActionResult> RetrainModel()
        {
            try
            {
                await _modelService.TrainModelAsync();
                return Ok("Model retrained successfully");
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error retraining model");
                return StatusCode(500, "An error occurred while retraining the model");
            }
        }

        /// <summary>
        /// Get current system status including vector database and model status
        /// </summary>
        [HttpGet("system-status")]
        [ProducesResponseType(StatusCodes.Status200OK)]
        public ActionResult GetSystemStatus()
        {
            var status = new
            {
                VectorDbStatus = _dataService.GetDatabaseStatus(),
                ModelStatus = _modelService.GetModelStatus(),
                SystemHealth = "OK"
            };

            return Ok(status);
        }
    }
}