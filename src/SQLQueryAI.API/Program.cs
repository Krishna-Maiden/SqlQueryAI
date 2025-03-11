using Microsoft.AspNetCore.Hosting;
using Microsoft.Extensions.Logging;
using SQLQueryAI.Core.Interfaces;
using SQLQueryAI.Core.Services;

namespace SQLQueryAI.API
{
    public class Program
    {
        public static void Main(string[] args)
        {
            var host = CreateHostBuilder(args).Build();

            // Initialize services on startup if needed
            using (var scope = host.Services.CreateScope())
            {
                var services = scope.ServiceProvider;
                try
                {
                    var dataService = services.GetRequiredService<IDataPreparationService>();
                    var logger = services.GetRequiredService<ILogger<Program>>();
                    logger.LogInformation("Application started. Use the admin endpoints to initialize vector database if needed.");

                    // Uncomment to rebuild vector database on startup (time-consuming)
                    // dataService.RebuildVectorDatabaseAsync().Wait();
                }
                catch (Exception ex)
                {
                    var logger = services.GetRequiredService<ILogger<Program>>();
                    logger.LogError(ex, "An error occurred while initializing services.");
                }
            }

            host.Run();
        }

        public static IHostBuilder CreateHostBuilder(string[] args) =>
            Host.CreateDefaultBuilder(args)
        .ConfigureWebHostDefaults(webBuilder =>
                {
                    webBuilder.UseStartup<Startup>();
                });
    }
}