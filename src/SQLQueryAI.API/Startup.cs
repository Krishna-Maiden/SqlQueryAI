using Microsoft.OpenApi.Models;
using SQLQueryAI.Core.Interfaces;
using SQLQueryAI.Core.Services;

namespace SQLQueryAI.API
{
    public class Startup
    {
        public Startup(IConfiguration configuration)
        {
            Configuration = configuration;
        }

        public IConfiguration Configuration { get; }

        public void ConfigureServices(IServiceCollection services)
        {
            services.AddControllers();

            // Register application services
            services.AddSingleton<IDataPreparationService, DataPreparationService>();
            services.AddSingleton<IVectorDatabaseService, VectorDatabaseService>();
            services.AddSingleton<IModelService, ModelService>();
            services.AddSingleton<INLQueryService, NLQueryService>();

            // Add Swagger
            services.AddSwaggerGen(c =>
            {
                c.SwaggerDoc("v1", new OpenApiInfo
                {
                    Title = "SQL Natural Language Query API",
                    Version = "v1",
                    Description = "API for querying SQL data using natural language"
                });
            });

            // Add CORS
            services.AddCors(options =>
            {
                options.AddPolicy("AllowAllOrigins",
                    builder => builder.AllowAnyOrigin()
                                      .AllowAnyMethod()
                                      .AllowAnyHeader());
            });
        }

        public void Configure(IApplicationBuilder app, IWebHostEnvironment env)
        {
            if (env.IsDevelopment())
            {
                app.UseDeveloperExceptionPage();
            }

            app.UseSwagger();
            app.UseSwaggerUI(c => c.SwaggerEndpoint("/swagger/v1/swagger.json", "SQL Natural Language Query API v1"));

            app.UseHttpsRedirection();
            app.UseStaticFiles();
            app.UseRouting();
            app.UseCors("AllowAllOrigins");
            app.UseAuthorization();

            app.UseEndpoints(endpoints =>
            {
                endpoints.MapControllers();
                // Serve default file for root path
                endpoints.MapFallbackToFile("index.html");
            });
        }
    }
}