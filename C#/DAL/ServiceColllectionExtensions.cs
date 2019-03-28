using DataMining.DAL.Internal;
using DataMining.DAL.Internal.Implementation;
using MyLibrary.EntityFramework;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.DependencyInjection;
using MyLibrary.EntityFramework.Implementation;

namespace DataMining.DAL
{
    public static class ServiceCollectionExtensions
    {
        public static void RegisterCarDbContext(this IServiceCollection services, IConfiguration configuration)
        {
            services.RegisterDbContext<ICarDbContext, CarDbContext>(() => configuration.GetConnectionString("MTUSIDB"));
        }

    }
}
