using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using DataMining.DAL.Entities;
using DataMining.DAL.Internal;
using Microsoft.EntityFrameworkCore;
using MyLibrary.EntityFramework.Implementation;

namespace DataMining.DAL.Repositories.Implementation
{
    public class CarRepository :  BaseRepository<Car>, ICarRepository
    {
        private readonly ICarDbContext _carDbContext;

        public CarRepository(ICarDbContext carDbContext) : base(carDbContext.Cars)
        {
            _carDbContext = carDbContext;
        }

        
        public IEnumerable<Car> Cars => _carDbContext.Cars;


        public Car GetById(int id)
        {
            return  DbSet.SingleOrDefault(x => x.Id == id);
        }

        public async Task<Car> GetByIdAsync(int id)
        {
            return await DbSet.SingleOrDefaultAsync(x => x.Id == id);
        }

        public async Task<Car> GetByNameAndDateAsync(string name, DateTime date)
        {
            return await DbSet.FirstOrDefaultAsync(x => x.Date == date && x.Name == name);
        }
    }
}
