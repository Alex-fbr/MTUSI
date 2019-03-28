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
            throw new NotImplementedException();
        }

        public Task<Car> GetByIdAsync(int id)
        {
            throw new NotImplementedException();
        }

       
        }
}
