using DataMining.DAL.Entities;
using DataMining.DAL.Repositories;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

namespace WebApp.Services.Implementation
{
    public class DataManager: IDataManager
    {
        private readonly ICarRepository _carRepository;
        public DataManager(ICarRepository carRepository)
        {
            _carRepository = carRepository;
        }
        public async Task UpdateDataBase(List<Car> data)
        {
            foreach (var item in data)
            {
                await _carRepository.InsertAsync(item);
            }
        }
    }
}
