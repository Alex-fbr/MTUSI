using DataMining.DAL.Entities;
using MyLibrary.EntityFramework;
using System;
using System.Collections.Generic;
using System.Text;
using System.Threading.Tasks;

namespace DataMining.DAL.Repositories
{
    public interface ICarRepository:IBaseRepository<Car>
    {
        Car GetById(int id);
        Task<Car> GetByIdAsync(int id);
        IEnumerable<Car> Cars { get;}
    }
}
