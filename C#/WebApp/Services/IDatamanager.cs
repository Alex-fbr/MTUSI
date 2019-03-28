using DataMining.DAL.Entities;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

namespace WebApp.Services
{
    public interface IDataManager
    {
        Task UpdateDataBase(List<Car> data);
    }
}
