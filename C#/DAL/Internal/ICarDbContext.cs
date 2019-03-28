using DataMining.DAL.Entities;
using Microsoft.EntityFrameworkCore;
using System;
using System.Collections.Generic;
using System.Text;

namespace DataMining.DAL.Internal
{
    public interface ICarDbContext
    {
        DbSet<Car> Cars { get; set; }
    }
}
