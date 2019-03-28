using System;
using System.Collections.Generic;
using System.Text;
using DataMining.DAL.Entities;
using Microsoft.EntityFrameworkCore;

namespace DataMining.DAL.Internal.Implementation
{
    public class CarDbContext : DbContext, ICarDbContext
    {
        public CarDbContext(DbContextOptions<CarDbContext> options) : base(options)
        {}
     
        public DbSet<Car> Cars { get;set;}
  
    }
 
}

