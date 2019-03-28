using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Threading.Tasks;
using DataMining.DAL.Entities;
using Microsoft.AspNetCore.Mvc;
using WebApp.Models;
using WebApp.Services;
using WebApp.Services.Implementation;

namespace WebApp.Controllers
{
    public class HomeController : Controller
    {     
        private readonly IDataManager _dataManager;
        public HomeController(IDataManager dataManager)
        {
            _dataManager = dataManager;
        }
        public IActionResult Index()
        {
            return View();
        }

        public async Task<IActionResult> Parse()
        {
           var data = Parser.ParseFromFile("D:\\Данные по диплому\\logdet07-11-11.log");
           await _dataManager.UpdateDataBase(data);
           return Ok();
        }

      
    }
}
