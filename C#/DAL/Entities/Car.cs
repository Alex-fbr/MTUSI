using System;
using System.Collections.Generic;
using System.ComponentModel.DataAnnotations.Schema;
using System.Text;

namespace DataMining.DAL.Entities
{
    [Table("Cars", Schema = "dbo")]
    public class Car
    {
        public int Id { get; set; }

        /// <summary>
        /// номер полосы;
        /// </summary>
        public string Name { get; set; }

        /// <summary>
        ///  количество автомобилей(шт);
        /// </summary>
        public int Volume { get; set; }

        /// <summary>
        /// нагрузка(%);
        /// </summary>
        public double Occupancy { get; set; }

        /// <summary>
        ///  Speed - скорость(км/ч);
        /// </summary>
        public double Speed { get; set; }

        /// <summary>
        /// 85%  Speed - скорость(км/ч);
        /// </summary>
        public double SpeedKPH { get; set; }

        /// <summary>
        /// тип автомобиля
        /// </summary>
        public int? C1 { get; set; }
        public int? C2 { get; set; }
        public int? C3 { get; set; }
        public int? C4 { get; set; }
        public int? C5 { get; set; }
        public int? C6 { get; set; }
        public int? C7 { get; set; }
        public int? C8 { get; set; }


        /// <summary>
        ///  дистанция от переднего бампера ведомого до переднего бампера   ведущего(c);
        /// </summary>
        public double Headway { get; set; }

        /// <summary>
        ///  дистанция от переднего бампера ведомого до заднего бампера ведущего(c)
        /// </summary>
        public double Gap { get; set; }

        public DateTime Date { get; set; }
    }
}
