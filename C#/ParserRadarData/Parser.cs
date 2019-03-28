using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;

namespace ParserRadarData
{
    /// <summary>
    /// парсер строки вида:
    ///   LANE_01        0      0,0    20,8    20,9    0    0    -    -    -    -    -    -      0,0      0,0   2011-11-07 00:00:00       60    -     -     -     -     -     -     -     -     -      -      -      -      -      -      -        -        -
    /// </summary>
    public static class Parser
    {
        public static List<Car> ParseFromFile(string filePath) 
        {
            var result = new List<Car>();
            if (!string.IsNullOrEmpty(filePath))
            {
                using (StreamReader sr = new StreamReader(filePath, System.Text.Encoding.Default))
                {
                    string line;
                    while ((line = sr.ReadLine()) != null)
                    {
                      var car =  ParseString(line);
                      result.Add(car);
                    }
                }
            }

            return result;
        }
        public static Car ParseString(string str)
        {
            var result = new Car();
            
            if (str.Any())
            {
                var car = new Car();

                car.Name = GetNextField(ref str);
                car.Volume = int.Parse(GetNextField(ref str));
                car.Occupancy = double.Parse(GetNextField(ref str));
                car.Speed = double.Parse(GetNextField(ref str));
                car.SpeedKPH = double.Parse(GetNextField(ref str));
                car.C1 = int.Parse(GetNextField(ref str));
                car.C2 = int.Parse(GetNextField(ref str));
               
                car.C3 = !string.IsNullOrEmpty(GetNextField(ref str)) ? int.Parse(GetNextField(ref str)) : 0;
                car.C4 = !string.IsNullOrEmpty(GetNextField(ref str)) ? int.Parse(GetNextField(ref str)) : 0;
                car.C5 = !string.IsNullOrEmpty(GetNextField(ref str)) ? int.Parse(GetNextField(ref str)) : 0;
                car.C6 = !string.IsNullOrEmpty(GetNextField(ref str)) ? int.Parse(GetNextField(ref str)) : 0;
                car.C7 = !string.IsNullOrEmpty(GetNextField(ref str)) ? int.Parse(GetNextField(ref str)) : 0;
                car.C8 = !string.IsNullOrEmpty(GetNextField(ref str)) ? int.Parse(GetNextField(ref str)) : 0;

                car.Headway = double.Parse(GetNextField(ref str));
                car.Gap = double.Parse(GetNextField(ref str));
                car.Date = (DateTime.Parse(GetNextField(ref str))).Date;

                return car;
            }

            return result;
        }

        private static string GetNextField(ref string str)
        {
            string field = "";
            int firstSymbol = 0;
            int firstWhiteSpace = 0;
            for (int i = 0; i < str.Length; i++)
            {
                if (str[i] != ' ')
                {
                    firstSymbol = i;
                    str = str.Substring(firstSymbol);
                    firstWhiteSpace = str.IndexOf(' ');
                    break;
                }
            }
       
            field = str.Substring(0, firstWhiteSpace);
            if (field == "-") field = null;
            str = str.Substring(firstWhiteSpace);
            return field;
        }
    }
}
