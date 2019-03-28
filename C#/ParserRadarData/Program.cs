using System;
using System.Diagnostics;

namespace ParserRadarData
{
    class Program
    {
        static void Main(string[] args)
        {
            Console.WriteLine("Hello World!");

            // Create new stopwatch
            Stopwatch stopwatch = new Stopwatch();

            // Begin timing
            stopwatch.Start();

            // Do something
            Parser.ParseFromFile("D:\\Данные по диплому\\logdet07-11-11.log");
            // Stop timing
            stopwatch.Stop();

            // Write result
            Console.WriteLine("Time elapsed: {0}", stopwatch.ElapsedMilliseconds);
        }
    }
}
