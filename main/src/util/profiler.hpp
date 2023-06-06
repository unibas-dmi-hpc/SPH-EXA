#pragma once

#include <vector>
#include <fstream>
#include <mpi.h>

namespace sphexa
{

class Profiler
{
private:
    std::vector<float*> timeSteps; // vector of timesteps for all ranks
    std::vector<float>  funcTimes; // vector of timesteps for each rank
    int                 _numRanks;
    int                 _rank;
    std::ofstream       profilingFile;

    // Metrics
    // std::vector<float>  meanPerStep;   // mean (average)
    // std::vector<double> stdevPerStep;  // standard deviation
    // std::vector<double> covPerStep;    // c.o.v = std / mean
    // std::vector<double> lambdaPerStep; // lambda = (Lmax / Lmean - 1) * 100%
    // std::vector<float*> vectorMetric;  // vector based metric calculation
    // std::vector<double> I_2PerStep;    // distance to zero for vector based metric, name tentative
    // std::vector<float>  g1PerStep;     // skewness
    // std::vector<float>  g2PerStep;     // kurtosis
    // std::vector<float>  totalTimeStep; // total time-step durations

public:
    Profiler(int rank)
        : _rank(rank)
    {
        MPI_Comm_size(MPI_COMM_WORLD, &_numRanks);
        std::string filename = "profiling-" + std::to_string(_numRanks) + "Ranks.txt";
        if (_rank == 0) profilingFile.open(filename);
    }
    ~Profiler() {}

    // void printMetrics(int iteration)
    // {
    //     profilingFile << meanPerStep.at(iteration) << "," << stdevPerStep.at(iteration) << ","
    //                   << covPerStep.at(iteration) << "," << I_2PerStep.at(iteration) << ","
    //                   << lambdaPerStep.at(iteration);
    //     for (int i = 0; i < _numRanks; i++)
    //     {
    //         profilingFile << "," << vectorMetric.at(iteration)[i];
    //     }
    // }

    void printProfilingInfo()
    {
        if (_rank == 0)
        {
            size_t iter = 1;
            // profilingFile << std::setw(3) << " ";
            for (int i = 0; i < _numRanks; i++)
            {
                profilingFile << "RANK" << i << ",";
            }
            // profilingFile << "mean,"
            //               << "stdev,"
            //               << "CoV,"
            //               << "I_2,"
            //               << "lambda";
            // for (int i = 0; i < _numRanks; i++)
            // {
            //     profilingFile << ",Vector" << i;
            // }
            profilingFile << std::endl;
            for (auto& element : timeSteps)
            {
                // calculateMetrics(element);
                // profilingFile << std::setw(3) << iter << " ";
                for (int i = 0; i < _numRanks; i++)
                {
                    profilingFile << element[i] << ",";
                }
                // printMetrics(iter - 1);
                iter++;
                profilingFile << std::endl;
                delete[] element;
            }
            profilingFile << std::endl;
            std::cout << "Profiling data written." << std::endl;
        }
        profilingFile.close();
    }

    void saveTimings(float duration) { funcTimes.push_back(duration); }

    void saveTimestep(float duration)
    {
        funcTimes.push_back(duration); // save the total timestep timing
    }

    void gatherTimings()
    {
        std::cout << "Gathering profiling data." << std::endl;
        std::vector<float> durs;
        durs.resize(_numRanks * funcTimes.size());

        float* durations = durs.data();
        MPI_Gather(funcTimes.data(), funcTimes.size(), MPI_FLOAT, &durations[_rank * funcTimes.size()],
                   funcTimes.size(), MPI_FLOAT, 0, MPI_COMM_WORLD);

        if (_rank == 0)
        {
            for (int i = 0; i < funcTimes.size(); i++)
            {
                float* funcDurations = new float[_numRanks];
                for (int r = 0; r < _numRanks; r++)
                {
                    funcDurations[r] = durations[r * funcTimes.size() + i];
                }
                timeSteps.push_back(funcDurations);
            }
        }
    }

    // LiB metrics calculations
    // void calculateMetrics(float* durations)
    // {
    //     const std::vector<float> durData(durations, durations + _numRanks);
    //     float*                   vecMetric = new float[_numRanks];

    //     float mean   = std::reduce(durData.begin(), durData.end()) / durData.size();
    //     float sq_sum = std::inner_product(durData.begin(), durData.end(), durData.begin(), 0.0);
    //     float stdev  = std::sqrt(sq_sum / durData.size() - mean * mean);
    //     float Lmax   = *std::max_element(durData.begin(), durData.end());
    //     float lambda = (Lmax / mean - 1.0f) * 100;

    //     for (int i = 0; i < _numRanks; i++)
    //     {
    //         vecMetric[i] = 1 - (durData.at(i) / mean);
    //     }
    //     float sumsqVector = std::inner_product(vecMetric, vecMetric + _numRanks, vecMetric, 0.0);

    //     meanPerStep.push_back(mean);
    //     stdevPerStep.push_back(stdev);
    //     lambdaPerStep.push_back(lambda);
    //     covPerStep.push_back(stdev / mean);
    //     vectorMetric.push_back(vecMetric);
    //     I_2PerStep.push_back(std::sqrt(sumsqVector));
    // }
};

} // namespace sphexa
