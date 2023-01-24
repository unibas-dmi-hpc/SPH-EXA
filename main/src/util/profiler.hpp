#pragma once

#include <vector>
#include <iostream>
#include <mpi.h>

namespace sphexa
{

class Profiler
{
private:
    std::vector<float*> timeSteps; // vector of timesteps for all ranks
    std::vector<float> meanPerStep; // mean (average)
    std::vector<float> stdevPerStep; // standard deviation
    std::vector<float> covPerStep; // c.o.v = std / mean
    std::vector<float> I_2PerStep; // distance to zero for vector based metric, name tentative
    std::vector<float> g1PerStep; // skewness
    std::vector<float> g2PerStep; // kurtosis
    int _numRanks;
    int _rank;

public:
    Profiler(int rank)
        :_rank(rank)
    {
        MPI_Comm_size(MPI_COMM_WORLD, &_numRanks);
    }
    ~Profiler(){}

    void printMetrics(int iteration)
    {
        std::cout << std::setw(15) <<  meanPerStep.at(iteration) << "," << std::setw(15) <<  stdevPerStep.at(iteration) << ",";
        std::cout << std::setw(15) <<  covPerStep.at(iteration) << "," << std::setw(15) <<  I_2PerStep.at(iteration) << ",";
    }

    void printProfilingInfo()
    {
        if (_rank == 0)
        {
            size_t iter = 1;
            for (auto & element : timeSteps)
            {
                std::cout << std::endl << "time-step = " << iter << ",\t";
                for(int i = 0; i < _numRanks; i++)
                {
                    std::cout << std::setw(15) << element[i] << ",";
                }
                printMetrics(iter-1);
                iter++;
            }
            std::cout << std::endl;
        }
    }
    
    void gatherTimings(float duration, size_t iteration)
    {
        // float* dur = timeSteps.at(iteration-1);
        // float x = dur[iteration-1];
        float* durations = new float[_numRanks];
        MPI_Gather(&duration, 1, MPI_FLOAT, &durations[_rank], 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
        // std::cout << "gather_timings" << " iter = " << iteration << ", rank = " << _rank << ", duration = " << duration <<  std::endl;

        if (_rank == 0)
        {
            timeSteps.push_back(durations);
            // calculate mean, stdev, c.o.v, g1, g2, I_2
            calculateMetrics(durations);
            // std::cout << "dur1 = " << durations[0] << " , dur2 = " << durations[1] << std::endl;
            // std::cout << "ts1 = " << timeSteps.back()[0] << " , ts2 = " << timeSteps.back()[1] << std::endl;
        }
    }

    // LiB metrics calculations
    void calculateMetrics(float* durations)
    {
        std::vector<float> durData(durations, durations + _numRanks);

        float mean = std::reduce(durData.begin(), durData.end()) / durData.size();
        float sq_sum = std::inner_product(durData.begin(), durData.end(), durData.begin(), 0.0);
        float stdev = std::sqrt(sq_sum / durData.size() - mean * mean);

        meanPerStep.push_back(mean);
        stdevPerStep.push_back(stdev);
        covPerStep.push_back(stdev/mean);
        I_2PerStep.push_back(std::sqrt(sq_sum));
    }
};

} // namespace sphexa
