#pragma once

#include <vector>
#include <fstream>
#include <mpi.h>

#ifdef USE_PMT
#include "pmt.h"
#endif

namespace sphexa
{

class EnergyReader
{
#ifdef USE_PMT
    std::unique_ptr<pmt::PMT> cpu_sensor;
    std::unique_ptr<pmt::PMT> gpu_sensor;
    pmt::State                cpu_pmt_start, cpu_pmt_end;
    pmt::State                gpu_pmt_start, gpu_pmt_end;
#endif

public:
    EnergyReader()
    {
#ifdef USE_PMT
        cpu_sensor = pmt::cray::Cray::create();
        gpu_sensor = pmt::nvml::NVML::create();
#endif
    }

    void startReader()
    {
#ifdef USE_PMT
        pmt_cpu_pmt_startstart = cpu_pmt_end = cpu_sensor->read();
        gpu_pmt_start = gpu_pmt_end = gpu_sensor->read();
#endif
    }

    float readEnergy()
    {
        float cpuEnergy = 0.0;
        float gpuEnergy = 0.0;
#ifdef USE_PMT
        cpu_pmt_end   = cpu_sensor->read();
        gpuEnergy     = gpu_sensor->joules(gpu_pmt_start, gpu_pmt_end);
        cpuEnergy     = cpu_sensor->joules(cpu_pmt_start, cpu_pmt_end);
        cpu_pmt_start = cpu_sensor->read();
#endif
        return cpuEnergy + gpuEnergy;
    }
};

class Profiler
{
private:
    std::vector<float*> timeSteps;    // vector of timesteps for all ranks
    std::vector<float>  funcTimes;    // vector of timesteps for each rank
    std::vector<float>  funcEnergies; // vector of energy measurements for each rank
    std::vector<float*> energies;     // vector of energy measurements for each rank
    int                 _numRanks;
    int                 _rank;
    std::ofstream       profilingFile;
    std::ofstream       energyFile;
    EnergyReader        enReader;

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

    void saveFunctionEnergy()
    {
        float totalEnergy = enReader.readEnergy();
        funcEnergies.push_back(totalEnergy);
    }

public:
    Profiler(int rank)
        : _rank(rank)
        , enReader()
    {
        MPI_Comm_size(MPI_COMM_WORLD, &_numRanks);
        std::string filename  = "profiling-" + std::to_string(_numRanks) + "Ranks.txt";
        std::string efilename = "energy-" + std::to_string(_numRanks) + "Ranks.txt";
        if (_rank == 0)
        {
            profilingFile.open(filename);
            energyFile.open(efilename);
        }
    }
    ~Profiler() {}

    void printEnergyMeasurements()
    {
        if (_rank == 0)
        {
            size_t iter = 1;
            for (int i = 0; i < _numRanks; i++)
            {
                energyFile << "RANK" << i << ",";
            }
            energyFile << std::endl;
            for (auto& element : energies)
            {
                for (int i = 0; i < _numRanks; i++)
                {
                    energyFile << element[i] << ",";
                }

                iter++;
                energyFile << std::endl;
                delete[] element;
            }
            energyFile << std::endl;
            std::cout << "Energy data written." << std::endl;
        }
        energyFile.close();
    }

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
            profilingFile << std::endl;
            for (auto& element : timeSteps)
            {
                for (int i = 0; i < _numRanks; i++)
                {
                    profilingFile << element[i] << ",";
                }

                iter++;
                profilingFile << std::endl;
                delete[] element;
            }
            profilingFile << std::endl;
            std::cout << "Profiling data written." << std::endl;
        }
        profilingFile.close();
    }

    void saveFunctionTimings(float duration)
    {
        funcTimes.push_back(duration);
        saveFunctionEnergy();
    }

    void startEnergyMeasurement() { enReader.startReader(); }

    // save the total timestep timing
    void saveTimestep(float duration) { funcTimes.push_back(duration); }

    void gatherEnergies()
    {
        std::cout << "Gathering energy data." << std::endl;
        std::vector<float> durs;
        durs.resize(_numRanks * funcEnergies.size());

        float* durations = durs.data();
        MPI_Gather(funcEnergies.data(), funcEnergies.size(), MPI_FLOAT, &durations[_rank * funcEnergies.size()],
                   funcEnergies.size(), MPI_FLOAT, 0, MPI_COMM_WORLD);

        if (_rank == 0)
        {
            for (int i = 0; i < funcEnergies.size(); i++)
            {
                float* funcDurations = new float[_numRanks];
                for (int r = 0; r < _numRanks; r++)
                {
                    funcDurations[r] = durations[r * funcEnergies.size() + i];
                }
                energies.push_back(funcDurations);
            }
        }
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
};

} // namespace sphexa
