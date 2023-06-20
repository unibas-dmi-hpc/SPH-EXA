#pragma once

#include <vector>
#include <fstream>
#include <mpi.h>

#ifdef USE_PMT
#include "pmt.h"
#endif

#define NUM_GPUS_PER_NODE 4

namespace sphexa
{

enum deviceType
{
    GPU = 0,
    CPU = 4,
    MEM = 5,
    CN  = 6
};

class CrayPmtReader
{
    deviceType          devType;
    int                 _rank;
    std::vector<float>  funcEnergies; // energy measurements for each rank
    std::vector<float*> energies;     // vector of energy measurements for each rank
#ifdef USE_PMT
    std::unique_ptr<pmt::PMT> sensor;
    pmt::State                pmt_start, pmt_end;
#endif

    CrayPmtReader()
    {
#ifdef USE_PMT
        if (devType == GPU)
        {
            int gpu_id = _rank % NUM_GPUS_PER_NODE;
            sensor     = pmt::cray::Cray::create(gpu_id); // Change for reading
        }
        else
            sensor = pmt::cray::Cray::create(devType);
#endif
    }

public:
    CrayPmtReader(deviceType dt, int rank)
        : _rank(rank)
    {
        devType = dt;
        CrayPmtReader();
    }

    void startReader()
    {
#ifdef USE_PMT
        pmt_start = pmt_end = sensor->read();
#endif
    }

    void readEnergy()
    {
        float energy = 0.0;
#ifdef USE_PMT
        pmt_end   = sensor->read();
        energy    = sensor->joules(pmt_start, pmt_end);
        pmt_start = sensor->read();
#endif
        funcEnergies.push_back(energy);
    }

    void gatherEnergies(int numRanks)
    {
        std::cout << "Gathering energy data." << std::endl;
        std::vector<float> durs;
        durs.resize(numRanks * funcEnergies.size());

        float* durations = durs.data();
        MPI_Gather(funcEnergies.data(), funcEnergies.size(), MPI_FLOAT, &durations[_rank * funcEnergies.size()],
                   funcEnergies.size(), MPI_FLOAT, 0, MPI_COMM_WORLD);

        if (_rank == 0)
        {
            for (int i = 0; i < funcEnergies.size(); i++)
            {
                float* funcDurations = new float[numRanks];
                for (int r = 0; r < numRanks; r++)
                {
                    funcDurations[r] = durations[r * funcEnergies.size() + i];
                }
                energies.push_back(funcDurations);
            }
        }
    }

    void printEnergyMeasurements(int numRanks)
    {
        if (_rank == 0)
        {
            std::ofstream energyFile;
            std::string   efilename;

            switch (devType)
            {
                case GPU: efilename = "energy-GPU" + std::to_string(numRanks) + "Ranks.txt"; break;
                case CPU: efilename = "energy-CPU" + std::to_string(numRanks) + "Ranks.txt"; break;
                case MEM: efilename = "energy-MEM" + std::to_string(numRanks) + "Ranks.txt"; break;
                case CN: efilename = "energy-CN" + std::to_string(numRanks) + "Ranks.txt"; break;
                default: efilename = "error-energy-file.txt"; break;
            }
            energyFile.open(efilename);

            for (int i = 0; i < numRanks; i++)
            {
                energyFile << "RANK" << i << ",";
            }
            energyFile << std::endl;
            for (auto& element : energies)
            {
                for (int i = 0; i < numRanks; i++)
                {
                    energyFile << element[i] << ",";
                }

                energyFile << std::endl;
                delete[] element;
            }

            std::cout << "Energy data written." << std::endl;
            energyFile.close();
        }
    }
};

class Profiler
{
private:
    std::vector<float*>        timeSteps; // vector of timesteps for all ranks
    std::vector<float>         funcTimes; // vector of timesteps for each rank
    int                        _numRanks;
    int                        _rank;
    std::vector<CrayPmtReader> vecEnergyReaders;

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
        CrayPmtReader gpuReader(GPU, rank);
        CrayPmtReader cpuReader(CPU, rank);
        CrayPmtReader memReader(MEM, rank);
        CrayPmtReader cnReader(CN, rank);
        vecEnergyReaders.push_back(gpuReader);
        vecEnergyReaders.push_back(cpuReader);
        vecEnergyReaders.push_back(memReader);
        vecEnergyReaders.push_back(cnReader);
    }
    ~Profiler() {}

    void printProfilingInfo()
    {
        for (int i = 0; i < vecEnergyReaders.size(); i++)
            vecEnergyReaders.data()[i].printEnergyMeasurements(_numRanks);

        if (_rank == 0)
        {
            std::ofstream profilingFile;
            std::string   filename = "profiling-" + std::to_string(_numRanks) + "Ranks.txt";
            profilingFile.open(filename);

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

                profilingFile << std::endl;
                delete[] element;
            }

            std::cout << "Profiling data written." << std::endl;
            profilingFile.close();
        }
    }

    void saveFunctionTimings(float duration)
    {
        funcTimes.push_back(duration);
        for (int i = 0; i < vecEnergyReaders.size(); i++)
            vecEnergyReaders.data()[i].readEnergy();
    }

    void startEnergyMeasurement()
    {
        for (int i = 0; i < vecEnergyReaders.size(); i++)
            vecEnergyReaders.data()[i].startReader();
    }

    // save the total timestep timing
    void saveTimestep(float duration) { funcTimes.push_back(duration); }

    void gatherEnergies()
    {
        for (int i = 0; i < vecEnergyReaders.size(); i++)
            vecEnergyReaders.data()[i].gatherEnergies(_numRanks);
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
