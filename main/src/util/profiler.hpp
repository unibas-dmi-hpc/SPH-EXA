/*
 * MIT License
 *
 * Copyright (c) 2023 CSCS, ETH Zurich, University of Basel, University of Zurich
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

/*! @file
 * @brief Time and energy profiling
 *
 * @author Osman Seckin Simsek <osman.simsek@unibas.ch>
 */

#pragma once

#include <vector>
#include <fstream>
#include <mpi.h>
#include "io/h5part_wrapper.hpp"

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

public:
    CrayPmtReader()
    {
        devType = CPU;
#ifdef USE_PMT
        sensor = pmt::cray::Cray::create(devType);
#endif
    }

    CrayPmtReader(deviceType dt, int rank)
        : _rank(rank)
        , devType(dt)
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

    void printEnergyMeasurementsHDF5(H5PartFile* _energyFile, int numRanks, size_t timesteps)
    {
        H5PartSetNumParticles(_energyFile, funcEnergies.size());
        int numFuncs = funcEnergies.size() / timesteps;
        fileutils::writeH5PartStepAttrib(_energyFile, "NUM_RANKS", &numRanks, 1);
        fileutils::writeH5PartStepAttrib(_energyFile, "NUM_FUNCS", &numFuncs, 1);
        std::string fieldName = "RANK" + std::to_string(_rank);
        fileutils::writeH5PartField(_energyFile, fieldName, funcEnergies.data());
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
    H5PartFile*                _profilingFile;
    H5PartFile*                _energyFile;
    std::vector<CrayPmtReader> vecEnergyReaders;

public:
    Profiler(int rank)
        : _rank(rank)
    {
        MPI_Comm_size(MPI_COMM_WORLD, &_numRanks);
        std::string profilingFilename = "profiling.h5";
        std::string energyFilename    = "energy.h5";

        if (rank == 0 && std::filesystem::exists(profilingFilename)) { std::filesystem::remove(profilingFilename); }
        if (rank == 0 && std::filesystem::exists(energyFilename)) { std::filesystem::remove(energyFilename); }
        MPI_Barrier(MPI_COMM_WORLD);

        _profilingFile = fileutils::openH5Part(profilingFilename, H5PART_WRITE | H5PART_VFD_MPIIO_IND, MPI_COMM_WORLD);
        _energyFile    = fileutils::openH5Part(energyFilename, H5PART_WRITE | H5PART_VFD_MPIIO_IND, MPI_COMM_WORLD);

        CrayPmtReader gpuReader(GPU, rank);
        CrayPmtReader cpuReader(CPU, rank);
        CrayPmtReader memReader(MEM, rank);
        CrayPmtReader cnReader(CN, rank);
        vecEnergyReaders.push_back(std::move(gpuReader));
        vecEnergyReaders.push_back(std::move(cpuReader));
        vecEnergyReaders.push_back(std::move(memReader));
        vecEnergyReaders.push_back(std::move(cnReader));
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

    void printProfilingInfoHDF5(size_t timesteps)
    {
        H5PartSetStep(_profilingFile, 0);
        H5PartSetNumParticles(_profilingFile, funcTimes.size());
        int numFuncs = funcTimes.size() / timesteps;
        fileutils::writeH5PartStepAttrib(_profilingFile, "NUM_RANKS", &_numRanks, 1);
        fileutils::writeH5PartStepAttrib(_profilingFile, "NUM_FUNCS", &numFuncs, 1);
        std::string fieldName = "RANK" + std::to_string(_rank);
        fileutils::writeH5PartField(_profilingFile, fieldName, funcTimes.data());
        H5PartCloseFile(_profilingFile);

        for (int i = 0; i < vecEnergyReaders.size(); i++)
        {
            H5PartSetStep(_energyFile, i);
            vecEnergyReaders.data()[i].printEnergyMeasurementsHDF5(_energyFile, _numRanks, timesteps);
        }
        H5PartCloseFile(_energyFile);
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
