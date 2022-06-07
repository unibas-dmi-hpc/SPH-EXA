/*
 * MIT License
 *
 * Copyright (c) 2021 CSCS, ETH Zurich
 *               2021 University of Basel
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
 * @brief SPH-EXA application front-end and main function
 *
 * @author Ruben Cabezon <ruben.cabezon@unibas.ch>
 * @author Aurelien Cavelan
 * @author Jose A. Escartin <ja.escartin@gmail.com>
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#include <iostream>
#include <string>
#include <memory>
#include <vector>

// hard code MPI for now
#ifndef USE_MPI
#define USE_MPI
#endif

#include "cstone/domain/domain.hpp"
#include "init/factory.hpp"
#include "observables/factory.hpp"
#include "propagator/factory.hpp"
#include "io/arg_parser.hpp"
#include "io/ifile_writer.hpp"
#include "util/timer.hpp"
#include "util/utils.hpp"

#include "insitu_viz.h"

#ifdef USE_CUDA
using AccType = cstone::GpuTag;
#else
using AccType = cstone::CpuTag;
#endif

using namespace sphexa;

bool stopSimulation(size_t iteration, double time, const std::string& maxStepStr);
void printHelp(char* binName, int rank);

int main(int argc, char** argv)
{
    auto [rank, numRanks] = initMpi();
    const ArgParser parser(argc, argv);

    if (parser.exists("-h") || parser.exists("--h") || parser.exists("-help") || parser.exists("--help"))
    {
        printHelp(argv[0], rank);
        return exitSuccess();
    }

    using Real    = double;
    using KeyType = uint64_t;
    using Dataset = ParticlesData<Real, KeyType, AccType>;
    using Domain  = cstone::Domain<KeyType, Real, AccType>;

    const std::string        initCond          = parser.get("--init");
    const size_t             problemSize       = parser.get("-n", 50);
    const std::string        glassBlock        = parser.get("--glass");
    const std::string        propChoice        = parser.get("--prop", std::string("ve"));
    const std::string        maxStepStr        = parser.get("-s", std::string("200"));
    const std::string        writeFrequencyStr = parser.get("-w", std::string("0"));
    std::vector<std::string> writeExtra        = parser.getCommaList("--wextra");
    std::vector<std::string> outputFields      = parser.getCommaList("-f");
    const bool               ascii             = parser.exists("--ascii");
    const std::string        outDirectory      = parser.get("--outDir");
    const bool               quiet             = parser.exists("--quiet");

    if (outputFields.empty()) { outputFields = {"x", "y", "z", "vx", "vy", "vz", "h", "rho", "u", "p", "c"}; }

    const std::string outFile = outDirectory + "dump_" + initCond;

    size_t ngmax = 150;
    size_t ng0   = 100;

    std::ofstream nullOutput("/dev/null");
    std::ostream& output = quiet ? nullOutput : std::cout;
    std::ofstream constantsFile(outDirectory + "constants.txt");

    //! @brief evaluate user choice for different kind of actions
    auto simInit     = initializerFactory<Dataset>(initCond, glassBlock);
    auto propagator  = propagatorFactory<Domain, Dataset>(propChoice, ngmax, ng0, output, rank);
    auto fileWriter  = fileWriterFactory<Dataset>(ascii);
    auto observables = observablesFactory<Dataset>(initCond, constantsFile);

    Dataset d;
    d.comm = MPI_COMM_WORLD;
    propagator->activateFields(d);
    cstone::Box<Real> box = simInit->init(rank, numRanks, problemSize, d);
    d.setOutputFields(outputFields);

    bool  haveGrav = (d.g != 0.0);
    float theta    = parser.get("--theta", haveGrav ? 0.5f : 1.0f);

    if (rank == 0 && (writeFrequencyStr != "0" || !writeExtra.empty()))
    {
        fileWriter->constants(simInit->constants(), outFile);
    }
    if (rank == 0) { std::cout << "Data generated for " << d.numParticlesGlobal << " global particles\n"; }

    size_t bucketSizeFocus = 64;
    // we want about 100 global nodes per rank to decompose the domain with +-1% accuracy
    size_t bucketSize = std::max(bucketSizeFocus, d.numParticlesGlobal / (100 * numRanks));
    Domain domain(rank, numRanks, bucketSize, bucketSizeFocus, theta, box);

    propagator->sync(domain, d);
    if (rank == 0) std::cout << "Domain synchronized, nLocalParticles " << d.x.size() << std::endl;

    viz::init_catalyst(argc, argv);
    viz::init_ascent(d, domain.startIndex());

    MasterProcessTimer totalTimer(output, rank);
    totalTimer.start();
    size_t startIteration = d.iteration;
    for (; !stopSimulation(d.iteration - 1, d.ttot, maxStepStr); d.iteration++)
    {
        propagator->step(domain, d);

        observables->computeAndWrite(d, domain.startIndex(), domain.endIndex(), box);
        propagator->printIterationTimings(domain, d);

        if (isPeriodicOutputStep(d.iteration, writeFrequencyStr) ||
            isPeriodicOutputTime(d.ttot - d.minDt, d.ttot, writeFrequencyStr) ||
            isExtraOutputStep(d.iteration, d.ttot - d.minDt, d.ttot, writeExtra))
        {
            propagator->prepareOutput(d, domain.startIndex(), domain.endIndex());
            fileWriter->dump(d, domain.startIndex(), domain.endIndex(), box, outFile);
            propagator->finishOutput(d);
        }

        viz::execute(d, domain.startIndex(), domain.endIndex());
    }

    totalTimer.step("Total execution time of " + std::to_string(d.iteration - startIteration) + " iterations of " +
                    initCond + " up to t = " + std::to_string(d.ttot));

    constantsFile.close();
    viz::finalize();
    return exitSuccess();
}

//! @brief decide whether to stop the simulation based on evolved time (not wall-clock) or iteration count
bool stopSimulation(size_t iteration, double time, const std::string& maxStepStr)
{
    bool lastIteration = strIsIntegral(maxStepStr) && iteration == std::stoi(maxStepStr);
    bool simTimeLimit  = !strIsIntegral(maxStepStr) && time > std::stod(maxStepStr);

    return lastIteration || simTimeLimit;
}

void printHelp(char* name, int rank)
{
    if (rank == 0)
    {
        printf("\nUsage:\n\n");
        printf("%s [OPTIONS]\n", name);
        printf("\nWhere possible options are:\n\n");

        printf("\t--init \t\t Test case selection (evrard, sedov, noh, isobaric-cube, wind-shock) or an HDF5 file "
               "with initial conditions\n");
        printf("\t-n NUM \t\t Initialize data with (approx when using glass blocks) NUM^3 global particles [50]\n");
        printf("\t--glass FILE\t Use glass block as template to generate initial x,y,z configuration\n\n");

        printf("\t--theta NUM \t Gravity accuracy parameter [default 0.5 when self-gravity is active]\n\n");

        printf("\t--prop STRING \t Choice of SPH propagator [default: modern SPH]. For standard SPH, use \"std\" \n\n");

        printf("\t-s NUM \t\t int(NUM):  Number of iterations (time-steps) [200],\n\
                \t real(NUM): Time   of simulation (time-model)\n\n");

        printf("\t--wextra LIST \t Comma-separated list of steps (integers) or ~times (floating point)"
               " at which to trigger output to file []\n\
                   \t\t e.g.: --wextra 1,0.77,1.29,2.58\n\n");

        printf("\t-w NUM \t\t NUM<=0:    Disable file output [default],\n\
                \t int(NUM):  Dump particle data every NUM iteration steps,\n\
                \t real(NUM): Dump particle data every NUM seconds of simulation (not wall-clock) time \n\n");

        printf("\t-f LIST \t Comma-separated list of field names to write for each dump [x,y,z,vx,vy,vz,h,rho,u,p,c]\n\
                    \t\t e.g: -f x,y,z,h,rho\n\n");

        printf("\t--ascii \t Dump file in ASCII format [binary HDF5 by default]\n\n");

        printf("\t--outDir PATH \t Path to directory where output will be saved [./].\n\
                    \t\t Note that directory must exist and be provided with ending slash,\n\
                    \t\t e.g: --outDir /home/user/folderToSaveOutputFiles/\n\n");

        printf("\t--quiet \t Don't print anything to stdout\n\n");
    }
}
