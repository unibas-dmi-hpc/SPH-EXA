#include <iostream>
#include <fstream>
#include <string>
#include <vector>

// hard code MPI for now
#ifndef USE_MPI
#define USE_MPI
#endif

#include "cstone/domain/domain.hpp"
#include "sphexa.hpp"
#include "sph/findNeighborsSfc.hpp"
#include "tests/test_file_writer.hpp"
#include "propagator.hpp"

#include "NohDataGenerator.hpp"

using namespace cstone;
using namespace sphexa;
using namespace sphexa::sph;

#ifdef SPH_EXA_USE_CATALYST2
#include "CatalystAdaptor.h"
#endif

#ifdef SPH_EXA_USE_ASCENT
#include "AscentAdaptor.h"
#endif

void printHelp(char* binName, int rank);

int main(int argc, char** argv)
{
    const int rank = initAndGetRankId();

#ifdef SPH_EXA_USE_CATALYST2
    CatalystAdaptor::Initialize(argc, argv);
    std::cout << "CatalystInitialize\n";
#endif

    const ArgParser parser(argc, argv);

    if (parser.exists("-h") || parser.exists("--h") || parser.exists("-help") || parser.exists("--help"))
    {
        printHelp(argv[0], rank);
        return exitSuccess();
    }

    const size_t      cubeSide       = parser.getInt("-n", 100);
    const size_t      maxStep        = parser.getInt("-s", 1000);
    const int         writeFrequency = parser.getInt("-w", -1);
    const bool        quiet          = parser.exists("--quiet");
    const std::string outDirectory   = parser.getString("--outDir");

    std::ofstream nullOutput("/dev/null");
    std::ostream& output = quiet ? nullOutput : std::cout;

    using Real    = double;
    using KeyType = uint64_t;
    using Dataset = ParticlesData<Real, KeyType>;

    const IFileWriter<Dataset>& fileWriter = TestMPIFileWriter<Dataset>();

    auto d = NohDataGenerator<Real, KeyType>::generate(cubeSide);

    if (d.rank == 0) std::cout << "Data generated." << std::endl;

    MasterProcessTimer totalTimer(output, d.rank);

    std::ofstream constantsFile(outDirectory + "constants_noh.txt");

    size_t bucketSizeFocus = 64;
    // we want about 100 global nodes per rank to decompose the domain with +-1% accuracy
    size_t bucketSize = std::max(bucketSizeFocus, d.n / (100 * d.nrank));

    double    radius = NohDataGenerator<Real, KeyType>::r1;
    Box<Real> box(-radius, radius, false);

    float theta = 1.0;

#ifdef USE_CUDA
    Domain<KeyType, Real, CudaTag> domain(rank, d.nrank, bucketSize, bucketSizeFocus, theta, box);
#else
    Domain<KeyType, Real> domain(rank, d.nrank, bucketSize, bucketSizeFocus, theta, box);
#endif

    if (d.rank == 0) std::cout << "Domain created." << std::endl;

    domain.sync(
        d.codes, d.x, d.y, d.z, d.h, d.m, d.mui, d.u, d.vx, d.vy, d.vz, d.x_m1, d.y_m1, d.z_m1, d.du_m1, d.dt_m1);

    if (d.rank == 0) std::cout << "Domain synchronized, nLocalParticles " << d.x.size() << std::endl;

#ifdef SPH_EXA_USE_ASCENT
    AscentAdaptor::Initialize(d, domain.startIndex());
    std::cout << "AscentInitialize\n";
#endif

    const size_t nTasks = 64;
    const size_t ngmax  = 150;
    const size_t ng0    = 100;

    Propagator propagator(nTasks, ngmax, ng0, output, d.rank);

    if (d.rank == 0) std::cout << "Starting main loop." << std::endl;

    totalTimer.start();
    for (d.iteration = 0; d.iteration <= maxStep; d.iteration++)
    {
        propagator.hydroStep(domain, d);

        Printer::printConstants(d.iteration, d.ttot, d.minDt, d.etot, d.ecin, d.eint, d.egrav, constantsFile);

        if ((writeFrequency > 0 && d.iteration % writeFrequency == 0) || writeFrequency == 0)
        {
#ifdef SPH_EXA_HAVE_H5PART
            fileWriter.dumpParticleDataToH5File(
                d, domain.startIndex(), domain.endIndex(), outDirectory + "dump_noh.h5part");
#else
            fileWriter.dumpParticleDataToAsciiFile(d,
                                                   domain.startIndex(),
                                                   domain.endIndex(),
                                                   outDirectory + "dump_noh" + std::to_string(d.iteration) + ".txt");
#endif
        }

#ifdef SPH_EXA_USE_CATALYST2
        CatalystAdaptor::Execute(d, domain.startIndex(), domain.endIndex());
#endif

#ifdef SPH_EXA_USE_ASCENT
        if (d.iteration % 5 == 0) { AscentAdaptor::Execute(d, domain.startIndex(), domain.endIndex()); }
#endif
    }

    totalTimer.step("Total execution time of " + std::to_string(maxStep) + " iterations of Noh");

    constantsFile.close();

#ifdef SPH_EXA_USE_CATALYST2
    CatalystAdaptor::Finalize();
#endif

#ifdef SPH_EXA_USE_ASCENT
    AscentAdaptor::Finalize();
#endif

    return exitSuccess();
}

void printHelp(char* name, int rank)
{
    if (rank == 0)
    {
        printf("\nUsage:\n\n");
        printf("%s [OPTIONS]\n", name);
        printf("\nWhere possible options are:\n\n");

        printf("\t-n NUM \t\t\t NUM^3 Number of particles [100]\n");
        printf("\t-s NUM \t\t\t NUM Number of iterations (time-steps) [1000]\n\n");

        printf("\t-w NUM \t\t\t Dump particles data every NUM iterations (time-steps) [-1]\n\n");

        printf("\t--quiet \t\t Don't print anything to stdout [false]\n\n");

        printf("\t--outDir PATH \t\t Path to directory where output will be saved [./].\
                    \n\t\t\t\t Note that directory must exist and be provided with ending slash.\
                    \n\t\t\t\t Example: --outDir /home/user/folderToSaveOutputFiles/\n");

        printf("\nFor example:\n");
        printf("\t$ %s -n 100 -s 1000 -w 50 --outDir ./bin/\n\n", name);
    }
}
