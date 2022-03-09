#include <iostream>
#include <fstream>
#include <string>
#include <vector>

// hard code MPI for now
#ifndef USE_MPI
#define USE_MPI
#endif

#include "cstone/domain/domain.hpp"
#include "sph/propagator.hpp"
#include "io/arg_parser.hpp"
#include "io/ifile_writer.hpp"
#include "util/timer.hpp"
#include "util/utils.hpp"

#include "init/noh_constants.hpp"
#include "insitu_viz.h"

#ifdef USE_CUDA
using AccType = cstone::GpuTag;
#else
using AccType = cstone::CpuTag;
#endif

using namespace sphexa;
using namespace sphexa::sph;

void printHelp(char* binName, int rank);

int main(int argc, char** argv)
{
    auto [rank, numRanks] = initMpi();
    const ArgParser parser(argc, argv);
    using Gen = NohConstants;

    if (parser.exists("-h") || parser.exists("--h") || parser.exists("-help") || parser.exists("--help"))
    {
        printHelp(argv[0], rank);
        return exitSuccess();
    }

    const size_t      cubeSide       = parser.getInt("-n", 100);
    const size_t      maxStep        = parser.getInt("-s", 1000);
    const int         writeFrequency = parser.getInt("-w", -1);
    const bool        ascii          = parser.exists("--ascii");
    const bool        quiet          = parser.exists("--quiet");
    const std::string outDirectory   = parser.getString("--outDir");
    const std::string outFile        = outDirectory + "dump_noh";

    using Real    = double;
    using KeyType = uint64_t;
    using Dataset = ParticlesData<Real, KeyType, AccType>;

    float  theta           = 1.0;
    size_t ngmax           = 150;
    size_t ng0             = 100;
    size_t bucketSizeFocus = 64;
    // we want about 100 global nodes per rank to decompose the domain with +-1% accuracy
    size_t bucketSize = std::max(bucketSizeFocus, size_t(std::pow(cubeSide, 3) / (100 * numRanks)));

    std::vector<std::string> outputFields = parser.getCommaList("-f");
    if (outputFields.empty()) { outputFields = {"x", "y", "z", "vx", "vy", "vz", "h", "rho", "u", "p", "c"}; }

    std::ofstream nullOutput("/dev/null");
    std::ostream& output = quiet ? nullOutput : std::cout;

    std::unique_ptr<IFileWriter<Dataset>> fileWriter;
    if (ascii) { fileWriter = std::make_unique<AsciiWriter<Dataset>>(); }
    else
    {
        fileWriter = std::make_unique<H5PartWriter<Dataset>>();
    }
    std::ofstream constantsFile(outDirectory + "constants_noh.txt");

    auto d = Gen::generate<Dataset>(cubeSide);
    d.setOutputFields(outputFields);
    if (d.rank == 0 && writeFrequency > 0)
    {
        fileWriter->constants({"r0", "r1", "dim", "gamma", "rho0", "u0", "p0", "vr0", "cs0"},
                              {Gen::r0, Gen::r1, Gen::dim, Gen::gamma, Gen::rho0, Gen::u0, Gen::p0, Gen::vr0, Gen::cs0},
                              outFile);
    }

    double            radius = NohConstants::r1;
    cstone::Box<Real> box(-radius, radius, false);

    cstone::Domain<KeyType, Real, AccType> domain(rank, d.nrank, bucketSize, bucketSizeFocus, theta, box);
    domain.sync(d.codes, d.x, d.y, d.z, d.h, d.m, d.u, d.vx, d.vy, d.vz, d.x_m1, d.y_m1, d.z_m1, d.du_m1, d.dt_m1);
    if (d.rank == 0) std::cout << "Domain synchronized, numLocalParticles " << d.x.size() << std::endl;

    viz::init_catalyst(argc, argv);
    viz::init_ascent(d, domain.startIndex());

    Propagator propagator(ngmax, ng0, output, d.rank);

    MasterProcessTimer totalTimer(output, d.rank);
    totalTimer.start();
    for (d.iteration = 0; d.iteration <= maxStep; d.iteration++)
    {
        propagator.hydroStep(domain, d);

        fileutils::writeColumns(constantsFile, ' ', d.iteration, d.ttot, d.minDt, d.etot, d.ecin, d.eint, d.egrav);

        if ((writeFrequency > 0 && d.iteration % writeFrequency == 0) || writeFrequency == 0)
        {
            fileWriter->dump(d, domain.startIndex(), domain.endIndex(), outFile);
        }

        if (d.iteration % 5 == 0) { viz::execute(d, domain.startIndex(), domain.endIndex()); }
    }

    totalTimer.step("Total execution time of " + std::to_string(maxStep) + " iterations of Noh");

    constantsFile.close();
    viz::finalize();
    return exitSuccess();
}

void printHelp(char* name, int rank)
{
    if (rank == 0)
    {
        printf("\nUsage:\n\n");
        printf("%s [OPTIONS]\n", name);
        printf("\nWhere possible options are:\n\n");

        printf("\t-n NUM \t\t\t NUM^3 Number of particles [100].\n");
        printf("\t-s NUM \t\t\t NUM Number of iterations (time-steps) [1000].\n\n");

        printf("\t-w NUM \t\t\t Dump particles data every NUM iterations (time-steps) [-1].\n");
        printf("\t--ascii \t\t  Write file in ASCII format [false].\n\n");

        printf("\t--quiet \t\t Don't print anything to stdout [false].\n\n");

        printf("\t--outDir PATH \t\t Path to directory where output will be saved [./].\
                    \n\t\t\t\t Note that directory must exist and be provided with ending slash.\
                    \n\t\t\t\t Example: --outDir /home/user/folderToSaveOutputFiles/\n");

        printf("\nFor example:\n");
        printf("\t$ %s -n 100 -s 1000 -w 50 --outDir ./bin/\n\n", name);
    }
}
