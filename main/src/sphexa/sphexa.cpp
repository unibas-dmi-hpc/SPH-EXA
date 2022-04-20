#include <iostream>
#include <fstream>
#include <string>
#include <memory>
#include <vector>

// hard code MPI for now
#ifndef USE_MPI
#define USE_MPI
#endif

#include "cstone/domain/domain.hpp"
#include "propagator.hpp"
#include "init/factory.hpp"
#include "observables/factory.hpp"
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
using namespace sphexa::sph;

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
    if (!parser.exists("--init") && rank == 0)
    {
        throw std::runtime_error("no initial conditions specified (--init flag missing)\n");
    }

    const std::string        initCond       = parser.getString("--init");
    const size_t             problemSize    = parser.getInt("-n", 50);
    const std::string        glassBlock     = parser.getString("--glass");
    const bool               ve             = parser.exists("--ve");
    const size_t             maxStep        = parser.getInt("-s", 200);
    const int                writeFrequency = parser.getInt("-w", -1);
    std::vector<std::string> outputFields   = parser.getCommaList("-f");
    const bool               ascii          = parser.exists("--ascii");
    const std::string        outDirectory   = parser.getString("--outDir");
    const bool               quiet          = parser.exists("--quiet");

    if (outputFields.empty()) { outputFields = {"x", "y", "z", "vx", "vy", "vz", "h", "rho", "u", "p", "c"}; }

    const std::string outFile = outDirectory + "dump_" + initCond;

    using Real    = double;
    using KeyType = uint64_t;
    using Dataset = ParticlesData<Real, KeyType, AccType>;
    using Domain  = cstone::Domain<KeyType, Real, AccType>;

    size_t ngmax = 150;
    size_t ng0   = 100;

    std::ofstream nullOutput("/dev/null");
    std::ostream& output = quiet ? nullOutput : std::cout;

    std::unique_ptr<IFileWriter<Dataset>> fileWriter;
    if (ascii) { fileWriter = std::make_unique<AsciiWriter<Dataset>>(); }
    else
    {
        fileWriter = std::make_unique<H5PartWriter<Dataset>>();
    }
    std::ofstream constantsFile(outDirectory + "constants.txt");

    std::unique_ptr<ISimInitializer<Dataset>> simInit = initializerFactory<Dataset>(initCond, glassBlock);

    Dataset d;
    d.comm = MPI_COMM_WORLD;
    if (ve)
    {
        d.setConservedFieldsVE();
        d.setDependentFieldsVE();
    }
    cstone::Box<Real> box = simInit->init(rank, numRanks, problemSize, d);
    d.setOutputFields(outputFields);

    bool  haveGrav = (d.g != 0.0);
    float theta    = parser.exists("--theta") ? parser.getDouble("--theta") : (haveGrav ? 0.5 : 1.0);

    if (rank == 0 && writeFrequency > 0) { fileWriter->constants(simInit->constants(), outFile); }
    if (rank == 0) { std::cout << "Data generated for " << d.numParticlesGlobal << " global particles\n"; }

    size_t bucketSizeFocus = 64;
    // we want about 100 global nodes per rank to decompose the domain with +-1% accuracy
    size_t bucketSize = std::max(bucketSizeFocus, d.numParticlesGlobal / (100 * numRanks));
    Domain domain(rank, numRanks, bucketSize, bucketSizeFocus, theta, box);
    auto   propagator = propagatorFactory<Domain, Dataset>(ve, ngmax, ng0, output, rank);

    if (ve)
        domain.sync(d.codes, d.x, d.y, d.z, d.h, d.m, d.u, d.vx, d.vy, d.vz, d.x_m1, d.y_m1, d.z_m1, d.du_m1, d.alpha);
    else
        domain.sync(d.codes, d.x, d.y, d.z, d.h, d.m, d.u, d.vx, d.vy, d.vz, d.x_m1, d.y_m1, d.z_m1, d.du_m1);

    if (rank == 0) std::cout << "Domain synchronized, nLocalParticles " << d.x.size() << std::endl;

    viz::init_catalyst(argc, argv);
    viz::init_ascent(d, domain.startIndex());

    std::unique_ptr<IObservables<Dataset>> observables = observablesFactory<Dataset>(initCond, constantsFile);

    MasterProcessTimer totalTimer(output, rank);
    totalTimer.start();
    size_t startIteration = d.iteration;
    for (; d.iteration <= maxStep; d.iteration++)
    {
        propagator->step(domain, d);

        observables -> computeAndWrite(d, domain.startIndex(), domain.endIndex(), box);

        if ((writeFrequency > 0 && d.iteration % writeFrequency == 0) || writeFrequency == 0)
        {
            fileWriter->dump(d, domain.startIndex(), domain.endIndex(), box, outFile);
        }

        if (d.iteration % 50 == 0) { viz::execute(d, domain.startIndex(), domain.endIndex()); }
    }

    totalTimer.step("Total execution time of " + std::to_string(maxStep - startIteration + 1) + " iterations of " +
                    initCond);

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

        printf(
            "\t--init \t\t Test case selection (sedov, noh, isobaric-cube) or an HDF5 file with initial conditions\n");
        printf("\t-n NUM \t\t Initialize data with (approx when using glass blocks) NUM^3 global particles [50]\n");
        printf("\t--glass \t Use glass block at tests\n\n");

        printf("\t--theta NUM \t Gravity accuracy parameter [default 0.5 when self-gravity is active]\n\n");

        printf("\t--ve \t\t Activate SPH with generalized volume elements\n\n");

        printf("\t-s NUM \t\t NUM Number of iterations (time-steps) [200]\n\n");

        printf("\t-w NUM \t\t Dump particles data every NUM iterations (time-steps) [-1]\n");
        printf("\t-f list \t Comma-separated list of field names to write for each dump,\n\
                    \t\t e.g: -f x,y,z,h,rho\n\n");

        printf("\t--ascii \t Dump file in ASCII format [binary HDF5 by default]\n\n");

        printf("\t--outDir PATH \t Path to directory where output will be saved [./].\n\
                    \t\t Note that directory must exist and be provided with ending slash,\n\
                    \t\t e.g: --outDir /home/user/folderToSaveOutputFiles/\n\n");

        printf("\t--quiet \t Don't print anything to stdout\n\n");
    }
}
