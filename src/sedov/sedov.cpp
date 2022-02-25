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

#include "sphexa.hpp"
#include "sph/find_neighbors.hpp"
#include "SedovDataGenerator.hpp"
#include "ifile_writer.hpp"

#include "propagator.hpp"
#include "insitu_viz.h"

using namespace cstone;
using namespace sphexa;
using namespace sphexa::sph;

void printHelp(char* binName, int rank);

int main(int argc, char** argv)
{
    const int rank = initAndGetRankId();
    const ArgParser parser(argc, argv);

    if (parser.exists("-h") || parser.exists("--h") || parser.exists("-help") || parser.exists("--help"))
    {
        printHelp(argv[0], rank);
        return exitSuccess();
    }

    const size_t cubeSide          = parser.getInt("-n", 50);
    const size_t maxStep           = parser.getInt("-s", 200);
    const int writeFrequency       = parser.getInt("-w", -1);
    const bool quiet               = parser.exists("--quiet");
    const bool ascii               = parser.exists("--ascii");
    const std::string outDirectory = parser.getString("--outDir");

    std::vector<std::string> outputFields = parser.getCommaList("-f");
    if (outputFields.empty())
    {
        outputFields = {"x", "y", "z", "vx", "vy", "vz", "h", "ro", "u", "p", "c", "grad_P_x", "grad_P_y", "grad_P_z"};
    }

    std::ofstream nullOutput("/dev/null");
    std::ostream& output = quiet ? nullOutput : std::cout;

    using Real    = double;
    using KeyType = uint64_t;
    using Dataset = ParticlesData<Real, KeyType>;

    std::unique_ptr<IFileWriter<Dataset>> fileWriter;
    if (ascii) { fileWriter = std::make_unique<AsciiWriter<Dataset>>(); }
    else { fileWriter = std::make_unique<H5PartWriter<Dataset>>(); }

    auto d = SedovDataGenerator<Real, KeyType>::generate(cubeSide);
    d.outputFields = std::move(outputFields);

    if (d.rank == 0) std::cout << "Data generated." << std::endl;

    MasterProcessTimer totalTimer(output, d.rank);

    std::ofstream constantsFile(outDirectory + "constants.txt");

    size_t bucketSizeFocus = 64;
    // we want about 100 global nodes per rank to decompose the domain with +-1% accuracy
    size_t bucketSize = std::max(bucketSizeFocus, d.n / (100 * d.nrank));

    Box<Real> box(0, 1);
    box = makeGlobalBox(d.x.begin(), d.x.end(), d.y.begin(), d.z.begin(), box);

    // enable PBC and enlarge bounds
    Real dx = 0.5 / cubeSide;
    box = Box<Real>(box.xmin() - dx, box.xmax() + dx,
                    box.ymin() - dx, box.ymax() + dx,
                    box.zmin() - dx, box.zmax() + dx, true, true, true);

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

    viz::init_catalyst(argc, argv);
    viz::init_ascent(d, domain.startIndex());

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
            fileWriter->dump(d, domain.startIndex(), domain.endIndex(), outDirectory + "dump_sedov");
        }

        if (d.iteration % 5 == 0) { viz::execute(d, domain.startIndex(), domain.endIndex()); }
    }

    totalTimer.step("Total execution time of " + std::to_string(maxStep) + " iterations of Sedov");

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

        printf("\t-n NUM \t\t\t NUM^3 Number of particles [50]\n");
        printf("\t-s NUM \t\t\t NUM Number of iterations (time-steps) [200]\n\n");

        printf("\t-w NUM \t\t\t Dump particles data every NUM iterations (time-steps) [-1]\n\n");
        printf("\t-f list \t\t Comma-separated list of field names to write for each dump, "
               "e.g -f x,y,z,h,ro\n\n");

        printf("\t--quiet \t\t Don't print anything to stdout [false]\n\n");

        printf("\t--outDir PATH \t\t Path to directory where output will be saved [./].\
                    \n\t\t\t\t Note that directory must exist and be provided with ending slash.\
                    \n\t\t\t\t Example: --outDir /home/user/folderToSaveOutputFiles/\n");
    }
}
