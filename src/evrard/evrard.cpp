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
#include "propagator.hpp"
#include "insitu_viz.h"

#include "evrard_data_file_reader.hpp"
#include "evrard_data_file_writer.hpp"

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

    const size_t      nParticles          = parser.getInt("-n", 65536);
    const size_t      maxStep             = parser.getInt("-s", 10);
    const int         writeFrequency      = parser.getInt("-w", -1);
    const int         checkpointFrequency = parser.getInt("-c", -1);
    const bool        quiet               = parser.exists("--quiet");
    const std::string checkpointInput     = parser.getString("--cinput");
    const std::string inputFilePath       = parser.getString("--input", "./Test3DEvrardRel.bin");
    const std::string outDirectory        = parser.getString("--outDir");

    std::ofstream nullOutput("/dev/null");
    std::ostream& output = quiet ? nullOutput : std::cout;

    using Real    = double;
    using KeyType = uint64_t;
    using Dataset = ParticlesData<Real, KeyType>;

    const IFileReader<Dataset>& fileReader = EvrardDataMPIFileReader<Dataset>();
    const IFileWriter<Dataset>& fileWriter = EvrardDataMPIFileWriter<Dataset>();

    auto d = checkpointInput.empty() ? fileReader.readParticleDataFromBinFile(inputFilePath, nParticles)
                                     : fileReader.readParticleDataFromCheckpointBinFile(checkpointInput);

    std::cout << d.x[0] << " " << d.y[0] << " " << d.z[0] << std::endl;
    std::cout << d.x[1] << " " << d.y[1] << " " << d.z[1] << std::endl;
    std::cout << d.x[2] << " " << d.y[2] << " " << d.z[2] << std::endl;

    if (d.rank == 0) std::cout << "Data generated." << std::endl;

    MasterProcessTimer totalTimer(output, d.rank);

    std::ofstream constantsFile(outDirectory + "constants_evrard.txt");

    size_t bucketSizeFocus = 64;
    // we want about 100 global nodes per rank to decompose the domain with +-1% accuracy
    size_t bucketSize = std::max(bucketSizeFocus, nParticles / (100 * d.nrank));

    // no PBC, global box will be recomputed every step
    Box<Real> box(0, 1, false);

    float theta = 0.5;

#ifdef USE_CUDA
    DomainType<KeyType, Real, CudaTag> domain(rank, d.nrank, bucketSize, bucketSizeFocus, theta, box);
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
        propagator.hydroStepGravity(domain, d);

        Printer::printConstants(d.iteration, d.ttot, d.minDt, d.etot, d.ecin, d.eint, d.egrav, constantsFile);

        if ((writeFrequency > 0 && d.iteration % writeFrequency == 0) || writeFrequency == 0)
        {
#ifdef SPH_EXA_HAVE_H5PART
            fileWriter.dumpParticleDataToH5File(
                d, domain.startIndex(), domain.endIndex(), outDirectory + "dump_evrard.h5part");
#else
            fileWriter.dumpParticleDataToAsciiFile(d,
                                                   domain.startIndex(),
                                                   domain.endIndex(),
                                                   outDirectory + "dump_evrard" + std::to_string(d.iteration) + ".txt");
#endif
        }

        if (checkpointFrequency > 0 && d.iteration % checkpointFrequency == 0)
        {
            fileWriter.dumpCheckpointDataToBinFile(
                d, outDirectory + "checkpoint_evrard" + std::to_string(d.iteration) + ".bin");
        }

        if (d.iteration % 5 == 0) { viz::execute(d, domain.startIndex(), domain.endIndex()); }
    }

    totalTimer.step("Total execution time of " + std::to_string(maxStep) + " iterations of Evrard");

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

        printf("\t-n NUM \t\t\t NUM Number of particles [65536]\n");
        printf("\t-s NUM \t\t\t NUM Number of iterations (time-steps) [10]\n\n");

        printf("\t-w NUM \t\t\t Dump Frequency data every NUM iterations (time-steps) [-1]\n");
        printf("\t-c NUM \t\t\t Create checkpoint every NUM iterations (time-steps) [-1]\n\n");

        printf("\t--quiet \t\t Don't print anything to stdout [false]\n\n");

        printf("\t--cinput \t\t Read ParticleData from CheckpointBinFile input [false]\n\n");

        printf("\t--input  PATH \t\t Path to input file [./Test3DEvrardRel.bin]\n");

        printf("\t--outDir PATH \t\t Path to directory where output will be saved [./].\
                    \n\t\t\t\t Note that directory must exist and be provided with ending slash.\
                    \n\t\t\t\t Example: --outDir /home/user/folderToSaveOutputFiles/\n\n");
    }
}
