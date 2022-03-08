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
#include "sph/find_neighbors.hpp"
#include "ifile_writer.hpp"
#include "evrard_reader.hpp"

#include "propagator.hpp"
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

    const size_t nParticles     = parser.getInt("-n", 65536);
    const size_t maxStep        = parser.getInt("-s", 10);
    const int    writeFrequency = parser.getInt("-w", -1);
    // const int checkpointFrequency     = parser.getInt("-c", -1);
    const bool        quiet         = parser.exists("--quiet");
    const bool        ascii         = parser.exists("--ascii");
    const std::string inputFilePath = parser.getString("--input", "./Test3DEvrardRel.bin");
    const std::string outDirectory  = parser.getString("--outDir");

    std::ofstream nullOutput("/dev/null");
    std::ostream& output = quiet ? nullOutput : std::cout;

    using Real    = double;
    using KeyType = uint64_t;
    using Dataset = ParticlesData<Real, KeyType, AccType>;

    const IFileReader<Dataset>& fileReader = EvrardFileReader<Dataset>();

    std::unique_ptr<IFileWriter<Dataset>> fileWriter;
    if (ascii) { fileWriter = std::make_unique<AsciiWriter<Dataset>>(); }
    else
    {
        fileWriter = std::make_unique<H5PartWriter<Dataset>>();
    }

    auto d = fileReader.read(inputFilePath, nParticles);

    std::vector<std::string> outputFields = {
        "x", "y", "z", "vx", "vy", "vz", "h", "rho", "u", "p", "c", "grad_P_x", "grad_P_y", "grad_P_z"};
    if (parser.exists("-f")) { outputFields = parser.getCommaList("-f"); }
    d.setOutputFields(outputFields);

    std::cout << d.x[0] << " " << d.y[0] << " " << d.z[0] << std::endl;
    std::cout << d.x[1] << " " << d.y[1] << " " << d.z[1] << std::endl;
    std::cout << d.x[2] << " " << d.y[2] << " " << d.z[2] << std::endl;

    if (d.rank == 0) std::cout << "Data generated." << std::endl;

    MasterProcessTimer totalTimer(output, d.rank);

    std::ofstream constantsFile(outDirectory + "constants.txt");

    size_t bucketSizeFocus = 64;
    // we want about 100 global nodes per rank to decompose the domain with +-1% accuracy
    size_t bucketSize = std::max(bucketSizeFocus, nParticles / (100 * d.nrank));

    // no PBC, global box will be recomputed every step
    cstone::Box<Real> box(0, 1, false);

    float theta = 0.5;

    cstone::Domain<KeyType, Real, AccType> domain(rank, d.nrank, bucketSize, bucketSizeFocus, theta, box);

    if (d.rank == 0) std::cout << "Domain created." << std::endl;

    domain.sync(d.codes, d.x, d.y, d.z, d.h, d.m, d.u, d.vx, d.vy, d.vz, d.x_m1, d.y_m1, d.z_m1, d.du_m1, d.dt_m1);

    if (d.rank == 0) std::cout << "Domain synchronized, nLocalParticles " << d.x.size() << std::endl;

    viz::init_catalyst(argc, argv);
    viz::init_ascent(d, domain.startIndex());

    const size_t ngmax = 150;
    const size_t ng0   = 100;

    Propagator propagator(ngmax, ng0, output, d.rank);

    totalTimer.start();
    for (d.iteration = 0; d.iteration <= maxStep; d.iteration++)
    {
        propagator.hydroStepGravity(domain, d);

        fileutils::writeColumns(constantsFile, ' ', d.iteration, d.ttot, d.minDt, d.etot, d.ecin, d.eint, d.egrav);

        if ((writeFrequency > 0 && d.iteration % writeFrequency == 0) || writeFrequency == 0)
        {
            fileWriter->dump(d, domain.startIndex(), domain.endIndex(), outDirectory + "dump_evrard");
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
        printf("\t-f list \t\t Comma-separated list of field names to write for each dump, "
               "e.g -f x,y,z,h,ro\n\n");

        printf("\t--quiet \t\t Don't print anything to stdout [false]\n\n");

        printf("\t--cinput \t\t Read ParticleData from CheckpointBinFile input [false]\n\n");

        printf("\t--input  PATH \t\t Path to input file [./Test3DEvrardRel.bin]\n");

        printf("\t--outDir PATH \t\t Path to directory where output will be saved [./].\
                    \n\t\t\t\t Note that directory must exist and be provided with ending slash.\
                    \n\t\t\t\t Example: --outDir /home/user/folderToSaveOutputFiles/\n\n");
    }
}
