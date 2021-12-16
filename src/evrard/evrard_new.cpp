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
#include "EvrardCollapseInputFileReader.hpp"
#include "EvrardCollapseFileWriter.hpp"

#include "sph/findNeighborsSfc.hpp"

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

    const size_t nParticles = parser.getInt("-n", 65536);
    const size_t maxStep = parser.getInt("-s", 10);
    const int writeFrequency = parser.getInt("-w", -1);
    const int checkpointFrequency = parser.getInt("-c", -1);
    const bool quiet = parser.exists("--quiet");
    const std::string checkpointInput = parser.getString("--cinput");
    const std::string inputFilePath = parser.getString("--input", "bigfiles/Test3DEvrardRel.bin");
    const std::string outDirectory = parser.getString("--outDir");

    std::ofstream nullOutput("/dev/null");
    std::ostream& output = quiet ? nullOutput : std::cout;

    using Real = double;
    using KeyType = uint64_t;
    using Dataset = ParticlesData<Real, KeyType>;

    const IFileReader<Dataset>& fileReader = EvrardCollapseMPIInputFileReader<Dataset>();
    const IFileWriter<Dataset>& fileWriter = EvrardCollapseMPIFileWriter<Dataset>();

    auto d = checkpointInput.empty() ? fileReader.readParticleDataFromBinFile(inputFilePath, nParticles)
                                     : fileReader.readParticleDataFromCheckpointBinFile(checkpointInput);

    std::cout << d.x[0] << " " << d.y[0] << " " << d.z[0] << std::endl;
    std::cout << d.x[1] << " " << d.y[1] << " " << d.z[1] << std::endl;
    std::cout << d.x[2] << " " << d.y[2] << " " << d.z[2] << std::endl;

    if (d.rank == 0) std::cout << "Data generated." << std::endl;

    MasterProcessTimer timer(output, d.rank), totalTimer(output, d.rank);

    std::ofstream constantsFile(outDirectory + "constants.txt");

    size_t bucketSizeFocus = 64;
    // we want about 100 global nodes per rank to decompose the domain with +-1% accuracy
    size_t bucketSize = std::max(bucketSizeFocus, nParticles / (100 * d.nrank));

    // no PBC, global box will be recomputed every step
    Box<Real> box(0, 1, false);

    float theta = 0.5;

#ifdef USE_CUDA
    Domain<KeyType, Real, CudaTag> domain(rank, d.nrank, bucketSize, bucketSizeFocus, theta, box);
#else
    Domain<KeyType, Real> domain(rank, d.nrank, bucketSize, bucketSizeFocus, theta, box);
#endif

    if (d.rank == 0) std::cout << "Domain created." << std::endl;

    domain.sync(d.x, d.y, d.z, d.h, d.codes, d.m, d.mui, d.u, d.vx, d.vy, d.vz, d.x_m1, d.y_m1, d.z_m1, d.du_m1,
                d.dt_m1);

    if (d.rank == 0) std::cout << "Domain synchronized, nLocalParticles " << d.x.size() << std::endl;

    const size_t nTasks = 64;
    const size_t ngmax = 150;
    const size_t ng0 = 100;
    TaskList taskList = TaskList(0, domain.nParticles(), nTasks, ngmax, ng0);

#ifdef SPH_EXA_USE_ASCENT
    AscentAdaptor::Initialize(d, domain.startIndex());
    std::cout << "AscentInitialize\n";
#endif
    if (d.rank == 0) std::cout << "Starting main loop." << std::endl;

    totalTimer.start();
    for (d.iteration = 0; d.iteration <= maxStep; d.iteration++)
    {
        timer.start();
        domain.sync(d.x, d.y, d.z, d.h, d.codes, d.m, d.mui, d.u, d.vx, d.vy, d.vz, d.x_m1, d.y_m1, d.z_m1, d.du_m1,
                    d.dt_m1);
        timer.step("domain::sync");

        d.resize(domain.nParticlesWithHalos()); // also resize arrays not listed in sync, even though space for halos is
                                                // not needed
        // domain.exchangeHalos(d.m);
        std::fill(begin(d.m), begin(d.m) + domain.startIndex(), d.m[domain.startIndex()]);
        std::fill(begin(d.m) + domain.endIndex(), begin(d.m) + domain.nParticlesWithHalos(), d.m[domain.startIndex()]);

        taskList.update(domain.startIndex(), domain.endIndex());
        timer.step("updateTasks");
        findNeighborsSfc(taskList.tasks, d.x, d.y, d.z, d.h, d.codes, domain.box());
        timer.step("FindNeighbors");
        computeDensity<Real>(taskList.tasks, d, domain.box());
        timer.step("Density");
        computeEquationOfStateEvrard<Real>(taskList.tasks, d);
        timer.step("EquationOfState");
        domain.exchangeHalos(d.vx, d.vy, d.vz, d.ro, d.p, d.c);
        timer.step("mpi::synchronizeHalos");
        computeIAD<Real>(taskList.tasks, d, domain.box());
        timer.step("IAD");
        domain.exchangeHalos(d.c11, d.c12, d.c13, d.c22, d.c23, d.c33);
        timer.step("mpi::synchronizeHalos");
        computeMomentumAndEnergyIAD<Real>(taskList.tasks, d, domain.box());
        timer.step("MomentumEnergyIAD");
        d.egrav = domain.addGravityAcceleration(d.x, d.y, d.z, d.h, d.m, d.g, d.grad_P_x, d.grad_P_y, d.grad_P_z);
        // temporary sign fix, see note in ParticlesData
        d.egrav = (d.g > 0.0) ? d.egrav : -d.egrav;
        timer.step("Gravity");
        computeTimestep<Real, TimestepPress2ndOrder<Real, Dataset>>(taskList.tasks, d);
        timer.step("Timestep"); // AllReduce(min:dt)
        computePositions<Real, computeAcceleration<Real, Dataset>>(taskList.tasks, d, domain.box());
        timer.step("UpdateQuantities");
        computeTotalEnergy<Real>(taskList.tasks, d);
        d.etot += d.egrav;
        timer.step("EnergyConservation"); // AllReduce(sum:ecin,ein)
        updateSmoothingLength<Real>(taskList.tasks, d);
        timer.step("UpdateSmoothingLength");

        size_t totalNeighbors = neighborsSum(taskList.tasks);

        if (d.rank == 0)
        {
            Printer::printCheck(d.ttot,
                                d.minDt,
                                d.etot,
                                d.eint,
                                d.ecin,
                                d.egrav,
                                domain.box(),
                                d.n,
                                domain.nParticles(),
                                nNodes(domain.tree()),
                                d.x.size() - domain.nParticles(),
                                totalNeighbors,
                                output);
            std::cout << "### Check ### Focus Tree Nodes: " << nNodes(domain.focusedTree()) << std::endl;
            Printer::printConstants(
                d.iteration, d.ttot, d.minDt, d.etot, d.ecin, d.eint, d.egrav, totalNeighbors, constantsFile);
        }

        if ((writeFrequency > 0 && d.iteration % writeFrequency == 0) || writeFrequency == 0)
        {
#ifdef SPH_EXA_HAVE_H5PART
            fileWriter.dumpParticleDataToH5File(d, domain.startIndex(), domain.endIndex(),
                                                   outDirectory + "dump_evrard.h5part");
#else
            fileWriter.dumpParticleDataToAsciiFile(d, domain.startIndex(), domain.endIndex(),
                                                   outDirectory + "dump_evrard" + std::to_string(d.iteration) + ".txt");
#endif
            timer.step("writeFile");
        }
        if (checkpointFrequency > 0 && d.iteration % checkpointFrequency == 0)
        {
            fileWriter.dumpCheckpointDataToBinFile(d, outDirectory + "checkpoint_evrard" + std::to_string(d.iteration) +
                                                          ".bin");
            timer.step("Save Checkpoint File");
        }

        timer.stop();

        if (d.rank == 0)
        {
            Printer::printTotalIterationTime(d.iteration, timer.duration(), output);
        }
#ifdef SPH_EXA_USE_CATALYST2
        CatalystAdaptor::Execute(d, domain.startIndex(), domain.endIndex());
#endif
#ifdef SPH_EXA_USE_ASCENT
	if((d.iteration % 5) == 0)
          AscentAdaptor::Execute(d, domain.startIndex(), domain.endIndex());
#endif
    }

    totalTimer.step("Total execution time of " + std::to_string(maxStep) + " iterations of Evrard");

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
        printf("\nWhere possible options are:\n");

        printf("\t-n NUM \t\t\t NUM^3 Number of particles [65536]\n");
        printf("\t-s NUM \t\t\t NUM Number of iterations (time-steps) [10]\n");
        printf("\t-w NUM \t\t\t Dump Frequency data every NUM iterations (time-steps) [-1]\n\n");
        printf("\t-c NUM \t\t\t Check point Frequency [-1]\n");

        printf("\t--quiet \t\t Don't print anything to stdout [false]\n\n");

        printf("\t--cinput \t\tRead ParticleData from CheckpointBinFile [false]\n\n");

        printf("\t--input PATH \t\t Path to directory where input BinFile is storaged [bigfiles/Test3DEvrardRel.bin].\
                    \n\t\t\t\t Example: --input ../../../bigfiles/Test3DEvrardRel.bin\n");

        printf("\t--outDir PATH \t\t Path to directory where output will be saved [./].\
                    \n\t\t\t\t Note that directory must exist and be provided with ending slash.\
                    \n\t\t\t\t Example: --outDir /home/user/folderToSaveOutputFiles/\n");
    }
}
