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
#include "ifile_writer.hpp"
#include "sedov_generator.hpp"

using namespace cstone;
using namespace sphexa;
using namespace sphexa::sph;

#ifdef SPH_EXA_USE_CATALYST2
#include "CatalystAdaptor.h"
#endif

#ifdef SPH_EXA_USE_ASCENT
#include "AscentAdaptor.h"
#endif

#ifdef USE_CUDA
using AccType = cstone::GpuTag;
#else
using AccType = cstone::CpuTag;
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

    const size_t      cubeSide       = parser.getInt("-n", 50);
    const size_t      maxStep        = parser.getInt("-s", 10);
    const int         writeFrequency = parser.getInt("-w", -1);
    const bool        quiet          = parser.exists("--quiet");
    const bool        ascii          = parser.exists("--ascii");
    const std::string outDirectory   = parser.getString("--outDir");

    std::ofstream nullOutput("/dev/null");
    std::ostream& output = quiet ? nullOutput : std::cout;

    using Real    = double;
    using KeyType = uint64_t;
    using Dataset = ParticlesData<Real, KeyType, AccType>;

    std::vector<std::string> outputFields = parser.getCommaList("-f");
    if (outputFields.empty()) { outputFields = {"x", "y", "z", "vx", "vy", "vz", "h", "rho", "u", "p", "c"}; }

    std::unique_ptr<IFileWriter<Dataset>> fileWriter;
    if (ascii) { fileWriter = std::make_unique<AsciiWriter<Dataset>>(); }
    else
    {
        fileWriter = std::make_unique<H5PartWriter<Dataset>>();
    }
    std::ofstream constantsFile(outDirectory + "constants.txt");

    // Feed max min here.
    // If makeGlobalBox is called later, it wont touch it.
    Dataset d;
    d.side = cubeSide;
    d.setConservedFieldsVE();
    d.setDependentFieldsVE();
    SedovDataGenerator::generate(d);
    d.setOutputFields(outputFields);

    if (d.rank == 0) std::cout << "Data generated." << std::endl;

    size_t bucketSizeFocus = 64;
    // we want about 100 global nodes per rank to decompose the domain with +-1% accuracy
    size_t bucketSize = std::max(bucketSizeFocus, (cubeSide * cubeSide * cubeSide) / (100 * d.nrank));

    Box<Real> box(0, 1);
    // Define box first and not call this.
    box = makeGlobalBox(d.x.begin(), d.x.end(), d.y.begin(), d.z.begin(), box);

    // Enable PBC and enlarge bounds
    Real dx = 0.5 / cubeSide;
    box     = Box<Real>(box.xmin() - dx,
                    box.xmax() + dx,
                    box.ymin() - dx,
                    box.ymax() + dx,
                    box.zmin() - dx,
                    box.zmax() + dx,
                    true,
                    true,
                    true);

    float theta = 1.0;

#ifdef USE_CUDA
    Domain<KeyType, Real, CudaTag> domain(rank, d.nrank, bucketSize, bucketSizeFocus, theta, box);
#else
    Domain<KeyType, Real> domain(rank, d.nrank, bucketSize, bucketSizeFocus, theta, box);
#endif

    domain.sync(
        d.codes, d.x, d.y, d.z, d.h, d.m, d.u, d.vx, d.vy, d.vz, d.x_m1, d.y_m1, d.z_m1, d.du_m1, d.dt_m1, d.alpha);

    if (d.rank == 0) std::cout << "Domain synchronized, nLocalParticles " << d.x.size() << std::endl;

    const size_t nTasks   = 64;
    const size_t ngmax    = 150;
    const size_t ng0      = 100;
    TaskList     taskList = TaskList(0, domain.nParticles(), nTasks, ngmax, ng0);

#ifdef SPH_EXA_USE_ASCENT
    AscentAdaptor::Initialize(d, domain.startIndex());
    std::cout << "AscentInitialize\n";
#endif

    MasterProcessTimer timer(output, d.rank), totalTimer(output, d.rank);
    totalTimer.start();
    for (d.iteration = 0; d.iteration <= maxStep; d.iteration++)
    {
        timer.start();
        domain.sync(
            d.codes, d.x, d.y, d.z, d.h, d.m, d.u, d.vx, d.vy, d.vz, d.x_m1, d.y_m1, d.z_m1, d.du_m1, d.dt_m1, d.alpha);
        timer.step("domain::sync");

        resize(d, domain.nParticlesWithHalos());
        size_t first = domain.startIndex();
        size_t last  = domain.endIndex();

        // domain.exchangeHalos(d.m);
        std::fill(begin(d.m), begin(d.m) + domain.startIndex(), d.m[domain.startIndex()]);
        std::fill(begin(d.m) + domain.endIndex(), begin(d.m) + domain.nParticlesWithHalos(), d.m[domain.startIndex()]);

        taskList.update(domain.startIndex(), domain.endIndex());
        timer.step("updateTasks");
        findNeighborsSfc(taskList.tasks, d.x, d.y, d.z, d.h, d.codes, domain.box());
        timer.step("FindNeighbors");
        computeRho0<Real>(taskList.tasks, d, domain.box());
        timer.step("Rho0");
        domain.exchangeHalos(d.rho0);
        timer.step("mpi::synchronizeHalos");
        computeDensity<Real>(taskList.tasks, d, domain.box());
        timer.step("Density");
        computeEquationOfState(first, last, d);
        timer.step("EquationOfState");
        domain.exchangeHalos(d.vx, d.vy, d.vz, d.rho, d.p, d.c, d.kx);
        timer.step("mpi::synchronizeHalos");
        computeIAD<Real>(taskList.tasks, d, domain.box());
        timer.step("IAD");
        computedivv_curlv<Real>(taskList.tasks, d, domain.box());
        timer.step("divv_curlv");
        domain.exchangeHalos(d.c11, d.c12, d.c13, d.c22, d.c23, d.c33, d.divv, d.curlv);
        timer.step("mpi::synchronizeHalos");
        computeAVswitches<Real>(taskList.tasks, d, domain.box());
        timer.step("AVswitches");
        domain.exchangeHalos(d.alpha);
        timer.step("mpi::synchronizeHalos");
        computeMomentumAndEnergyIAD<Real>(taskList.tasks, d, domain.box());
        timer.step("MomentumEnergyIAD");
        computeTimestep(first, last, d);
        timer.step("Timestep"); // AllReduce(min:dt)
        computePositions(first, last, d, domain.box());
        timer.step("UpdateQuantities");
        computeTotalEnergy(first, last, d);
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
            std::cout << "### Check ### Focus Tree Nodes: " << domain.focusTree().numLeafNodes() << std::endl;
            Printer::printConstants(
                d.iteration, d.ttot, d.minDt, d.etot, d.ecin, d.eint, d.egrav, totalNeighbors, constantsFile);
        }

        if ((writeFrequency > 0 && d.iteration % writeFrequency == 0) || writeFrequency == 0)
        {
            fileWriter->dump(d, domain.startIndex(), domain.endIndex(), outDirectory + "dump_sedov");
        }

        timer.stop();

        if (d.rank == 0) { Printer::printTotalIterationTime(d.iteration, timer.duration(), output); }
#ifdef SPH_EXA_USE_CATALYST2
        CatalystAdaptor::Execute(d, domain.startIndex(), domain.endIndex());
#endif
#ifdef SPH_EXA_USE_ASCENT
        if ((d.iteration % 5) == 0) AscentAdaptor::Execute(d, domain.startIndex(), domain.endIndex());
#endif
    }

    totalTimer.step("Total execution time of " + std::to_string(maxStep) + " iterations of Sedov");

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

        printf("\t-n NUM \t\t\t NUM^3 Number of particles [50]\n");
        printf("\t-s NUM \t\t\t NUM Number of iterations (time-steps) [10]\n\n");

        printf("\t-w NUM \t\t\t Dump particles data every NUM iterations (time-steps) [-1]\n\n");

        printf("\t--quiet \t\t Don't print anything to stdout [false]\n\n");

        printf("\t--outDir PATH \t\t Path to directory where output will be saved [./].\
                    \n\t\t\t\t Note that directory must exist and be provided with ending slash.\
                    \n\t\t\t\t Example: --outDir /home/user/folderToSaveOutputFiles/\n");
    }
}
