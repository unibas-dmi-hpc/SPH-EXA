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
#include "NohDataGenerator.hpp"
#include "NohDataFileWriter.hpp"
#include "NohAnalyticalSolution.hpp"

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

    const size_t cubeSide = parser.getInt("-n", 50);
    const size_t maxStep = parser.getInt("-s", 300);
    const int writeFrequency = parser.getInt("-w", -1);
    const bool solution = parser.exists("--sol");
    const bool quiet = parser.exists("--quiet");
    const std::string outDirectory = parser.getString("--outDir");

    std::ofstream nullOutput("/dev/null");
    std::ostream& output = quiet ? nullOutput : std::cout;

    using Real = double;
    using KeyType = uint64_t;
    using Dataset = ParticlesData<Real, KeyType>;

    const IFileWriter<Dataset>& fileWriter = NohMPIFileWriter<Dataset>();

    auto d = NohDataGenerator<Real, KeyType>::generate(cubeSide);

    const size_t dim    = 3;
    const double gamma  = NohDataGenerator<Real, KeyType>::gamma;
    const double r0     = NohDataGenerator<Real, KeyType>::r0;
    const double r1     = NohDataGenerator<Real, KeyType>::r1;
    const double rho0   = NohDataGenerator<Real, KeyType>::rho0;
    const double vr0    = NohDataGenerator<Real, KeyType>::vel0;

    if (d.rank == 0) std::cout << "Data generated." << std::endl;

    MasterProcessTimer timer(output, d.rank), totalTimer(output, d.rank);

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
        computeTimestep<Real, TimestepPress2ndOrder<Real, Dataset>>(taskList.tasks, d);
        timer.step("Timestep"); // AllReduce(min:dt)
        computePositions<Real, computeAcceleration<Real, Dataset>>(taskList.tasks, d, domain.box());
        timer.step("UpdateQuantities");
        computeTotalEnergy<Real>(taskList.tasks, d);
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
            std::string solutionFilename;
            std::string simulationFilename;

#ifdef SPH_EXA_HAVE_H5PART
            fileWriter.dumpParticleDataToH5File(
                d, domain.startIndex(), domain.endIndex(), outDirectory + "dump_noh.h5part");
            solutionFilename   = outDirectory + "dump_noh_sol.h5part";
            simulationFilename = outDirectory + "dump_noh_sim.h5part";
#else
            fileWriter.dumpParticleDataToAsciiFile(
                d, domain.startIndex(), domain.endIndex(), "dump_noh" + std::to_string(d.iteration) + ".txt");
            solutionFilename   = "dump_noh" + std::to_string(d.iteration) + "_sol.txt";
            simulationFilename = "dump_noh" + std::to_string(d.iteration) + "_sim.txt";
#endif

            if (solution)
            {
                /*
                NohAnalyticalSolution::create(  1,                      // xgeom
                                                0.,                     // r0
                                                1.,                     // r1
                                                1000,                   // nstep
                                                0.6,                    // time
                                                5./3.,                  // gamma
                                                1.,                     // rho0
                                                -1.,                    // vel0
                                                "theoretical_1D.dat");  // outfile

                NohAnalyticalSolution::create(  2,                      // xgeom
                                                0.,                     // r0
                                                1.,                     // r1
                                                1000,                   // nstep
                                                0.6,                    // time
                                                5./3.,                  // gamma
                                                1.,                     // rho0
                                                -1.,                    // vel0
                                                "theoretical_2D.dat");  // outfile

                NohAnalyticalSolution::create(  3,                      // xgeom
                                                0.,                     // r0
                                                1.,                     // r1
                                                1000,                   // nstep
                                                0.6,                    // time
                                                5./3.,                  // gamma
                                                1.,                     // rho0
                                                -1.,                    // vel0
                                                "theoretical_3D.dat");  // outfile

                exit(-1);
                */

                // Calculate and write theoretical solution in 1D
                size_t nSteps = 1000;  // Instead of 'domain.nParticles()'. It is not needed more precission to compare.
                NohAnalyticalSolution::create(  dim,
                                                r0,
                                                r1,
                                                nSteps,
                                                d.ttot,
                                                gamma,
                                                rho0,
                                                vr0,
                                                solutionFilename);

                // Calculate modules for position and velocity of the particles
                vector<double> r  (d.count);
                vector<double> vel(d.count);
                for(size_t i = 0; i < d.count; i++)
                {
                    r[i]   = std::sqrt( std::pow(d.x[i],  2.) + std::pow(d.y[i],  2.) + std::pow(d.z[i],  2.) );
                    vel[i] = std::sqrt( std::pow(d.vx[i], 2.) + std::pow(d.vy[i], 2.) + std::pow(d.vz[i], 2.) );
                }

                // Write 1D simulation solution to compare with the theoretical solution
                NohSolutionWriter::dump1DToAsciiFile(d.count, r, d.ro, d.u, d.p, vel, simulationFilename);
            }

            timer.step("writeFile");
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

        printf("\t-n NUM \t\t\t NUM^3 Number of particles [50]\n");
        printf("\t-s NUM \t\t\t NUM Number of iterations (time-steps) [300]\n\n");

        printf("\t-w NUM \t\t\t Dump particles data every NUM iterations (time-steps) [-1]\n");
        printf("\t--sol   \t\t Print anytical solution every dump [false]\n\n");

        printf("\t--quiet \t\t Don't print anything to stdout [false]\n\n");

        printf("\t--outDir PATH \t\t Path to directory where output will be saved [./].\
                    \n\t\t\t\t Note that directory must exist and be provided with ending slash.\
                    \n\t\t\t\t Example: --outDir /home/user/folderToSaveOutputFiles/\n");
    }
}