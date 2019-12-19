#include <iostream>
#include <fstream>
#include <string>
#include <vector>

#include "sphexa.hpp"
#include "SqPatchDataGenerator.hpp"

using namespace std;
using namespace sphexa;

int main(int argc, char **argv)
{
    ArgParser parser(argc, argv);
    const size_t cubeSide = parser.getInt("-n", 50);
    const size_t maxStep = parser.getInt("-s", 10);
    const int writeFrequency = parser.getInt("-w", -1);

    using Real = double;
    using Dataset = ParticlesData<Real>;
    using Tree = Octree<Real>;

#ifdef USE_MPI
    MPI_Init(NULL, NULL);
    DistributedDomain<Real, Dataset, Tree> domain;
#else
    Domain<Real, Dataset, Tree> domain;
#endif

    auto d = SqPatchDataGenerator<Real>::generate(cubeSide);
    Printer<Dataset> printer(d);
    MasterProcessTimer timer(d.rank);

    std::ofstream constantsFile("constants.txt");

    Tree::bucketSize = 64;
    Tree::minGlobalBucketSize = 512;
    Tree::maxGlobalBucketSize = 2048;
    domain.create(d);

    const size_t nTasks = 64;
    const size_t ngmax = 300;
    const size_t ng0 = 250;
    TaskList taskList = TaskList(domain.clist, nTasks, ngmax, ng0);

    for (d.iteration = 0; d.iteration <= maxStep; d.iteration++)
    {
        timer.start();

        domain.update(d);
        timer.step("domain::distribute");
        domain.synchronizeHalos(&d.x, &d.y, &d.z, &d.h);
        timer.step("mpi::synchronizeHalos");
        domain.buildTree(d);
        timer.step("domain::buildTree");
        taskList.update(domain.clist);
        timer.step("updateTasks");
        sph::findNeighbors(domain.octree, taskList.tasks, d);
        timer.step("FindNeighbors");
        sph::computeDensity<Real>(taskList.tasks, d);
        if (d.iteration == 0) { sph::initFluidDensityAtRest<Real>(taskList.tasks, d); }
        timer.step("Density");
        sph::computeEquationOfState<Real>(taskList.tasks, d);
        timer.step("EquationOfState");
        domain.synchronizeHalos(&d.vx, &d.vy, &d.vz, &d.ro, &d.p, &d.c);
        timer.step("mpi::synchronizeHalos");
        sph::computeIAD<Real>(taskList.tasks, d);
        timer.step("IAD");
        domain.synchronizeHalos(&d.c11, &d.c12, &d.c13, &d.c22, &d.c23, &d.c33);
        timer.step("mpi::synchronizeHalos");
        sph::computeMomentumAndEnergyIAD<Real>(taskList.tasks, d);
        timer.step("MomentumEnergyIAD");
        sph::computeTimestep<Real, sph::TimestepPress2ndOrder<Real, Dataset>>(taskList.tasks, d);
        timer.step("Timestep"); // AllReduce(min:dt)
        sph::computePositions<Real, sph::computeAcceleration<Real, Dataset>>(taskList.tasks, d);
        timer.step("UpdateQuantities");
        sph::computeTotalEnergy<Real>(taskList.tasks, d);
        timer.step("EnergyConservation"); // AllReduce(sum:ecin,ein)
        sph::updateSmoothingLength<Real>(taskList.tasks, d);
        timer.step("UpdateSmoothingLength");

        size_t totalNeighbors = sph::neighborsSum(taskList.tasks);
        if (d.rank == 0)
        {
            printer.printCheck(d.count, domain.octree.globalNodeCount, d.x.size() - d.count, totalNeighbors, std::cout);
            printer.printConstants(d.iteration, totalNeighbors, constantsFile);
        }

        if ((writeFrequency > 0 && d.iteration % writeFrequency == 0) || writeFrequency == 0)
        {
            printer.printAllDataToFile(domain.clist, "dump" + to_string(d.iteration) + ".txt");
            timer.step("writeFile");
        }

        timer.stop();

        if (d.rank == 0) printer.printTotalIterationTime(timer.duration(), std::cout);
    }

    constantsFile.close();

#ifdef USE_MPI
    MPI_Finalize();
#endif

    return 0;
}
