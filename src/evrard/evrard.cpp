#include <fstream>
#include <string>

#include "sphexa.hpp"
#include "EvrardCollapseInputFileReader.hpp"

using namespace sphexa;

int main(int argc, char **argv)
{
    ArgParser parser(argc, argv);

    const size_t maxStep = parser.getInt("-s", 10);
    const size_t writeFrequency = parser.getInt("-w", -1);
    const bool quiet = parser.exists("--quiet");
    
    std::ofstream nullOutput("/dev/null");
    std::ostream& output = quiet ? nullOutput : std::cout;

    using Real = double;
    using Dataset = ParticlesDataEvrard<Real>;
    using Tree = GravityOctree<Real>;
#ifdef USE_MPI
    MPI_Init(NULL, NULL);
    DistributedDomain<Real, Dataset, Tree> domain;
#else
    Domain<Real, Dataset, Tree> domain;
#endif

    const size_t nParticles = 65536;
    auto d = EvrardCollapseInputFileReader<Real>::load(nParticles, "bigfiles/Test3DEvrardRel.bin");

    Printer<Dataset> printer(d);
    MasterProcessTimer timer(output, d.rank);

    std::ofstream constantsFile("constants.txt");

    Tree::bucketSize = 1;
    Tree::minGlobalBucketSize = 1;
    Tree::maxGlobalBucketSize = 1;
    domain.create(d);

    const size_t nTasks = 64;
    const size_t ng0 = 100;
    const size_t ngmax = 150;
    TaskList taskList = TaskList(domain.clist, nTasks, ngmax, ng0);

    for (d.iteration = 0; d.iteration <= maxStep; d.iteration++)
    {
        timer.start();
        domain.update(d);
        timer.step("domain::distribute");
        domain.synchronizeHalos(&d.x, &d.y, &d.z, &d.h, &d.m);
        timer.step("mpi::synchronizeHalos");
        domain.buildTree(d);
        timer.step("BuildTree");
        taskList.update(domain.clist);
        timer.step("updateTasks");
        sph::gravityTreeWalk(taskList.tasks, domain.octree, d);
        timer.step("Gravity");
        sph::findNeighbors(domain.octree, taskList.tasks, d);
        timer.step("FindNeighbors");
        sph::computeDensity<Real>(taskList.tasks, d);
        if (d.iteration == 0) { sph::initFluidDensityAtRest<Real>(taskList.tasks, d); }
        timer.step("Density");
        sph::computeEquationOfStateEvrard<Real>(taskList.tasks, d);
        timer.step("EquationOfState");
        domain.synchronizeHalos(&d.vx, &d.vy, &d.vz, &d.ro, &d.p, &d.c);
        timer.step("mpi::synchronizeHalos");
        sph::computeIAD<Real>(taskList.tasks, d);
        timer.step("IAD");
        domain.synchronizeHalos(&d.c11, &d.c12, &d.c13, &d.c22, &d.c23, &d.c33);
        timer.step("mpi::synchronizeHalos");
        sph::computeMomentumAndEnergyIAD<Real>(taskList.tasks, d);
        timer.step("MomentumEnergyIAD");
        sph::computeTimestep<Real, sph::TimestepKCourant<Real, Dataset>>(taskList.tasks, d);
        timer.step("Timestep"); // AllReduce(min:dt)
        sph::computePositions<Real, sph::computeAccelerationWithGravity<Real, Dataset>, Dataset>(taskList.tasks, d);
        timer.step("UpdateQuantities");
        sph::computeTotalEnergyWithGravity<Real>(taskList.tasks, d);
        timer.step("EnergyConservation"); // AllReduce(sum:ecin,ein)
        sph::updateSmoothingLength<Real>(taskList.tasks, d);
        timer.step("UpdateSmoothingLength"); // AllReduce(sum:ecin,ein)

        size_t totalNeighbors = sph::neighborsSum(taskList.tasks);
        if (d.rank == 0)
        {
            // printer.printRadiusAndGravityForce(domain.clist, fxFile);
            // printer.printTree(domain.octree, treeFile);
            printer.printCheck(d.count, domain.octree.globalNodeCount, d.x.size() - d.count, totalNeighbors, output);
            printer.printConstants(d.iteration, totalNeighbors, constantsFile);
        }

        if (writeFrequency > 0 && d.iteration % writeFrequency == 0)
        {
            printer.printAllDataToFile(domain.clist, "dump" + std::to_string(d.iteration) + ".txt");
            timer.step("writeFile");
        }

        timer.stop();

        if (d.rank == 0) printer.printTotalIterationTime(timer.duration(), output);
    }

    constantsFile.close();

#ifdef USE_MPI
    MPI_Finalize();
#endif

    return 0;
}
