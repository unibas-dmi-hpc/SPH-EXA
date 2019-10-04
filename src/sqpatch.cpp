#include <iostream>
#include <fstream>
#include <string>

#include "sphexa.hpp"
#include "SqPatchDataGenerator.hpp"

using namespace std;
using namespace sphexa;

lookup_tables::GpuLookupTableInitializer<double> ltinit;

int main(int argc, char **argv)
{
    ArgParser parser(argc, argv);
    const int cubeSide = parser.getInt("-n", 50);
    const int maxStep = parser.getInt("-s", 10);
    const int writeFrequency = parser.getInt("-w", -1);

#ifdef _JENKINS
    maxStep = 0;
    writeFrequency = -1;
#endif

    using Real = double;
    using Tree = Octree<Real>;
    using Dataset = ParticlesData<Real>;

#ifdef USE_MPI
    MPI_Init(NULL, NULL);
#endif

    auto d = SqPatchDataGenerator<Real>::generate(cubeSide);
    DistributedDomain<Real> distributedDomain;
    Domain<Real, Tree> domain(d.ngmin, d.ng0, d.ngmax);
    Printer<Dataset> printer(d);
    MPITimer timer(d.rank);

    std::ofstream constantsFile("constants.txt");

    vector<int> clist(d.count);
    for (unsigned int i = 0; i < clist.size(); i++)
        clist[i] = i;

    for (d.iteration = 0; d.iteration <= maxStep; d.iteration++)
    {
        timer.start();
        d.resize(d.count); // Discard halos
        distributedDomain.distribute(clist, d);
        timer.step("domain::build");
        distributedDomain.synchronizeHalos(&d.x, &d.y, &d.z, &d.h, &d.m);
        timer.step("mpi::synchronizeHalos");
        domain.buildTree(d);
        timer.step("BuildTree");
        domain.findNeighbors(clist, d);
        timer.step("FindNeighbors");

        sph::computeDensity<Real>(clist, d);
        if (d.iteration == 0) { sph::initFluidDensityAtRest<Real>(clist, d); }
        timer.step("Density");

        sph::computeEquationOfState<Real>(clist, d);
        timer.step("EquationOfState");

        distributedDomain.resizeArrays(d.count, &d.vx, &d.vy, &d.vz, &d.ro, &d.p, &d.c); // Discard halos
        distributedDomain.synchronizeHalos(&d.vx, &d.vy, &d.vz, &d.ro, &d.p, &d.c);
        timer.step("mpi::synchronizeHalos");

        sph::computeIAD<Real>(clist, d);
        timer.step("IAD");

        distributedDomain.resizeArrays(d.count, &d.c11, &d.c12, &d.c13, &d.c22, &d.c23, &d.c33); // Discard halos
        distributedDomain.synchronizeHalos(&d.c11, &d.c12, &d.c13, &d.c22, &d.c23, &d.c33);
        timer.step("mpi::synchronizeHalos");

        sph::computeMomentumAndEnergyIAD<Real>(clist, d);
        timer.step("MomentumEnergyIAD");

        sph::computeTimestep<Real>(clist, d);
        timer.step("Timestep"); // AllReduce(min:dt)
        sph::computePositions<Real>(clist, d);
        timer.step("UpdateQuantities");
        sph::computeTotalEnergy<Real>(clist, d);
        timer.step("EnergyConservation"); // AllReduce(sum:ecin,ein)

        long long int totalNeighbors = domain.neighborsSum(clist, d);
        if (d.rank == 0)
        {
            printer.printCheck(clist.size(), distributedDomain.haloCount, totalNeighbors, std::cout);
            printer.printConstants(d.iteration, totalNeighbors, constantsFile);
        }

        if (writeFrequency > 0 && d.iteration % writeFrequency == 0)
        {
            printer.printAllDataToFile(clist, "dump" + to_string(d.iteration) + ".txt");
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
