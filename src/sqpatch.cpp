#include <iostream>
#include <fstream>
#include <string>

#include "sphexa.hpp"
#include "SqPatch.hpp"

using namespace std;
using namespace sphexa;

int main(int argc, char **argv)
{
    ArgParser parser(argc, argv);
    int cubeSide = parser.getInt("-n", 50);
    int maxStep = parser.getInt("-s", 10);
    int writeFrequency = parser.getInt("-w", -1);

#ifdef _JENKINS
    maxStep = 0;
    writeFrequency = -1;
#endif

    typedef double Real;
    typedef SqPatch<Real> Dataset;

#ifdef USE_MPI
    MPI_Init(NULL, NULL);
#endif

    Dataset d(cubeSide);
    DistributedDomain<Real> distributedDomain;
    //Domain<Real, Tree> domain(d.ngmin, d.ng0, d.ngmax);

    vector<int> clist(d.count);
    for (unsigned int i = 0; i < clist.size(); i++)
        clist[i] = i;

    std::ofstream constants("constants.txt");

    distributedDomain.setBox(0, 0, 0, 0, d.bbox.zmin, d.bbox.zmax, false, false, true);
    distributedDomain.approximate(d);

    MPITimer timer(d.rank);
    for (d.iteration = 0; d.iteration <= maxStep; d.iteration++)
    {
        timer.start();

        distributedDomain.distribute(d);
        timer.step("domain::distribute");

       /* {
            char fname[256];
            sprintf(fname, "particles%d", distributedDomain.comm_rank);
            FILE *fout = fopen(fname, "w");
            for(int i=0; i<(int)d.x.size(); i++)
                fprintf(fout, "%f %f %f\n", d.x[i], d.y[i], d.z[i]);
            fclose(fout);
        }*/

        distributedDomain.synchronizeHalos(&d.x, &d.y, &d.z, &d.h, &d.m);
        timer.step("mpi::synchronizeHalos");
        
        // {
        //     char fname[256];
        //     sprintf(fname, "particlesSync%d", distributedDomain.comm_rank);
        //     FILE *fout = fopen(fname, "w");
        //     for(int i=0; i<(int)d.x.size(); i++)
        //         fprintf(fout, "%f %f %f\n", d.x[i], d.y[i], d.z[i]);
        //     fclose(fout);
        // }

        // distributedDomain.findNeighbors(clist, d);
        // timer.step("FindNeighbors");

        /*domain.buildTree(d);
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

        sph::computeMomentumAndEnergy<Real>(clist, d);
        timer.step("MomentumEnergy");
        sph::computeTimestep<Real>(clist, d);
        timer.step("Timestep"); // AllReduce(min:dt)
        sph::computePositions<Real>(clist, d);
        timer.step("UpdateQuantities");
        sph::computeTotalEnergy<Real>(clist, d);
        timer.step("EnergyConservation"); // AllReduce(sum:ecin,ein)

        long long int totalNeighbors = domain.neighborsSum(clist, d);
        if (d.rank == 0)
        {
            cout << "### Check ### Particles: " << clist.size() << ", Halos: " << distributedDomain.haloCount << endl;
            cout << "### Check ### Computational domain: " << d.bbox.xmin << " " << d.bbox.xmax << " " << d.bbox.ymin << " " << d.bbox.ymax
                 << " " << d.bbox.zmin << " " << d.bbox.zmax << endl;
            cout << "### Check ### Avg neighbor count per particle: " << totalNeighbors / d.n << endl;
            cout << "### Check ### Total time: " << d.ttot << ", current time-step: " << d.dt[0] << endl;
            cout << "### Check ### Total energy: " << d.etot << ", (internal: " << d.eint << ", cinetic: " << d.ecin << ")" << endl;
        }

        if (writeFrequency > 0 && d.iteration % writeFrequency == 0)
        {
            std::ofstream dump("dump" + to_string(d.iteration) + ".txt");
            d.writeData(clist, dump);
            timer.step("writeFile");
            dump.close();
        }
        d.writeConstants(d.iteration, totalNeighbors, constants);

        timer.stop();
        if (d.rank == 0) cout << "=== Total time for iteration(" << d.iteration << ") " << timer.duration() << "s" << endl << endl;*/
    }

    constants.close();

#ifdef USE_MPI
    MPI_Finalize();
#endif

    return 0;
}
