#include <iostream>
#include <fstream>
#include <string>
#include <vector>

#include "sphexa.hpp"
#include "WindblobDataGenerator.hpp"
#include "WindblobFileWriter.hpp"

#include <fenv.h>
#pragma STDC FENV_ACCESS ON


using namespace sphexa;

int main(int argc, char **argv)
{
#ifndef USE_MPI
#error "Windblob Requires MPI at the moment due to MPI-Only version of file reader"
#endif

    std::feclearexcept(FE_ALL_EXCEPT);

    const int rank = initAndGetRankId();

    const ArgParser parser(argc, argv);
    const size_t maxStep = parser.getInt("-s", 10);
    const int writeFrequency = parser.getInt("-w", -1);
    const std::string outDirectory = "";
    const bool oldAV = parser.exists("--oldAV");
    const bool gVE = parser.exists("--gVE");
    const size_t hNRStart = parser.getInt("--hNRStart", maxStep);
    const size_t hNgBMStop = parser.getInt("--hNgBMStop", maxStep);
    const size_t ngmax_cli = std::max(parser.getInt("--ngmax", 750), 0);
    const size_t ngmin_cli = std::max(parser.getInt("--ngmin", 150), 0);
    const size_t hackyNgMinMaxFixTries = parser.getInt("--hackyNgMinMaxFixTries", 5);

    std::ostream &output = std::cout;

#ifdef _JENKINS
    maxStep = 0;
    writeFrequency = -1;
#endif

    using Real = double;
    using Dataset = ParticlesData<Real>;
    using Tree = Octree<Real>;


    DistributedDomain<Real, Dataset, Tree> domain;
    const IFileWriter<Dataset> &fileWriter = WindblobMPIFileWriter<Dataset>();

    auto d = WindblobDataGenerator<Real>::generate("data/windblob_3M.bin");
    d.oldAV = oldAV;
    const Printer<Dataset> printer(d);

    MasterProcessTimer timer(output, d.rank), totalTimer(output, d.rank);

    std::ofstream constantsFile(outDirectory + "constants.txt");

    Tree::bucketSize = 64;
    Tree::minGlobalBucketSize = 512;
    Tree::maxGlobalBucketSize = 2048;
    domain.create(d);

    const size_t nTasks = 64;
    const size_t ng0 = 250;
//    const size_t ngmax = 750; // increased to fight bug
    const size_t ngmax = std::max(ng0 + 100, ngmax_cli);
    TaskList taskList = TaskList(domain.clist, nTasks, ngmax, ng0);

    // want to dump on floating point exceptions
    bool fpe_raised = false;

    totalTimer.start();

    for(size_t i=0; i<10; i++)
    {
        domain.update(d);
        timer.step("domain::distribute");
        domain.synchronizeHalos(&d.x, &d.y, &d.z, &d.h, &d.xmass);
        timer.step("mpi::synchronizeHalos");
        domain.buildTree(d);
        timer.step("domain::buildTree");
        taskList.update(domain.clist);
        timer.step("updateTasks");
        sph::findNeighbors(domain.octree, taskList.tasks, d);
        timer.step("FindNeighbors");
        sph::updateSmoothingLength<Real>(taskList.tasks, d);
        timer.step("UpdateSmoothingLength");
    }


    for (d.iteration = 0; d.iteration <= maxStep; d.iteration++)
    {
        timer.start();

        domain.update(d);
        timer.step("domain::distribute");
        domain.synchronizeHalos(&d.x, &d.y, &d.z, &d.h, &d.xmass);
        timer.step("mpi::synchronizeHalos");
        domain.buildTree(d);
        timer.step("domain::buildTree");
        taskList.update(domain.clist);
        timer.step("updateTasks");
        sph::findNeighbors(domain.octree, taskList.tasks, d);
        timer.step("FindNeighbors");

        size_t maxNeighbors, minNeighbors;
        std::tie(minNeighbors, maxNeighbors) = sph::neighborsStats<Real>(taskList.tasks, d); // AllReduce
        size_t tries = 0;
        while (tries < hackyNgMinMaxFixTries && (maxNeighbors > ngmax || minNeighbors < ngmin_cli)) {
            tries++;
            if (d.rank == 0) output << "-- minNeighbors " << minNeighbors << " maxNeighbors " << maxNeighbors << " try: " << tries <<  std::endl;
            // updating for all is often unstable and oscillates...
            // the update guarantees to reduce h if it's higher than ng0 ->
            // a particle that had n < ng0 is now overshooting
            // another option could be to have a "soft"-max, i.e.
            // the threshold for recalculating is lower than the actually supported max...
            sph::updateSmoothingLengthForExceeding<Real>(taskList.tasks, d, ngmin_cli);
            timer.step("HackyUpdateSmoothingLengthForThoseWithTooManyOrTooFewNeighbors");
            domain.update(d);
            timer.step("domain::distribute");
            domain.synchronizeHalos(&d.x, &d.y, &d.z, &d.h, &d.xmass);
            timer.step("mpi::synchronizeHalos");
            domain.buildTree(d);
            timer.step("domain::buildTree");
            taskList.update(domain.clist);
            timer.step("updateTasks");
            sph::findNeighbors(domain.octree, taskList.tasks, d);
            timer.step("FindNeighbors");
            std::tie(minNeighbors, maxNeighbors) = sph::neighborsStats<Real>(taskList.tasks, d); // AllReduce        const int maxRetries = 10;
        }

        sph::computeDensity<Real>(taskList.tasks, d);
        timer.step("Density");
        if (d.iteration == 0) { sph::initFluidDensityAtRest<Real>(taskList.tasks, d); }
        if (d.iteration > hNRStart) {
            sph::newtonRaphson<Real>(taskList.tasks, d);
            timer.step("hNR");
            for (int iterNR = 0; iterNR < 2; iterNR++) {
                sph::computeDensity<Real>(taskList.tasks, d);
                timer.step("Density");
                sph::newtonRaphson<Real>(taskList.tasks, d);
                timer.step("hNR");
            }
        }
        sph::calcGradhTerms<Real>(taskList.tasks, d);
        timer.step("calcGradhTerms");
        sph::computeEquationOfStateWindblob<Real>(taskList.tasks, d);
        timer.step("EquationOfState");
        domain.synchronizeHalos(&d.vx, &d.vy, &d.vz, &d.ro, &d.p, &d.c, &d.sumkx, &d.gradh, &d.h, &d.vol);
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
        if (d.iteration < hNgBMStop) {
            sph::updateSmoothingLength<Real>(taskList.tasks, d);
            timer.step("UpdateSmoothingLength");
        }

        if (gVE && d.iteration > hNRStart - 5)
            sph::updateVEEstimator<Real, sph::XmassSPHYNXVE<Real, Dataset>>(taskList.tasks, d);
        else
            sph::updateVEEstimator<Real, sph::XmassStdVE<Real, Dataset>>(taskList.tasks, d);
        timer.step("UpdateVEEstimator");

        const size_t totalNeighbors = sph::neighborsSum(taskList.tasks);

        if (d.rank == 0)
        {
            printer.printCheck(d.count, domain.octree.globalNodeCount, d.x.size() - d.count, totalNeighbors,
                               minNeighbors, maxNeighbors, ngmax, output);
            printer.printConstants(d.iteration, totalNeighbors, minNeighbors, maxNeighbors, ngmax, constantsFile);
        }

        fpe_raised = all_check_FPE("after print, rank " + std::to_string(d.rank));
        if (fpe_raised) break;

        if ((writeFrequency > 0 && d.iteration % writeFrequency == 0) || writeFrequency == 0)
        {
            fileWriter.dumpParticleDataToAsciiFile(d, domain.clist, outDirectory + "dump_windblob" + std::to_string(d.iteration) + ".txt");
            timer.step("writeFile");
        }

        timer.stop();

        if (d.rank == 0) printer.printTotalIterationTime(timer.duration(), output);
    }

    if (fpe_raised) {
        fileWriter.dumpParticleDataToAsciiFile(d, domain.clist, outDirectory + "fperrordump_windblob" + std::to_string(d.iteration) + "_" + std::to_string(std::time(0)) + ".txt");
    }

    totalTimer.step("Total execution time for " + std::to_string(d.iteration) + " iterations of Windblob");

    constantsFile.close();

    return exitSuccess();
}
