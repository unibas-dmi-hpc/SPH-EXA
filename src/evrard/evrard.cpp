#include <fstream>
#include <string>

#include "sphexa.hpp"
#include "EvrardCollapseInputFileReader.hpp"
#include "EvrardCollapseFileWriter.hpp"

using namespace sphexa;

int main(int argc, char **argv)
{
    ArgParser parser(argc, argv);

    const size_t maxStep = parser.getInt("-s", 10);
    const size_t nParticles = parser.getInt("-n", 65536);
    const int writeFrequency = parser.getInt("-w", -1);
    const int checkpointFrequency = parser.getInt("-c", -1);
    const bool quiet = parser.exists("--quiet");
    const std::string checkpointInput = parser.getString("--cinput");
    const std::string inputFilePath = parser.getString("--input", "bigfiles/Test3DEvrardRel.bin");
    const std::string outDirectory = parser.getString("--outDir");

    std::ofstream nullOutput("/dev/null");
    std::ostream &output = quiet ? nullOutput : std::cout;

    using Real = double;
    using Dataset = ParticlesDataEvrard<Real>;
    using Tree = GravityOctree<Real>;

#ifdef USE_MPI
    MPI_Init(NULL, NULL);
    DistributedDomain<Real, Dataset, Tree> domain;
    const IFileReader<Dataset> &fileReader = EvrardCollapseMPIInputFileReader<Dataset>();
    const IFileWriter<Dataset> &fileWriter = EvrardCollapseMPIFileWriter<Dataset>();
#else
    Domain<Real, Dataset, Tree> domain;
    const IFileReader<Dataset> &fileReader = EvrardCollapseInputFileReader<Dataset>();
    const IFileWriter<Dataset> &fileWriter = EvrardCollapseFileWriter<Dataset>();
#endif

    auto d = checkpointInput.empty() ? fileReader.readParticleDataFromBinFile(inputFilePath, nParticles)
                                     : fileReader.readParticleDataFromCheckpointBinFile(checkpointInput);
    const Printer<Dataset> printer(d);

    MasterProcessTimer timer(output, d.rank), totalTimer(output, d.rank);

    std::ofstream constantsFile("constants.txt");

    Tree::bucketSize = 1;
    Tree::minGlobalBucketSize = 1;
    Tree::maxGlobalBucketSize = 1;
    domain.create(d);

    const size_t nTasks = 64;
    const size_t ng0 = 100;
    const size_t ngmax = 150;
    TaskList taskList = TaskList(domain.clist, nTasks, ngmax, ng0);

    totalTimer.start();
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

        const size_t totalNeighbors = sph::neighborsSum(taskList.tasks);
        if (d.rank == 0)
        {
            // printer.printRadiusAndGravityForce(domain.clist, fxFile);
            // printer.printTree(domain.octree, treeFile);
            printer.printCheck(d.count, domain.octree.globalNodeCount, d.x.size() - d.count, totalNeighbors, output);
            printer.printConstants(d.iteration, totalNeighbors, constantsFile);
        }

        if (writeFrequency > 0 && d.iteration % writeFrequency == 0)
        {
            fileWriter.dumpParticleDataToAsciiFile(d, domain.clist, outDirectory + "dump_evrard" + std::to_string(d.iteration) + ".txt");
            timer.step("writeFile");
        }
        if (checkpointFrequency > 0 && d.iteration % checkpointFrequency == 0)
        {
            fileWriter.dumpCheckpointDataToBinFile(d, outDirectory + "checkpoint_evrard" + std::to_string(d.iteration) + ".bin");
            timer.step("Save Checkpoint File");
        }

        timer.stop();

        if (d.rank == 0) printer.printTotalIterationTime(timer.duration(), output);
    }

    totalTimer.step("Total execution time of " + std::to_string(maxStep) + " iterations of Evrard Collapse");
    constantsFile.close();

#ifdef USE_MPI
    MPI_Finalize();
#endif

    return 0;
}
