#include <chrono>

#include "types.h"
#include "buildtree.h"
#include "dataset.h"
#include "grouptargets.h"
#include "traversal.h"
#include "upwardpass.h"

int main(int argc, char** argv)
{
#if MASS
    const int numBodies = (1 << 10) - 1;
    const int images    = 0;
    const float theta   = 0.75;
#else
    const int numBodies = (1 << 10) - 1;
    const int images    = 0;
    const float theta   = 0.75;
#endif
    const float eps   = 0.05;
    const int ncrit   = 64;
    const float cycle = 2 * M_PI;

    fprintf(stdout, "--- FMM Parameters ---------------\n");
    fprintf(stdout, "numBodies            : %d\n", numBodies);
    fprintf(stdout, "P                    : %d\n", P);
    fprintf(stdout, "theta                : %f\n", theta);
    fprintf(stdout, "ncrit                : %d\n", ncrit);
    const Dataset data(numBodies);

    cudaVec<fvec4> bodyPos(numBodies, true);
    cudaVec<fvec4> bodyPos2(numBodies);
    cudaVec<fvec4> bodyAcc(numBodies, true);
    cudaVec<fvec4> bodyAcc2(numBodies, true);
    for (int i = 0; i < numBodies; i++)
    {
        bodyPos[i][0] = data.pos[i][0];
        bodyPos[i][1] = data.pos[i][1];
        bodyPos[i][2] = data.pos[i][2];
        bodyPos[i][3] = data.pos[i][3];
    }
    bodyPos.h2d();
    bodyAcc.h2d();

    fprintf(stdout, "--- FMM Profiling ----------------\n");

    auto t0 = std::chrono::high_resolution_clock::now();

    Box box;
    cudaVec<int2> levelRange(32, true);
    cudaVec<CellData> sourceCells(numBodies);

    int3 counts = Build::tree<ncrit>(bodyPos, bodyPos2, box, levelRange, sourceCells);

    int numLevels  = counts.x;
    int numSources = counts.y;
    int numLeafs   = counts.z;

    cudaVec<int2> targetRange(numBodies);
    cudaVec<fvec4> sourceCenter(numSources);
    cudaVec<fvec4> Multipole(NVEC4 * numSources);
    Group group;
    int numTargets = group.targets(bodyPos, bodyPos2, box, targetRange, 5);

    Pass::upward(numLeafs, numLevels, theta, levelRange, bodyPos, sourceCells, sourceCenter, Multipole);

    fvec4 interactions = Traversal::approx(numTargets,
                                           images,
                                           eps,
                                           cycle,
                                           bodyPos,
                                           bodyPos2,
                                           bodyAcc,
                                           targetRange,
                                           sourceCells,
                                           sourceCenter,
                                           Multipole,
                                           levelRange);

    auto t1      = std::chrono::high_resolution_clock::now();
    double dt    = std::chrono::duration<double>(t1 - t0).count();
    double flops = (interactions[0] * 20 + interactions[2] * 2 * pow(P, 3)) * numBodies / dt / 1e12;

    fprintf(stdout, "--- Total runtime ----------------\n");
    fprintf(stdout, "Total FMM            : %.7f s (%.7f TFlops)\n", dt, flops);

    const int numTarget = std::min(512, numBodies); // Number of threads per block will be set to this value
    const int numBlock  = std::min(128, (numBodies - 1) / numTarget + 1);

    t0 = std::chrono::high_resolution_clock::now();
    Traversal::direct(numTarget, numBlock, images, eps, cycle, bodyPos2, bodyAcc2);
    t1 = std::chrono::high_resolution_clock::now();
    dt = std::chrono::duration<double>(t1 - t0).count();

    flops = 35. * numTarget * numBodies * pow(2 * images + 1, 3) / dt / 1e12;
    fprintf(stdout, "Total Direct         : %.7f s (%.7f TFlops)\n", dt, flops);

    bodyAcc.d2h();
    bodyAcc2.d2h();

    //for (int i = 0; i < numTarget; i++)
    //{
    //    fvec4 tempAcc = bodyAcc2[i];
    //    for (int j = 1; j < numBlock; j++)
    //    {
    //        tempAcc += bodyAcc2[i + numTarget * j];
    //    }
    //    bodyAcc2[i] = tempAcc;
    //}

    double diffp = 0, diffa = 0;
    double normp = 0, norma = 0;

    std::vector<double> delta(numBodies);

    for (int i = 0; i < numBodies; i++)
    {
        diffp += (bodyAcc[i][0] - bodyAcc2[i][0]) * (bodyAcc[i][0] - bodyAcc2[i][0]);
        normp += bodyAcc2[i][0] * bodyAcc2[i][0];

        double da = (bodyAcc[i][1] - bodyAcc2[i][1]) * (bodyAcc[i][1] - bodyAcc2[i][1]) +
                    (bodyAcc[i][2] - bodyAcc2[i][2]) * (bodyAcc[i][2] - bodyAcc2[i][2]) +
                    (bodyAcc[i][3] - bodyAcc2[i][3]) * (bodyAcc[i][3] - bodyAcc2[i][3]);
        double na = bodyAcc2[i][1] * bodyAcc2[i][1] + bodyAcc2[i][2] * bodyAcc2[i][2] + bodyAcc2[i][3] * bodyAcc2[i][3];

        diffa += da;
        norma += na;
        delta[i] = std::sqrt(da/na);
    }

    std::sort(begin(delta), end(delta));
    std::cout << "min Error: "       << delta[0] << std::endl;
    std::cout << "50th percentile: " << delta[numTarget/2] << std::endl;
    std::cout << "10th percentile: " << delta[numTarget*0.9] << std::endl;
    std::cout << "1st percentile: "  << delta[numTarget*0.99] << std::endl;
    std::cout << "max Error: "       << delta[numTarget-1] << std::endl;

    fprintf(stdout, "--- FMM vs. direct ---------------\n");
    fprintf(stdout, "Rel. L2 Error (pot)  : %.7e\n", sqrt(diffp / normp));
    fprintf(stdout, "Rel. L2 Error (acc)  : %.7e\n", sqrt(diffa / norma));
    fprintf(stdout, "--- Tree stats -------------------\n");
    fprintf(stdout, "Bodies               : %d\n", numBodies);
    fprintf(stdout, "Cells                : %d\n", numSources);
    fprintf(stdout, "Tree depth           : %d\n", numLevels);
    fprintf(stdout, "--- Traversal stats --------------\n");
    fprintf(stdout, "P2P mean list length : %d (max %d)\n", int(interactions[0]), int(interactions[1]));
    fprintf(stdout, "M2P mean list length : %d (max %d)\n", int(interactions[2]), int(interactions[3]));

    return 0;
}