#include <chrono>

#include "ryoanji/types.h"
#include "ryoanji/buildtree_cs.hpp"
#include "ryoanji/traversal_cpu.hpp"
#include "ryoanji/dataset.h"
#include "ryoanji/traversal.h"
#include "ryoanji/direct.cuh"
#include "ryoanji/upwardpass.h"

int main(int argc, char** argv)
{
    int numBodies = (1 << 16) - 1;
    int images    = 0;
    float theta   = 0.5;
    float boxSize = 3;

    const float eps   = 0.05;
    const int ncrit   = 64;
    const float cycle = 2 * M_PI;

    fprintf(stdout, "--- FMM Parameters ---------------\n");
    fprintf(stdout, "numBodies            : %d\n", numBodies);
    fprintf(stdout, "P                    : %d\n", P);
    fprintf(stdout, "theta                : %f\n", theta);
    fprintf(stdout, "ncrit                : %d\n", ncrit);

    auto bodies = makeCubeBodies(numBodies, boxSize);

    Box box{ {0.0f}, boxSize * 1.0f };

    auto [highestLevel, tree, levelRangeCs] = buildFromCstone(bodies, box);

    int numSources = tree.size();

    cudaVec<fvec4> bodyPos(numBodies, true);
    std::copy(bodies.begin(), bodies.end(), bodyPos.h());
    bodyPos.h2d();

    cudaVec<int2> levelRange(levelRangeCs.size(), true);
    std::copy(levelRangeCs.begin(), levelRangeCs.end(), levelRange.h());
    levelRange.h2d();

    cudaVec<CellData> sources(numSources, true);
    std::copy(tree.begin(), tree.end(), sources.h());
    sources.h2d();

    cudaVec<fvec4> sourceCenter(numSources, true);
    cudaVec<fvec4> Multipole(NVEC4 * numSources, true);

    int numLeaves = -1;
    Pass::upward(numLeaves, highestLevel, theta, levelRange, bodyPos, sources, sourceCenter, Multipole);
    sourceCenter.d2h();
    Multipole.d2h();

    cudaVec<fvec4> bodyAcc(numBodies, true);

    fprintf(stdout, "--- FMM Profiling ----------------\n");

    auto t0 = std::chrono::high_resolution_clock::now();

    // the groupSize cannot be changed as it is hard-coded in the traversal function
    constexpr int groupSize = 2 * WARP_SIZE;
    int numTargets = (numBodies - 1) / groupSize + 1;
    cudaVec<int2> targetRange(numBodies, true);

    for (int target = 0; target < numTargets; ++target)
    {
        targetRange[target] = {target * groupSize, groupSize};
    }
    // if numBodies is not a multiple of the groupSize, the last target only gets the remainder
    if (numBodies % groupSize != 0)
    {
        targetRange[numTargets - 1].y = numBodies % groupSize;
    }
    targetRange.h2d();

    fvec4 interactions = Traversal::approx(numTargets,
                                           images,
                                           eps,
                                           cycle,
                                           bodyPos,
                                           bodyPos,
                                           bodyAcc,
                                           targetRange,
                                           sources,
                                           sourceCenter,
                                           Multipole,
                                           levelRange);

    auto t1      = std::chrono::high_resolution_clock::now();
    double dt    = std::chrono::duration<double>(t1 - t0).count();
    double flops = (interactions[0] * 20 + interactions[2] * 2 * pow(P, 3)) * numBodies / dt / 1e12;

    fprintf(stdout, "--- Total runtime ----------------\n");
    fprintf(stdout, "Total FMM            : %.7f s (%.7f TFlops)\n", dt, flops);

    cudaVec<fvec4> bodyAccDirect(numBodies, true);

    t0 = std::chrono::high_resolution_clock::now();
    directSum(eps, bodyPos, bodyAccDirect);
    t1 = std::chrono::high_resolution_clock::now();
    dt = std::chrono::duration<double>(t1 - t0).count();

    flops = 24. * numBodies * numBodies / dt / 1e12;
    fprintf(stdout, "Total Direct         : %.7f s (%.7f TFlops)\n", dt, flops);

    bodyAcc.d2h();
    bodyAccDirect.d2h();

    std::vector<double> delta(numBodies);

    for (int i = 0; i < numBodies; i++)
    {
        fvec3 ref   = {bodyAccDirect[i][1], bodyAccDirect[i][2], bodyAccDirect[i][3]};
        fvec3 probe = {bodyAcc[i][1], bodyAcc[i][2], bodyAcc[i][3]};
        delta[i]    = std::sqrt(norm(ref - probe) / norm(ref));
    }

    //int mei = std::max_element(delta.begin(), delta.end()) - delta.begin();
    //fvec4 test = walkParticle(mei, eps, sources, sourceCenter, Multipole, bodyPos);
    //std::cout << bodyAcc[mei][1] << " " << test[1] << std::endl;

    std::sort(begin(delta), end(delta));

    fprintf(stdout, "--- FMM vs. direct ---------------\n");

    std::cout << "min Error: "       << delta[0] << std::endl;
    std::cout << "50th percentile: " << delta[numBodies/2] << std::endl;
    std::cout << "10th percentile: " << delta[numBodies*0.9] << std::endl;
    std::cout << "1st percentile: "  << delta[numBodies*0.99] << std::endl;
    std::cout << "max Error: "       << delta[numBodies-1] << std::endl;

    //fprintf(stdout, "--- FMM vs. direct ---------------\n");
    //fprintf(stdout, "Rel. L2 Error (pot)  : %.7e\n", sqrt(diffp / normp));
    //fprintf(stdout, "Rel. L2 Error (acc)  : %.7e\n", sqrt(diffa / norma));

    fprintf(stdout, "--- Tree stats -------------------\n");
    fprintf(stdout, "Bodies               : %d\n", numBodies);
    fprintf(stdout, "Cells                : %d\n", numSources);
    fprintf(stdout, "Tree depth           : %d\n", 0);
    fprintf(stdout, "--- Traversal stats --------------\n");
    fprintf(stdout, "P2P mean list length : %d (max %d)\n", int(interactions[0]), int(interactions[1]));
    fprintf(stdout, "M2P mean list length : %d (max %d)\n", int(interactions[2]), int(interactions[3]));

    return 0;
}