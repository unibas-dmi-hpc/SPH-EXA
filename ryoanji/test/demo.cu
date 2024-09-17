/*
 * MIT License
 *
 * Copyright (c) 2021 CSCS, ETH Zurich
 *               2021 University of Basel
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#include <chrono>
#include <numeric>

#ifdef __HIPCC__
#include <hip/hip_runtime.h>
#endif

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include "cstone/cuda/cuda_utils.cuh"
#include "cstone/cuda/gpu_config.cuh"
#include "cstone/cuda/thrust_util.cuh"

#include "nbody/dataset.hpp"
#include "ryoanji/interface/treebuilder.cuh"
#include "ryoanji/nbody/types.h"
#include "ryoanji/nbody/traversal.cuh"
#include "ryoanji/nbody/direct.cuh"
#include "ryoanji/nbody/upwardpass.cuh"

using namespace ryoanji;

int main(int argc, char** argv)
{
    constexpr int P     = 4;
    using T             = float;
    using MultipoleType = SphericalMultipole<T, P>;

    int power     = argc > 1 ? std::stoi(argv[1]) : 17;
    int directRef = argc > 2 ? std::stoi(argv[2]) : 1;
    int numShells = argc > 3 ? std::stoi(argv[3]) : 0;

    std::size_t numBodies = (1 << power) - 1;
    T           theta     = 0.6;
    T           boxSize   = 3;
    T           G         = 1.0;

    const int ncrit = 64;

    fprintf(stdout, "--- BH Parameters ---------------\n");
    fprintf(stdout, "numBodies            : %lu\n", numBodies);
    fprintf(stdout, "P                    : %d\n", P);
    fprintf(stdout, "theta                : %f\n", theta);
    fprintf(stdout, "ncrit                : %d\n", ncrit);

    thrust::host_vector<T> x(numBodies), y(numBodies), z(numBodies), m(numBodies), h(numBodies);
    makeCubeBodies(x.data(), y.data(), z.data(), m.data(), h.data(), numBodies, boxSize);

    // upload bodies to device
    thrust::device_vector<T> d_x = x, d_y = y, d_z = z, d_m = m, d_h = h;

    cstone::Box<T> box(-boxSize, boxSize);

    TreeBuilder<uint64_t> treeBuilder;
    int                   numSources = treeBuilder.update(rawPtr(d_x), rawPtr(d_y), rawPtr(d_z), numBodies, box);

    std::vector<int2> levelRange(treeBuilder.maxTreeLevel() + 1);
    int               highestLevel = treeBuilder.extract(levelRange.data());

    thrust::device_vector<Vec4<T>>       sourceCenter(numSources);
    thrust::device_vector<MultipoleType> Multipole(numSources);

    upsweep(numSources, treeBuilder.numLeafNodes(), highestLevel, theta, levelRange.data(), rawPtr(d_x), rawPtr(d_y),
            rawPtr(d_z), rawPtr(d_m), rawPtr(d_h), treeBuilder.layout(), treeBuilder.childOffsets(),
            treeBuilder.leafToInternal(), rawPtr(sourceCenter), rawPtr(Multipole));

    thrust::device_vector<T> d_p(numBodies, 0), d_ax(numBodies, 0), d_ay(numBodies, 0), d_az(numBodies, 0);

    fprintf(stdout, "--- BH Profiling ----------------\n");

    auto t0 = std::chrono::high_resolution_clock::now();

    auto interactions = computeAcceleration(0, numBodies, rawPtr(d_x), rawPtr(d_y), rawPtr(d_z), rawPtr(d_m),
                                            rawPtr(d_h), G, numShells, box, rawPtr(d_p), rawPtr(d_ax), rawPtr(d_ay),
                                            rawPtr(d_az), treeBuilder.childOffsets(), treeBuilder.internalToLeaf(),
                                            treeBuilder.layout(), rawPtr(sourceCenter), rawPtr(Multipole));

    auto   t1    = std::chrono::high_resolution_clock::now();
    double dt    = std::chrono::duration<double>(t1 - t0).count();
    double flops = (interactions[0] * 20 + interactions[2] * 2 * pow(P, 3)) * numBodies / dt / 1e12;

    fprintf(stdout, "--- Total runtime ----------------\n");
    fprintf(stdout, "Total BH            : %.7f s (%.7f TFlops)\n", dt, flops);

    if (!directRef) { return 0; }

    thrust::device_vector<T> refP(numBodies), refAx(numBodies), refAy(numBodies), refAz(numBodies);

    t0 = std::chrono::high_resolution_clock::now();
    directSum(0, numBodies, numBodies, Vec3<T>{box.lx(), box.ly(), box.lz()}, numShells, rawPtr(d_x), rawPtr(d_y),
              rawPtr(d_z), rawPtr(d_m), rawPtr(d_h), rawPtr(refP), rawPtr(refAx), rawPtr(refAy), rawPtr(refAz));

    t1 = std::chrono::high_resolution_clock::now();
    dt = std::chrono::duration<double>(t1 - t0).count();

    flops = std::pow((2 * numShells + 1), 3) * 24. * numBodies * numBodies / dt / 1e12;
    fprintf(stdout, "Total Direct         : %.7f s (%.7f TFlops)\n", dt, flops);

    thrust::host_vector<T> h_p  = d_p;
    thrust::host_vector<T> h_ax = d_ax;
    thrust::host_vector<T> h_ay = d_ay;
    thrust::host_vector<T> h_az = d_az;

    double                 referencePotential = 0.5 * G * thrust::reduce(refP.begin(), refP.end(), 0.0);
    thrust::host_vector<T> h_refAx            = refAx;
    thrust::host_vector<T> h_refAy            = refAy;
    thrust::host_vector<T> h_refAz            = refAz;

    std::vector<double> delta(numBodies);

    double potentialSum = 0;
    for (int i = 0; i < numBodies; i++)
    {
        potentialSum += h_p[i];
        Vec3<T> ref   = {h_refAx[i], h_refAy[i], h_refAz[i]};
        Vec3<T> probe = {h_ax[i], h_ay[i], h_az[i]};
        delta[i]      = std::sqrt(norm2(ref - probe) / norm2(ref));
    }

    std::sort(begin(delta), end(delta));

    fprintf(stdout, "--- BH vs. direct ---------------\n");

    std::cout << "potentials, body-sum: " << 0.5 * G * potentialSum << " atomic sum: " << 0.5 * G * interactions[4]
              << " reference: " << referencePotential << std::endl;
    std::cout << "min Error: " << delta[0] << std::endl;
    std::cout << "50th percentile: " << delta[numBodies / 2] << std::endl;
    std::cout << "10th percentile: " << delta[numBodies * 0.9] << std::endl;
    std::cout << "1st percentile: " << delta[numBodies * 0.99] << std::endl;
    std::cout << "max Error: " << delta[numBodies - 1] << std::endl;

    fprintf(stdout, "--- Tree stats -------------------\n");
    fprintf(stdout, "Bodies               : %lu\n", numBodies);
    fprintf(stdout, "Cells                : %d\n", numSources);
    fprintf(stdout, "Tree depth           : %d\n", highestLevel);
    fprintf(stdout, "--- Traversal stats --------------\n");
    fprintf(stdout, "P2P mean list length : %d (max %d)\n", int(interactions[0]), int(interactions[1]));
    fprintf(stdout, "M2P mean list length : %d (max %d)\n", int(interactions[2]), int(interactions[3]));

    return 0;
}
