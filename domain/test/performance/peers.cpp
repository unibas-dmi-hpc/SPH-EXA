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

/*! @file
 * @brief Test peer detection performance
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#include <chrono>
#include <iostream>

#include "coord_samples/random.hpp"
#include "cstone/traversal/peers.hpp"

using namespace cstone;

int main()
{
    using KeyType = uint64_t;
    Box<double> box{-1, 1};

    LocalIndex nParticles = 20000;
    int bucketSize        = 1;
    int numRanks          = 50;

    auto codes = makeRandomGaussianKeys<KeyType>(nParticles);

    auto [treeLeaves, counts] = computeOctree(codes.data(), codes.data() + nParticles, bucketSize);
    Octree<KeyType> octree;
    octree.update(treeLeaves.data(), nNodes(treeLeaves));

    auto assignment = makeSfcAssignment(numRanks, counts, treeLeaves.data());
    int probeRank   = numRanks / 2;

    auto tp0                  = std::chrono::high_resolution_clock::now();
    std::vector<int> peersDtt = findPeersMac(probeRank, assignment, octree, box, invThetaMinMac(0.5f));
    auto tp1                  = std::chrono::high_resolution_clock::now();

    double t2 = std::chrono::duration<double>(tp1 - tp0).count();
    std::cout << "find peers: " << t2 << " numPeers: " << peersDtt.size() << std::endl;
}
