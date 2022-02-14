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
 * @brief GTest MPI driver
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#include <mpi.h>

#include "cstone/domain/domain.hpp"
#include "cstone/findneighbors.hpp"
#include "coord_samples/random.hpp"

#include "ryoanji/types.h"
#include "ryoanji/cpu/multipole.hpp"
#include "ryoanji/cpu/upsweep.hpp"

using namespace ryoanji;

//! @brief can be used to calculate reasonable smoothing lengths for each particle
template<class KeyType, class Tc, class Th>
void adjustSmoothingLength(LocalIndex numParticles,
                           int ng0,
                           int ngmax,
                           const std::vector<Tc>& xGlob,
                           const std::vector<Tc>& yGlob,
                           const std::vector<Tc>& zGlob,
                           std::vector<Th>& hGlob,
                           const cstone::Box<Tc>& box)
{
    std::vector<KeyType> codesGlobal(numParticles);

    std::vector<Tc> x = xGlob;
    std::vector<Tc> y = yGlob;
    std::vector<Tc> z = zGlob;
    std::vector<Tc> h = hGlob;

    cstone::computeSfcKeys(x.data(), y.data(), z.data(), cstone::sfcKindPointer(codesGlobal.data()), numParticles, box);
    std::vector<LocalIndex> ordering(numParticles);
    std::iota(begin(ordering), end(ordering), LocalIndex(0));
    cstone::sort_by_key(begin(codesGlobal), end(codesGlobal), begin(ordering));
    cstone::reorderInPlace(ordering, x.data());
    cstone::reorderInPlace(ordering, y.data());
    cstone::reorderInPlace(ordering, z.data());
    cstone::reorderInPlace(ordering, h.data());

    std::vector<LocalIndex> inverseOrdering(numParticles);
    std::iota(begin(inverseOrdering), end(inverseOrdering), 0);
    std::vector orderCpy = ordering;
    cstone::sort_by_key(begin(orderCpy), end(orderCpy), begin(inverseOrdering));

    std::vector<int> neighbors(numParticles * ngmax);
    std::vector<int> neighborCounts(numParticles);

    // adjust h[i] such that each particle has between ng0/2 and ngmax neighbors
    for (LocalIndex i = 0; i < numParticles; ++i)
    {
        do
        {
            cstone::findNeighbors(i,
                                  x.data(),
                                  y.data(),
                                  z.data(),
                                  h.data(),
                                  box,
                                  cstone::sfcKindPointer(codesGlobal.data()),
                                  neighbors.data() + i * ngmax,
                                  neighborCounts.data() + i,
                                  numParticles,
                                  ngmax);

            const Tc c0 = 7.0;
            int nn      = std::max(neighborCounts[i], 1);
            h[i]        = h[i] * 0.5 * pow(1.0 + (c0 * ng0) / nn, 1.0 / 3.0);
        } while (neighborCounts[i] < ng0/2 || neighborCounts[i] >= ngmax);
    }

    for (LocalIndex i = 0; i < numParticles; ++i)
    {
        hGlob[i] = h[inverseOrdering[i]];
    }
}

template<class T, class KeyType>
static void globalMultipoleExchange(int thisRank, int numRanks)
{
    using MultipoleType = CartesianQuadrupole<T>;
    const LocalIndex numParticles    = 1000;
    unsigned         bucketSize      = 64;
    unsigned         bucketSizeLocal = 16;
    float            theta           = 10.0;

    cstone::Box<T> box{-1, 1};

    // common pool of coordinates, identical on all ranks
    RandomGaussianCoordinates<T, cstone::SfcKind<KeyType>> coords(numRanks * numParticles, box);

    std::vector<T> globalH(numRanks * numParticles, 0.1);
    adjustSmoothingLength<KeyType>(globalH.size(), 100, 150, coords.x(), coords.y(), coords.z(), globalH, box);

    std::vector<T> globalMasses(numRanks * numParticles, 1.0 / (numRanks * numParticles));

    auto firstIndex = numParticles * thisRank;
    auto lastIndex  = numParticles * thisRank + numParticles;

    // extract a slice of the common pool, each rank takes a different slice, but all slices together
    // are equal to the common pool
    std::vector<T> x(coords.x().begin() + firstIndex, coords.x().begin() + lastIndex);
    std::vector<T> y(coords.y().begin() + firstIndex, coords.y().begin() + lastIndex);
    std::vector<T> z(coords.z().begin() + firstIndex, coords.z().begin() + lastIndex);
    std::vector<T> h(globalH.begin() + firstIndex, globalH.begin() + lastIndex);
    std::vector<T> m(globalMasses.begin() + firstIndex, globalMasses.begin() + lastIndex);

    for (int rank = 0; rank < numRanks; ++rank)
    {
        if (thisRank == rank)
        {
            std::cout << "localMass rank " << rank << " " << std::accumulate(m.begin(), m.end(), 0.0) << std::endl;
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }

    std::vector<KeyType> particleKeys(x.size());

    cstone::Domain<KeyType, T> domain(thisRank, numRanks, bucketSize, bucketSizeLocal, theta, box);

    domain.syncGrav(particleKeys, x, y, z, h, m);

    for (int rank = 0; rank < numRanks; ++rank)
    {
        if (thisRank == rank)
        {
            std::cout << "localMass rank " << rank << " " << std::accumulate(m.begin() + domain.startIndex(), m.begin() + domain.endIndex(), 0.0) << " ";
            std::cout << "start/end " << domain.startIndex() << " " << domain.endIndex() << " " << domain.nParticlesWithHalos() << " ";

            gsl::span<const cstone::SourceCenterType<T>> centers = domain.expansionCenters();
            auto rc = centers[0];
            std::cout << rc[0] << " " << rc[1] << " " << rc[2] << std::endl;
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }

    const cstone::Octree<KeyType>& focusTree = domain.focusTree();
    gsl::span<const cstone::SourceCenterType<T>> centers = domain.expansionCenters();

    MultipoleType zero;
    zero = 0;
    std::vector<MultipoleType> multipoles(focusTree.numTreeNodes(), zero);
    ryoanji::computeLeafMultipoles(
        focusTree, domain.layout(), x.data(), y.data(), z.data(), m.data(), centers.data(), multipoles.data());

    ryoanji::CombineMultipole<MultipoleType> combineMultipole(centers.data());
    upsweep(focusTree, multipoles.data(), combineMultipole);

    if (thisRank == 0)
        std::cout << "root mass before global exchange " << multipoles[0][0] << std::endl;

    domain.template exchangeFocusGlobal<MultipoleType>(multipoles, combineMultipole);
    upsweep(focusTree, multipoles.data(), combineMultipole);

    if (thisRank == 0)
        std::cout << "root mass after global exchange " << multipoles[0][0] << std::endl;

    MultipoleType globalRootMultipole = multipoles[focusTree.levelOffset(0)];

    // compute reference root cell multipole from global particle data
    MultipoleType reference;
    particle2Multipole(coords.x().data(),
                       coords.y().data(),
                       coords.z().data(),
                       globalMasses.data(),
                       0,
                       numParticles * numRanks,
                       makeVec3(centers[focusTree.levelOffset(0)]),
                       reference);

    MultipoleType diff = reference - globalRootMultipole;

    bool pass = true;
    for (size_t i = 0; i < diff.size(); ++i)
    {
        if (reference[i] != 0)
        {
            if (std::abs(reference[i] - globalRootMultipole[i]) > 1e-6)
            {
                pass = false;
                break;
            }
        }
        else
        {
            if (globalRootMultipole[i] != 0)
            {
                pass = false;
                break;
            }
        }
    }

    std::string testResult = pass ? "PASS" : "FAIL";
    std::cout << "Root multipole moment comparison: " << testResult << std::endl;
}

int main(int argc, char** argv)
{
    MPI_Init(NULL, NULL);

    int rank = 0, numRanks = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numRanks);

    globalMultipoleExchange<double, unsigned>(rank, numRanks);
}
