/*! @file
 * @brief Detection and exchange of halo peer ranks
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#pragma once

#include <vector>
#include <mpi.h>

#include "cstone/domain/index_ranges.hpp"
#include "cstone/tree/definitions.h"
#include "cstone/util/gsl-lite.hpp"

namespace cstone
{
namespace detail
{
//! @brief Check if each flagged halo node could be associated with a peer rank
template<class KeyType>
int checkHalos(int myRank,
               gsl::span<const TreeIndexPair> focusAssignment,
               gsl::span<const int> haloFlags,
               gsl::span<const KeyType> ftree)
{
    TreeNodeIndex firstAssignedNode = focusAssignment[myRank].start();
    TreeNodeIndex lastAssignedNode  = focusAssignment[myRank].end();

    std::array<TreeNodeIndex, 2> checkRanges[2] = {{0, firstAssignedNode},
                                                   {lastAssignedNode, TreeNodeIndex(haloFlags.size())}};

    int ret = 0;
    for (int range = 0; range < 2; ++range)
    {
#pragma omp parallel for
        for (TreeNodeIndex i = checkRanges[range][0]; i < checkRanges[range][1]; ++i)
        {
            if (haloFlags[i])
            {
                bool peerFound = false;
                for (auto peerRange : focusAssignment)
                {
                    if (peerRange.start() <= i && i < peerRange.end()) { peerFound = true; }
                }
                if (!peerFound)
                {
                    std::cout << "Assignment rank " << myRank << " " << std::oct << ftree[firstAssignedNode] << " - "
                              << ftree[lastAssignedNode] << std::dec << std::endl;
                    std::cout << "Failed node " << i << " " << std::oct << ftree[i] << " - " << ftree[i + 1] << std::dec
                              << std::endl;
                    ret = 1;
                }
            }
        }
    }
    return ret;
}

void compactPeers(gsl::span<const int> flags, std::vector<int>& peers)
{
    peers.clear();
    for (int rank = 0; rank < flags.size(); ++rank)
    {
        if (flags[rank]) { peers.push_back(rank); };
    }
}
} // namespace detail

inline std::vector<int>
haloPeers(int myRank, gsl::span<const int> haloFlags, gsl::span<const TreeIndexPair> fAssignment)
{
    int numRanks = fAssignment.size();
    std::vector<int> peerFlags(numRanks, 0);
#pragma omp parallel for
    for (int rank = 0; rank < numRanks; ++rank)
    {
        if (rank == myRank) { continue; }

        TreeNodeIndex focStart = fAssignment[rank].start();
        TreeNodeIndex focEnd   = fAssignment[rank].end();
        if (focEnd < focStart) { focEnd = focStart; }

        peerFlags[rank] = bool(std::accumulate(haloFlags.begin() + focStart, haloFlags.begin() + focEnd, 0));
    }
    return peerFlags;
}

void exchangePeers(MPI_Win rmaWin,
                   int myRank,
                   gsl::span<const int> extPeerFlags,
                   gsl::span<int> intPeerFlags,
                   std::vector<int>& exteriorPeers,
                   std::vector<int>& interiorPeers)
{
    std::fill(intPeerFlags.begin(), intPeerFlags.end(), 0);
    //MPI_Alltoall(extPeerFlags.data(), 1, MPI_INT, intPeerFlags.data(), 1, MPI_INT, MPI_COMM_WORLD);

    MPI_Win_fence(0, rmaWin);
    for (int rank = 0; rank < extPeerFlags.size(); ++rank)
    {
        if (extPeerFlags[rank])
        {
            int one = 1;
            MPI_Put(&one, 1, MPI_INT, rank, myRank, 1, MPI_INT, rmaWin);
        }
    }
    MPI_Win_fence(0, rmaWin);

    detail::compactPeers(extPeerFlags, exteriorPeers);
    detail::compactPeers(intPeerFlags, interiorPeers);
}

} // namespace cstone
