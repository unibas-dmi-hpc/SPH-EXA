#pragma once

#include <algorithm>
#include <vector>

#include "sfc/collisions_cpu.hpp"
#include "sfc/domaindecomp.hpp"

namespace sphexa
{

template <class I>
struct HaloRange
{
    I codeStart;
    I codeEnd;
    int sourceRank;

    friend bool operator<(const HaloRange& a, const HaloRange& b)
    {
        return a.codeStart < b.codeStart;
    }

    friend bool operator==(const HaloRange& a, const HaloRange& b)
    {
        return a.codeStart == b.codeStart && a.codeEnd == b.codeEnd && a.sourceRank == b.sourceRank;
    }
};


/*! \brief Compute halo node pairs
 *
 * @tparam I                   32- or 64-bit unsigned integer
 * @tparam T                   float or double
 * @param tree                 cornerstone octree
 * @param interactionRadii     halo search radii per octree (leaf) node
 * @param box                  coordinate bounding box
 * @param assignment           list if Morton code ranges assignments per rank
 * @param rank                 compute pairs from perspective of \a rank
 * @param[out] haloPairs       output list of halo node index pairs
 * @return
 *
 * A pair of indices (i,j) in [0...nNodes(tree)], is a halo pair for rank r if
 *   - tree[i] is assigned to rank r
 *   - tree[j] is not assigned to rank r
 *   - tree[i] enlarged by the search radius interactionRadii[i] overlaps with tree[j]
 *   - tree[j] enlarged by the search radius interactionRadii[j] overlaps with tree[i]
 *
 * This means that the first element in each index pair in \a haloPairs is the index of a
 * node (in \a tree) that belongs to rank \a rank and must be sent out to another rank.
 *
 * The second element of each pair is the index of a remote node that is a halo for rank \a rank.
 * We can easily find the source rank of the halo with binary search in the space curve assignment.
 * The source rank of the halo is also the destination where the internal node referenced in the first
 * pair element must be sent to.
 */
template <class I, class T>
void findHalos(const std::vector<I>&           tree,
               const std::vector<T>&           interactionRadii,
               const Box<T>&                   box,
               const SpaceCurveAssignment<I>&  assignment,
               int                             rank,
               std::vector<pair<int>>&         haloPairs)
{
    std::vector<BinaryNode<I>> internalTree = createInternalTree(tree);

    // go through all ranges assigned to rank
    for (int range = 0; range < assignment.nRanges(rank); ++range)
    {
        int firstNode = std::lower_bound(begin(tree), end(tree), assignment.rangeStart(rank, range)) - begin(tree);
        int lastNode  = std::lower_bound(begin(tree), end(tree), assignment.rangeEnd(rank, range)) - begin(tree);

        // loop over all the nodes in range
        for (int nodeIdx = firstNode; nodeIdx < lastNode; ++nodeIdx)
        {
            CollisionList collisions;
            T radius = interactionRadii[nodeIdx];

            // find out with which other nodes in the octree that the node at nodeIdx
            // enlarged by the halo radius collides with
            Box<int> haloBox = makeHaloBox(tree[nodeIdx], tree[nodeIdx + 1], radius, box);
            findCollisions(internalTree.data(), tree.data(), collisions, haloBox);

            // Go through all colliding nodes to determine which of them fall into a part of the SFC
            // that is not assigned to the executing rank. These nodes will be marked as halos.
            for (int i = 0; i < collisions.size(); ++i)
            {
                int collidingNodeIdx = collisions[i];

                I collidingNodeStart = tree[collidingNodeIdx];
                I collidingNodeEnd   = tree[collidingNodeIdx + 1];

                bool isHalo = false;
                for (int a = 0; a < assignment.nRanges(rank); ++a)
                {
                    I assignmentStart = assignment.rangeStart(rank, a);
                    I assignmentEnd   = assignment.rangeEnd(rank, a);

                    if (collidingNodeStart < assignmentStart || collidingNodeEnd > assignmentEnd)
                    {
                        // node with index collidingNodeIdx is a halo node
                        isHalo = true;
                    }
                }
                if (isHalo)
                {
                    // check if remote node +halo also overlaps with internal node
                    Box<int> remoteNodeBox = makeHaloBox(collidingNodeStart, collidingNodeEnd,
                                                         interactionRadii[collidingNodeIdx], box);
                    if (overlap(tree[nodeIdx], tree[nodeIdx+1], remoteNodeBox))
                    {
                        haloPairs.emplace_back(nodeIdx, collidingNodeIdx);
                    }
                }
            }
        }
    }
}

} // namespace sphexa