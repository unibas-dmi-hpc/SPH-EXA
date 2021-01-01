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


/*! \brief compose a list of Morton code ranges that contain halos from rank's perspective
 *
 * @tparam I
 * @tparam T
 * @param tree
 * @param nodeCounts
 * @param interactionRadii
 * @param box
 * @param assignment
 * @param rank
 * @return
 *
 * This means that the returned Morton code ranges are not assigned to \a rank, but
 * contain halos of the code ranges assigned to \a rank.
 *
 * Unparallelized prototype implementation
 */
template <class I, class T>
std::vector<HaloRange<I>> findIncomingHalos(const std::vector<I>&           tree,
                                            const std::vector<T>&           interactionRadii,
                                            const Box<T>&                   box,
                                            const SpaceCurveAssignment<I>&  assignment,
                                            int                             rank)
{
    SfcLookupKey<I> sfcLookup(assignment);

    std::vector<BinaryNode<I>> internalTree = createInternalTree(tree);
    std::vector<HaloRange<I>> ret;

    // go through all ranges assigned to rank
    for (int range = 0; range < assignment.nRanges(rank); ++range)
    {
        int nodeStart = std::lower_bound(begin(tree), end(tree), assignment.rangeStart(rank, range)) - begin(tree);
        int nodeEnd   = std::lower_bound(begin(tree), end(tree), assignment.rangeEnd(rank, range)) - begin(tree);

        // loop over all the nodes in range
        for (int nodeIdx = nodeStart; nodeIdx < nodeEnd; ++nodeIdx)
        {
            CollisionList collisions;
            T radius = interactionRadii[nodeIdx];

            int dx = detail::toNBitInt<I>(normalize(radius, box.xmin(), box.xmax()));
            int dy = detail::toNBitInt<I>(normalize(radius, box.ymin(), box.ymax()));
            int dz = detail::toNBitInt<I>(normalize(radius, box.zmin(), box.zmax()));

            // find out with which other nodes in the octree that the node at nodeIdx
            // enlarged by the halo radius collides with
            Box<int> haloBox = makeHaloBox(tree[nodeIdx], tree[nodeIdx + 1], dx, dy, dz);
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
                    ret.push_back({collidingNodeStart, collidingNodeEnd, sfcLookup.findRank(collidingNodeStart)});
                }
            }
        }
    }

    // eliminate duplicates
    std::sort(begin(ret), end(ret));
    auto uit = std::unique(begin(ret), end(ret));
    ret.erase(uit, end(ret));

    return ret;
}

} // namespace sphexa