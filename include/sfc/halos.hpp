#pragma once

#include <algorithm>
#include <vector>

#include "sfc/collisions_cpu.hpp"
#include "sfc/domaindecomp.hpp"

namespace sphexa
{

template <class I>
struct IncomingHaloRange
{
    I codeStart;
    I codeEnd;
    std::size_t count;
    int sourceRank;
};



template <class I, class T>
std::vector<IncomingHaloRange<I>> findIncomingHalos(const std::vector<I>&           tree,
                                                    const std::vector<std::size_t>& nodeCounts,
                                                    const std::vector<T>&           interactionRadii,
                                                    const Box<T>&                   box,
                                                    const SpaceCurveAssignment<I>&  assignment,
                                                    int                             rank)
{
    std::vector<BinaryNode<I>> internalTree = createInternalTree(tree);

    std::vector<IncomingHaloRange<I>> ret;

    // loop over all the nodes in the octree
    // (omp) parallel
    for (int nodeIdx = 0; nodeIdx < nNodes(tree); ++nodeIdx)
    {
        CollisionList collisions;
        T radius = interactionRadii[nodeIdx];

        int dx = detail::toNBitInt<I>(normalize(radius, box.xmin(), box.xmax()));
        int dy = detail::toNBitInt<I>(normalize(radius, box.ymin(), box.ymax()));
        int dz = detail::toNBitInt<I>(normalize(radius, box.zmin(), box.zmax()));

        // find out with which other nodes in the octree that the node at nodeIdx
        // enlarged by the halo radius collides with
        Box<int> haloBox = makeHaloBox(tree[nodeIdx], tree[nodeIdx +1], dx, dy, dz);
        findCollisions(internalTree.data(), tree.data(), collisions, haloBox);

        // Go through all colliding nodes to determine which of them fall into a part of the SFC
        // that is not assigned to the executing rank. These nodes will be marked as halos.
        for (int i = 0; i < collisions.size(); ++i)
        {
            int collidingNodeIdx = collisions[i];

            I collidingNodeStart = tree[collidingNodeIdx];
            I collidingNodeEnd   = tree[collidingNodeIdx+1];

            bool isHalo = false;
            for (int a = 0; a < assignment[rank].nRanges(); ++a)
            {
                I assignmentStart = assignment[rank].rangeStart(a);
                I assignmentEnd = assignment[rank].rangeEnd(a);

                if (collidingNodeStart < assignmentStart || collidingNodeEnd > assignmentEnd)
                {
                    // node with index collidingNodeIdx is a halo node
                    isHalo = true;
                }
            }
            if (isHalo)
            {
                ret.push_back({collidingNodeStart, collidingNodeEnd, nodeCounts[collidingNodeIdx], 0});
            }
        }
    }

    return ret;
}

} // namespace sphexa