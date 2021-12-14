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
 * @brief Generation of locally essential global octrees in cornerstone format
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 *
 * A locally essential octree has a certain global resolution specified by a maximum
 * particle count per leaf node. In addition, it features a focus area defined as a
 * sub-range of the global space filling curve. In this focus sub-range, the resolution
 * can be higher, expressed through a smaller maximum particle count per leaf node.
 * Crucially, the resolution is also higher in the halo-areas of the focus sub-range.
 * These halo-areas can be defined as the overlap with the smoothing-length spheres around
 * the contained particles in the focus sub-range (SPH) or as the nodes whose opening angle
 * is too big to satisfy a multipole acceptance criterion from any perspective within the
 * focus sub-range (N-body).
 */

#pragma once

#include <vector>

#include "cstone/domain/domaindecomp.hpp"
#include "cstone/traversal/boxoverlap.hpp"
#include "cstone/util/gsl-lite.hpp"
#include "cstone/util/index_ranges.hpp"

#include "cstone/traversal/macs.hpp"
#include "cstone/tree/octree_internal.hpp"
#include "cstone/traversal/traversal.hpp"

namespace cstone
{

/*! @brief combines the particle count and multipole criteria for rebalancing
 *
 * @return
 *      - 0 if node to be merged
 *      - 1 if node to stay unchanged
 *      - 8 if node to be split
 */
template<class KeyType>
inline HOST_DEVICE_FUN
int mergeCountAndMacOp(TreeNodeIndex leafIdx, const KeyType* cstoneTree,
                       TreeNodeIndex numInternalNodes,
                       const TreeNodeIndex* leafParents,
                       const unsigned* leafCounts, const char* macs,
                       TreeNodeIndex firstFocusNode, TreeNodeIndex lastFocusNode,
                       unsigned bucketSize)
{
    auto [siblingIdx, level] = siblingAndLevel(cstoneTree, leafIdx);

    if (siblingIdx > 0) // 8 siblings next to each other, node can potentially be merged
    {
        // pointer to first node in sibling group
        auto g = leafCounts + leafIdx - siblingIdx;

        bool countMerge = (g[0]+g[1]+g[2]+g[3]+g[4]+g[5]+g[6]+g[7]) <= bucketSize;
        bool macMerge   = macs[leafParents[leafIdx]] == 0;

        TreeNodeIndex firstSibling = leafIdx - siblingIdx;
        TreeNodeIndex lastSibling  = firstSibling + 8;

        // inFringe: leadIdx not in focus, but at least one sibling is in the focus
        // in that case we cannot remove the nodes based on a MAC criterion
        bool inFringe = overlapTwoRanges(firstSibling, lastSibling, firstFocusNode, lastFocusNode);

        if (countMerge || (macMerge && !inFringe)) { return 0; } // merge
    }

    bool inFocus  = (leafIdx >= firstFocusNode && leafIdx < lastFocusNode);
    if (level < maxTreeLevel<KeyType>{} && leafCounts[leafIdx] > bucketSize
        && (macs[numInternalNodes + leafIdx] || inFocus))
    { return 8; } // split

    return 1; // default: do nothing
}

/*! @brief Compute locally essential split or fuse decision for each octree node in parallel
 *
 * @tparam    KeyType          32- or 64-bit unsigned integer type
 * @param[in] cstoneTree       cornerstone octree leaves, length = @p numLeafNodes
 * @param[in] numInternalNodes number of internal octree nodes
 * @param[in] numLeafNodes     number of leaf octree nodes
 * @param[in] leafParents      stores the parent node index of each leaf, length = @p numLeafNodes
 * @param[in] leafCounts       particle counts per leaf node, length = @p numLeafNodes
 * @param[in] macs             multipole pass or fail per node, length = @p numInternalNodes + numLeafNodes
 * @param[in] firstFocusNode   first focus node in @p cstoneTree, range = [0:numLeafNodes]
 * @param[in] lastFocusNode    last focus node in @p cstoneTree, range = [0:numLeafNodes]
 * @param[in] bucketSize       maximum particle count per (leaf) node and
 *                             minimum particle count (strictly >) for (implicit) internal nodes
 * @param[out] nodeOps         stores rebalance decision result for each node, length = @p numLeafNodes()
 * @return                     true if converged, false otherwise
 *
 * For each node i in the tree, in nodeOps[i], stores
 *  - 0 if to be merged
 *  - 1 if unchanged,
 *  - 8 if to be split.
 */
template<class KeyType, class LocalIndex>
bool rebalanceDecisionEssential(const KeyType* cstoneTree, TreeNodeIndex numInternalNodes, TreeNodeIndex numLeafNodes,
                                const TreeNodeIndex* leafParents,
                                const unsigned* leafCounts, const char* macs,
                                TreeNodeIndex firstFocusNode, TreeNodeIndex lastFocusNode,
                                unsigned bucketSize, LocalIndex* nodeOps)
{
    bool converged = true;
    #pragma omp parallel
    {
        bool convergedThread = true;
        #pragma omp for
        for (TreeNodeIndex leafIdx = 0; leafIdx < numLeafNodes; ++leafIdx)
        {
            int opDecision = mergeCountAndMacOp(leafIdx, cstoneTree, numInternalNodes, leafParents, leafCounts,
                                                macs, firstFocusNode, lastFocusNode, bucketSize);
            if (opDecision != 1) { convergedThread = false; }

            nodeOps[leafIdx] = opDecision;
        }
        if (!convergedThread) { converged = false; }
    }
    return converged;
}


template<class KeyType>
class FocusedOctreeCore
{
public:
    FocusedOctreeCore(unsigned bucketSize)
        : bucketSize_(bucketSize)
    {
        tree_.update(std::vector<KeyType>{0, nodeRange<KeyType>(0)});
    }

    /*! @brief perform a local update step
     *
     * @param[in] particleKeys    locally present particle SFC keys
     * @param[in] focusStart      start of the focus area
     * @param[in] focusEnd        end of the focus area
     * @param[in] mandatoryKeys   List of SFC keys that have to be present in the focus tree after this function
     *                            returns. @p focusStart and @p focusEnd are always mandatory, so they don't need to be
     *                            specified here. @p mandatoryKeys need not be sorted and can tolerate duplicates.
     *                            This is used e.g. to guarantee that the assignment boundaries of peer ranks are
     *                            resolved, even if the update did not converge.
     * @param[in] counts          leaf particle counts, length = tree_.numLeafNodes()
     * @param[in] macs            MAC pass/fail results for each node, length = tree_.numTreeNodes()
     * @return                    true if the tree structure did not change
     */
    bool update(gsl::span<const KeyType> particleKeys,
                KeyType focusStart,
                KeyType focusEnd,
                gsl::span<const KeyType> mandatoryKeys,
                gsl::span<const unsigned> counts,
                gsl::span<const char> macs)
    {
        assert(TreeNodeIndex(counts.size()) == tree_.numLeafNodes());
        assert(TreeNodeIndex(macs.size()) == tree_.numTreeNodes());
        assert(std::is_sorted(particleKeys.begin(), particleKeys.end()));

        gsl::span<const KeyType> leaves = tree_.treeLeaves();

        TreeNodeIndex firstFocusNode = findNodeBelow(leaves, focusStart);
        TreeNodeIndex lastFocusNode  = findNodeAbove(leaves, focusEnd);

        std::vector<TreeNodeIndex> nodeOps(tree_.numLeafNodes() + 1);
        bool converged = rebalanceDecisionEssential(leaves.data(), tree_.numInternalNodes(), tree_.numLeafNodes(),
                                                    tree_.leafParents(), counts.data(), macs.data(), firstFocusNode,
                                                    lastFocusNode, bucketSize_, nodeOps.data());

        std::vector<KeyType> allMandatoryKeys{focusStart, focusEnd};
        std::copy(mandatoryKeys.begin(), mandatoryKeys.end(), std::back_inserter(allMandatoryKeys));

        auto status = enforceKeys<KeyType>(leaves, allMandatoryKeys, nodeOps);

        if (status == ResolutionStatus::cancelMerge)
        {
            converged = std::all_of(begin(nodeOps), end(nodeOps) - 1, [](TreeNodeIndex i) { return i == 1; });
        }
        else if (status == ResolutionStatus::rebalance)
        {
            converged = false;
        }

        std::vector<KeyType> newLeaves;
        rebalanceTree(leaves, newLeaves, nodeOps.data());

        // if rebalancing couldn't introduce the mandatory keys, we force-inject them now into the tree
        if (status == ResolutionStatus::failed)
        {
            converged = false;
            injectKeys(newLeaves, allMandatoryKeys);
        }

        tree_.update(std::move(newLeaves));

        return converged;
    }

    //! @brief provide access to the linked octree
    const Octree<KeyType>& octree() const { return tree_; }

    //! @brief returns a view of the tree leaves
    gsl::span<const KeyType> treeLeaves() const { return tree_.treeLeaves(); }

private:

    //! @brief max number of particles per node in focus
    unsigned bucketSize_;

    //! @brief the focused tree
    Octree<KeyType> tree_;
};

/*! @brief A fully traversable octree, locally focused w.r.t a MinMac criterion
 *
 * This single rank version is only useful in unit tests.
 */
template<class KeyType>
class FocusedOctreeSingleNode
{
public:
    FocusedOctreeSingleNode(unsigned bucketSize, float theta)
        : theta_(theta)
        , tree_(bucketSize)
        , counts_{bucketSize + 1}
        , macs_{1}
    {
    }

    //! @brief perform a local update step, see FocusedOctreeCore
    template<class T>
    bool update(const Box<T>& box,
                gsl::span<const KeyType> particleKeys,
                KeyType focusStart,
                KeyType focusEnd,
                gsl::span<const KeyType> mandatoryKeys)
    {
        bool converged = tree_.update(particleKeys, focusStart, focusEnd, mandatoryKeys, counts_, macs_);

        gsl::span<const KeyType> leaves = tree_.treeLeaves();

        macs_.resize(tree_.octree().numTreeNodes());
        markMac(tree_.octree(), box, focusStart, focusEnd, 1.0 / theta_, macs_.data());

        counts_.resize(nNodes(leaves));
        computeNodeCounts(leaves.data(), counts_.data(), nNodes(leaves), particleKeys.data(),
                          particleKeys.data() + particleKeys.size(), std::numeric_limits<unsigned>::max(), true);

        return converged;
    }

    const Octree<KeyType>& octree() const { return tree_.octree(); }

    gsl::span<const KeyType>  treeLeaves() const { return tree_.treeLeaves(); }
    gsl::span<const unsigned> leafCounts() const { return counts_; }

private:
    //! @brief opening angle refinement criterion
    float theta_;

    FocusedOctreeCore<KeyType> tree_;

    //! @brief particle counts of the focused tree leaves
    std::vector<unsigned> counts_;
    //! @brief mac evaluation result relative to focus area (pass or fail)
    std::vector<char> macs_;
};

} // namespace cstone
