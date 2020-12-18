#pragma once

/*! \brief \file binary radix tree implementation
 *
 * Algorithm published in https://dl.acm.org/doi/10.5555/2383795.2383801
 * and further illustrated at
 * https://developer.nvidia.com/blog/thinking-parallel-part-iii-tree-construction-gpu/
 *
 * Most SFC-based functionality for SPH like domain decomposition, global octree build
 * and neighbor search does not need tree-traversal and explicit construction
 * of tree-internal nodes.
 *
 * The halo search problem is an exception. Finding all nodes in the global octree
 * that overlap with a given node that is enlarged by the search radius in all
 * dimensions is essentially the same as collision detection between objects
 * in 3D graphics, where bounding volume hierarchies are constructed from binary radix trees.
 *
 * The reason why the halo search problem differs from the other SFC-based algorithms
 * for neighbor search and domain decomposition is that an octree node that is enlarged
 * by an arbitrary distance in each dimension cannot necessarily be composed from octree
 * nodes of similar size as the query node and therefore also not as a range of Morton codes,
 * or the sum of a small number of Morton code ranges. This is especially true at the boundaries
 * between octree nodes of different division level, i.e. when large nodes are next to
 * very small nodes. For this reason, nodes enlarged by a halo radius are best represented
 * by separate x,y,z coordinate ranges.
 *
 * The best way to implement collision detection for the 3D box defined x,y,z coordinate ranges
 * is by constructing the internal tree part over the
 * global octree leaf nodes as a binary radix tree. Each internal node of the binary
 * radix tree can be constructed independently and thus the algorithm is ideally suited
 * for GPUs. Subsequent tree traversal to detect collisions can also be done for all leaf
 * nodes in parallel. It is possible to convert the internal binary tree into an octree,
 * as 3 levels in the binary tree correspond to one level in the equivalent octree.
 * Doing so could potentially speed up traversal by a bit, but it is not clear whether it
 * would make up for the overhead of constructing the internal octree.
 */

#include "sfc/clz.hpp"
#include "sfc/mortoncode.hpp"

namespace sphexa {

/*! \brief binary radix tree node
 *
 * @tparam I 32 or 64 bit unsigned integer
 *
 * (final content TBD)
 */
template<class I>
struct BinaryNode
{
    BinaryNode* leftChild;
    BinaryNode* rightChild;

    I   prefix;
    int prefixLength;

    int leftLeafIndex;
    int rightLeafIndex;
};

/*! \brief calculate common prefix (cpr) of two morton keys
 *
 * @tparam I    32 or 64 bit unsigned integer
 * @param key1  first morton code key
 * @param key2  second morton code key
 * @return      number of continuous identical bits, counting from MSB
 *              minus the 2 unused bits in 32 bit codes or minus the 1 unused bit
 *              in 64 bit codes.
 */
template<class I>
int cpr(I key1, I key2)
{
    return int(countLeadingZeros(key1 ^ key2)) - unusedBits<I>{};
}

/*! \brief find position of first differing bit
 *
 * @tparam I                 32 or 64 bit unsigned integer
 * @param sortedMortonCodes
 * @param first              first range index
 * @param last               last rang index
 * @return                   position of morton
 */
template<class I>
int findSplit(I*  sortedMortonCodes,
              int first,
              int last)
{
    // Identical Morton codes => split the range in the middle.
    I firstCode = sortedMortonCodes[first];
    I lastCode  = sortedMortonCodes[last];

    if (firstCode == lastCode)
        return (first + last) >> 1;

    // Calculate the number of highest bits that are the same for all objects
    int commonPrefix = cpr(firstCode, lastCode);

    // Use binary search to find where the next bit differs.
    // Specifically, we are looking for the highest object that
    // shares more than commonPrefix bits with the first one.
    int split = first; // initial guess
    int step = last - first;

    do
    {
        step = (step + 1) >> 1;      // exponential decrease
        int newSplit = split + step; // proposed new position

        if (newSplit < last)
        {
            I splitCode = sortedMortonCodes[newSplit];
            int splitPrefix = cpr(firstCode, splitCode);
            if (splitPrefix > commonPrefix)
                split = newSplit; // accept proposal
        }
    }
    while (step > 1);

    return split;
}

/*! \brief construct the internal binary tree node with index idx
 *
 * @tparam I
 * @param codes
 * @param leaves
 * @param nLeaves
 * @param internalNodes
 * @param idx
 */
template<class I>
void constructInternalNode(const I* codes, int nLeaves, BinaryNode<I>* internalNodes, int idx)
{
    BinaryNode<I>* idxNode = internalNodes + idx;

    int d = 1;
    int minPrefixLength = -1;

    if (idx > 0)
    {
        d = (cpr(codes[idx], codes[idx + 1]) > cpr(codes[idx], codes[idx - 1])) ? 1 : -1;
        minPrefixLength = cpr(codes[idx], codes[idx-d]);
    }

    //int jdx = idx + d;
    //while (0 < jdx && jdx < nLeaves - 1
    //       && cpr(codes[jdx+d], codes[idx]) > minPrefixLength)
    //{
    //    jdx += d;
    //}

    // find max search range
    int jSearchRange = 2;
    int upperJ       = idx + jSearchRange * d;
    while(0 <= upperJ && upperJ < nLeaves
          && cpr(codes[idx], codes[upperJ]) > minPrefixLength)
    {
        jSearchRange *= 2;
        upperJ = idx + jSearchRange * d;
    }

    // prune max search range if out of bounds
    if (upperJ >= nLeaves)
    {
        jSearchRange = nLeaves - idx;
    }
    if (upperJ < 0)
    {
        jSearchRange = idx;
    }

    // binary search to determine the second node range index jdx
    int nodeLength = 0;
    int step = jSearchRange;
    do
    {
        step = (step + 1) / 2;
        if (cpr(codes[idx], codes[idx + (nodeLength+step)*d]) > minPrefixLength)
        {
            nodeLength += step;
        }

    } while (step > 1);

    int jdx = idx + nodeLength * d;

    idxNode->prefixLength = cpr(codes[idx], codes[jdx]);
    idxNode->prefix       = zeroLowBits(codes[idx], idxNode->prefixLength);

    // find position of highest differing bit between [idx, jdx]
    int gamma = findSplit(codes, std::min(jdx, idx), std::max(jdx, idx));

    // establish child relationships
    if (std::min(jdx, idx) == gamma)
    {
        // left child is a leaf
        idxNode->leftChild     = nullptr;
        idxNode->leftLeafIndex = gamma;
    }
    else
    {
        //left child is an internal binary node
        idxNode->leftChild     = internalNodes + gamma;
        idxNode->leftLeafIndex = -1;
    }

    if (std::max(jdx,idx) == gamma + 1)
    {
        // right child is a leaf
        idxNode->rightChild     = nullptr;
        idxNode->rightLeafIndex = gamma + 1;
    }
    else
    {
        // right child is an internal binary node
        idxNode->rightChild     = internalNodes + gamma + 1;
        idxNode->rightLeafIndex = -1;
    }
}

//! \brief stores indices of colliding octree leaf nodes
class CollisionList
{
public:
    //! \brief add an index to the list of colliding leaf tree nodes
    void add(int i)
    {
        list_[n_] = i;
        n_ = (n_ < collisionMax-1) ? n_+1 : n_;
    }

    //! \brief access collision list as a range
    [[nodiscard]] const int* begin() const { return list_; }
    [[nodiscard]] const int* end()   const { return list_ + n_; }

    //! \brief access collision list elements
    int operator[](int i) const
    {
        assert(i < collisionMax);
        return list_[i];
    }

    /*! \brief returns number of collisions
     *
     * Can (should) also be used to check whether the internal storage
     * was exhausted during collision detection.
     */
    [[nodiscard]] int size() const { return n_; };

private:
    static constexpr int collisionMax = 64;
    int n_{0};
    int list_[collisionMax]{0};
};

/*! \brief check for overlap between a binary or octree node and an box 3D space
 *
 * @tparam I
 * @param prefix            Morton code node prefix, defines the corner of node
 *                          closest to origin. Also equals the lower Morton code bound
 *                          of the node.
 * @param length            Number of bits in the prefix to treat as the key. Defines
 *                          the Morton code range of the node.
 * @param [x,y,z][min,max]  3D coordinate range, defines an arbitrary box in space to
 *                          test for overlap.
 * @return                  true or false
 *
 */
template<class I>
bool overlap(I prefix, int length, int xmin, int xmax, int ymin, int ymax, int zmin, int zmax)
{
    std::array<int, 2> xRange = decodeXRange(prefix, length);
    std::array<int, 2> yRange = decodeYRange(prefix, length);
    std::array<int, 2> zRange = decodeZRange(prefix, length);

    bool xOverlap = xmax > xRange[0] && xRange[1] > xmin;
    bool yOverlap = ymax > yRange[0] && yRange[1] > ymin;
    bool zOverlap = zmax > zRange[0] && zRange[1] > zmin;

    return xOverlap && yOverlap && zOverlap;
}



template<class I>
void findCollisions(const BinaryNode<I>* internalRoot, const I* leafNodes,
                    CollisionList& collisionList,
                    int xmin, int xmax, int ymin, int ymax, int zmin, int zmax)
{
    using NodePtr = BinaryNode<I>*;
    assert(0 <= xmin && xmax < (1u<<maxTreeLevel<I>{}));
    assert(0 <= ymin && ymax < (1u<<maxTreeLevel<I>{}));
    assert(0 <= zmin && zmax < (1u<<maxTreeLevel<I>{}));

    NodePtr  stack[64];
    NodePtr* stackPtr = stack;

    *stackPtr++ = nullptr;

    const BinaryNode<I>* node = internalRoot;

    do {
        if (node->leftChild)
        {
            if (overlap(node->leftChild->prefix, node->leftChild->prefixLength,
                        xmin, xmax, ymin, ymax, zmin, zmax))
            {
                assert(stackPtr - stack < 64 && "local stack overflow");
                *stackPtr++ = node->leftChild;
            }
        }
        else {
            int leafIndex    = node->leftLeafIndex;
            I leafCode       = leafNodes[leafIndex];
            I leafUpperBound = leafNodes[leafIndex+1];

            int prefixNBits = treeLevel(leafUpperBound - leafCode) * 3;

            if (overlap(leafCode, prefixNBits, xmin, xmax, ymin, ymax, zmin, zmax))
            {
                collisionList.add(leafIndex);
            }
        }
        if (node->rightChild)
        {
            if (overlap(node->rightChild->prefix, node->rightChild->prefixLength,
                        xmin, xmax, ymin, ymax, zmin, zmax))
            {
                assert(stackPtr - stack < 64 && "local stack overflow");
                *stackPtr++ = node->rightChild;
            }
        }
        else {
            int leafIndex    = node->rightLeafIndex;
            I leafCode       = leafNodes[leafIndex];
            I leafUpperBound = leafNodes[leafIndex+1];

            int prefixNBits = treeLevel(leafUpperBound - leafCode) * 3;

            if (overlap(leafCode, prefixNBits, xmin, xmax, ymin, ymax, zmin, zmax))
            {
                collisionList.add(leafIndex);
            }
        }

        node = *--stackPtr;

    } while (node != nullptr);
}

/*! \brief find all collisions between a leaf node enlarged by (dx,dy,dz) and the rest of the tree
 *
 * @tparam I                  32- or 64-bit unsigned integer
 * @param[in]  internalRoot   root of the internal binary tree
 * @param[in]  leafNodes      octree leaf nodes
 * @param[out] collisionList  output list of indices of colliding nodes
 * @param[in]  leafIndex      query node index to look for collisions
 * @param[in]  dx             enlarge X range of node by +-dx
 * @param[in]  dy             enlarge Y range of node by +-dy
 * @param[in]  dz             enlarge Z range of node by +-dz
 *
 */
template<class I>
void findCollisions(const BinaryNode<I>* internalRoot, const I* leafNodes,
                    CollisionList& collisionList, int leafIndex, int dx, int dy, int dz)
{
    I leafCode       = leafNodes[leafIndex];
    I leafUpperBound = leafNodes[leafIndex+1];

    int prefixNBits = treeLevel(leafUpperBound - leafCode) * 3;

    std::array<int, 2> xrange = decodeXRange(leafCode, prefixNBits);
    std::array<int, 2> yrange = decodeYRange(leafCode, prefixNBits);
    std::array<int, 2> zrange = decodeZRange(leafCode, prefixNBits);

    constexpr int maxCoordinate = (1u << maxTreeLevel<I>{}) - 1;

    // add halo range to the coordinate ranges of the node to be collided
    int xmin = std::max(0, xrange[0] - dx);
    int xmax = std::min(maxCoordinate, xrange[1] + dx);
    int ymin = std::max(0, yrange[0] - dy);
    int ymax = std::min(maxCoordinate, yrange[1] + dy);
    int zmin = std::max(0, zrange[0] - dz);
    int zmax = std::min(maxCoordinate, zrange[1] + dz);

    // find collisions between the leafIndex node enlarged by the halo (dx,dy,dz) range
    // and all the other leaf nodes
    findCollisions(internalRoot, leafNodes, collisionList, xmin, xmax, ymin, ymax, zmin, zmax);
}

/*! \brief create the internal part of an octree as internal nodes
 *
 * @tparam I    32- or 64-bit unsigned integer
 * @param tree  sorted Morton codes representing the leaves of the (global) octree
 * @return      the internal part of the input tree constructed as binary nodes
 *
 * This is a CPU version that can be OpenMP parallelized.
 * In the GPU version, the for-loop body is designed such that one GPU-thread
 * can be launched for each for-loop element.
 */
template<class I>
std::vector<BinaryNode<I>> createInternalTree(const std::vector<I>& tree)
{
    std::vector<BinaryNode<I>> ret(tree.size() - 1);

    // (omp) parallel
    for (int idx = 0; idx < ret.size(); ++idx)
    {
        constructInternalNode(tree.data(), tree.size(), ret.data(), idx);
    }

    return ret;
}

/*! \brief For each leaf node enlarged by its halo radius, find all colliding leaf nodes
 *
 * @tparam I            32- or 64-bit unsigned integer
 * @param internalTree  internal binary tree
 * @param tree          sorted Morton codes representing the leaves of the (global) octree
 * @param haloRadii     halo search radii per leaf node, length = nNodes(tree)
 * @return              list of colliding node indices for each leaf node
 *
 * This is a CPU version that can be OpenMP parallelized.
 * In the GPU version, the for-loop body is designed such that one GPU-thread
 * can be launched for each for-loop element.
 */
template<class I, class T>
std::vector<CollisionList> findAllCollisions(const std::vector<BinaryNode<I>>& internalTree, const std::vector<I>& tree,
                                             const std::vector<T>& haloRadii, const Box<T>& globalBox)
{
    assert(internalTree.size() == tree.size() - 1 && "internal tree does not match leaves");
    assert(internalTree.size() == haloRadii.size() && "need one halo radius per leaf node");

    std::vector<CollisionList> collisions(tree.size() - 1);

    // (omp) parallel
    for (int leafIdx = 0; leafIdx < internalTree.size(); ++leafIdx)
    {
        T radius = haloRadii[leafIdx];

        int dx = detail::toNBitInt<I>(normalize(radius, globalBox.xmin(), globalBox.xmax()));
        int dy = detail::toNBitInt<I>(normalize(radius, globalBox.ymin(), globalBox.ymax()));
        int dz = detail::toNBitInt<I>(normalize(radius, globalBox.zmin(), globalBox.zmax()));

        findCollisions(internalTree.data(), tree.data(), collisions[leafIdx], leafIdx, dx, dy, dz);
    }

    return collisions;
}

} // namespace sphexa
