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
 * @brief binary tree traversal implementation
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#pragma once

#include "cstone/traversal/boxoverlap.hpp"
#include "cstone/tree/btree.hpp"

namespace cstone
{

//! @brief stores indices of colliding octree leaf nodes
class CollisionList
{
public:
    //! @brief add an index to the list of colliding leaf tree nodes
    void add(int i)
    {
        list_[n_] = i;
        n_        = (n_ < collisionMax - 1) ? n_ + 1 : n_;
    }

    //! @brief access collision list as a range
    [[nodiscard]] const int* begin() const { return list_; }
    [[nodiscard]] const int* end() const { return list_ + n_; }

    //! @brief access collision list elements
    int operator[](int i) const
    {
        assert(i < collisionMax);
        return list_[i];
    }

    /*! @brief returns number of collisions
     *
     * Can (should) also be used to check whether the internal storage
     * was exhausted during collision detection.
     */
    [[nodiscard]] std::size_t size() const { return n_; };

    [[nodiscard]] bool exhausted() const { return n_ == collisionMax - 1; }

private:
    static constexpr int collisionMax = 512;
    std::size_t n_{0};
    int list_[collisionMax]{0};
};

template<class KeyType>
HOST_DEVICE_FUN inline bool traverseNode(const BinaryNode<KeyType>* root,
                                         TreeNodeIndex idx,
                                         const IBox& collisionBox,
                                         util::array<KeyType, 2> excludeRange)
{
    return (!isLeafIndex(idx)) && !containedIn(root[idx].prefix, excludeRange[0], excludeRange[1]) &&
           overlap<KeyType>(
               sfcIBox(sfcKey(decodePlaceholderBit(root[idx].prefix)), decodePrefixLength(root[idx].prefix) / 3),
               collisionBox);
}

template<class KeyType>
HOST_DEVICE_FUN inline bool
leafOverlap(int leafIndex, const KeyType* leafNodes, const IBox& collisionBox, util::array<KeyType, 2> excludeRange)
{
    if (!isLeafIndex(leafIndex)) { return false; }

    TreeNodeIndex effectiveIndex = loadLeafIndex(leafIndex);
    KeyType leafCode             = leafNodes[effectiveIndex];
    KeyType leafUpperBound       = leafNodes[effectiveIndex + 1];

    bool notExcluded = !containedIn(leafCode, leafUpperBound, excludeRange[0], excludeRange[1]);
    return notExcluded &&
           overlap<KeyType>(sfcIBox(sfcKey(leafCode), treeLevel(leafUpperBound - leafCode)), collisionBox);
}

/*! @brief find all collisions between a leaf node enlarged by (dx,dy,dz) and the rest of the tree
 *
 * @tparam KeyType              32- or 64-bit unsigned integer
 * @param[in]    internalRoot   root of the internal binary radix tree
 * @param[in]    leafNodes      octree leaf nodes
 * @param[inout] collisionList  endpoint action to perform with each colliding leaf node
 *                              callable with signature void(TreeNodeIndex)
 * @param[in]    collisionBox   query box to look for collisions
 *                              with leaf nodes
 * @param[in]  excludeRange     range defined by two SFC codes to exclude from collision search
 *                              any leaf nodes fully contained in the specified range will not be
 *                              reported as collisions
 *
 * At all traversal steps through the hierarchy of the internal binary radix tree,
 * all 3 x,y,z dimensions are checked to determine overlap with a binary node.
 * If the Morton codes used to construct the binary tree satisfy the requirements of
 * a cornerstone octree, it would be sufficient to only check
 *
 * x at nodes with prefixLength % 3 == 1
 * y at nodes with prefixLength % 3 == 2
 * z at nodes with prefixLength % 3 == 0
 *
 * because the cornerstone invariants guarantee that each internal binary node will
 * have one more bit in its prefix than its parent.
 *
 * However, the construction of the internal tree and the following traversal as
 * implemented also works if an arbitrary sorted sequence of Morton codes was used
 * for the construction of the internal tree, i.e. holes or omission of empty nodes
 * would be possible. Since this capability might be useful in the future, and the
 * cost to check all 3 dimensions at each step should not be very high, we keep
 * the implementation general.
 */
template<class KeyType, class Endpoint>
HOST_DEVICE_FUN void findCollisions(const BinaryNode<KeyType>* root,
                                    const KeyType* leafNodes,
                                    Endpoint&& reportCollision,
                                    const IBox& collisionBox,
                                    util::array<KeyType, 2> excludeRange)
{
    using Node = BinaryNode<KeyType>;

    TreeNodeIndex stack[64];
    stack[0] = 0;

    TreeNodeIndex stackPos = 1;
    TreeNodeIndex node     = 0; // start at the root

    do
    {
        TreeNodeIndex leftChild  = root[node].child[Node::left];
        TreeNodeIndex rightChild = root[node].child[Node::right];
        bool traverseL           = traverseNode(root, leftChild, collisionBox, excludeRange);
        bool traverseR           = traverseNode(root, rightChild, collisionBox, excludeRange);

        bool overlapLeafL = leafOverlap(leftChild, leafNodes, collisionBox, excludeRange);
        bool overlapLeafR = leafOverlap(rightChild, leafNodes, collisionBox, excludeRange);

        if (overlapLeafL) { reportCollision(loadLeafIndex(leftChild)); }
        if (overlapLeafR) { reportCollision(loadLeafIndex(rightChild)); }

        if (!traverseL and !traverseR) { node = stack[--stackPos]; }
        else
        {
            if (traverseL && traverseR)
            {
                if (stackPos >= 64)
                {
                    printf("btree traversal stack exhausted\n");
                    return;
                }
                stack[stackPos++] = rightChild; // push
            }

            node = (traverseL) ? leftChild : rightChild;
        }

    } while (node != 0); // the root can only be obtained when the tree has been fully traversed
}

//! @brief convenience overload for storing colliding indices
template<class KeyType>
void findCollisions(const BinaryNode<KeyType>* root,
                    const KeyType* leafNodes,
                    CollisionList& collisions,
                    const IBox& collisionBox,
                    util::array<KeyType, 2> excludeRange)
{
    auto storeCollisions = [&collisions](TreeNodeIndex i) { collisions.add(i); };
    findCollisions(root, leafNodes, storeCollisions, collisionBox, excludeRange);
}

//! @brief convenience overload for marking colliding node indices
template<class KeyType>
HOST_DEVICE_FUN inline void findCollisions(const BinaryNode<KeyType>* root,
                                           const KeyType* leafNodes,
                                           int* flags,
                                           const IBox& collisionBox,
                                           util::array<KeyType, 2> excludeRange)
{
    auto markCollisions = [flags](TreeNodeIndex i) { flags[i] = 1; };
    findCollisions(root, leafNodes, markCollisions, collisionBox, excludeRange);
}

} // namespace cstone
