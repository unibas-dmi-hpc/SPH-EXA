#pragma once

#include "sfc/btree.hpp"

//! \brief \file binary tree traversal implementation

namespace sphexa
{

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


/*! \brief find all collisions between a leaf node enlarged by (dx,dy,dz) and the rest of the tree
 *
 * @tparam I                  32- or 64-bit unsigned integer
 * @param[in]  internalRoot   root of the internal binary radix tree
 * @param[in]  leafNodes      octree leaf nodes
 * @param[out] collisionList  output list of indices of colliding nodes
 * @param[in]  collisionBox   query box to look for collisions
 *                            with leaf nodes
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
template <class I>
void findCollisions(const BinaryNode<I> *internalRoot, const I *leafNodes, CollisionList& collisionList,
                    const Box<int>& collisionBox)
{
    using NodePtr = BinaryNode<I> *;
    assert(0 <= collisionBox.xmin() && collisionBox.xmax() <= (1u << maxTreeLevel<I>{}));
    assert(0 <= collisionBox.ymin() && collisionBox.ymax() <= (1u << maxTreeLevel<I>{}));
    assert(0 <= collisionBox.zmin() && collisionBox.zmax() <= (1u << maxTreeLevel<I>{}));

    NodePtr stack[64];
    NodePtr *stackPtr = stack;

    *stackPtr++ = nullptr;

    const BinaryNode<I> *node = internalRoot;

    do
    {
        if (node->leftChild)
        {
            if (overlap(node->leftChild->prefix, node->leftChild->prefixLength, collisionBox))
            {
                assert(stackPtr - stack < 64 && "local stack overflow");
                *stackPtr++ = node->leftChild;
            }
        }
        else
        {
            int leafIndex = node->leftLeafIndex;
            I leafCode = leafNodes[leafIndex];
            I leafUpperBound = leafNodes[leafIndex + 1];

            int prefixNBits = treeLevel(leafUpperBound - leafCode) * 3;

            if (overlap(leafCode, prefixNBits, collisionBox)) { collisionList.add(leafIndex); }
        }
        if (node->rightChild)
        {
            if (overlap(node->rightChild->prefix, node->rightChild->prefixLength, collisionBox))
            {
                assert(stackPtr - stack < 64 && "local stack overflow");
                *stackPtr++ = node->rightChild;
            }
        }
        else
        {
            int leafIndex = node->rightLeafIndex;
            I leafCode = leafNodes[leafIndex];
            I leafUpperBound = leafNodes[leafIndex + 1];

            int prefixNBits = treeLevel(leafUpperBound - leafCode) * 3;

            if (overlap(leafCode, prefixNBits, collisionBox)) { collisionList.add(leafIndex); }
        }

        node = *--stackPtr;

    } while (node != nullptr);
}

} // namespace sphexa