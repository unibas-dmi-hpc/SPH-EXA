#pragma once

#include "Octree.hpp"

#include "sfc/clz.hpp"
#include "sfc/mortoncode.hpp"

namespace sphexa
{

//! \brief Defines data to describe an octree node
template<class I>
struct SfcNode
{
    /*! The morton start and end codes define the scope of the node.
     *  They are equivalent to an integer (ix,iy,iz) index triple that describes the
     *  spatial location of the box plus the octree division level (or the size of the box)
     */
    I startCode;
    I endCode;

    /*! The particle content of the node.
     *
     *  coordinateIndex: Stores the index of the first morton code in the full array
     *                   of sorted morton codes that falls into the node.
     *  count:           Number of morton codes in this node. Since the morton code
     *                   array is assumed to be sorted, all particles in the range
     *                   [coordinateIndex, coordinateIndex + count] fall into the node.
     *
     *  Both indices are also valid for the x,y,z coordinate arrays, provided that they
     *  are sorted according to the ascending morton code ordering.
     */
    unsigned coordinateIndex;
    unsigned count;
};

template<class I>
inline bool operator<(const SfcNode<I>& lhs, const SfcNode<I>& rhs)
{
    return lhs.startCode < rhs.startCode;
}

template<class I>
inline bool operator==(const SfcNode<I>& lhs, const SfcNode<I>& rhs)
{
    return std::tie(lhs.startCode, lhs.endCode, lhs.coordinateIndex, lhs.count)
           == std::tie(rhs.startCode, rhs.endCode, rhs.coordinateIndex, rhs.count);

}

/*! \brief aggregate mortonCodes into octree leaf nodes of increased size
 *
 * \tparam I              32- or 64-bit unsigned integer
 * \param[in] mortonCodes input mortonCode array
 * \param[in] bucketSize  determine size of octree nodes such that
 *                        (leaf node).count <= bucketSize
 *                        and for their parents (<=> internal nodes)
 *                        (parent node).count > bucketSize
 *
 * \return vector with the sorted octree leaf nodes
 */
template<class I>
std::vector<SfcNode<I>> trimZCurve(const std::vector<I>& mortonCodes, unsigned bucketSize)
{
    std::vector<SfcNode<I>> ret;

    unsigned n = mortonCodes.size();
    unsigned i = 0;

    I previousBoxEnd = 0;

    while (i < n)
    {
        I code = mortonCodes[i];

        // the smallest code more than bucketSize away
        // need to find a box that stays below it
        I codeLimit = (i + bucketSize < n) ? mortonCodes[i + bucketSize] : nodeRange<I>(0);

        // find smallest j in [i, i + bucketSize], such that codeLimit < get<1>(smallestCommonBox(mCodes[i], mCodes[j]))
        auto isInBox = [code](I c1_, I c2_){ return c1_ < std::get<1>(smallestCommonBox(code, c2_)); };
        auto jIt = std::upper_bound(cbegin(mortonCodes) + i, cbegin(mortonCodes) + std::min(n, i + bucketSize), codeLimit, isInBox);
        unsigned j = jIt - cbegin(mortonCodes);

        // find smallest k in [i, i + bucketSize], such that not(get<0>(smallestCommonBox(mCodes[i], mCodes[k])) < previousBoxEnd)
        auto isBelowBox = [code](I c1_, I c2_){ return !(std::get<0>(smallestCommonBox(code, c1_)) < c2_); };
        auto kIt = std::lower_bound(cbegin(mortonCodes) + i, cbegin(mortonCodes) + std::min(n, i + bucketSize), previousBoxEnd, isBelowBox);
        unsigned k = kIt - cbegin(mortonCodes);

        // the smaller of the two indices is the one that produces a range of codes
        // with an enclosing octree box that both doesn't overlap with the previous one
        // and does not include more than bucketSize particles
        j = std::min(j, k);

        std::tuple<I, I> box = smallestCommonBox(code, mortonCodes[j-1]);
        ret.push_back(SfcNode<I>{std::get<0>(box), std::get<1>(box), i, j-i});
        i = j;
        previousBoxEnd = std::get<1>(box);
    }

    return ret;
}

} // namespace sphexa
