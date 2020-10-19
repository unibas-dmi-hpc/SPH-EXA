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
    return std::tie(lhs.startCode, lhs.endCode) < std::tie(rhs.startCode, rhs.endCode);
}

template<class I>
inline bool operator==(const SfcNode<I>& lhs, const SfcNode<I>& rhs)
{
    return std::tie(lhs.startCode, lhs.endCode, lhs.coordinateIndex, lhs.count)
           == std::tie(rhs.startCode, rhs.endCode, rhs.coordinateIndex, rhs.count);

}

//! \brief Defines data to describe an octree node, no coordinate reference
template<class I>
struct GlobalSfcNode
{
    GlobalSfcNode(I start, I end, [[maybe_unused]] unsigned ignore, unsigned c)
        : startCode(start), endCode(end), count(c) { }

    //! start and end codes
    I startCode;
    I endCode;

    //! The particle content of the node.
    unsigned count;
};

template<class I>
inline bool operator<(const GlobalSfcNode<I>& lhs, const GlobalSfcNode<I>& rhs)
{
    return std::tie(lhs.startCode, lhs.endCode) < std::tie(rhs.startCode, rhs.endCode);
}

template<class I>
inline bool operator==(const GlobalSfcNode<I>& lhs, const GlobalSfcNode<I>& rhs)
{
    return std::tie(lhs.startCode, lhs.endCode, lhs.count)
           == std::tie(rhs.startCode, rhs.endCode, rhs.count);

}

/*! \brief aggregate mortonCodes into octree leaf nodes of increased size
 *
 * \tparam NodeType       SfcNode, either with or without offset into coordinate arrays
 * \tparam I              32- or 64-bit unsigned integer
 * \param[in] mortonCodes input mortonCode array
 * \param[in] bucketSize  determine size of octree nodes such that
 *                        (leaf node).count <= bucketSize
 *                        and for their parents (<=> internal nodes)
 *                        (parent node).count > bucketSize
 *
 * \return vector with the sorted octree leaf nodes
 */
template<template<class> class NodeType, class I>
std::vector<NodeType<I>> trimZCurve(const std::vector<I>& mortonCodes, unsigned bucketSize)
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
        ret.push_back(NodeType<I>{std::get<0>(box), std::get<1>(box), i, j-i});
        i = j;
        previousBoxEnd = std::get<1>(box);
    }

    return ret;
}

namespace detail
{

/*! \brief merge two overlapping ranges of GlobalSfcNodes
 *
 * \tparam SfcInputIterator input random access iterator
 * \tparam I                32- or 64-bit unsigned integer type
 * \param superRange        iterator to the enclosing octree Morton code range
 * \param subRangeStart     superRange->startCode <= subRangeStart->startCode
 * \param subRangeEnd       subIt->endCode <= superRange->endCode for all subIt with subRangeStart <= subIt < subRangeEnd
 * \param outputNodes       output for the merged GlobalSfcNodes<I>
 */
template<class SfcInputIterator, class I>
void mergeOverlappingRange(SfcInputIterator superRange, SfcInputIterator subRangeStart, SfcInputIterator subRangeEnd,
                           std::vector<GlobalSfcNode<I>>& outputNodes)
{
}

} // namespace detail

template<class I>
std::vector<GlobalSfcNode<I>> mergeZCurves(const std::vector<GlobalSfcNode<I>>& a,
                                           const std::vector<GlobalSfcNode<I>>& b)
{
}


} // namespace sphexa
