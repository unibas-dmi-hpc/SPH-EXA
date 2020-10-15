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

/*! \brief aggregate mortonCodes into octree leaf nodes
 *
 * \tparam I              32- or 64-bit unsigned integer
 * \param[in] mortonCodes input mortonCode array
 * \param[in] bucketSize  determine size of octree nodes such that
 *                        (leaf node).count < bucketSize
 *                        and for their parents (<=> internal nodes)
 *                        (parent node).count >= bucketSize
 *
 * \return vector with the sorted octree leaf nodes
 */
template<class I>
std::vector<SfcNode<I>> generateOctree(const std::vector<I>& mortonCodes, unsigned bucketSize)
{
    std::vector<SfcNode<I>> ret;

    unsigned n = mortonCodes.size();
    unsigned i = 0;
    //while (i < n)
    {
        I code = mortonCodes[i];
        unsigned upperIndex = std::min(n, i + bucketSize - 1);



        I upperCode = mortonCodes[upperIndex];

        // if the resolution of the SFC is not enough
        // to tell bucketSize particles apart at the lowest level
        if (code == upperCode)
        {
            upperIndex = std::upper_bound(cbegin(mortonCodes) + i, cend(mortonCodes), code)
                           - cbegin(mortonCodes);
            if (upperIndex == n)
            {
                ret.push_back(SfcNode<I>{code, upperCode, i, upperIndex});
                return ret;
            }
            else
            {
                upperCode = mortonCodes[upperIndex];
            }
        }

        // guaranteed to be > 0
        I codeDifference = upperCode - code;
        unsigned containingTreeLevel = (countLeadingZeros(codeDifference) - unusedBits<I>{})/3 + 1;
        I range = 1u << (3*(maxTreeLevel<I>{} - containingTreeLevel));

        I nodeStart = detail::enclosingBoxCode(code, containingTreeLevel);
        I nodeEnd   = nodeStart + range;

        // count the number of codes in the range [nodeStart, nodeEnd)
        upperIndex = std::lower_bound(cbegin(mortonCodes) + i, cbegin(mortonCodes) + upperIndex, nodeEnd)
                       - cbegin(mortonCodes);

        unsigned count = upperIndex - i;
        ret.push_back(SfcNode<I>{nodeStart, nodeEnd, i, count});

        i += count;
    }

    return ret;
}

} // namespace sphexa
