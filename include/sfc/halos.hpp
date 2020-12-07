#pragma once

#include <algorithm>
#include <vector>

#include "sfc/domaindecomp.hpp"

namespace sphexa
{

template <class I>
struct IncomingHaloRange
{
    I codeStart;
    I codeEnd;
    int count;
    int sourceRank;
};



template <class I, class T>
std::vector<IncomingHaloRange<I>> findIncomingHalos(const std::vector<I>&           tree,
                                                    const std::vector<std::size_t>& counts,
                                                    const std::vector<T>&           interactionRadii,
                                                    const Box<T>&                   box,
                                                    const SpaceCurveAssignment<I>&  assignment,
                                                    int                             rank)
{
    std::vector<IncomingHaloRange<I>> ret;

    for (int node = 0; node < nNodes(tree); ++node)
    {
        T h  = interactionRadii[node]; // smoothing length
        T hn = normalize(h, box.xmin(), box.xmax());
    }

    return ret;
}

} // namespace sphexa