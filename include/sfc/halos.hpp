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
std::vector<IncomingHaloRange<I>> findIncomingHalos(std::vector<I>&                 globalTree,
                                                    const std::vector<std::size_t>& globalCounts,
                                                    const std::vector<T>&           interactionRadii,
                                                    const Box<T>&                   globalBox,
                                                    const SpaceCurveAssignment<I>&  assignment)
{


}

} // namespace sphexa