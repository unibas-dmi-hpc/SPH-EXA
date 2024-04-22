/*! @file
 * @brief Timestep definition
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#pragma once

#include <array>
#include <string>

#include "cstone/tree/definitions.h"

namespace sph
{

struct Timestep
{
    static constexpr int maxNumRungs = 4;
    //! @brief maxDt = minDt * 2^numRungs;
    float minDt;
    int   numRungs{1};
    //! @brief 0,...,2^numRungs
    int substep{0};

    std::array<cstone::LocalIndex, maxNumRungs + 1> rungRanges;

    template<class Archive>
    void loadOrStore(Archive* ar, const std::string& prefix)
    {
        ar->stepAttribute(prefix + "minDt", &minDt, 1);
        ar->stepAttribute(prefix + "numRungs", &numRungs, 1);
        ar->stepAttribute(prefix + "substep", &substep, 1);
    }
};

} // namespace sph