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
 * @brief A Propagator class for modern SPH with generalized volume elements
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 * @author Jose A. Escartin <ja.escartin@gmail.com>
 */

#pragma once

#include <variant>

#include "sph/sph.hpp"
#include "sph/hydro_turb/turbulence_data.hpp"

#include "ipropagator.hpp"
#include "ve_hydro.hpp"
#include "gravity_wrapper.hpp"

namespace sphexa
{

using namespace sph;

//! @brief VE hydro propagator that adds turbulence stirring to the acceleration prior to position update
template<class DomainType, class ParticleDataType>
class TurbVeProp final : public HydroVeProp<DomainType, ParticleDataType>
{
    using Base = HydroVeProp<DomainType, ParticleDataType>;
    using Base::ng0_;
    using Base::ngmax_;
    using Base::timer;

    sph::TurbulenceData<typename ParticleDataType::RealType, typename ParticleDataType::AcceleratorType> turbulenceData;

public:
    TurbVeProp(size_t ngmax, size_t ng0, std::ostream& output, size_t rank)
        : Base(ngmax, ng0, output, rank)
        , turbulenceData(TurbulenceConstants(), rank == 0)
    {
    }

    void step(DomainType& domain, ParticleDataType& d) override
    {
        Base::computeForces(domain, d);

        size_t first = domain.startIndex();
        size_t last  = domain.endIndex();

        computeTimestep(d);
        timer.step("Timestep");
        driveTurbulence(first, last, d, turbulenceData);
        timer.step("Turbulence Stirring");

        transferToHost(d, first, last, {"ax", "ay", "az", "du"});
        computePositions(first, last, d, domain.box());
        timer.step("UpdateQuantities");
        updateSmoothingLength(first, last, d, ng0_);
        timer.step("UpdateSmoothingLength");

        timer.stop();
    }
};

} // namespace sphexa
