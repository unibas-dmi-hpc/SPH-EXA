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
 * @brief A Propagator class for standard SPH
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 * @author Jose A. Escartin <ja.escartin@gmail.com>
 */

#pragma once

#include <variant>

#include "ipropagator.hpp"
#include "gravity_wrapper.hpp"

namespace sphexa
{

using namespace sph;

template<class DomainType, class ParticleDataType>
class HydroProp final : public Propagator<DomainType, ParticleDataType>
{
    using Base = Propagator<DomainType, ParticleDataType>;
    using Base::ng0_;
    using Base::ngmax_;
    using Base::timer;

    using T             = typename ParticleDataType::RealType;
    using KeyType       = typename ParticleDataType::KeyType;
    using MultipoleType = ryoanji::CartesianQuadrupole<float>;

    using Acc = typename ParticleDataType::AcceleratorType;
    using MHolder_t =
        typename detail::AccelSwitchType<Acc, MultipoleHolderCpu, MultipoleHolderGpu>::template type<MultipoleType,
                                                                                                     KeyType, T, T, T>;
    MHolder_t mHolder_;

public:
    HydroProp(size_t ngmax, size_t ng0, std::ostream& output, size_t rank)
        : Base(ngmax, ng0, output, rank)
    {
    }

    void activateFields(ParticleDataType& d) override
    {
        d.setConserved("x", "y", "z", "h", "m", "u", "vx", "vy", "vz", "x_m1", "y_m1", "z_m1", "du_m1");
        d.setDependent("rho", "p", "c", "ax", "ay", "az", "du", "c11", "c12", "c13", "c22", "c23", "c33", "keys", "nc");

        d.devData.setConserved("x", "y", "z", "h", "m", "vx", "vy", "vz");
        d.devData.setDependent(
            "rho", "p", "c", "ax", "ay", "az", "du", "c11", "c12", "c13", "c22", "c23", "c33", "keys");
    }

    void sync(DomainType& domain, ParticleDataType& d) override
    {
        if (d.g != 0.0)
        {
            domain.syncGrav(d.codes, d.x, d.y, d.z, d.h, d.m, d.u, d.vx, d.vy, d.vz, d.x_m1, d.y_m1, d.z_m1, d.du_m1);
        }
        else { domain.sync(d.codes, d.x, d.y, d.z, d.h, d.m, d.u, d.vx, d.vy, d.vz, d.x_m1, d.y_m1, d.z_m1, d.du_m1); }
    }

    void step(DomainType& domain, ParticleDataType& d) override
    {
        timer.start();
        sync(domain, d);
        timer.step("domain::sync");

        d.resize(domain.nParticlesWithHalos());
        resizeNeighbors(d, domain.nParticles() * ngmax_);
        size_t first = domain.startIndex();
        size_t last  = domain.endIndex();

        std::fill(begin(d.m), begin(d.m) + first, d.m[first]);
        std::fill(begin(d.m) + last, end(d.m), d.m[first]);

        findNeighborsSfc<T, KeyType>(
            first, last, ngmax_, d.x, d.y, d.z, d.h, d.codes, d.neighbors, d.neighborsCount, domain.box());
        timer.step("FindNeighbors");

        transferToDevice(d, 0, domain.nParticlesWithHalos(), {"x", "y", "z", "h", "m", "keys"});
        computeDensity(first, last, ngmax_, d, domain.box());
        transferToHost(d, first, last, {"rho"});
        timer.step("Density");
        computeEOS_HydroStd(first, last, d);
        timer.step("EquationOfState");
        domain.exchangeHalos(d.vx, d.vy, d.vz, d.rho, d.p, d.c);
        timer.step("mpi::synchronizeHalos");

        transferToDevice(d, 0, first, {"rho"});
        transferToDevice(d, last, domain.nParticlesWithHalos(), {"rho"});
        computeIAD(first, last, ngmax_, d, domain.box());
        transferToHost(d, first, last, {"c11", "c12", "c13", "c22", "c23", "c33"});
        timer.step("IAD");

        domain.exchangeHalos(d.c11, d.c12, d.c13, d.c22, d.c23, d.c33);
        timer.step("mpi::synchronizeHalos");

        transferToDevice(d, 0, domain.nParticlesWithHalos(), {"vx", "vy", "vz", "p", "c"});
        transferToDevice(d, 0, first, {"c11", "c12", "c13", "c22", "c23", "c33"});
        transferToDevice(d, last, domain.nParticlesWithHalos(), {"c11", "c12", "c13", "c22", "c23", "c33"});
        computeMomentumEnergySTD(first, last, ngmax_, d, domain.box());
        timer.step("MomentumEnergyIAD");

        if (d.g != 0.0)
        {
            mHolder_.upsweep(d, domain);
            timer.step("Upsweep");
            mHolder_.traverse(d, domain);
            timer.step("Gravity");
        }
        transferToHost(d, first, last, {"ax", "ay", "az", "du"});

        computeTimestep(first, last, d);
        timer.step("Timestep");
        computePositions(first, last, d, domain.box());
        timer.step("UpdateQuantities");
        updateSmoothingLength(first, last, d, ng0_);
        timer.step("UpdateSmoothingLength");

        timer.stop();
    }
};

} // namespace sphexa
