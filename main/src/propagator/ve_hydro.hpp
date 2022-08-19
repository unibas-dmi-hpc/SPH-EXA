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

#include "ipropagator.hpp"
#include "gravity_wrapper.hpp"

namespace sphexa
{

using namespace sph;

template<class DomainType, class ParticleDataType>
class HydroVeProp : public Propagator<DomainType, ParticleDataType>
{
protected:
    using Base = Propagator<DomainType, ParticleDataType>;
    using Base::ng0_;
    using Base::ngmax_;
    using Base::timer;

    using T             = typename ParticleDataType::RealType;
    using KeyType       = typename ParticleDataType::KeyType;
    using MultipoleType = ryoanji::CartesianQuadrupole<float>;

    using Acc = typename ParticleDataType::AcceleratorType;
    using MHolder_t =
        typename cstone::AccelSwitchType<Acc, MultipoleHolderCpu, MultipoleHolderGpu>::template type<MultipoleType,
                                                                                                     KeyType, T, T, T>;

    MHolder_t mHolder_;

    /*! @brief the list of conserved particles fields with values preserved between iterations
     *
     * x, y, z, h and m are automatically considered conserved and must not be specified in this list
     */
    using ConservedFields = FieldList<"u", "vx", "vy", "vz", "x_m1", "y_m1", "z_m1", "du_m1", "alpha">;

    //! @brief the list of dependent particle fields, these may be used as scratch space during domain sync
    using DependentFields = FieldList<"prho", "c", "ax", "ay", "az", "du", "c11", "c12", "c13", "c22", "c23", "c33",
                                      "xm", "kx", "divv", "curlv", "nc">;

public:
    HydroVeProp(size_t ngmax, size_t ng0, std::ostream& output, size_t rank)
        : Base(ngmax, ng0, output, rank)
    {
    }

    std::vector<std::string> conservedFields() const override
    {
        std::vector<std::string> ret{"x", "y", "z", "h", "m"};
        for_each_tuple([&ret](auto f) { ret.push_back(f.value); }, make_tuple(ConservedFields{}));
        return ret;
    }

    void activateFields(ParticleDataType& d) override
    {
        //! @brief Fields accessed in domain sync are not part of extensible lists.
        d.setConserved("x", "y", "z", "h", "m");
        d.setDependent("keys");

        std::apply([&d](auto... f) { d.setConserved(f.value...); }, make_tuple(ConservedFields{}));
        std::apply([&d](auto... f) { d.setDependent(f.value...); }, make_tuple(DependentFields{}));

        d.devData.setConserved("x", "y", "z", "h", "m", "vx", "vy", "vz", "alpha");
        d.devData.setDependent("prho", "c", "kx", "xm", "ax", "ay", "az", "du", "c11", "c12", "c13", "c22", "c23",
                               "c33", "keys");
    }

    void sync(DomainType& domain, ParticleDataType& d) override
    {
        if (d.g != 0.0)
        {
            domain.syncGrav(d.codes, d.x, d.y, d.z, d.h, d.m, getHost<ConservedFields>(d), getHost<DependentFields>(d));
        }
        else
        {
            domain.sync(d.codes, d.x, d.y, d.z, d.h, std::tuple_cat(std::tie(d.m), getHost<ConservedFields>(d)),
                        getHost<DependentFields>(d));
        }
    }

    void computeForces(DomainType& domain, ParticleDataType& d)
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

        findNeighborsSfc<T, KeyType>(first, last, ngmax_, d.x, d.y, d.z, d.h, d.codes, d.neighbors, d.neighborsCount,
                                     domain.box());
        timer.step("FindNeighbors");

        transferToDevice(d, 0, domain.nParticlesWithHalos(), {"x", "y", "z", "h", "m", "keys"});
        computeXMass(first, last, ngmax_, d, domain.box());
        timer.step("XMass");
        domain.exchangeHalosAuto(get<"xm">(d), std::get<0>(get<"ax">(d)), std::get<0>(get<"ay">(d)));
        timer.step("mpi::synchronizeHalos");

        d.release("ax");
        d.acquire("gradh");
        d.devData.release("ax", "du");
        d.devData.acquire("gradh", "u");
        computeVeDefGradh(first, last, ngmax_, d, domain.box());
        timer.step("Normalization & Gradh");

        transferToDevice(d, first, last, {"vx", "vy", "vz", "u", "alpha"});
        computeEOS(first, last, d);
        timer.step("EquationOfState");

        domain.exchangeHalosAuto(get<"vx", "vy", "vz", "prho", "c", "kx">(d), std::get<0>(get<"gradh">(d)),
                                 std::get<0>(get<"ay">(d)));
        timer.step("mpi::synchronizeHalos");

        d.release("gradh");
        d.acquire("ax");
        d.devData.release("gradh", "ay", "u");
        d.devData.acquire("divv", "curlv", "du");
        computeIadDivvCurlv(first, last, ngmax_, d, domain.box());
        transferToHost(d, first, last, {"divv", "curlv"});
        timer.step("IadVelocityDivCurl");

        domain.exchangeHalosAuto(get<"c11", "c12", "c13", "c22", "c23", "c33", "divv">(d), std::get<0>(get<"az">(d)),
                                 std::get<0>(get<"du">(d)));
        timer.step("mpi::synchronizeHalos");

        computeAVswitches(first, last, ngmax_, d, domain.box());
        transferToHost(d, first, last, {"alpha"});
        timer.step("AVswitches");

        domain.exchangeHalosAuto(get<"alpha">(d), std::get<0>(get<"az">(d)), std::get<0>(get<"du">(d)));
        timer.step("mpi::synchronizeHalos");

        d.devData.release("divv", "curlv");
        d.devData.acquire("ax", "ay");
        computeMomentumEnergy(first, last, ngmax_, d, domain.box());
        timer.step("MomentumAndEnergy");

        if (d.g != 0.0)
        {
            mHolder_.upsweep(d, domain);
            timer.step("Upsweep");
            mHolder_.traverse(d, domain);
            timer.step("Gravity");
        }
    }

    void step(DomainType& domain, ParticleDataType& d) override
    {
        computeForces(domain, d);

        size_t first = domain.startIndex();
        size_t last  = domain.endIndex();
        transferToHost(d, first, last, {"ax", "ay", "az", "du"});

        //fix for observables:
        transferToHost(d, first, last, {"kx", "xm"});

        computeTimestep(d);
        timer.step("Timestep");
        computePositions(first, last, d, domain.box());
        timer.step("UpdateQuantities");
        updateSmoothingLength(first, last, d, ng0_);
        timer.step("UpdateSmoothingLength");

        timer.stop();
    }

    //! @brief configure the dataset for output by calling EOS again to recover rho and p
    void prepareOutput(ParticleDataType& d, size_t startIndex, size_t endIndex) override
    {
        d.release("c11", "c12", "c13");
        d.acquire("rho", "p", "gradh");
        transferToHost(d, startIndex, endIndex, {"kx", "xm"});
        computeEOS_Impl(startIndex, endIndex, d);
    }

    //! @brief undo output configuration and restore compute configuration
    void finishOutput(ParticleDataType& d) override
    {
        d.release("rho", "p", "gradh");
        d.acquire("c11", "c12", "c13");
    }
};

} // namespace sphexa
