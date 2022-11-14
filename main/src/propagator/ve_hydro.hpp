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

#include "cstone/fields/particles_get.hpp"
#include "sph/particles_data.hpp"
#include "sph/sph.hpp"

#include "ipropagator.hpp"
#include "gravity_wrapper.hpp"

namespace sphexa
{

using namespace sph;
using cstone::FieldList;

template<class DomainType, class DataType>
class HydroVeProp : public Propagator<DomainType, DataType>
{
protected:
    using Base = Propagator<DomainType, DataType>;
    using Base::ng0_;
    using Base::ngmax_;
    using Base::timer;

    using T             = typename DataType::RealType;
    using KeyType       = typename DataType::KeyType;
    using Tmass         = typename DataType::HydroData::Tmass;
    using MultipoleType = ryoanji::CartesianQuadrupole<Tmass>;

    using Acc = typename DataType::AcceleratorType;
    using MHolder_t =
        typename cstone::AccelSwitchType<Acc, MultipoleHolderCpu,
                                         MultipoleHolderGpu>::template type<MultipoleType, KeyType, T, T, Tmass, T, T>;

    MHolder_t mHolder_;

    /*! @brief the list of conserved particles fields with values preserved between iterations
     *
     * x, y, z, h and m are automatically considered conserved and must not be specified in this list
     */
    using ConservedFields = FieldList<"temp", "vx", "vy", "vz", "x_m1", "y_m1", "z_m1", "du_m1", "alpha">;

    //! @brief the list of dependent particle fields, these may be used as scratch space during domain sync
    using DependentFields = FieldList<"prho", "c", "ax", "ay", "az", "du", "c11", "c12", "c13", "c22", "c23", "c33",
                                      "xm", "kx", "divv", "curlv", "nc">;

    //! @brief not all dependent CPU fields are simultaneously needed on the GPU
    using DependentFieldsGpu =
        FieldList<"prho", "c", "ax", "ay", "az", "du", "c11", "c12", "c13", "c22", "c23", "c33", "xm", "kx", "nc">;

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

    void activateFields(DataType& simData) override
    {
        auto& d = simData.hydro;
        //! @brief Fields accessed in domain sync (x,y,z,h,m,keys) are not part of extensible lists.
        d.setConserved("x", "y", "z", "h", "m");
        d.setDependent("keys");
        std::apply([&d](auto... f) { d.setConserved(f.value...); }, make_tuple(ConservedFields{}));
        std::apply([&d](auto... f) { d.setDependent(f.value...); }, make_tuple(DependentFields{}));

        d.devData.setConserved("x", "y", "z", "h", "m");
        d.devData.setDependent("keys");
        std::apply([&d](auto... f) { d.devData.setConserved(f.value...); }, make_tuple(ConservedFields{}));
        std::apply([&d](auto... f) { d.devData.setDependent(f.value...); }, make_tuple(DependentFieldsGpu{}));
    }

    void sync(DomainType& domain, DataType& simData) override
    {
        auto& d = simData.hydro;
        if (d.g != 0.0)
        {
            domain.syncGrav(get<"keys">(d), get<"x">(d), get<"y">(d), get<"z">(d), get<"h">(d), get<"m">(d),
                            get<ConservedFields>(d), get<DependentFieldsGpu>(d));
        }
        else
        {
            domain.sync(get<"keys">(d), get<"x">(d), get<"y">(d), get<"z">(d), get<"h">(d),
                        std::tuple_cat(std::tie(get<"m">(d)), get<ConservedFields>(d)), get<DependentFieldsGpu>(d));
        }
    }

    void computeForces(DomainType& domain, DataType& simData)
    {
        timer.start();
        sync(domain, simData);
        timer.step("domain::sync");

        auto& d = simData.hydro;
        d.resize(domain.nParticlesWithHalos());
        resizeNeighbors(d, domain.nParticles() * ngmax_);
        size_t first = domain.startIndex();
        size_t last  = domain.endIndex();

        transferToHost(d, first, first + 1, {"m"});
        fill(get<"m">(d), 0, first, d.m[first]);
        fill(get<"m">(d), last, domain.nParticlesWithHalos(), d.m[first]);

        findNeighborsSfc<T, KeyType>(first, last, ngmax_, d.x, d.y, d.z, d.h, d.keys, d.neighbors, d.nc, domain.box());
        timer.step("FindNeighbors");

        computeXMass(first, last, ngmax_, d, domain.box());
        timer.step("XMass");
        domain.exchangeHalos(std::tie(get<"xm">(d)), get<"ax">(d), get<"ay">(d));
        timer.step("mpi::synchronizeHalos");

        d.release("ax");
        d.acquire("gradh");
        d.devData.release("ax");
        d.devData.acquire("gradh");
        computeVeDefGradh(first, last, ngmax_, d, domain.box());
        timer.step("Normalization & Gradh");

        computeEOS(first, last, d);
        timer.step("EquationOfState");

        domain.exchangeHalos(get<"vx", "vy", "vz", "prho", "c", "kx">(d), get<"gradh">(d), get<"ay">(d));
        timer.step("mpi::synchronizeHalos");

        d.release("gradh");
        d.acquire("ax");
        d.devData.release("gradh", "ay");
        d.devData.acquire("divv", "curlv");
        computeIadDivvCurlv(first, last, ngmax_, d, domain.box());
        timer.step("IadVelocityDivCurl");

        domain.exchangeHalos(get<"c11", "c12", "c13", "c22", "c23", "c33", "divv">(d), get<"az">(d), get<"du">(d));
        timer.step("mpi::synchronizeHalos");

        computeAVswitches(first, last, ngmax_, d, domain.box());
        timer.step("AVswitches");

        domain.exchangeHalos(std::tie(get<"alpha">(d)), get<"az">(d), get<"du">(d));
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

    void step(DomainType& domain, DataType& simData) override
    {
        computeForces(domain, simData);

        auto&  d     = simData.hydro;
        size_t first = domain.startIndex();
        size_t last  = domain.endIndex();

        computeTimestep(d);
        timer.step("Timestep");
        computePositions(first, last, d, domain.box());
        timer.step("UpdateQuantities");
        updateSmoothingLength(first, last, d, ng0_);
        timer.step("UpdateSmoothingLength");

        timer.stop();
    }

    //! @brief configure the dataset for output by calling EOS again to recover rho and p
    void prepareOutput(DataType& simData, size_t startIndex, size_t endIndex, const cstone::Box<T>& box) override
    {
        auto& d = simData.hydro;
        transferToHost(d, startIndex, endIndex, conservedFields());
        transferToHost(d, startIndex, endIndex, {"ax", "ay", "az", "du", "kx", "xm", "nc"});
        d.release("c11", "c12", "c13");
        d.acquire("rho", "p", "gradh");
        computeEOS_Impl(startIndex, endIndex, d);

        if constexpr (cstone::HaveGpu<typename DataType::AcceleratorType>{})
        {
            const auto& outFields = d.outputFieldNames;
            bool        outDiv    = std::find(outFields.begin(), outFields.end(), "divv") != outFields.end();
            bool        outCurl   = std::find(outFields.begin(), outFields.end(), "curlv") != outFields.end();
            if (outDiv || outCurl)
            {
                d.devData.release("ax", "ay");
                d.devData.acquire("divv", "curlv");
                computeIadDivvCurlv(startIndex, endIndex, ngmax_, d, box);
                transferToHost(d, startIndex, endIndex, {"divv", "curlv"});
                d.devData.release("divv", "curlv");
                d.devData.acquire("ax", "ay");
            }
        }
    }

    //! @brief undo output configuration and restore compute configuration
    void finishOutput(DataType& simData) override
    {
        auto& d = simData.hydro;
        d.release("rho", "p", "gradh");
        d.acquire("c11", "c12", "c13");
    }
};

} // namespace sphexa
