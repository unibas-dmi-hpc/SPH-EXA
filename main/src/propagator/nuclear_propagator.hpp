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
 * @brief propagator for nuclear networks. TODO
 * @author Joseph Touzet <joseph.touzet@ens-paris-saclay.fr>
 */

#include "nnet/sphexa/nuclear-net.hpp"

#include "ipropagator.hpp"

namespace sphexa
{

using namespace sph;
using cstone::FieldList;

template<class DomainType, class DataType, const nnet::reaction_list& reactions>
class NuclearProp final : public Propagator<DomainType, DataType>
{
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
    using ConservedFields = FieldList<"u", "vx", "vy", "vz", "x_m1", "y_m1", "z_m1", "du_m1">;

    //! @brief the list of conserved nuclear fields with values preserved between iterations
    using NuclearConservedFields = FieldList</* TODO */>;

    //! @brief the list of dependent particle fields, these may be used as scratch space during domain sync
    using DependentFields =
        FieldList<"rho", "p", "c", "ax", "ay", "az", "du", "c11", "c12", "c13", "c22", "c23", "c33", "nc">;

    //! @brief the list of dependent nuclear fields, these may be used as scratch space during domain sync
    using NuclearDependentFields = FieldList</* TODO */>;

public:
    NuclearProp(size_t ngmax, size_t ng0, std::ostream& output, size_t rank)
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

        //! @brief Fields accessed in domain sync are not part of extensible lists.
        d.setConserved("x", "y", "z", "h", "m");
        d.setDependent("keys");
        std::apply([&d](auto... f) { d.setConserved(f.value...); }, make_tuple(ConservedFields{}));
        std::apply([&d](auto... f) { d.setDependent(f.value...); }, make_tuple(DependentFields{}));

        d.devData.setConserved(/* TODO */);
        d.devData.setDependent(/* TODO */);
        std::apply([&d](auto... f) { d.devData.setConserved(f.value...); }, make_tuple(ConservedFields{}));
        std::apply([&d](auto... f) { d.devData.setDependent(f.value...); }, make_tuple(DependentFields{}));

        auto& n = simData.nuclearData;

        //! @brief Fields accessed in domain sync are not part of extensible lists.
        n.setConserved(/* TODO */);
        n.setDependent(/* TODO */);
        std::apply([&n](auto... f) { n.setConserved(f.value...); }, make_tuple(NuclearConservedFields{}));
        std::apply([&n](auto... f) { n.setDependent(f.value...); }, make_tuple(NuclearDependentFields{}));

        n.devData.setConserved(/* TODO */);
        n.devData.setDependent(/* TODO */);
        std::apply([&n](auto... f) { n.devData.setConserved(f.value...); }, make_tuple(NuclearConservedFields{}));
        std::apply([&n](auto... f) { n.devData.setDependent(f.value...); }, make_tuple(NuclearDependentFields{}));
    }

    void sync(DomainType& domain, DataType& simData) override
    {
        auto& d = simData.hydro;
        if (d.g != 0.0)
        {
            domain.syncGrav(get<"keys">(d), get<"x">(d), get<"y">(d), get<"z">(d), get<"h">(d), get<"m">(d),
                            get<ConservedFields>(d), get<DependentFields>(d));
        }
        else
        {
            domain.sync(get<"keys">(d), get<"x">(d), get<"y">(d), get<"z">(d), get<"h">(d),
                        std::tuple_cat(std::tie(get<"m">(d)), get<ConservedFields>(d)), get<DependentFields>(d));
        }
    }

    void step(DomainType& domain, DataType& simData) override
    {
        timer.start();
        sync(domain, simData);
        timer.step("domain::sync");

        hydro_step_before(domain, simData);

        nuclear_sync_before(domain, simData);
        nuclear_step(domain, simData);
        nuclear_sync_after(domain, simData);

        hydro_step_after(domain, simData);
    }

    void prepareOutput(DataType& simData, size_t first, size_t last, const cstone::Box<T>& box) override
    {
        auto& d = simData.hydro;
        transferToHost(d, first, last, conservedFields());
        transferToHost(d, first, last, {"rho", "p", "c", "du", "ax", "ay", "az", "nc"});

        auto& n = simData.nuclearData;
        /* TODO */
    }

private:
    void hydro_step_before(DomainType& domain, DataType& simData)
    {
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

        computeDensity(first, last, ngmax_, d, domain.box());
        timer.step("Density");
    }

    void hydro_step_after(DomainType& domain, DataType& simData)
    {
        auto&  d     = simData.hydro;
        size_t first = domain.startIndex();
        size_t last  = domain.endIndex();

        computeEOS_HydroStd(first, last, d);
        timer.step("EquationOfState");

        domain.exchangeHalos(get<"vx", "vy", "vz", "rho", "p", "c">(d), get<"ax">(d), get<"ay">(d));

        timer.step("mpi::synchronizeHalos");

        computeIAD(first, last, ngmax_, d, domain.box());
        timer.step("IAD");

        domain.exchangeHalos(get<"c11", "c12", "c13", "c22", "c23", "c33">(d), get<"ax">(d), get<"ay">(d));

        timer.step("mpi::synchronizeHalos");

        computeMomentumEnergySTD(first, last, ngmax_, d, domain.box());
        timer.step("MomentumEnergyIAD");

        if (d.g != 0.0)
        {
            mHolder_.upsweep(d, domain);
            timer.step("Upsweep");
            mHolder_.traverse(d, domain);
            timer.step("Gravity");
        }

        computeTimestep(d);
        timer.step("Timestep");
        computePositions(first, last, d, domain.box());
        timer.step("UpdateQuantities");
        updateSmoothingLength(first, last, d, ng0_);
        timer.step("UpdateSmoothingLength");

        timer.stop();
    }

    void nuclear_step(DomainType& domain, DataType& simData) { /* TODO */ }

    void nuclear_sync_before(DomainType& domain, DataType& simData) { /* TODO */ }

    void nuclear_sync_after(DomainType& domain, DataType& simData) { /* TODO */ }
};

} // namespace sphexa