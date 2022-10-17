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

#include "nnet/parameterization/net87/net87.hpp"
#include "nnet/parameterization/net14/net14.hpp"

#include "nnet/parameterization/eos/helmholtz.hpp"
#include "nnet/parameterization/eos/ideal_gas.hpp"

#include "ipropagator.hpp"

namespace sphexa
{

using namespace sph;
using cstone::FieldList;

template<class DomainType, class DataType, int n_species, bool use_helm>
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
    using NuclearConservedFields = FieldList<"temp", "rho", "dt", "node_id", "particle_id">;

    //! @brief the list of dependent particle fields, these may be used as scratch space during domain sync
    using DependentFields =
        FieldList<"rho", "p", "c", "ax", "ay", "az", "du", "c11", "c12", "c13", "c22", "c23", "c33", "nc">;

    //! @brief the list of dependent nuclear fields, these may be used as scratch space during domain sync
    using NuclearDependentFields =
        FieldList<"m", "u", "c", "p", "cv", "dpdT", "nuclear_node_id", "nuclear_particle_id", "rho_m1" /* TODO */>;

    //! @brief nuclear network reaction list
    nnet::reaction_list const* reactions;
    //! @brief nuclear network parameterization
    nnet::compute_reaction_rates_functor<T> const* construct_rates_BE;
    //! @brief eos
    nnet::eos_functor<T> const* eos;
    //! @brief Z
    std::vector<T> Z;

public:
    NuclearProp(size_t ngmax, size_t ng0, std::ostream& output, size_t rank)
        : Base(ngmax, ng0, output, rank)
    {
        if (n_species == 14)
        {
            reactions          = &nnet::net14::reaction_list;
            construct_rates_BE = &nnet::net14::compute_reaction_rates;

            Z.resize(14);
            for (int i = 0; i < 14; ++i)
            {
                Z[i] = nnet::net14::constants::Z[i];
            }
        }
        else if (n_species == 86)
        {
            reactions          = &nnet::net86::reaction_list;
            construct_rates_BE = &nnet::net86::compute_reaction_rates;

            Z.resize(86);
            for (int i = 0; i < 86; ++i)
            {
                Z[i] = nnet::net86::constants::Z[i];
            }
        }
        else if (n_species == 87)
        {
            reactions          = &nnet::net87::reaction_list;
            construct_rates_BE = &nnet::net87::compute_reaction_rates;

            Z.resize(87);
            for (int i = 0; i < 87; ++i)
            {
                Z[i] = nnet::net87::constants::Z[i];
            }
        }
        else
        {
            throw std::runtime_error("not able to initialize propagator " + std::to_string(n_species) +
                                     " nuclear species !");
        }

        if (use_helm) { eos = new nnet::eos::helmholtz_functor<T>(Z); }
        else { eos = new nnet::eos::ideal_gas_functor<T>(10.0); }
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

        auto& n      = simData.nuclearData;
        n.numSpecies = n_species;

        //! @brief set nuclear data dependent
        for (int i = 0; i < n.numSpecies; ++i)
        {
            n.setDependent("Y" + std::to_string(i));
            n.devData.setDependent("Y" + std::to_string(i));
        }

        //! @brief Fields accessed in domain sync are not part of extensible lists.
        n.setConserved("node_id", "particle_id", "rho", "temp", "dt" /* TODO */);
        n.setDependent("u", "c", "p", "cv", "dpdT", "rho_m1", "nuclear_node_id", "nuclear_particle_id" /* TODO */);
        std::apply([&n](auto... f) { n.setConserved(f.value...); }, make_tuple(NuclearConservedFields{}));
        std::apply([&n](auto... f) { n.setDependent(f.value...); }, make_tuple(NuclearDependentFields{}));

        n.devData.setConserved("rho", "temp", "dt" /* TODO */);
        n.devData.setDependent("u", "c", "p", "cv", "dpdT" /* TODO */);
        std::apply([&n](auto... f) { n.devData.setConserved(f.value...); }, make_tuple(NuclearConservedFields{}));
        std::apply([&n](auto... f) { n.devData.setDependent(f.value...); }, make_tuple(NuclearDependentFields{}));
    }

    void sync(DomainType& domain, DataType& simData) override
    {
        auto& d = simData.hydro;
        auto& n = simData.nuclearData;

        if (d.g != 0.0)
        {
            domain.syncGrav(get<"keys">(d), get<"x">(d), get<"y">(d), get<"z">(d), get<"h">(d), get<"m">(d),
                            std::tuple_cat(std::tie(get<"node_id">(n), get<"particle_id">(n)), get<ConservedFields>(d)),
                            std::tuple_cat(get<DependentFields>(d), get<NuclearDependentFields>(n)));
        }
        else
        {
            domain.sync(get<"keys">(d), get<"x">(d), get<"y">(d), get<"z">(d), get<"h">(d),
                        std::tuple_cat(std::tie(get<"node_id">(n), get<"particle_id">(n)), std::tie(get<"m">(d)),
                                       get<ConservedFields>(d)),
                        std::tuple_cat(get<DependentFields>(d), get<NuclearDependentFields>(n)));
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

        auto&  n                   = simData.nuclearData;
        size_t n_nuclear_particles = n.Y[0].size();

        int node_id = 0;
#ifdef USE_MPI
        MPI_Comm_rank(simData.comm, &node_id);
#endif

        std::iota(getHost<"nuclear_particle_id">(n).begin(),
                  getHost<"nuclear_particle_id">(n).begin() + n_nuclear_particles, 0);
        std::fill(getHost<"nuclear_node_id">(n).begin(), getHost<"nuclear_node_id">(n).begin() + n_nuclear_particles,
                  node_id);

        sphexa::sphnnet::syncHydroToNuclear(simData, {"m"});
    }

private:
    void hydro_step_before(DomainType& domain, DataType& simData)
    {
        auto& d = simData.hydro;
        auto& n = simData.nuclearData;

        d.resize(domain.nParticlesWithHalos());
        n.resize_hydro(domain.nParticlesWithHalos());

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

    void nuclear_sync_before(DomainType& domain, DataType& simData)
    {
        auto& d = simData.hydro;
        auto& n = simData.nuclearData;

        size_t first = domain.startIndex();
        size_t last  = domain.endIndex();

        sphexa::sphnnet::computePartition(first, last, n);
        timer.step("sphnnet::computePartition");

        sphexa::sphnnet::syncHydroToNuclear(simData, {"rho" /*, "temp", */ /* TODO */});
        timer.step("sphnnet::syncHydroToNuclear");
    }

    void nuclear_step(DomainType& domain, DataType& simData)
    {
        auto&  d                   = simData.hydro;
        auto&  n                   = simData.nuclearData;
        size_t n_nuclear_particles = n.Y[0].size();

        sphexa::transferToDevice(n, 0, n_nuclear_particles, {/*"rho_m1", "rho", "temp"*/});
        timer.step("transferToDevice");

        sphexa::sphnnet::computeNuclearReactions(n, 0, n_nuclear_particles, d.minDt, d.minDt_m1, *reactions,
                                                 *construct_rates_BE, *eos,
                                                 /*considering expansion:*/ false);
        timer.step("sphnnet::computeNuclearReactions");

        if (use_helm)
        {
            sphexa::sphnnet::computeHelmEOS(n, 0, n_nuclear_particles, Z);
            timer.step("sphnnet::computeHelmEOS");
        }
    }

    void nuclear_sync_after(DomainType& domain, DataType& simData)
    {
        sphexa::sphnnet::syncNuclearToHydro(simData, {/*, "temp", */ /* TODO */});

        if (use_helm) { sphexa::sphnnet::syncNuclearToHydro(simData, {"u", "c", "p" /*, "cv", "dpdT"*/}); }
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
};

} // namespace sphexa
