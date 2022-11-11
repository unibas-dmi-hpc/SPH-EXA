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

#include "sphnnet/nuclear_net.hpp"

#include "nnet/parameterization/net87/net87.hpp"
#include "nnet/parameterization/net14/net14.hpp"

#include "nnet/parameterization/eos/helmholtz.hpp"
#include "nnet/parameterization/eos/ideal_gas.hpp"

#include "ipropagator.hpp"

namespace sphexa
{

using namespace sph;
using cstone::FieldList;

template<class DomainType, class DataType>
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
    using ConservedFields = FieldList<"temp", "vx", "vy", "vz", "x_m1", "y_m1", "z_m1", "du_m1">;

    //! @brief the list of conserved nuclear fields with values preserved between iterations
    using NuclearConservedFields = FieldList<"temp", "rho", "dt", "m">;

    //! @brief the list of dependent particle fields, these may be used as scratch space during domain sync
    using DependentFields =
        FieldList<"rho", "p", "c", "ax", "ay", "az", "du", "c11", "c12", "c13", "c22", "c23", "c33", "nc">;

    //! @brief the list of dependent nuclear fields, these may be used as scratch space during domain sync
    using NuclearDependentFields = FieldList<"u", "c", "p", "cv", "dpdT">;

    //! @brief number of nuclear species in use
    int numSpecies;
    //! @brief selector for Helmholtz EOS
    bool useHelm;
    //! @brief selector for nuclear reactions, only for debugging
    bool useNuclear;

    //! @brief nuclear network reaction list
    nnet::ReactionList const* reactions;
    //! @brief nuclear network parameterization
    nnet::ComputeReactionRatesFunctor<T> const* construct_rates_BE;
    //! @brief eos
    nnet::EosFunctor<T> const* eos;
    //! @brief Z
    std::vector<T> Z;

    //! @brief possible propagator choices that are handled by this class
    inline static std::array variants{"std-net14",
                                      "std-net86",
                                      "std-net87",
                                      "std-net14-helm",
                                      "std-net86-helm",
                                      "std-net87-helm",
                                      "std-net14-no-nuclear",
                                      "std-net86-no-nuclear",
                                      "std-net87-no-nuclear",
                                      "std-net14-helm-no-nuclear",
                                      "std-net86-helm-no-nuclear",
                                      "std-net87-helm-no-nuclear"};

    //! @brief extract the number of species to use from the propagator choice
    static int getNumSpecies(const std::string& choice)
    {
        std::string ext;
        std::copy_if(choice.begin(), choice.end(), std::back_inserter(ext), [](char x) { return std::isdigit(x); });
        return std::atoi(ext.c_str());
    }

    static bool choiceContains(const std::string& choice, const std::string& patern)
    {
        return choice.find(patern) != std::string::npos;
    }
    static bool hasNuclear(const std::string& choice) { return !choiceContains(choice, "no-nuclear"); }
    static bool hasHelm(const std::string& choice) { return choiceContains(choice, "helm"); }

public:
    static bool isNuclear(const std::string& choice)
    {
        return std::find(variants.begin(), variants.end(), choice) != variants.end();
    }

    NuclearProp(size_t ngmax, size_t ng0, std::ostream& output, size_t rank, const std::string& choice)
        : numSpecies(getNumSpecies(choice))
        , useHelm(hasHelm(choice))
        , useNuclear(hasNuclear(choice))
        , Base(ngmax, ng0, output, rank)
    {
        Z.resize(numSpecies);

        if (numSpecies == 14)
        {
            reactions          = &nnet::net14::reactionList;
            construct_rates_BE = &nnet::net14::computeReactionRates;
            std::copy_n(nnet::net14::constants::Z.begin(), numSpecies, Z.begin());
        }
        else if (numSpecies == 86)
        {
            reactions          = &nnet::net86::reactionList;
            construct_rates_BE = &nnet::net86::computeReactionRates;
            std::copy_n(nnet::net86::constants::Z.begin(), numSpecies, Z.begin());
        }
        else if (numSpecies == 87)
        {
            reactions          = &nnet::net87::reactionList;
            construct_rates_BE = &nnet::net87::computeReactionRates;
            std::copy_n(nnet::net87::constants::Z.begin(), numSpecies, Z.begin());
        }
        else
        {
            throw std::runtime_error("not able to initialize propagator " + std::to_string(numSpecies) +
                                     " nuclear species !");
        }

        if (useHelm) { eos = new nnet::eos::HelmholtzFunctor<T>(Z); }
        else { eos = new nnet::eos::IdealGasFunctor<T>(10.0); }
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

        d.devData.setConserved("x", "y", "z", "h", "m");
        d.devData.setDependent("keys");
        std::apply([&d](auto... f) { d.devData.setConserved(f.value...); }, make_tuple(ConservedFields{}));
        std::apply([&d](auto... f) { d.devData.setDependent(f.value...); }, make_tuple(DependentFields{}));

        auto& n      = simData.nuclearData;
        n.numSpecies = numSpecies;

        //! @brief set nuclear data dependent
        for (int i = 0; i < n.numSpecies; ++i)
        {
            n.setDependent("Y" + std::to_string(i));
            n.devData.setDependent("Y" + std::to_string(i));
        }

        //! @brief Fields accessed in domain sync are not part of extensible lists.
        n.setConserved("node_id", "particle_id" /* TODO */);
        n.setDependent("nuclear_node_id", "nuclear_particle_id" /* TODO */);
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
        auto& n = simData.nuclearData;

        if (d.g != 0.0)
        {
            domain.syncGrav(get<"keys">(d), get<"x">(d), get<"y">(d), get<"z">(d), get<"h">(d), get<"m">(d),
                            std::tuple_cat(std::tie(get<"node_id">(n), get<"particle_id">(n)), get<ConservedFields>(d)),
                            std::tuple_cat(get<DependentFields>(d),
                                           std::tie(get<"nuclear_node_id">(n), get<"nuclear_particle_id">(n)),
                                           get<NuclearDependentFields>(n)));
        }
        else
        {
            domain.sync(get<"keys">(d), get<"x">(d), get<"y">(d), get<"z">(d), get<"h">(d),
                        std::tuple_cat(std::tie(get<"node_id">(n), get<"particle_id">(n)), std::tie(get<"m">(d)),
                                       get<ConservedFields>(d)),
                        std::tuple_cat(get<DependentFields>(d),
                                       std::tie(get<"nuclear_node_id">(n), get<"nuclear_particle_id">(n)),
                                       get<NuclearDependentFields>(n)));
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
    }

private:
    void hydro_step_before(DomainType& domain, DataType& simData)
    {
        auto& d = simData.hydro;
        auto& n = simData.nuclearData;

        size_t domain_size = domain.nParticlesWithHalos();
        d.resize(domain_size);
        n.resizeAttached(domain_size);

        resizeNeighbors(d, domain.nParticles() * ngmax_);
        size_t first = domain.startIndex();
        size_t last  = domain.endIndex();

        transferToHost(d, first, first + 1, {"m"});
        fill(get<"m">(d), 0, first, d.m[first]);
        fill(get<"m">(d), last, domain.nParticlesWithHalos(), d.m[first]);

        sphnnet::syncHydroToNuclear(simData, {"m"});

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

        sphnnet::computeNuclearPartition(first, last, simData);
        timer.step("sphnnet::computeNuclearPartition");

        sphnnet::syncHydroToNuclear(simData, {"rho", "temp" /*, TODO */});
        timer.step("sphnnet::syncHydroToNuclear");
    }

    void nuclear_step(DomainType& domain, DataType& simData)
    {
        auto&  d                   = simData.hydro;
        auto&  n                   = simData.nuclearData;
        size_t n_nuclear_particles = n.Y[0].size();

        sphexa::transferToDevice(n, 0, n_nuclear_particles, {/*"rho", "temp"*/});
        timer.step("transferToDevice");

        if (useNuclear)
        {
            sphnnet::computeNuclearReactions(n, 0, n_nuclear_particles, d.minDt, d.minDt_m1, *reactions,
                                             *construct_rates_BE, *eos,
                                             /*considering expansion:*/ false);
            timer.step("sphnnet::computeNuclearReactions");
        }

        if (useHelm)
        {
            sphnnet::computeHelmEOS(n, 0, n_nuclear_particles, Z);
            timer.step("sphnnet::computeHelmEOS");
        }
    }

    void nuclear_sync_after(DomainType& domain, DataType& simData)
    {
        sphnnet::syncNuclearToHydro(simData, {"temp" /*, TODO */});
        if (useHelm) { sphnnet::syncNuclearToHydro(simData, {"c", "p" /*, "cv", "u", "dpdT", TODO */}); }
        timer.step("sphnnet::syncNuclearToHydro");
    }

    void hydro_step_after(DomainType& domain, DataType& simData)
    {
        auto&  d     = simData.hydro;
        size_t first = domain.startIndex();
        size_t last  = domain.endIndex();

        if (!useHelm)
        {
            computeEOS_HydroStd(first, last, d);
            timer.step("EquationOfState");
        }

        domain.exchangeHalos(get<"vx", "vy", "vz", "rho", "p", "c", "temp">(d), get<"ax">(d), get<"ay">(d));
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
