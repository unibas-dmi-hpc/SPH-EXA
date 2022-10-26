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

#include "cstone/fields/particles_get.hpp"
#include "sph/particles_data.hpp"
#include "sph/sph.hpp"

#include "cooling/cooling.hpp"

#include "ipropagator.hpp"
#include "gravity_wrapper.hpp"

namespace sphexa
{

using namespace sph;
using cstone::FieldList;

template<class DomainType, class DataType>
class HydroGrackleProp final : public Propagator<DomainType, DataType>
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

    //! @brief the list of dependent particle fields, these may be used as scratch space during domain sync
    using DependentFields =
        FieldList<"rho", "p", "c", "ax", "ay", "az", "du", "c11", "c12", "c13", "c22", "c23", "c33", "nc">;
    //cooling::CoolingData<T> cd;
    //cooling::ChemistryData<T> chemistry;

public:
    HydroGrackleProp(size_t ngmax, size_t ng0, std::ostream& output, size_t rank/*, const std::string& grackleOptionFile*/)
        : Base(ngmax, ng0, output, rank)//, cd(grackleOptionFile, 1e16, 46400.)
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

        d.devData.setConserved("x", "y", "z", "h", "m");
        d.devData.setDependent("keys");
        std::apply([&d](auto... f) { d.devData.setConserved(f.value...); }, make_tuple(ConservedFields{}));
        std::apply([&d](auto... f) { d.devData.setDependent(f.value...); }, make_tuple(DependentFields{}));
        simData.chem.setConserved(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20);
    }

    void sync(DomainType& domain, DataType& simData) override
    {
        auto& d = simData.hydro;
        if (d.g != 0.0)
        {
            using cooling_fields = FieldList<"Y0", "Y1", "Y2", "Y3", "Y4", "Y5", "Y6",
                    "Y7", "Y8", "Y9", "Y10", "Y11", "Y12",
                    "Y13", "Y14", "Y15", "Y16", "Y17", "Y18", "Y19">;
            std::cout << "sizes: " << get<"x">(d).size() << "\t" << get<"Y0">(simData.chem).size() << std::endl;
            domain.syncGrav(get<"keys">(d), get<"x">(d), get<"y">(d), get<"z">(d), get<"h">(d), get<"m">(d),
                            std::tuple_cat(get<ConservedFields>(d), get<cooling_fields>(simData.chem)), get<DependentFields>(d));

        }
        else
        {
            using cooling_fields = FieldList<"Y0", "Y1", "Y2", "Y3", "Y4", "Y5", "Y6",
                    "Y7", "Y8", "Y9", "Y10", "Y11", "Y12",
                    "Y13", "Y14", "Y15", "Y16", "Y17", "Y18", "Y19">;
            domain.sync(get<"keys">(d), get<"x">(d), get<"y">(d), get<"z">(d), get<"h">(d),
                        std::tuple_cat(std::tie(get<"m">(d)), get<ConservedFields>(d), get<cooling_fields>(simData.chem)), get<DependentFields>(d));

        }
    }
    //Temporary solution
    void step(DomainType& domain, DataType& simData) override
    //void step(DomainType& domain, DataType& simData, cooling::CoolingData<T>& cooling_data) //override
    {
        timer.start();
        sync(domain, simData);
        timer.step("domain::sync");

        auto& d = simData.hydro;
        d.resize(domain.nParticlesWithHalos());
        //simData.chem.resize(domain.nParticlesWithHalos());
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

#pragma omp parallel for schedule(static)
        for (size_t i = first; i < last; i++) {
            using FieldNames = typename cooling::CoolingData<T>::FieldNames;
            T u_cool = d.u[i];
            cooling::cool_particle(simData.chem.cooling_data.global_values,
                                   d.minDt,
                                   d.rho[i],
                                   u_cool,
                                   simData.chem.fields[FieldNames::HI_fraction][i],
                                   simData.chem.fields[FieldNames::HII_fraction][i],
                                   simData.chem.fields[FieldNames::HM_fraction][i],
                                   simData.chem.fields[FieldNames::HeI_fraction][i],
                                   simData.chem.fields[FieldNames::HeII_fraction][i],
                                   simData.chem.fields[FieldNames::HeIII_fraction][i],
                                   simData.chem.fields[FieldNames::H2I_fraction][i],
                                   simData.chem.fields[FieldNames::H2II_fraction][i],
                                   simData.chem.fields[FieldNames::DI_fraction][i],
                                   simData.chem.fields[FieldNames::DII_fraction][i],
                                   simData.chem.fields[FieldNames::HDI_fraction][i],
                                   simData.chem.fields[FieldNames::e_fraction][i],
                                   simData.chem.fields[FieldNames::metal_fraction][i],
                                   simData.chem.fields[FieldNames::volumetric_heating_rate][i],
                                   simData.chem.fields[FieldNames::specific_heating_rate][i],
                                   simData.chem.fields[FieldNames::RT_heating_rate][i],
                                   simData.chem.fields[FieldNames::RT_HI_ionization_rate][i],
                                   simData.chem.fields[FieldNames::RT_HeI_ionization_rate][i],
                                   simData.chem.fields[FieldNames::RT_HeII_ionization_rate][i],
                                   simData.chem.fields[FieldNames::RT_H2_dissociation_rate][i],
                                   simData.chem.fields[FieldNames::H2_self_shielding_length][i]);
            const T du = (u_cool - d.u[i]) / d.minDt;
            //d.dLambda[i] = du;
            d.du[i] += du;
        }
        /*//For debug to check energy conservation
        T total_cooling = 0.;
        for (size_t i = first; i < last; i++) {
            total_cooling += d.dLambda[i];
        }
        std::cout << "Total cooling: " << total_cooling << std::endl;*/
        timer.step("GRACKLE chemistry and cooling");

        computePositions(first, last, d, domain.box());
        timer.step("UpdateQuantities");
        updateSmoothingLength(first, last, d, ng0_);
        timer.step("UpdateSmoothingLength");

        timer.stop();
    }

    void prepareOutput(DataType& simData, size_t first, size_t last, const cstone::Box<T>& box) override
    {
        auto& d = simData.hydro;
        transferToHost(d, first, last, conservedFields());
        transferToHost(d, first, last, {"rho", "p", "c", "du", "ax", "ay", "az", "nc"});
    }
};

} // namespace sphexa
