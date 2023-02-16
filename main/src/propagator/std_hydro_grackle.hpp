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

#include "cooling/cooler.hpp"

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
    using Base::timer;

    using T             = typename DataType::RealType;
    using KeyType       = typename DataType::KeyType;
    using Tmass         = typename DataType::HydroData::Tmass;
    using MultipoleType = ryoanji::CartesianQuadrupole<Tmass>;

    using Acc = typename DataType::AcceleratorType;
    using MHolder_t =
        typename cstone::AccelSwitchType<Acc, MultipoleHolderCpu,
                                         MultipoleHolderGpu>::template type<MultipoleType, KeyType, T, T, Tmass, T, T>;
    MHolder_t          mHolder_;
    cooling::Cooler<T> cooling_data;

    /*! @brief the list of conserved particles fields with values preserved between iterations
     *
     * x, y, z, h and m are automatically considered conserved and must not be specified in this list
     */
    using ConservedFields = FieldList<"temp", "vx", "vy", "vz", "x_m1", "y_m1", "z_m1", "du_m1">;

    //! @brief the list of dependent particle fields, these may be used as scratch space during domain sync
    using DependentFields =
        FieldList<"rho", "p", "c", "ax", "ay", "az", "du", "c11", "c12", "c13", "c22", "c23", "c33", "nc">;

    using CoolingFields =
        FieldList<"HI_fraction", "HII_fraction", "HM_fraction", "HeI_fraction", "HeII_fraction", "HeIII_fraction",
                  "H2I_fraction", "H2II_fraction", "DI_fraction", "DII_fraction", "HDI_fraction", "e_fraction",
                  "metal_fraction", "volumetric_heating_rate", "specific_heating_rate", "RT_heating_rate",
                  "RT_HI_ionization_rate", "RT_HeI_ionization_rate", "RT_HeII_ionization_rate",
                  "RT_H2_dissociation_rate", "H2_self_shielding_length">;

public:
    HydroGrackleProp(std::ostream& output, size_t rank)
        : Base(output, rank)
    {
        constexpr float                 ms_sim = 1e16;
        constexpr float                 kp_sim = 46400.;
        std::map<std::string, std::any> grackleOptions;
        grackleOptions["use_grackle"]            = 1;
        grackleOptions["with_radiative_cooling"] = 1;
        grackleOptions["dust_chemistry"]         = 0;
        grackleOptions["metal_cooling"]          = 0;
        grackleOptions["UVbackground"]           = 0;
        cooling_data.init(ms_sim, kp_sim, 0, grackleOptions, std::nullopt);
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
        std::apply([&simData](auto... f) { simData.chem.setConserved(f.value...); }, make_tuple(CoolingFields{}));

        d.devData.setConserved("x", "y", "z", "h", "m");
        d.devData.setDependent("keys");
        std::apply([&d](auto... f) { d.devData.setConserved(f.value...); }, make_tuple(ConservedFields{}));
        std::apply([&d](auto... f) { d.devData.setDependent(f.value...); }, make_tuple(DependentFields{}));
    }

    void sync(DomainType& domain, DataType& simData) override
    {
        auto& d = simData.hydro;
        if (d.g != 0.0)
        {
            std::cout << "sizes: " << get<"x">(d).size() << "\t" << get<"HI_fraction">(simData.chem).size()
                      << std::endl;
            domain.syncGrav(get<"keys">(d), get<"x">(d), get<"y">(d), get<"z">(d), get<"h">(d), get<"m">(d),
                            std::tuple_cat(get<ConservedFields>(d), get<CoolingFields>(simData.chem)),
                            get<DependentFields>(d));
        }
        else
        {
            domain.sync(
                get<"keys">(d), get<"x">(d), get<"y">(d), get<"z">(d), get<"h">(d),
                std::tuple_cat(std::tie(get<"m">(d)), get<ConservedFields>(d), get<CoolingFields>(simData.chem)),
                get<DependentFields>(d));
        }
        d.treeView = domain.octreeNsViewAcc();
    }

    void step(DomainType& domain, DataType& simData) override
    {
        timer.start();
        sync(domain, simData);
        timer.step("domain::sync");

        auto& d = simData.hydro;
        d.resize(domain.nParticlesWithHalos());
        resizeNeighbors(d, domain.nParticles() * d.ngmax);
        size_t first = domain.startIndex();
        size_t last  = domain.endIndex();

        transferToHost(d, first, first + 1, {"m"});
        fill(get<"m">(d), 0, first, d.m[first]);
        fill(get<"m">(d), last, domain.nParticlesWithHalos(), d.m[first]);

        findNeighborsSfc(first, last, d, domain.box());
        timer.step("FindNeighbors");

        computeDensity(first, last, d, domain.box());
        timer.step("Density");

        computeEOS_HydroStd(first, last, d);
        timer.step("EquationOfState");
        domain.exchangeHalos(get<"vx", "vy", "vz", "rho", "p", "c">(d), get<"ax">(d), get<"ay">(d));

        timer.step("mpi::synchronizeHalos");

        computeIAD(first, last, d, domain.box());
        timer.step("IAD");

        domain.exchangeHalos(get<"c11", "c12", "c13", "c22", "c23", "c33">(d), get<"ax">(d), get<"ay">(d));

        timer.step("mpi::synchronizeHalos");

        computeMomentumEnergySTD(first, last, d, domain.box());
        timer.step("MomentumEnergyIAD");

        if (d.g != 0.0)
        {
            mHolder_.upsweep(d, domain);
            timer.step("Upsweep");
            mHolder_.traverse(d, domain);
            timer.step("Gravity");
        }

        computeTimestep(first, last, d);
        timer.step("Timestep");

#pragma omp parallel for schedule(static)
        for (size_t i = first; i < last; i++)
        {
            bool haveMui = !d.mui.empty();
            T    cv      = idealGasCv(haveMui ? d.mui[i] : d.muiConst, d.gamma);

            T u_old  = cv * d.temp[i];
            T u_cool = u_old;
            cooling_data.cool_particle(
                d.minDt, d.rho[i], u_cool, get<"HI_fraction">(simData.chem)[i], get<"HII_fraction">(simData.chem)[i],
                get<"HM_fraction">(simData.chem)[i], get<"HeI_fraction">(simData.chem)[i],
                get<"HeII_fraction">(simData.chem)[i], get<"HeIII_fraction">(simData.chem)[i],
                get<"H2I_fraction">(simData.chem)[i], get<"H2II_fraction">(simData.chem)[i],
                get<"DI_fraction">(simData.chem)[i], get<"DII_fraction">(simData.chem)[i],
                get<"HDI_fraction">(simData.chem)[i], get<"e_fraction">(simData.chem)[i],
                get<"metal_fraction">(simData.chem)[i], get<"volumetric_heating_rate">(simData.chem)[i],
                get<"specific_heating_rate">(simData.chem)[i], get<"RT_heating_rate">(simData.chem)[i],
                get<"RT_HI_ionization_rate">(simData.chem)[i], get<"RT_HeI_ionization_rate">(simData.chem)[i],
                get<"RT_HeII_ionization_rate">(simData.chem)[i], get<"RT_H2_dissociation_rate">(simData.chem)[i],
                get<"H2_self_shielding_length">(simData.chem)[i]);
            const T du = (u_cool - u_old) / d.minDt;
            d.du[i] += du;
        }
        timer.step("GRACKLE chemistry and cooling");

        computePositions(first, last, d, domain.box());
        timer.step("UpdateQuantities");
        updateSmoothingLength(first, last, d);
        timer.step("UpdateSmoothingLength");

        timer.stop();
    }

    void saveFields(IFileWriter* writer, size_t first, size_t last, DataType& simData,
                    const cstone::Box<T>& box) override
    {
        auto&            d             = simData.hydro;
        auto             fieldPointers = d.data();
        std::vector<int> outputFields  = d.outputFieldIndices;

        auto output = [&]()
        {
            for (int i = int(outputFields.size()) - 1; i >= 0; --i)
            {
                int fidx = outputFields[i];
                if (d.isAllocated(fidx))
                {
                    int column = std::find(d.outputFieldIndices.begin(), d.outputFieldIndices.end(), fidx) -
                                 d.outputFieldIndices.begin();
                    transferToHost(d, first, last, {d.fieldNames[fidx]});
                    std::visit([writer, c = column, key = d.fieldNames[fidx]](auto field)
                               { writer->writeField(key, field->data(), c); },
                               fieldPointers[fidx]);
                    outputFields.erase(outputFields.begin() + i);
                }
            }
        };

        output();
    }
};

} // namespace sphexa
