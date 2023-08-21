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
#include "cooling/eos_cooling.hpp"

#include "ipropagator.hpp"
#include "gravity_wrapper.hpp"

namespace sphexa
{

using namespace sph;
using cstone::FieldList;

template<class DomainType, class DataType>
class HydroGrackleProp final : public HydroProp<DomainType, DataType>
{
    using Base = HydroProp<DomainType, DataType>;
    using Base::timer;

    using T       = typename DataType::RealType;
    using KeyType = typename DataType::KeyType;

    cooling::Cooler<T> cooling_data;

    /*! @brief the list of conserved particles fields with values preserved between iterations
     *
     * x, y, z, h and m are automatically considered conserved and must not be specified in this list
     */
    using ConservedFields = FieldList<"u", "vx", "vy", "vz", "x_m1", "y_m1", "z_m1", "du_m1">;

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
    }

    void load(const std::string& initCond, MPI_Comm comm) override
    {
        if (initCond == "evrard-cooling")
        {
            BuiltinWriter attributeSetter(evrardCoolingConstants());
            cooling_data.loadOrStoreAttributes(&attributeSetter);
        }
        else
        {
            std::string path = strBeforeSign(initCond, ":");
            if (std::filesystem::exists(path))
            {
                std::unique_ptr<IFileReader> reader;
                reader = std::make_unique<H5PartReader>(comm);
                reader->setStep(path, -1);
                cooling_data.loadOrStoreAttributes(reader.get());
                reader->closeStep();
            }
            else
                throw std::runtime_error("");
        }
        cooling_data.init(0);
    }

    void save(IFileWriter* writer) override { cooling_data.loadOrStoreAttributes(writer); }

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
        d.treeView = domain.octreeProperties();
    }

    void computeForces(DomainType& domain, DataType& simData)
    {
        size_t first = domain.startIndex();
        size_t last  = domain.endIndex();
        auto&  d     = simData.hydro;

        resizeNeighbors(d, domain.nParticles() * d.ngmax);
        findNeighborsSfc(first, last, d, domain.box());
        timer.step("FindNeighbors");

        computeDensity(first, last, d, domain.box());
        timer.step("Density");
        eos_cooling(first, last, d, simData.chem, cooling_data);
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
            Base::mHolder_.upsweep(d, domain);
            timer.step("Upsweep");
            Base::mHolder_.traverse(d, domain);
            timer.step("Gravity");
        }
    }

    void step(DomainType& domain, DataType& simData) override
    {
        auto& d = simData.hydro;
        timer.start();

        sync(domain, simData);
        // halo exchange for masses, allows for particles with variable masses
        domain.exchangeHalos(std::tie(get<"m">(d)), get<"ax">(d), get<"ay">(d));
        timer.step("domain::sync");

        d.resize(domain.nParticlesWithHalos());

        computeForces(domain, simData);

        size_t first = domain.startIndex();
        size_t last  = domain.endIndex();

        computeTimestep_cool(first, last, d, cooling_data, simData.chem);
        timer.step("Timestep");

#pragma omp parallel for schedule(static)
        for (size_t i = first; i < last; i++)
        {
            T u_old  = d.u[i];
            T u_cool = d.u[i];
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
};

} // namespace sphexa
