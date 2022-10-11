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

#include "sph/particles_get.hpp"
#include "sph/sph.hpp"

#include "ipropagator.hpp"
#include "gravity_wrapper.hpp"
#include "cooling.hpp"
namespace sphexa
{

using namespace sph;

template<class DomainType, class ParticleDataType>
class HydroGrackleProp final : public Propagator<DomainType, ParticleDataType>
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
        typename cstone::AccelSwitchType<Acc, MultipoleHolderCpu, MultipoleHolderGpu>::template type<MultipoleType,
                                                                                                     KeyType, T, T, T>;
    MHolder_t mHolder_;
    cooling::cooling_data<T> cd;

    /*! @brief the list of conserved particles fields with values preserved between iterations
     *
     * x, y, z, h and m are automatically considered conserved and must not be specified in this list
     */
    using ConservedFields = FieldList<"u", "vx", "vy", "vz", "x_m1", "y_m1", "z_m1", "du_m1">;

    //! @brief the list of dependent particle fields, these may be used as scratch space during domain sync
    using DependentFields =
        FieldList<"rho", "p", "c", "ax", "ay", "az", "du", "c11", "c12", "c13", "c22", "c23", "c33", "nc">;

public:
    HydroGrackleProp(size_t ngmax, size_t ng0, std::ostream& output, size_t rank, const std::string& grackleOptionFile)
        : Base(ngmax, ng0, output, rank), cd(grackleOptionFile, 1e16, 46400.)
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
        d.setConserved("x", "y", "z", "h", "m", "grackleData", "grackleTemp", "dLambda");
        d.setDependent("keys");

        std::apply([&d](auto... f) { d.setConserved(f.value...); }, make_tuple(ConservedFields{}));
        std::apply([&d](auto... f) { d.setDependent(f.value...); }, make_tuple(DependentFields{}));

        d.devData.setConserved("x", "y", "z", "h", "m", "vx", "vy", "vz");
        d.devData.setDependent("rho", "p", "c", "ax", "ay", "az", "du", "c11", "c12", "c13", "c22", "c23", "c33",
                               "keys");
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

        findNeighborsSfc<T, KeyType>(first, last, ngmax_, d.x, d.y, d.z, d.h, d.codes, d.neighbors, d.neighborsCount,
                                     domain.box());
        d.devData.release("du");
        d.devData.acquire("u");
        transferToDevice(d, first, last, {"vx", "vy", "vz", "u"});
        timer.step("FindNeighbors");

        transferToDevice(d, 0, domain.nParticlesWithHalos(), {"x", "y", "z", "h", "m", "keys"});
        computeDensity(first, last, ngmax_, d, domain.box());
        timer.step("Density");

        //Will have to be moved to a more sensible position
        if (d.iteration == 1) {
            cooling::initGrackleData(d, cd.global_values);
        }
        computeEOS_HydroStd(first, last, d);
        timer.step("EquationOfState");

        domain.exchangeHalosAuto(get<"vx", "vy", "vz", "rho", "p", "c">(d), std::get<0>(get<"ax">(d)),
                                 std::get<0>(get<"ay">(d)));

        timer.step("mpi::synchronizeHalos");
        d.devData.release("u");
        d.devData.acquire("du");

        computeIAD(first, last, ngmax_, d, domain.box());
        timer.step("IAD");

        domain.exchangeHalosAuto(get<"c11", "c12", "c13", "c22", "c23", "c33">(d), std::get<0>(get<"ax">(d)),
                                 std::get<0>(get<"ay">(d)));

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
        transferToHost(d, first, last, {"ax", "ay", "az", "du"});

        computeTimestep(d);
        timer.step("Timestep");

#pragma omp parallel for schedule(static)
        for (size_t i = first; i < last; i++)
        {
            using gr_data = typename cooling::cooling_data<T>::gr_data;
            T u_cool = d.u[i];
            cooling::cool_particle(cd.global_values,
                          d.minDt,//d.dt,
                          d.rho[i],
                          u_cool,
                          d.grackleData[i][gr_data::HI_fraction],
                          d.grackleData[i][gr_data::HII_fraction],
                          d.grackleData[i][gr_data::HM_fraction],
                          d.grackleData[i][gr_data::HeI_fraction],
                          d.grackleData[i][gr_data::HeII_fraction],
                          d.grackleData[i][gr_data::HeIII_fraction],
                          d.grackleData[i][gr_data::H2I_fraction],
                          d.grackleData[i][gr_data::H2II_fraction],
                          d.grackleData[i][gr_data::DI_fraction],
                          d.grackleData[i][gr_data::DII_fraction],
                          d.grackleData[i][gr_data::HDI_fraction],
                          d.grackleData[i][gr_data::e_fraction],
                          d.grackleData[i][gr_data::metal_fraction],
                          d.grackleData[i][gr_data::volumetric_heating_rate],
                          d.grackleData[i][gr_data::specific_heating_rate],
                          d.grackleData[i][gr_data::RT_heating_rate],
                          d.grackleData[i][gr_data::RT_HI_ionization_rate],
                          d.grackleData[i][gr_data::RT_HeI_ionization_rate],
                          d.grackleData[i][gr_data::RT_HeII_ionization_rate],
                          d.grackleData[i][gr_data::RT_H2_dissociation_rate],
                          d.grackleData[i][gr_data::H2_self_shielding_length]);

            const T du = (u_cool - d.u[i]) / d.minDt;
            d.dLambda[i] = du;
            d.du[i] += du;
        }
        //For debug to check energy conservation
        T total_cooling = 0.;
        for (size_t i = first; i < last; i++) {
            total_cooling += d.dLambda[i];
        }
        std::cout << "Total cooling: " << total_cooling << std::endl;
        timer.step("GRACKLE");


        computePositions(first, last, d, domain.box());
        timer.step("UpdateQuantities");
        updateSmoothingLength(first, last, d, ng0_);
        timer.step("UpdateSmoothingLength");

        timer.stop();
    }
};

} // namespace sphexa
