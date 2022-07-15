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
//#include "..grackle_deps/version.h"
//#include "../../../physics/cooling/include/cooling.hpp"
#include "cooling.hpp"
#include "grackle_deps/version.h"

namespace sphexa
{

using namespace sph;
void setGrackleOption(grackle_options &options, const char *key, const char *value)
{
    if (strstr(key, "grackle_data_file_path")) options.grackle_data_file_path = std::string(value);
    if (strstr(key, "with_radiative_cooling")) options.with_radiative_cooling = atoi(value);
    if (strstr(key, "primordial_chemistry")) options.primordial_chemistry = atoi(value);
    if (strstr(key, "h2_on_dust")) options.h2_on_dust = atoi(value);
    if (strstr(key, "metal_cooling")) options.metal_cooling = atoi(value);
    if (strstr(key, "cmb_temperature_floor")) options.cmb_temperature_floor = atoi(value);
    if (strstr(key, "UVbackground")) options.UVbackground = atoi(value);
    if (strstr(key, "UVbackground_redshift_on")) options.UVbackground_redshift_on = atoi(value);
    if (strstr(key, "UVbackground_redshift_fullon")) options.UVbackground_redshift_fullon = atoi(value);
    if (strstr(key, "UVbackground_redshift_drop")) options.UVbackground_redshift_drop = atoi(value);
    if (strstr(key, "UVbackground_redshift_off")) options.UVbackground_redshift_off = atoi(value);
    if (strstr(key, "Gamma")) options.Gamma = atoi(value);
    if (strstr(key, "three_body_rate")) options.three_body_rate = atoi(value);
    if (strstr(key, "cie_cooling")) options.cie_cooling = atoi(value);
    if (strstr(key, "h2_optical_depth_approximation")) options.h2_optical_depth_approximation = atoi(value);
    if (strstr(key, "photoelectric_heating_rate")) options.photoelectric_heating_rate = atoi(value);
    if (strstr(key, "Compton_xray_heating")) options.Compton_xray_heating = atoi(value);
    if (strstr(key, "LWbackground_intensity")) options.LWbackground_intensity = atoi(value);
    if (strstr(key, "LWbackground_sawtooth_suppression")) options.LWbackground_sawtooth_suppression = atoi(value);
    if (strstr(key, "use_volumetric_heating_rate")) options.use_volumetric_heating_rate = atoi(value);
    if (strstr(key, "use_specific_heating_rate")) options.use_specific_heating_rate = atoi(value);
    if (strstr(key, "use_radiative_transfer")) options.use_radiative_transfer = atoi(value);
    if (strstr(key, "radiative_transfer_coupled_rate_solver")) options.radiative_transfer_coupled_rate_solver = atoi(value);
    if (strstr(key, "radiative_transfer_intermediate_step")) options.radiative_transfer_intermediate_step = atoi(value);
    if (strstr(key, "radiative_transfer_hydrogen_only")) options.radiative_transfer_hydrogen_only = atoi(value);
    if (strstr(key, "H2_self_shielding")) options.H2_self_shielding = atoi(value);
    if (strstr(key, "dust_chemistry")) options.dust_chemistry = atoi(value);


}
grackle_options getGrackleArgumentsFromFile(const std::string path)
{
    grackle_options options;
    FILE *file = fopen(path.c_str(), "r");
    char key[32], value[64];
    while (fscanf(file, "%31s = %63s", key, value) == 2) {
        setGrackleOption(options, key, value);
    }
    std::cout << options.grackle_data_file_path << std::endl;
    return options;
}
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
        typename detail::AccelSwitchType<Acc, MultipoleHolderCpu, MultipoleHolderGpu>::template type<MultipoleType,
                                                                                                     KeyType, T, T, T>;
    MHolder_t mHolder_;

public:
    HydroGrackleProp(size_t ngmax, size_t ng0, std::ostream& output, size_t rank, const std::string& grackleOptionFile)
        : Base(ngmax, ng0, output, rank)
    {
        grackle_options options = getGrackleArgumentsFromFile(grackleOptionFile);
        initGrackle(options);
    }

    void activateFields(ParticleDataType& d) override
    {
        d.setConserved("x", "y", "z", "h", "m", "u", "vx", "vy", "vz", "x_m1", "y_m1", "z_m1", "du_m1", "grackleData");
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

        //Create sample data
        util::array<T, 21> gr_test;
        std::fill(gr_test.begin(), gr_test.end(), 0.);
        gr_test[gr_data::HI_fraction] = 0.76;
        gr_test[gr_data::HeI_fraction] = 0.24;
        gr_test[gr_data::DI_fraction] = 2.0 * 3.4e-5;
        gr_test[gr_data::metal_fraction] = 0.01295;
        for (size_t i = first; i < last; i++) {
            d.grackleData[i] = gr_test;
        }

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


#pragma omp parallel for schedule(static)
        for (size_t i = first; i < last; i++)
        {
            cool_particle(d.minDt,//d.dt,
                          d.rho[i],
                          d.u[i],
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

        }
        timer.stop();
    }
};

} // namespace sphexa
