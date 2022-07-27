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

/*// @file
 * @brief Turbulence simulation data initialization
 *
 * @author Axel Sanz <axelsanzlechuga@gmail.com>
 */

#pragma once

#include <map>
#include <cmath>
#include <algorithm>

#include "cstone/sfc/box.hpp"

#include "isim_init.hpp"
#include "grid.hpp"

#include "sph/hydro_turb/st_ounoise.hpp"

namespace sphexa
{

template<class Dataset, class T>
void stir_init(Dataset& d, T Lx, T Ly, T Lz, size_t st_maxmodes, T st_energy, T st_stirmax, T st_stirmin, size_t ndim,
               long int& st_seed, size_t st_spectform, T st_power_law_exp, T st_angles_exp)
{
    size_t ikx, iky, ikz, st_tot_nmodes;
    T      kx, ky, kz, k, kc, amplitude, parab_prefact;

    size_t  nang, ikmin, ikmax;
    T       rand, phi, theta;
    const T twopi = 2.0 * M_PI;

    d.stOUvar = std::sqrt(st_energy / d.stDecay);

    // prefactor for amplitude normalistion to 1 at kc = 0.5*(st_stirmin+st_stirmax)
    parab_prefact = -4.0 / ((st_stirmax - st_stirmin) * (st_stirmax - st_stirmin));

    // characteristic k for scaling the amplitude below
    kc = st_stirmin;
    if (st_spectform == 1) { kc = 0.5 * (st_stirmin + st_stirmax); }

    // this makes the rms force const irrespective of the solenoidal weight
    d.stSolWeightNorm = std::sqrt(3.0) * std::sqrt(3.0 / T(ndim)) /
                        std::sqrt(1.0 - 2.0 * d.stSolWeight + T(ndim) * d.stSolWeight * d.stSolWeight);

    size_t ikxmin = 0;
    size_t ikymin = 0;
    size_t ikzmin = 0;

    size_t ikxmax = 256;
    size_t ikymax = (ndim > 1) ? 256 : 0;
    size_t ikzmax = (ndim > 2) ? 256 : 0;

    // determine the number of required modes (in case of full sampling)
    d.stNModes = 0;
    for (ikx = ikxmin; ikx <= ikxmax; ikx++)
    {
        kx = twopi * ikx / Lx;
        for (iky = ikymin; iky <= ikymax; iky++)
        {
            ky = twopi * iky / Ly;
            for (ikz = ikzmin; ikz <= ikzmax; ikz++)
            {
                kz = twopi * ikz / Lz;
                k  = std::sqrt(kx * kx + ky * ky + kz * kz);
                if (k >= st_stirmin && k <= st_stirmax)
                {
                    d.stNModes += 1;
                    if (ndim > 1) { d.stNModes += 1; }
                    if (ndim > 2) { d.stNModes += 2; }
                }
            }
        }
    }
    st_tot_nmodes = d.stNModes;

    d.stNModes = -1;

    if (st_spectform != 2)
    {
        std::cout << "Generating " << st_tot_nmodes << " driving modes..." << std::endl;
        // for band and parabolic spectrum, use the standard full sampling
        // loop over all kx, ky, kz to generate driving modes
        for (ikx = ikxmin; ikx <= ikxmax; ikx++)
        {
            kx = twopi * ikx / Lx;
            for (iky = ikymin; iky <= ikymax; iky++)
            {
                ky = twopi * iky / Ly;
                for (ikz = ikzmin; ikz <= ikzmax; ikz++)
                {
                    kz = twopi * ikz / Lz;
                    k  = std::sqrt(kx * kx + ky * ky + kz * kz);

                    if ((k >= st_stirmin) && (k <= st_stirmax))
                    {

                        if ((d.stNModes + 1 + std::pow(2, ndim - 1)) > st_maxmodes)
                        {
                            std::cout << "init_stir:  number of modes: = " << d.stNModes + 1
                                      << " maxstirmodes = " << st_maxmodes << std::endl;
                            std::cout << "Too many stirring modes" << std::endl;
                            break;
                        }

                        if (st_spectform == 0) { amplitude = 1.0; } // Band
                        if (st_spectform == 1)
                        {
                            amplitude = std::abs(parab_prefact * (k - kc) * (k - kc) + 1.0);
                        } // Parabola

                        // note: power spectrum ~ amplitude^2 (1D), amplitude^2 * 2pi k (2D), amplitude^2 * 4pi k^2 (3D)
                        amplitude = 2.0 * std::sqrt(amplitude) * std::pow((kc / k), 0.5 * (ndim - 1));

                        d.stNModes += 1;
                        d.stAmpl[d.stNModes] = amplitude;

                        d.stMode[ndim * d.stNModes]     = kx;
                        d.stMode[ndim * d.stNModes + 1] = ky;
                        d.stMode[ndim * d.stNModes + 2] = kz;

                        if (ndim > 1)
                        {
                            d.stNModes += 1;

                            d.stAmpl[d.stNModes] = amplitude;

                            d.stMode[ndim * d.stNModes]     = kx;
                            d.stMode[ndim * d.stNModes + 1] = -ky;
                            d.stMode[ndim * d.stNModes + 2] = kz;
                        }

                        if (ndim > 2)
                        {
                            d.stNModes += 1;

                            d.stAmpl[d.stNModes] = amplitude;

                            d.stMode[ndim * d.stNModes]     = kx;
                            d.stMode[ndim * d.stNModes + 1] = ky;
                            d.stMode[ndim * d.stNModes + 2] = -kz;

                            d.stNModes += 1;

                            d.stAmpl[d.stNModes] = amplitude;

                            d.stMode[ndim * d.stNModes]     = kx;
                            d.stMode[ndim * d.stNModes + 1] = -ky;
                            d.stMode[ndim * d.stNModes + 2] = -kz;
                        }

                        if (d.stNModes % 1000 == 0)
                        {
                            std::cout << " ..." << d.stNModes << " of total " << st_tot_nmodes << " modes generated..."
                                      << std::endl;
                        }
                    } // in k range
                }     // ikz
            }         // iky
        }             // ikx
    }

    if (st_spectform == 2)
    {
        std::cout << "There would be " << st_tot_nmodes
                  << " driving modes, if k-space were fully sampled (st_angles_exp = 2.0)..." << std::endl;
        std::cout << "Here we are using st_angles_exp = " << st_angles_exp << std::endl;

        // loop between smallest and largest k
        ikmin = std::max(1, int(st_stirmin * Lx / twopi + 0.5));
        ikmax = int(st_stirmax * Lx / twopi + 0.5);

        std::cout << "Generating driving modes within k = [ " << ikmin << " , " << ikmax << " ]" << std::endl;

        for (int ik = ikmin; ik <= ikmax; ik++)
        {
            nang = std::pow(2, ndim) * ceil(std::pow(ik, st_angles_exp));
            std::cout << "ik = " << ik << " , number of angles = " << nang << std::endl;
            for (int iang = 1; iang <= nang; iang++)
            {
                phi = twopi * sph::ran1s<T>(st_seed); // phi = [0,2pi] sample the whole sphere
                if (ndim == 1)
                {
                    if (phi < twopi / 2) { phi = 0.0; }
                    if (phi >= twopi / 2) { phi = twopi / 2.0; }
                }

                theta = twopi / 4.0;
                if (ndim > 2)
                {
                    theta = std::acos(1.0 - 2.0 * sph::ran1s<T>(st_seed));
                } // theta = [0,pi] sample the whole sphere

                rand = ik + sph::ran1s<T>(st_seed) - 0.5;
                kx   = twopi * std::round(rand * std::sin(theta) * std::cos(phi)) / Lx;
                ky   = 0.0;
                if (ndim > 1) { ky = twopi * std::round(rand * std::sin(theta) * std::sin(phi)) / Ly; }
                kz = 0.0;
                if (ndim > 2) { kz = twopi * std::round(rand * std::cos(theta)) / Lz; }

                k = std::sqrt(kx * kx + ky * ky + kz * kz);

                if ((k >= st_stirmin) && (k <= st_stirmax))
                {
                    if ((d.stNModes + 1 + std::pow(2, ndim - 1)) > st_maxmodes)
                    {
                        std::cout << "init_stir:  number of modes: = " << d.stNModes + 1
                                  << " maxstirmodes = " << st_maxmodes << std::endl;
                        std::cout << "Too many stirring modes" << std::endl;
                        break;
                    }

                    amplitude = std::pow(k / kc, st_power_law_exp); // Power law

                    // note: power spectrum ~ amplitude^2 (1D), amplitude^2 * 2pi k (2D), amplitude^2 * 4pi k^2 (3D)
                    // ...and correct for the number of angles sampled relative to the full sampling (k^2 per k-shell in
                    // 3D)
                    amplitude = std::sqrt(amplitude * (std::pow(ik, ndim - 1) * 4.0 * (std::sqrt(3.0)) / nang)) *
                                std::pow(kc / k, (ndim - 1) / 2.0);

                    d.stNModes = d.stNModes + 1;

                    d.stAmpl[d.stNModes] = amplitude;

                    d.stMode[ndim * d.stNModes]     = kx;
                    d.stMode[ndim * d.stNModes + 1] = ky;
                    d.stMode[ndim * d.stNModes + 2] = kz;

                    if ((d.stNModes + 1) % 1000 == 0)
                    {
                        std::cout << "... " << d.stNModes << " modes generated..." << std::endl;
                    }

                } // in k range
            }     // loop over angles
        }         // loop over k
    }             // st_spectform .eq. 2
    d.stNModes += 1;
}

std::map<std::string, double> TurbulenceConstants()
{
    return {{"stSolWeight", 0.5},
            {"stMaxModes", 100000},
            {"Lbox", 1.0},
            {"stMachVelocity", 0.3e0},
            {"dim", 3},
            {"firstTimeStep", 1e-4},
            {"epsilon", 1e-15},
            {"stSeedIni", 251299},
            {"stSpectForm", 2},
            {"mTotal", 1.0},
            {"powerLawExp", 5 / 3},
            {"anglesExp", 2.0}};
}

template<class Dataset>
void initTurbulenceFields(Dataset& d, const std::map<std::string, double>& constants)
{
    using T = typename Dataset::RealType;

    size_t   ng0           = 100;
    T        firstTimeStep = constants.at("firstTimeStep");
    T        eps           = constants.at("epsilon");
    size_t   stMaxModes    = constants.at("stMaxModes");
    T        Lbox          = constants.at("Lbox");
    T        velocity      = constants.at("stMachVelocity");
    long int seed          = constants.at("stSeedIni");
    size_t   stSpectForm   = constants.at("stSpectForm");
    T        powerLawExp   = constants.at("powerLawExp");
    T        anglesExp     = constants.at("anglesExp");
    T        mPart         = constants.at("mTotal") / d.numParticlesGlobal;
    T        hInit         = std::cbrt(3.0 / (4. * M_PI) * ng0 * std::pow(Lbox, 3) / d.numParticlesGlobal) * 0.5;

    T twopi     = 2.0 * M_PI;
    T stEnergy  = 5.0e-3 * std::pow(velocity, 3) / Lbox;
    T stStirMin = (1.0 - eps) * twopi / Lbox;
    T stStirMax = (3.0 + eps) * twopi / Lbox;

    d.ndim        = constants.at("dim");
    d.stDecay     = Lbox / (2.0 * velocity);
    d.stSolWeight = constants.at("stSolWeight");
    d.stSeed      = seed;

    d.stAmpl.resize(stMaxModes);
    d.stMode.resize(stMaxModes * d.ndim);

    stir_init(d,
              Lbox,
              Lbox,
              Lbox,
              stMaxModes,
              stEnergy,
              stStirMax,
              stStirMin,
              d.ndim,
              d.stSeed,
              stSpectForm,
              powerLawExp,
              anglesExp);

    std::cout << "Total Number of Stirring Modes: " << d.stNModes << std::endl;
    d.stAmpl.resize(d.stNModes);
    d.stMode.resize(d.stNModes * d.ndim);
    d.stOUPhases.resize(6 * d.stNModes);

    sph::st_ounoiseinit(d.stOUPhases, 6 * d.stNModes, d.stOUvar, d.stSeed);

    std::fill(d.m.begin(), d.m.end(), mPart);
    std::fill(d.du_m1.begin(), d.du_m1.end(), 0.0);
    std::fill(d.h.begin(), d.h.end(), hInit);
    std::fill(d.mui.begin(), d.mui.end(), 10.0);
    std::fill(d.alpha.begin(), d.alpha.end(), d.alphamin);
    std::fill(d.u.begin(), d.u.end(), 1000.0);

    d.minDt    = firstTimeStep;
    d.minDt_m1 = firstTimeStep;

#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < d.x.size(); i++)
    {
        d.vx[i]   = 0.;
        d.vy[i]   = 0.;
        d.vz[i]   = 0.;
        d.x_m1[i] = d.x[i] - d.vx[i] * firstTimeStep;
        d.y_m1[i] = d.y[i] - d.vy[i] * firstTimeStep;
        d.z_m1[i] = d.z[i] - d.vz[i] * firstTimeStep;
    }
}

template<class Dataset>
class TurbulenceGlass : public ISimInitializer<Dataset>
{
    std::string                   glassBlock;
    std::map<std::string, double> constants_;

public:
    TurbulenceGlass(std::string initBlock)
        : glassBlock(initBlock)
    {
        constants_ = TurbulenceConstants();
    }

    cstone::Box<typename Dataset::RealType> init(int rank, int numRanks, size_t cbrtNumPart, Dataset& d) const override
    {
        using KeyType = typename Dataset::KeyType;
        using T       = typename Dataset::RealType;

        std::vector<T> xBlock, yBlock, zBlock;
        fileutils::readTemplateBlock(glassBlock, xBlock, yBlock, zBlock);
        size_t blockSize = xBlock.size();

        size_t multiplicity  = std::rint(cbrtNumPart / std::cbrt(blockSize));
        d.numParticlesGlobal = multiplicity * multiplicity * multiplicity * blockSize;

        cstone::Box<T> globalBox(-0.5, 0.5, true);
        auto [keyStart, keyEnd] = partitionRange(cstone::nodeRange<KeyType>(0), rank, numRanks);
        assembleCube<T>(keyStart, keyEnd, globalBox, multiplicity, xBlock, yBlock, zBlock, d.x, d.y, d.z);

        d.resize(d.x.size());

        initTurbulenceFields(d, constants_);

        return globalBox;
    }

    const std::map<std::string, double>& constants() const override { return constants_; }
};

} // namespace sphexa
