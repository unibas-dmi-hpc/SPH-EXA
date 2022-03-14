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
 * @brief Noh implosion simulation data initialization
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#pragma once

#include <map>

#include "cstone/sfc/box.hpp"

#include "isim_init.hpp"
#include "grid.hpp"

namespace sphexa
{

template<class Dataset>
class NohGrid : public ISimInitializer<Dataset>
{
    std::map<std::string, double> constants_{{"r0", 0},
                                             {"r1", 0.5},
                                             {"mTotal", 1.},
                                             {"dim", 3},
                                             {"gamma", 5.0 / 3.0},
                                             {"rho0", 1.},
                                             {"u0", 1e-20},
                                             {"p0", 0.},
                                             {"vr0", -1.},
                                             {"cs0", 0.},
                                             {"firstTimeStep", 1e-4}};

public:
    cstone::Box<typename Dataset::RealType> init(int rank, int numRanks, Dataset& d) const override
    {
        using T = typename Dataset::RealType;
        d.n     = d.side * d.side * d.side;

        auto [first, last] = partitionRange(d.n, rank, numRanks);
        d.count            = last - first;

        resize(d, d.count);

        if (rank == 0)
        {
            std::cout << "Approx: " << d.count * (d.data().size() * 64.) / (8. * 1000. * 1000. * 1000.)
                      << "GB allocated on rank 0." << std::endl;
        }

        T r = constants_.at("r1");

        regularGrid(r, d.side, first, last, d.x, d.y, d.z);
        initFields(d);

        return cstone::Box<T>(-r, r, false);
    }

    const std::map<std::string, double>& constants() const override { return constants_; }

private:
    void initFields(Dataset& d) const
    {
        using T = typename Dataset::RealType;

        double r1            = constants_.at("r1");
        double step          = (2. * r1) / d.side;
        double hIni          = 1.5 * step;
        double mPart         = constants_.at("mTotal") / d.n;
        double firstTimeStep = constants_.at("firstTimeStep");

        std::fill(d.m.begin(), d.m.end(), mPart);
        std::fill(d.h.begin(), d.h.end(), hIni);
        std::fill(d.du_m1.begin(), d.du_m1.end(), 0.0);
        std::fill(d.mui.begin(), d.mui.end(), 10.0);
        std::fill(d.dt.begin(), d.dt.end(), firstTimeStep);
        std::fill(d.dt_m1.begin(), d.dt_m1.end(), firstTimeStep);
        std::fill(d.alpha.begin(), d.alpha.end(), d.alphamin);
        d.minDt = firstTimeStep;

#pragma omp parallel for schedule(static)
        for (size_t i = 0; i < d.count; i++)
        {
            T radius = std::sqrt((d.x[i] * d.x[i]) + (d.y[i] * d.y[i]) + (d.z[i] * d.z[i]));
            radius   = std::max(radius, 1e-10);

            d.u[i] = constants_.at("u0");

            d.vx[i] = constants_.at("vr0") * (d.x[i] / radius);
            d.vy[i] = constants_.at("vr0") * (d.y[i] / radius);
            d.vz[i] = constants_.at("vr0") * (d.z[i] / radius);

            d.x_m1[i] = d.x[i] - d.vx[i] * firstTimeStep;
            d.y_m1[i] = d.y[i] - d.vy[i] * firstTimeStep;
            d.z_m1[i] = d.z[i] - d.vz[i] * firstTimeStep;
        }
    }
};

} // namespace sphexa
