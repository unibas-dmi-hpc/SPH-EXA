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
 * @brief Test-case simulation data initialization
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#pragma once

#include "sedov/sedov_generator.hpp"
#include "grid.hpp"

namespace sphexa
{

template<class Dataset>
class ISimInitializer
{
public:
    virtual void init(int rank, int numRanks, Dataset& d) const = 0;

    virtual ~ISimInitializer() = default;
};

template<class Dataset>
class SedovGrid : public ISimInitializer<Dataset>
{
public:
    void init(int rank, int numRanks, Dataset& d) const override
    {
        d.n = d.side * d.side * d.side;

        auto [first, last] = partitionRange(d.n, rank, numRanks);
        d.count            = last - first;

        resize(d, d.count);

        if (rank == 0)
        {
            std::cout << "Approx: " << d.count * (d.data().size() * 64.) / (8. * 1000. * 1000. * 1000.)
                      << "GB allocated on rank 0." << std::endl;
        }

        regularGrid(SedovConstants::r1, d.side, first, last, d.x, d.y, d.z);
        initFields(d);
    }

private:
    void initFields(Dataset& d) const
    {
        using T = typename Dataset::RealType;

        double step   = (2. * SedovConstants::r1) / d.side;
        double hIni   = 1.5 * step;
        double mPart  = SedovConstants::mTotal / d.n;
        double width  = SedovConstants::width;
        double width2 = width * width;

        double firstTimeStep = SedovConstants::firstTimeStep;

        std::fill(d.m.begin(), d.m.end(), mPart);
        std::fill(d.h.begin(), d.h.end(), hIni);
        std::fill(d.du_m1.begin(), d.du_m1.end(), 0.0);
        std::fill(d.mui.begin(), d.mui.end(), 10.0);
        std::fill(d.dt.begin(), d.dt.end(), firstTimeStep);
        std::fill(d.dt_m1.begin(), d.dt_m1.end(), firstTimeStep);
        std::fill(d.alpha.begin(), d.alpha.end(), d.alphamin);
        d.minDt = firstTimeStep;

        std::fill(d.vx.begin(), d.vx.end(), 0.0);
        std::fill(d.vy.begin(), d.vy.end(), 0.0);
        std::fill(d.vz.begin(), d.vz.end(), 0.0);

#pragma omp parallel for schedule(static)
        for (size_t i = 0; i < d.count; i++)
        {
            T xi = d.x[i];
            T yi = d.y[i];
            T zi = d.z[i];
            T r2 = xi * xi + yi * yi + zi * zi;

            d.u[i] = SedovConstants::ener0 * exp(-(r2 / width2)) + SedovConstants::u0;

            d.x_m1[i] = xi - d.vx[i] * firstTimeStep;
            d.y_m1[i] = yi - d.vy[i] * firstTimeStep;
            d.z_m1[i] = zi - d.vz[i] * firstTimeStep;
        }
    }
};

} // namespace sphexa
