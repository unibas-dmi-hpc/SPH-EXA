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
 * @brief Initial placement of particles on a regular grid
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#pragma once

#include "cstone/sfc/box.hpp"
#include "cstone/util/array.hpp"
#include "cstone/util/gsl-lite.hpp"

namespace sphexa
{

/*! @brief partition a range R into N segments
 *
 * @param R   the length of the range to be divided
 * @param i   the index of the segment to be computed
 * @param N   the total number of segments
 * @return    the start and end index (subrange of R) of the i-th segment
 */
auto partitionRange(size_t R, size_t i, size_t N)
{
    size_t s = R / N;
    size_t r = R % N;
    if (i < r)
    {
        size_t start = (s + 1) * i;
        size_t end   = start + s + 1;
        return std::make_tuple(start, end);
    }
    else
    {
        size_t start = (s + 1) * r + s * (i - r);
        size_t end   = start + s;
        return std::make_tuple(start, end);
    }
}

/*! @brief create regular cubic grid centered on (0,0,0), spanning [-r, r)^3, in the x,y,z arrays
 *
 * @tparam     Vector
 * @param[in]  r      half the cube side-length
 * @param[in]  side   number of particles along each dimension
 * @param[in]  first  index in [0, side^3] of first particle add to x,y,z
 * @param[in]  last   index in [first, side^3] of last particles to add to x,y,z
 * @param[out] x      output coordinates, length = last - first
 * @param[out] y
 * @param[out] z
 */
template<class Vector>
void regularGrid(double r, size_t side, size_t first, size_t last, Vector& x, Vector& y, Vector& z)
{
    double step = (2. * r) / side;

#pragma omp parallel for
    for (size_t i = first / (side * side); i < last / (side * side) + 1; ++i)
    {
        double lz = -r + (i * step);

        for (size_t j = 0; j < side; ++j)
        {
            double ly = -r + (j * step);

            for (size_t k = 0; k < side; ++k)
            {
                size_t lindex = (i * side * side) + (j * side) + k;

                if (first <= lindex && lindex < last)
                {
                    double lx = -r + (k * step);

                    z[lindex - first] = lz;
                    y[lindex - first] = ly;
                    x[lindex - first] = lx;
                }
            }
        }
    }
}

/*! @brief intersection of a box with a regular grid
 *
 * @tparam T   float or double
 * @param  a   sub box of the unit-cube
 * @param  m   number of grid-segments per dimension
 * @return     two integer triples marking which grid cells intersected with @p a
 */
template<class T>
auto gridIntersection(const cstone::Box<T> a, int m)
{
    auto                l = util::array<T, 3>{a.xmin(), a.ymin(), a.zmin()} * T(m);
    util::array<int, 3> lowerIdx{int(l[0]), int(l[1]), int(l[2])};

    auto                u = util::array<T, 3>{a.xmax(), a.ymax(), a.zmax()} * T(m);
    util::array<int, 3> upperIdx{int(std::ceil(u[0])), int(std::ceil(u[1])), int(std::ceil(u[2]))};

    return std::make_tuple(lowerIdx, upperIdx);
}

template<class T>
cstone::Vec3<T> scaleBlockToGlobal(cstone::Vec3<T> uX, cstone::Vec3<int> gridIdx, int m,
                                   const cstone::Box<T>& globalBox)
{

    cstone::Vec3<T> blockOrigin{gridIdx[0] * globalBox.lx(), gridIdx[1] * globalBox.ly(), gridIdx[2] * globalBox.lz()};
    blockOrigin /= T(m);

    cstone::Vec3<T> globalOrigin{globalBox.xmin(), globalBox.ymin(), globalBox.zmin()};

    auto gX = uX;
    gX[0] *= globalBox.lx();
    gX[1] *= globalBox.ly();
    gX[2] *= globalBox.lz();
    gX /= m;

    gX += globalOrigin + blockOrigin;

    return gX;
}

template<class T, class Vector>
void extractBlock(const cstone::Box<T>& selectBox, const cstone::Box<T>& globalBox, cstone::Vec3<int> gridIdx, int m,
                  gsl::span<const T> xBlock, gsl::span<const T> yBlock, gsl::span<const T> zBlock, Vector& x, Vector& y,
                  Vector& z)
{
    cstone::Vec3<T> origin{globalBox.xmin(), globalBox.ymin(), globalBox.zmin()};

    for (size_t i = 0; i < xBlock.size(); ++i)
    {
        auto sX = scaleBlockToGlobal({xBlock[i], yBlock[i], zBlock[i]}, gridIdx, m, globalBox);

        bool select = (selectBox.xmin() <= sX[0] && sX[0] < selectBox.xmax()) &&
                      (selectBox.ymin() <= sX[1] && sX[1] < selectBox.ymax()) &&
                      (selectBox.zmin() <= sX[2] && sX[2] < selectBox.zmax());
        if (select)
        {
            x.push_back(sX[0]);
            y.push_back(sX[1]);
            z.push_back(sX[2]);
        }
    }
}

} // namespace sphexa
