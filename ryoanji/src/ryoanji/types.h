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
 * @brief  Ryoanji tree cell types and utilities
 *
 * @author Rio Yokota <rioyokota@gsic.titech.ac.jp>
 */

#pragma once

#include "cstone/cuda/annotation.hpp"
#include "cstone/sfc/box.hpp"
#include "cstone/util/array.hpp"

namespace ryoanji
{

template<class T>
using Vec3 = cstone::Vec3<T>;

template<class T>
using Vec4 = cstone::Vec4<T>;

using TreeNodeIndex = cstone::TreeNodeIndex;
using LocalIndex    = cstone::LocalIndex;

template<size_t P>
struct TermSize : public stl::integral_constant<size_t, P * (P + 1) * (P + 2) / 6>
{
};

template<class T, size_t P>
using SphericalMultipole = util::array<T, TermSize<P>{}>;

template<int ArraySize, class = void>
struct ExpansionOrder
{
};

template<>
struct ExpansionOrder<TermSize<1>{}> : stl::integral_constant<size_t, 1>
{
};

template<>
struct ExpansionOrder<TermSize<2>{}> : stl::integral_constant<size_t, 2>
{
};

template<>
struct ExpansionOrder<TermSize<3>{}> : stl::integral_constant<size_t, 3>
{
};

template<>
struct ExpansionOrder<TermSize<4>{}> : stl::integral_constant<size_t, 4>
{
};

template<class T, size_t P>
void setZero(SphericalMultipole<T, P>& M)
{
    for (size_t i = 0; i < TermSize<P>{}; ++i)
    {
        M[i] = 0;
    }
}

class CellData
{
private:
    static const int CHILD_SHIFT = 29;
    static const int CHILD_MASK  = ~(0x7U << CHILD_SHIFT);
    static const int LEVEL_SHIFT = 27;
    static const int LEVEL_MASK  = ~(0x1FU << LEVEL_SHIFT);

    using DataType = util::array<unsigned, 4>;
    DataType data;

public:
    CellData() = default;

    HOST_DEVICE_FUN CellData(unsigned level, unsigned parent, unsigned body, unsigned nbody, unsigned child = 0,
                             unsigned nchild = 1)
    {
        unsigned parentPack = parent | (level << LEVEL_SHIFT);
        unsigned childPack  = child | ((nchild - 1) << CHILD_SHIFT);
        data                = DataType{parentPack, childPack, body, nbody};
    }

    HOST_DEVICE_FUN CellData(DataType data_)
        : data(data_)
    {
    }

    HOST_DEVICE_FUN int level() const { return data[0] >> LEVEL_SHIFT; }
    HOST_DEVICE_FUN int parent() const { return data[0] & LEVEL_MASK; }
    HOST_DEVICE_FUN int child() const { return data[1] & CHILD_MASK; }
    HOST_DEVICE_FUN int nchild() const { return (data[1] >> CHILD_SHIFT) + 1; }
    HOST_DEVICE_FUN int body() const { return data[2]; }
    HOST_DEVICE_FUN int nbody() const { return data[3]; }
    HOST_DEVICE_FUN bool isLeaf() const { return data[1] == 0; }
    HOST_DEVICE_FUN bool isNode() const { return !isLeaf(); }

    HOST_DEVICE_FUN void setParent(unsigned parent) { data[0] = parent | (level() << LEVEL_SHIFT); }
    HOST_DEVICE_FUN void setChild(unsigned child) { data[1] = child | ((nchild() - 1) << CHILD_SHIFT); }
    HOST_DEVICE_FUN void setBody(unsigned body_) { data[2] = body_; }
    HOST_DEVICE_FUN void setNBody(unsigned nbody_) { data[3] = nbody_; }
};

} // namespace ryoanji
