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

/*!@file
 * @brief utilities for initial condition generation
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#pragma once

#include <numeric>
#include <string>
#include <vector>

#include "cstone/primitives/gather.hpp"
#include "cstone/sfc/sfc.hpp"
#include "io/ifile_io.hpp"

namespace sphexa
{

//! @brief sort x,y,z coordinates in the unit cube by SFC keys
template<class KeyType, class T>
void sortBySfcKey(std::vector<T>& x, std::vector<T>& y, std::vector<T>& z)
{
    assert(x.size() == y.size() == z.size());
    size_t blockSize = x.size();

    cstone::Box<T> box(0, 1);

    std::vector<KeyType> keys(blockSize);
    computeSfcKeys(x.data(), y.data(), z.data(), cstone::sfcKindPointer(keys.data()), blockSize, box);

    std::vector<cstone::LocalIndex> sfcOrder(blockSize);
    std::iota(begin(sfcOrder), end(sfcOrder), cstone::LocalIndex(0));
    cstone::sort_by_key(begin(keys), end(keys), begin(sfcOrder));

    std::vector<T> buffer(blockSize);
    cstone::gather<cstone::LocalIndex>(sfcOrder, x.data(), buffer.data());
    std::swap(x, buffer);
    cstone::gather<cstone::LocalIndex>(sfcOrder, y.data(), buffer.data());
    std::swap(y, buffer);
    cstone::gather<cstone::LocalIndex>(sfcOrder, z.data(), buffer.data());
    std::swap(z, buffer);
}

//! @brief read x,y,z coordinates from an H5Part file (at step 0)
template<class Vector>
void readTemplateBlock(const std::string& block, IFileReader* reader, Vector& x, Vector& y, Vector& z)
{
    reader->setStep(block, -1, FileMode::independent);
    size_t blockSize = reader->numParticles();
    x.resize(blockSize);
    y.resize(blockSize);
    z.resize(blockSize);

    reader->readField("x", x.data());
    reader->readField("y", y.data());
    reader->readField("z", z.data());

    reader->closeStep();
}

} // namespace sphexa