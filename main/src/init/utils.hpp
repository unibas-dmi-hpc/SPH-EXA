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
 * @author Lukas Schmidt
 */

#include <numeric>
#include <vector>

#include "cstone/primitives/gather.hpp"
#include "cstone/sfc/sfc.hpp"

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

/*!@brief remove every n-th element of a sequence
 * @param   n  factor to reduce density
 * @param   x  a vector
 * @return     vector with 1/n-th the elements of @p v
 */
template<class T>
auto skipElements(size_t n, const std::vector<T>& x)
{
    size_t newSize = x.size() / n;

    std::vector<T> xSmall(newSize);
    for (size_t i = 0; i < newSize; i++)
    {
        xSmall[i] = x[2 * i];
    }

    return xSmall;
}

//! @brief reduce density of @a SFC sorted template glass block (x,y,z) coordinates by factor of @p n
template<class T>
auto makeLessDenseTemplate(size_t n, const std::vector<T>& x, const std::vector<T>& y, const std::vector<T>& z)
{
    assert(x.size() == y.size() == z.size());
    return std::make_tuple(skipElements(n, x), skipElements(n, y), skipElements(n, z));
}

} // namespace sphexa