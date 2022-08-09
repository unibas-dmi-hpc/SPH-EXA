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
 * @brief Random coordinates generation for testing
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#pragma once

#include <algorithm>
#include <random>
#include <vector>

#include "cstone/findneighbors.hpp"
#include "cstone/primitives/gather.hpp"
#include "cstone/sfc/sfc.hpp"
#include "cstone/tree/definitions.h"
#include "cstone/util/gsl-lite.hpp"

namespace cstone
{

template<class Integer>
std::vector<Integer> makeRandomUniformKeys(size_t numKeys, int seed = 42)
{
    Integer maxCoord = nodeRange<Integer>(0) - 1;
    std::mt19937 gen(seed);
    std::uniform_int_distribution<Integer> distribution(0, maxCoord);

    auto randInt = [&distribution, &gen]() { return distribution(gen); };
    std::vector<Integer> ret(numKeys);
    std::generate(ret.begin(), ret.end(), randInt);
    std::sort(ret.begin(), ret.end());

    return ret;
}

template<class Integer>
std::vector<Integer> makeRandomGaussianKeys(size_t numKeys, int seed = 42)
{
    Integer maxCoord = nodeRange<Integer>(0) - 1;
    std::mt19937 gen(seed);
    std::normal_distribution<double> distribution(double(maxCoord) / 2, double(maxCoord) / 5);

    auto randInt = [&distribution, &gen, maxCoord]()
    {
        auto x = Integer(distribution(gen));
        // we can't cut down x to maxCoord in case it's too big, otherwise there will be too many keys in the last cell
        while (x > maxCoord)
        {
            x = Integer(distribution(gen));
        }
        return x;
    };

    std::vector<Integer> ret(numKeys);
    std::generate(ret.begin(), ret.end(), randInt);
    std::sort(ret.begin(), ret.end());

    return ret;
}

template<class T, class KeyType_>
class RandomCoordinates
{
public:
    using KeyType = KeyType_;
    using Integer = typename KeyType::ValueType;

    RandomCoordinates(size_t n, Box<T> box, int seed = 42)
        : box_(std::move(box))
        , x_(n)
        , y_(n)
        , z_(n)
        , codes_(n)
    {
        // std::random_device rd;
        std::mt19937 gen(seed);
        std::uniform_real_distribution<T> disX(box_.xmin(), box_.xmax());
        std::uniform_real_distribution<T> disY(box_.ymin(), box_.ymax());
        std::uniform_real_distribution<T> disZ(box_.zmin(), box_.zmax());

        auto randX = [&disX, &gen]() { return disX(gen); };
        auto randY = [&disY, &gen]() { return disY(gen); };
        auto randZ = [&disZ, &gen]() { return disZ(gen); };

        std::generate(begin(x_), end(x_), randX);
        std::generate(begin(y_), end(y_), randY);
        std::generate(begin(z_), end(z_), randZ);

        auto keyData = (KeyType*)(codes_.data());
        computeSfcKeys(x_.data(), y_.data(), z_.data(), keyData, n, box);

        std::vector<LocalIndex> sfcOrder(n);
        std::iota(begin(sfcOrder), end(sfcOrder), LocalIndex(0));
        sort_by_key(begin(codes_), end(codes_), begin(sfcOrder));

        std::vector<T> temp(x_.size());
        gather<LocalIndex>(sfcOrder, x_.data(), temp.data());
        swap(x_, temp);
        gather<LocalIndex>(sfcOrder, y_.data(), temp.data());
        swap(y_, temp);
        gather<LocalIndex>(sfcOrder, z_.data(), temp.data());
        swap(z_, temp);
    }

    const std::vector<T>& x() const { return x_; }
    const std::vector<T>& y() const { return y_; }
    const std::vector<T>& z() const { return z_; }
    const std::vector<Integer>& particleKeys() const { return codes_; }

private:
    Box<T> box_;
    std::vector<T> x_, y_, z_;
    std::vector<Integer> codes_;
};

template<class T, class KeyType_>
class RandomGaussianCoordinates
{
public:
    using KeyType = KeyType_;
    using Integer = typename KeyType::ValueType;

    RandomGaussianCoordinates(unsigned n, Box<T> box, int seed = 42)
        : box_(std::move(box))
        , x_(n)
        , y_(n)
        , z_(n)
        , codes_(n)
    {
        // std::random_device rd;
        std::mt19937 gen(seed);
        // random gaussian distribution at the center
        std::normal_distribution<T> disX((box_.xmax() + box_.xmin()) / 2, (box_.xmax() - box_.xmin()) / 5);
        std::normal_distribution<T> disY((box_.ymax() + box_.ymin()) / 2, (box_.ymax() - box_.ymin()) / 5);
        std::normal_distribution<T> disZ((box_.zmax() + box_.zmin()) / 2, (box_.zmax() - box_.zmin()) / 5);

        auto randX = [cmin = box_.xmin(), cmax = box_.xmax(), &disX, &gen]()
        { return std::max(std::min(disX(gen), cmax), cmin); };

        auto randY = [cmin = box_.ymin(), cmax = box_.ymax(), &disY, &gen]()
        { return std::max(std::min(disY(gen), cmax), cmin); };

        auto randZ = [cmin = box_.zmin(), cmax = box_.zmax(), &disZ, &gen]()
        { return std::max(std::min(disZ(gen), cmax), cmin); };

        std::generate(begin(x_), end(x_), randX);
        std::generate(begin(y_), end(y_), randY);
        std::generate(begin(z_), end(z_), randZ);

        auto keyData = (KeyType*)(codes_.data());
        computeSfcKeys(x_.data(), y_.data(), z_.data(), keyData, n, box);

        std::vector<LocalIndex> sfcOrder(n);
        std::iota(begin(sfcOrder), end(sfcOrder), LocalIndex(0));
        sort_by_key(begin(codes_), end(codes_), begin(sfcOrder));

        std::vector<T> temp(x_.size());
        gather<LocalIndex>(sfcOrder, x_.data(), temp.data());
        swap(x_, temp);
        gather<LocalIndex>(sfcOrder, y_.data(), temp.data());
        swap(y_, temp);
        gather<LocalIndex>(sfcOrder, z_.data(), temp.data());
        swap(z_, temp);
    }

    const std::vector<T>& x() const { return x_; }
    const std::vector<T>& y() const { return y_; }
    const std::vector<T>& z() const { return z_; }
    const std::vector<Integer>& particleKeys() const { return codes_; }

private:
    Box<T> box_;
    std::vector<T> x_, y_, z_;
    std::vector<Integer> codes_;
};

//! @brief can be used to calculate reasonable smoothing lengths for each particle
template<class KeyType, class Tc, class Th>
void adjustSmoothingLength(LocalIndex numParticles,
                           int ng0,
                           int ngmax,
                           const std::vector<Tc>& xGlob,
                           const std::vector<Tc>& yGlob,
                           const std::vector<Tc>& zGlob,
                           std::vector<Th>& hGlob,
                           const Box<Tc>& box)
{
    std::vector<KeyType> codesGlobal(numParticles);

    std::vector<Tc> x = xGlob;
    std::vector<Tc> y = yGlob;
    std::vector<Tc> z = zGlob;
    std::vector<Tc> h = hGlob;

    computeSfcKeys(x.data(), y.data(), z.data(), sfcKindPointer(codesGlobal.data()), numParticles, box);
    std::vector<LocalIndex> ordering(numParticles);
    std::iota(ordering.begin(), ordering.end(), LocalIndex(0));
    sort_by_key(codesGlobal.begin(), codesGlobal.end(), ordering.begin());

    std::vector<Tc> temp(x.size());
    gather<LocalIndex>(ordering, x.data(), temp.data());
    swap(temp, x);
    gather<LocalIndex>(ordering, y.data(), temp.data());
    swap(temp, y);
    gather<LocalIndex>(ordering, z.data(), temp.data());
    swap(temp, z);
    gather<LocalIndex>(ordering, h.data(), temp.data());
    swap(temp, h);

    std::vector<LocalIndex> inverseOrdering(numParticles);
    std::iota(inverseOrdering.begin(), inverseOrdering.end(), 0);
    std::vector<LocalIndex> orderCpy = ordering;
    sort_by_key(orderCpy.begin(), orderCpy.end(), inverseOrdering.begin());

    std::vector<int> neighbors(numParticles * ngmax);
    std::vector<int> neighborCounts(numParticles);

    // adjust h[i] such that each particle has between ng0/2 and ngmax neighbors
    for (LocalIndex i = 0; i < numParticles; ++i)
    {
        int iteration = 0;
        do
        {
            findNeighbors(i, x.data(), y.data(), z.data(), h.data(), box, sfcKindPointer(codesGlobal.data()),
                          neighbors.data() + i * ngmax, neighborCounts.data() + i, numParticles, ngmax);

            const Tc c0 = 7.0;
            int nn      = std::max(neighborCounts[i], 1);
            h[i]        = h[i] * 0.5 * pow(1.0 + (c0 * ng0) / nn, 1.0 / 3.0);
        } while ((neighborCounts[i] < ng0 / 4 || neighborCounts[i] >= ngmax) && iteration++ < 10);
    }

    for (LocalIndex i = 0; i < numParticles; ++i)
    {
        hGlob[i] = h[inverseOrdering[i]];
    }
}

} // namespace cstone