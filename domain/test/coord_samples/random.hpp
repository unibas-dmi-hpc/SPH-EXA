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

/*! \file
 * \brief Random coordinates generation for testing
 *
 * \author Sebastian Keller <sebastian.f.keller@gmail.com>
 */


#pragma once

#include <algorithm>
#include <random>
#include <vector>

#include "cstone/mortoncode.hpp"
#include "cstone/zorder.hpp"

using namespace cstone;

template<class T, class I>
class RandomCoordinates
{
public:

    RandomCoordinates(unsigned n, Box<T> box)
        : box_(std::move(box)), x_(n), y_(n), z_(n), codes_(n)
    {
        //std::random_device rd;
        std::mt19937 gen(42);
        std::uniform_real_distribution<T> disX(box_.xmin(), box_.xmax());
        std::uniform_real_distribution<T> disY(box_.ymin(), box_.ymax());
        std::uniform_real_distribution<T> disZ(box_.zmin(), box_.zmax());

        auto randX = [&disX, &gen]() { return disX(gen); };
        auto randY = [&disY, &gen]() { return disY(gen); };
        auto randZ = [&disZ, &gen]() { return disZ(gen); };

        std::generate(begin(x_), end(x_), randX);
        std::generate(begin(y_), end(y_), randY);
        std::generate(begin(z_), end(z_), randZ);

        computeMortonCodes(begin(x_), end(x_), begin(y_), begin(z_),
                                   begin(codes_), box);

        std::vector<I> mortonOrder(n);
        sort_invert(cbegin(codes_), cend(codes_), begin(mortonOrder));

        reorder(mortonOrder, codes_);
        reorder(mortonOrder, x_);
        reorder(mortonOrder, y_);
        reorder(mortonOrder, z_);
    }

    const std::vector<T>& x() const { return x_; }
    const std::vector<T>& y() const { return y_; }
    const std::vector<T>& z() const { return z_; }
    const std::vector<I>& mortonCodes() const { return codes_; }

private:

    Box<T> box_;
    std::vector<T> x_, y_, z_;
    std::vector<I> codes_;
};

template<class T, class I>
class RandomGaussianCoordinates
{
public:

    RandomGaussianCoordinates(unsigned n, Box<T> box)
        : box_(std::move(box)), x_(n), y_(n), z_(n), codes_(n)
    {
        //std::random_device rd;
        std::mt19937 gen(42);
        // random gaussian distribution at the center
        std::normal_distribution<T> disX((box_.xmax() + box_.xmin())/2, (box_.xmax() - box_.xmin())/5);
        std::normal_distribution<T> disY((box_.ymax() + box_.ymin())/2, (box_.ymax() - box_.ymin())/5);
        std::normal_distribution<T> disZ((box_.zmax() + box_.zmin())/2, (box_.zmax() - box_.zmin())/5);

        auto randX = [cmin=box_.xmin(), cmax=box_.xmax(), &disX, &gen]() { return std::max(std::min(disX(gen), cmax), cmin); };
        auto randY = [cmin=box_.ymin(), cmax=box_.ymax(), &disY, &gen]() { return std::max(std::min(disY(gen), cmax), cmin); };
        auto randZ = [cmin=box_.zmin(), cmax=box_.zmax(), &disZ, &gen]() { return std::max(std::min(disZ(gen), cmax), cmin); };

        std::generate(begin(x_), end(x_), randX);
        std::generate(begin(y_), end(y_), randY);
        std::generate(begin(z_), end(z_), randZ);

        computeMortonCodes(begin(x_), end(x_), begin(y_), begin(z_),
                                   begin(codes_), box);

        std::vector<I> mortonOrder(n);
        sort_invert(cbegin(codes_), cend(codes_), begin(mortonOrder));

        reorder(mortonOrder, codes_);
        reorder(mortonOrder, x_);
        reorder(mortonOrder, y_);
        reorder(mortonOrder, z_);
    }

    const std::vector<T>& x() const { return x_; }
    const std::vector<T>& y() const { return y_; }
    const std::vector<T>& z() const { return z_; }
    const std::vector<I>& mortonCodes() const { return codes_; }

private:

    Box<T> box_;
    std::vector<T> x_, y_, z_;
    std::vector<I> codes_;
};


template<class T, class I>
class RegularGridCoordinates
{
public:

    RegularGridCoordinates(unsigned gridSize)
        : box_(0,1), codes_(gridSize)
    {
        assert(isPowerOf8(gridSize));
        //x_.reserve(gridSize);
        //y_.reserve(gridSize);
        //z_.reserve(gridSize);

        unsigned n_ = 1u << log8ceil(gridSize);
        for (int i = 0; i < n_; ++i)
            for (int j = 0; j < n_; ++j)
                for (int k = 0; k < n_; ++k)
                {
                    x_.push_back(i);
                    y_.push_back(j);
                    z_.push_back(k);
                }

        box_ = Box<T>(0, n_);
        computeMortonCodes(begin(x_), end(x_), begin(y_), begin(z_),
                                   begin(codes_), box_);

        std::vector<I> mortonOrder(gridSize);
        sort_invert(cbegin(codes_), cend(codes_), begin(mortonOrder));

        reorder(mortonOrder, codes_);
        reorder(mortonOrder, x_);
        reorder(mortonOrder, y_);
        reorder(mortonOrder, z_);
    }

    const std::vector<T>& x() const { return x_; }
    const std::vector<T>& y() const { return y_; }
    const std::vector<T>& z() const { return z_; }
    const std::vector<I>& mortonCodes() const { return codes_; }

private:
    Box<T> box_;
    std::vector<T> x_, y_, z_;
    std::vector<I> codes_;
};