#pragma once

#include <algorithm>
#include <random>
#include <vector>

#include "sfc/mortoncode.hpp"
#include "sfc/zorder.hpp"

template<class T, class I>
class RandomCoordinates
{
public:

    RandomCoordinates(unsigned n, sphexa::Box<T> box)
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

        sphexa::computeMortonCodes(begin(x_), end(x_), begin(y_), begin(z_),
                                   begin(codes_), box);

        std::vector<I> mortonOrder(n);
        sphexa::sort_invert(cbegin(codes_), cend(codes_), begin(mortonOrder));

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

    template<class ValueType>
    void reorder(const std::vector<I>& ordering, std::vector<ValueType>& array)
    {
        std::vector<ValueType> tmp(array.size());
        for (std::size_t i = 0; i < array.size(); ++i)
        {
            tmp[i] = array[ordering[i]];
        }
        std::swap(tmp, array);
    }

    sphexa::Box<T> box_;
    std::vector<T> x_, y_, z_;
    std::vector<I> codes_;
};

