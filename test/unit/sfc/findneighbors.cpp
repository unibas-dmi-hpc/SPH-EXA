#include <iostream>
#include <random>
#include <vector>

#include "gtest/gtest.h"

#include "sfc/findneighbors.hpp"

template<class T, class I>
class RandomCoordinates
{
public:

    RandomCoordinates(int n, T x1, T x2, T y1, T y2, T z1, T z2)
        : xmin(x1), xmax(x2), ymin(y1), ymax(y2), zmin(z1), zmax(z2),
        x_(n), y_(n), z_(n), codes_(n)
    {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<T> disX(xmin, xmax);
        std::uniform_real_distribution<T> disY(ymin, ymax);
        std::uniform_real_distribution<T> disZ(zmin, zmax);

        auto randX = [&disX, &gen]() { return disX(gen); };
        auto randY = [&disY, &gen]() { return disY(gen); };
        auto randZ = [&disZ, &gen]() { return disZ(gen); };

        std::generate(begin(x_), end(x_), randX);
        std::generate(begin(y_), end(y_), randY);
        std::generate(begin(z_), end(z_), randZ);

        sphexa::computeMortonCodes(begin(x_), end(x_), begin(y_), begin(z_), begin(codes_),
                                   xmin, xmax, ymin, ymax, zmin, zmax);

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

    T xmin, xmax, ymin, ymax, zmin, zmax;
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

    std::vector<T> x_, y_, z_;
    std::vector<I> codes_;
};

template<class T>
static inline T distancesq(const T x1, const T y1, const T z1, const T x2, const T y2, const T z2)
{
    T xx = x1 - x2;
    T yy = y1 - y2;
    T zz = z1 - z2;

    return xx * xx + yy * yy + zz * zz;
}

//! \brief simple N^2 all-to-all neighbor search
template<class T>
void all2allNeighbors(const T* x, const T* y, const T* z, int n, T radius,
                      int *neighbors, int *neighborsCount, int ngmax)
{
    for (int i = 0; i < n; ++i)
    {
        T xi = x[i], yi = y[i], zi = z[i];

        int ngcount = 0;
        for (int j = i + 1; j < n; ++j)
        {
            if (ngcount < ngmax && distancesq(xi, yi, zi, x[j], y[j], z[j]) < radius)
            {
                neighbors[i * ngmax + ngcount++] = j;
            }
        }
        neighborsCount[i] = ngcount;
    }
}

void sortNeighbors(int *neighbors, int *neighborsCount, int n, int ngmax)
{
    for (int i = 0; i < n; ++i)
    {
        std::sort(neighbors + i*ngmax, neighbors + i*ngmax + neighborsCount[i]);
    }
}

TEST(FindNeighbors, treeLevel)
{
    EXPECT_EQ(3, sphexa::treeLevel(0.124, 0., 1., 0., 1., 0., 1.));
    EXPECT_EQ(2, sphexa::treeLevel(0.126, 0., 1., 0., 1., 0., 1.));
}

TEST(FindNeighbors, fixtureIsSorted)
{
    using real = double;
    using CodeType = unsigned;
    int n = 10;

    real xmin = 0, xmax = 1, ymin = 0, ymax = 1, zmin = 0, zmax = 1;
    RandomCoordinates<real, CodeType> c(n, xmin, xmax, ymin, ymax, zmin, zmax);

    std::vector<CodeType> testCodes(n);
    sphexa::computeMortonCodes(begin(c.x()), end(c.x()), begin(c.y()), begin(c.z()),
                               begin(testCodes), xmin, xmax, ymin, ymax, zmin, zmax);

    EXPECT_EQ(testCodes, c.mortonCodes());

    std::vector<CodeType> testCodesSorted = testCodes;
    std::sort(begin(testCodesSorted), end(testCodesSorted));

    EXPECT_EQ(testCodes, testCodesSorted);
}

TEST(FindNeighbors, randomUniform)
{
    using real = double;
    using CodeType = unsigned;

    int ngmax = 500;
    int n = 1000;

    real xmin = 0, xmax = 1, ymin = 0, ymax = 1, zmin = 0, zmax = 1;
    RandomCoordinates<real, CodeType> coords(n, xmin, xmax, ymin, ymax, zmin, zmax);

    std::vector<int> neighborsRef(n * ngmax), neighborsCountRef(n);

    real radius = 0.124;
    all2allNeighbors(coords.x().data(), coords.y().data(), coords.z().data(), n, radius,
                     neighborsRef.data(), neighborsCountRef.data(), ngmax);
    sortNeighbors(neighborsRef.data(), neighborsCountRef.data(), n, ngmax);

    //std::copy(begin(neighbors), begin(neighbors) + neighborsCount[0],
    //          std::ostream_iterator<int>(std::cout, " "));
    //std::copy(begin(neighborsCountRef), end(neighborsCountRef),
    //          std::ostream_iterator<int>(std::cout, " "));
    //std::cout << std::endl;

    //real avg = std::accumulate(begin(neighborsCountRef), end(neighborsCountRef), 0) / real(n);
    //std::cout << "avg count " << avg << std::endl;

    EXPECT_EQ(coords.x().size(), n);
}
