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
 * \brief particle count histograms along x,y,z dimensions
 *
 * \author Sebastian Keller <sebastian.f.keller@gmail.com>
 *
 * This code can be used to compute histograms of particle counts
 * along x,y,z dimensions. I.e. for a 32-bit Morton SFC octree, there
 * are 2^10=1024 bins along each dimension. Computing the particle
 * counts for those bins can be used to decompose a domain into
 * n_x * n_y * n_z subdomains, with n_x * n_y * n_z being a factorization
 * of the total rank count and n_i splits along dimension i.
 *
 * This generally leads to fewer halos than assigning a single Morton code
 * range along the SFC to each rank, because the SFC has spatial discontinuities.
 * It however leads to more octree nodes being split across domains which is a
 * disadvantage for FMM schemes.
 *
 * Since this code is not currently used, it lives in the testing space for now.
 *
 * \author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#include <algorithm>
#include <numeric>

#include "gtest/gtest.h"


#include "sfc/mortoncode.hpp"

namespace sphexa
{

template <class I>
struct CompareX
{
    inline bool operator()(I a, I b) { return decodeMortonX(a) < decodeMortonX(b); }
};

template <class I>
void histogramX(const I *codesStart, const I *codesEnd, std::array<unsigned, 1u << maxTreeLevel<I>{}> &histogram)
{
    constexpr int nBins = 1u << maxTreeLevel<I>{};

    for (int bin = 0; bin < nBins; ++bin)
    {
        auto lower = std::lower_bound(codesStart, codesEnd, bin, CompareX<I>{});
        auto upper = std::upper_bound(codesStart, codesEnd, bin, CompareX<I>{});
        histogram[bin] = upper - lower;
    }
}

} // namespace sphexa


TEST(DomainDecomposition, binX)
{
    using CodeType = unsigned;
    std::vector<CodeType> codes;

    constexpr unsigned nBins = 1u<<sphexa::maxTreeLevel<CodeType>{};

    for (unsigned i = 0; i < nBins/2; ++i)
    {
        codes.push_back(sphexa::codeFromBox<CodeType>(i,0,0, sphexa::maxTreeLevel<CodeType>{}));
        codes.push_back(sphexa::codeFromBox<CodeType>(i,0,1, sphexa::maxTreeLevel<CodeType>{}));
        codes.push_back(sphexa::codeFromBox<CodeType>(i+512,0,1, sphexa::maxTreeLevel<CodeType>{}));
        codes.push_back(sphexa::codeFromBox<CodeType>(i+512,1,1, sphexa::maxTreeLevel<CodeType>{}));
    }

    std::array<unsigned, nBins> xCounts{};

    std::sort(codes.data(), codes.data()+codes.size(), sphexa::CompareX<CodeType>{});
    sphexa::histogramX(codes.data(), codes.data()+codes.size(), xCounts);

    for (auto c : xCounts)
    {
        EXPECT_EQ(c, 2);
    }
}